import asyncio
import fractions
import logging
import threading
import time

from aiortc import MediaStreamTrack
from aiortc.mediastreams import VIDEO_CLOCK_RATE, VIDEO_TIME_BASE, MediaStreamError
from av import VideoFrame

from .frame_processor import FrameProcessor
from .pipeline_manager import PipelineManager

logger = logging.getLogger(__name__)


class VideoProcessingTrack(MediaStreamTrack):
    kind = "video"

    def __init__(
        self,
        pipeline_manager: PipelineManager,
        fps: int = 30,
        initial_parameters: dict = None,
        notification_callback: callable = None,
        session_id: str | None = None,
        user_id: str | None = None,
        connection_id: str | None = None,
        connection_info: dict | None = None,
    ):
        super().__init__()
        self.pipeline_manager = pipeline_manager
        self.initial_parameters = initial_parameters or {}
        self.notification_callback = notification_callback
        self.session_id = session_id
        self.user_id = user_id
        self.connection_id = connection_id
        self.connection_info = connection_info
        # FPS variables (will be updated from FrameProcessor or input measurement)
        self.fps = fps
        self.frame_ptime = 1.0 / fps

        self.frame_processor = None
        self.input_task = None
        self.input_task_running = False
        self._paused = False
        self._paused_lock = threading.Lock()
        self._last_frame = None

        # Drain rate tracking for diagnostics
        self._drain_frames = 0
        self._drain_log_time: float | None = None
        self._drain_empty_waits = 0  # how many loops had no frame (queue empty)

        # Server-side input mode - when enabled, frames come from the backend
        # instead of WebRTC (no browser video track needed)
        self._input_source_enabled = False
        if initial_parameters:
            input_source = initial_parameters.get("input_source")
            if input_source and input_source.get("enabled"):
                self._input_source_enabled = True
                logger.info(
                    f"Input source mode enabled: {input_source.get('source_type')}"
                )

    async def input_loop(self):
        """Background loop that continuously feeds frames to the processor"""
        while self.input_task_running:
            try:
                input_frame = await self.track.recv()

                # Store raw VideoFrame for later processing (tracks input FPS internally)
                self.frame_processor.put(input_frame)

            except asyncio.CancelledError:
                break
            except Exception as e:
                # Stop the input loop on connection errors to avoid spam
                logger.error(f"Error in input loop, stopping: {e}")
                self.input_task_running = False
                break

    # Copied from https://github.com/livepeer/fastworld/blob/e649ef788cd33d78af6d8e1da915cd933761535e/backend/track.py#L267
    async def next_timestamp(self) -> tuple[int, fractions.Fraction]:
        """Override to control frame rate"""
        if self.readyState != "live":
            raise MediaStreamError

        if hasattr(self, "_stream_start"):
            # Calculate wait time based on current frame rate
            now = time.monotonic()
            time_since_last_frame = now - self._last_pts_time

            # Wait for the appropriate interval based on current FPS
            target_interval = self.frame_ptime  # Current frame period
            wait_time = target_interval - time_since_last_frame

            if wait_time > 0:
                await asyncio.sleep(wait_time)

            # Derive PTS from wall clock relative to stream start — eliminates
            # accumulated sleep drift from timestamp computation.
            self.timestamp = int(
                (time.monotonic() - self._stream_start) * VIDEO_CLOCK_RATE
            )
            self._last_pts_time = time.monotonic()
        else:
            self._stream_start = time.monotonic()
            self._last_pts_time = time.monotonic()
            self.timestamp = 0
            # Keep self.start for any legacy compatibility
            self.start = time.time()

        return self.timestamp, VIDEO_TIME_BASE

    def initialize_output_processing(self):
        if not self.frame_processor:
            self.frame_processor = FrameProcessor(
                pipeline_manager=self.pipeline_manager,
                initial_parameters=self.initial_parameters,
                notification_callback=self.notification_callback,
                session_id=self.session_id,
                user_id=self.user_id,
                connection_id=self.connection_id,
                connection_info=self.connection_info,
            )
            self.frame_processor.start()

    def initialize_input_processing(self, track: MediaStreamTrack):
        self.track = track
        self.input_task_running = True
        self.input_task = asyncio.create_task(self.input_loop())

    async def recv(self) -> VideoFrame:
        """Return the next available processed frame"""
        # Lazy initialization on first call
        self.initialize_output_processing()

        # Keep running while any input source is active
        while self.input_task_running or self._input_source_enabled:
            try:
                # Update FPS: use the FPS from the pipeline chain
                if self.frame_processor:
                    self.fps = self.frame_processor.get_fps()
                    self.frame_ptime = 1.0 / self.fps

                # If paused, wait for the appropriate frame interval before returning
                with self._paused_lock:
                    paused = self._paused

                frame = None
                if paused:
                    # When video is paused, return the last frame to freeze the playback video
                    frame = self._last_frame
                else:
                    # When video is not paused, get the next frame from the frame processor
                    frame_tensor = self.frame_processor.get()
                    if frame_tensor is not None:
                        frame = VideoFrame.from_ndarray(
                            frame_tensor.numpy(), format="rgb24"
                        )

                if frame is not None:
                    pts, time_base = await self.next_timestamp()
                    frame.pts = pts
                    frame.time_base = time_base

                    self._last_frame = frame

                    # Drain rate diagnostics — log every 5s
                    self._drain_frames += 1
                    now = time.monotonic()
                    if self._drain_log_time is None:
                        self._drain_log_time = now
                    elif now - self._drain_log_time >= 5.0:
                        actual_fps = self._drain_frames / (now - self._drain_log_time)
                        q = (
                            self.frame_processor.pipeline_processors[-1].output_queue
                            if (
                                self.frame_processor
                                and self.frame_processor.pipeline_processors
                            )
                            else None
                        )
                        q_info = f"{q.qsize()}/{q.maxsize}" if q else "n/a"
                        logger.info(
                            f"[track] drain: actual={actual_fps:.1f}fps "
                            f"target={self.fps:.1f}fps "
                            f"ptime={self.frame_ptime * 1000:.1f}ms "
                            f"queue={q_info} "
                            f"empty_polls={self._drain_empty_waits}"
                        )
                        if self._drain_empty_waits > 0:
                            freeze_pct = self._drain_empty_waits / max(
                                1, self._drain_frames + self._drain_empty_waits
                            )
                            if freeze_pct > 0.05:
                                logger.warning(
                                    f"[track] FREEZE RISK: queue empty {self._drain_empty_waits}x "
                                    f"({freeze_pct:.0%} of polls) — pipeline may be too slow to "
                                    f"sustain target={self.fps:.1f}fps"
                                )
                        self._drain_frames = 0
                        self._drain_empty_waits = 0
                        self._drain_log_time = now

                    return frame

                # No frame available — advance the PTS clock and return the last
                # frame as a freeze-frame.  This prevents the PTS from falling
                # behind wall-clock time: without this, when the next burst
                # arrives all frames get timestamped in rapid succession (PTS
                # snap-forward), which the browser decoder renders as a freeze
                # followed by a jump cut.
                self._drain_empty_waits += 1
                if self._last_frame is not None:
                    pts, time_base = await self.next_timestamp()
                    # Reformat creates a new VideoFrame so we don't mutate _last_frame's pts
                    freeze_frame = self._last_frame.reformat(
                        width=self._last_frame.width,
                        height=self._last_frame.height,
                        format=self._last_frame.format.name,
                    )
                    freeze_frame.pts = pts
                    freeze_frame.time_base = time_base
                    return freeze_frame
                await asyncio.sleep(0.002)

            except Exception as e:
                logger.error(f"Error getting processed frame: {e}")
                raise

        raise Exception("Track stopped")

    def pause(self, paused: bool):
        """Pause or resume the video track processing"""
        with self._paused_lock:
            self._paused = paused

        logger.info(f"Video track {'paused' if paused else 'resumed'}")

    async def stop(self):
        self.input_task_running = False
        self._input_source_enabled = False

        if self.input_task is not None:
            self.input_task.cancel()
            try:
                await self.input_task
            except asyncio.CancelledError:
                pass

        if self.frame_processor is not None:
            self.frame_processor.stop()

        super().stop()
