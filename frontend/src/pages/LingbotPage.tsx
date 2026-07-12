import {
  ArrowRight,
  Camera,
  CircleStop,
  Flame,
  ImagePlus,
  LoaderCircle,
  MousePointer2,
  Play,
  RotateCcw,
  Send,
  Sparkles,
  Upload,
  Video,
  WandSparkles,
} from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";
import type { FormEvent } from "react";
import { toast } from "sonner";

import { Button } from "../components/ui/button";
import { useControllerInput } from "../hooks/useControllerInput";
import {
  getAssetUrl,
  getIceServers,
  getPipelineStatus,
  listAssets,
  loadPipeline,
  sendIceCandidates,
  sendWebRTCOffer,
  uploadAsset,
  type AssetFileInfo,
  type PipelineStatusResponse,
} from "../lib/api";

const DEFAULT_PROMPT =
  "A beautiful young woman with long dark hair in a black ribbed sweater sits beside a blue bed in a dim bedroom, intimate close-up fixed camera";

const EVENTS = [
  {
    key: "1",
    label: "Run through hair",
    description: "A subtle, natural appearance gesture",
    icon: Sparkles,
    prompt:
      "gently runs one hand through her hair and looks naturally toward the camera",
  },
  {
    key: "2",
    label: "Rest chin in hands",
    description: "Lean closer into the beauty pose",
    icon: Camera,
    prompt:
      "leans closer and rests her chin softly in both hands while looking at the camera",
  },
  {
    key: "3",
    label: "Hold a candle",
    description: "Add warm light and a new prop",
    icon: Flame,
    prompt:
      "holds a small lit candle carefully in both hands; its warm flame illuminates her face",
  },
  {
    key: "F",
    label: "Butterfly alights",
    description: "Introduce a delicate world event",
    icon: WandSparkles,
    prompt: "watches as a delicate butterfly alights gently on her hand",
  },
  {
    key: "G",
    label: "Snow blankets room",
    description: "Transform the atmosphere",
    icon: Sparkles,
    prompt: "watches soft snow begin to fall and blanket the same room",
  },
] as const;

type ConnectionState =
  | "idle"
  | "loading-model"
  | "connecting"
  | "warming-up"
  | "live"
  | "error";

const STATE_LABELS: Record<ConnectionState, string> = {
  idle: "Ready to begin",
  "loading-model": "Loading world model",
  connecting: "Connecting video stream",
  "warming-up": "Creating the first frames",
  live: "World is live",
  error: "Connection needs attention",
};

function wait(ms: number) {
  return new Promise(resolve => window.setTimeout(resolve, ms));
}

export function LingbotPage() {
  const [assets, setAssets] = useState<AssetFileInfo[]>([]);
  const [selectedImage, setSelectedImage] = useState<string>(
    "/assets/beauty_seed.png"
  );
  const [prompt, setPrompt] = useState(DEFAULT_PROMPT);
  const [customEvent, setCustomEvent] = useState("");
  const [activeEvent, setActiveEvent] = useState<string | null>(null);
  const [connectionState, setConnectionState] =
    useState<ConnectionState>("idle");
  const [statusDetail, setStatusDetail] = useState(
    "Choose a start image, then create your world."
  );
  const [pipelineStatus, setPipelineStatus] =
    useState<PipelineStatusResponse | null>(null);
  const [remoteStream, setRemoteStream] = useState<MediaStream | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [isGalleryOpen, setIsGalleryOpen] = useState(false);

  const peerRef = useRef<RTCPeerConnection | null>(null);
  const channelRef = useRef<RTCDataChannel | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const sessionIdRef = useRef<string | null>(null);
  const queuedCandidatesRef = useRef<RTCIceCandidate[]>([]);
  const videoRef = useRef<HTMLVideoElement>(null);
  const stageRef = useRef<HTMLDivElement>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  const isStreaming = ["connecting", "warming-up", "live"].includes(
    connectionState
  );

  const sendParameters = useCallback((parameters: Record<string, unknown>) => {
    const channel = channelRef.current;
    if (!channel || channel.readyState !== "open") return false;
    channel.send(JSON.stringify(parameters));
    return true;
  }, []);

  const { isPointerLocked, requestPointerLock, pressedKeys } =
    useControllerInput(sendParameters, connectionState === "live", stageRef, {
      sendRateHz: 30,
      mouseSensitivity: 1.0,
    });

  const refreshAssets = useCallback(async () => {
    try {
      const response = await listAssets("image");
      setAssets(response.assets);
      if (!selectedImage && response.assets.length > 0) {
        setSelectedImage(response.assets[0].path);
      }
    } catch (error) {
      console.error(error);
      toast.error("Could not load images");
    }
  }, [selectedImage]);

  useEffect(() => {
    void refreshAssets();
    void getPipelineStatus().then(setPipelineStatus).catch(console.error);
  }, [refreshAssets]);

  useEffect(() => {
    if (videoRef.current && remoteStream) {
      videoRef.current.srcObject = remoteStream;
      void videoRef.current.play().catch(() => undefined);
    }
  }, [remoteStream]);

  const closeConnection = useCallback(() => {
    channelRef.current?.close();
    peerRef.current?.close();
    peerRef.current = null;
    channelRef.current = null;
    sessionIdRef.current = null;
    queuedCandidatesRef.current = [];
    streamRef.current?.getTracks().forEach(track => track.stop());
    streamRef.current = null;
    setRemoteStream(null);
    setActiveEvent(null);
  }, []);

  useEffect(() => () => closeConnection(), [closeConnection]);

  const ensurePipeline = async () => {
    let status = await getPipelineStatus();
    setPipelineStatus(status);
    if (status.status === "loaded" && status.pipeline_id === "lingbot-world") {
      return;
    }

    setConnectionState("loading-model");
    setStatusDetail("Loading the 14B LingBot model onto the H200…");
    await loadPipeline({
      pipeline_ids: ["lingbot-world"],
      load_params: { height: 480, width: 832 },
    });

    for (let attempt = 0; attempt < 90; attempt += 1) {
      await wait(2000);
      status = await getPipelineStatus();
      setPipelineStatus(status);
      if (status.status === "loaded") return;
      if (status.status === "error") {
        throw new Error(status.error || "The world model failed to load");
      }
    }
    throw new Error("The world model did not finish loading in time");
  };

  const startWorld = async () => {
    if (!selectedImage) {
      toast.error("Choose a start image first");
      return;
    }

    closeConnection();
    setConnectionState("connecting");
    setStatusDetail("Preparing the H200 and negotiating a video stream…");

    try {
      await ensurePipeline();
      setConnectionState("connecting");

      const { iceServers } = await getIceServers();
      const pc = new RTCPeerConnection({ iceServers });
      peerRef.current = pc;

      const channel = pc.createDataChannel("parameters", { ordered: true });
      channelRef.current = channel;
      channel.onopen = () => {
        setConnectionState("warming-up");
        setStatusDetail(
          "Connected. The first image encode takes about 15–25 seconds."
        );
      };
      channel.onmessage = event => {
        try {
          const message = JSON.parse(event.data as string) as {
            type?: string;
            error_message?: string;
          };
          if (message.type === "stream_stopped") {
            throw new Error(message.error_message || "The stream stopped");
          }
        } catch (error) {
          if (error instanceof Error) {
            setConnectionState("error");
            setStatusDetail(error.message);
          }
        }
      };

      const transceiver = pc.addTransceiver("video", { direction: "recvonly" });
      const vp8 = RTCRtpReceiver.getCapabilities("video")?.codecs.filter(
        codec => codec.mimeType.toLowerCase() === "video/vp8"
      );
      if (vp8?.length) transceiver.setCodecPreferences(vp8);

      pc.ontrack = event => {
        const stream = event.streams[0] || new MediaStream([event.track]);
        streamRef.current = stream;
        setRemoteStream(stream);
      };
      pc.onconnectionstatechange = () => {
        if (pc.connectionState === "failed") {
          setConnectionState("error");
          setStatusDetail("WebRTC failed. Stop the world and try again.");
        }
        if (pc.connectionState === "disconnected") {
          setStatusDetail("Video connection interrupted; reconnecting…");
        }
      };
      pc.onicecandidate = event => {
        if (!event.candidate) return;
        const sessionId = sessionIdRef.current;
        if (sessionId) {
          void sendIceCandidates(sessionId, event.candidate).catch(
            console.error
          );
        } else {
          queuedCandidatesRef.current.push(event.candidate);
        }
      };

      const offer = await pc.createOffer();
      await pc.setLocalDescription(offer);
      const answer = await sendWebRTCOffer({
        sdp: pc.localDescription?.sdp,
        type: pc.localDescription?.type,
        initialParameters: {
          pipeline_ids: ["lingbot-world"],
          prompts: [{ text: prompt.trim() || DEFAULT_PROMPT, weight: 1 }],
          images: [],
          first_frame_image: selectedImage,
          input_mode: "text",
        },
      });
      sessionIdRef.current = answer.sessionId;

      if (queuedCandidatesRef.current.length) {
        await sendIceCandidates(answer.sessionId, queuedCandidatesRef.current);
        queuedCandidatesRef.current = [];
      }
      await pc.setRemoteDescription({
        sdp: answer.sdp,
        type: answer.type as RTCSdpType,
      });
    } catch (error) {
      console.error(error);
      closeConnection();
      setConnectionState("error");
      setStatusDetail(
        error instanceof Error ? error.message : "Could not start the world"
      );
      toast.error("Could not start the world");
    }
  };

  const stopWorld = () => {
    closeConnection();
    setConnectionState("idle");
    setStatusDetail("Stopped. Your model remains loaded for a fast restart.");
  };

  const triggerEvent = useCallback(
    (key: string, eventPrompt: string) => {
      if (!sendParameters({ event_prompt: eventPrompt })) {
        toast.error("Start the world before sending an action");
        return;
      }
      setActiveEvent(key);
      setStatusDetail("Action queued for the next generated chunk.");
    },
    [sendParameters]
  );

  useEffect(() => {
    const handleEventKey = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement;
      if (
        event.repeat ||
        target.tagName === "INPUT" ||
        target.tagName === "TEXTAREA" ||
        target.isContentEditable
      ) {
        return;
      }
      const action = EVENTS.find(
        item => item.key.toLowerCase() === event.key.toLowerCase()
      );
      if (action && connectionState === "live") {
        event.preventDefault();
        triggerEvent(action.key, action.prompt);
      }
    };
    window.addEventListener("keydown", handleEventKey);
    return () => window.removeEventListener("keydown", handleEventKey);
  }, [connectionState, triggerEvent]);

  const submitCustomEvent = (event: FormEvent) => {
    event.preventDefault();
    const value = customEvent.trim();
    if (!value) return;
    triggerEvent("custom", value);
    setCustomEvent("");
  };

  const handleUpload = async (file?: File) => {
    if (!file) return;
    setIsUploading(true);
    try {
      const uploaded = await uploadAsset(file);
      setSelectedImage(uploaded.path);
      await refreshAssets();
      setIsGalleryOpen(false);
      toast.success("Start image uploaded");
    } catch (error) {
      console.error(error);
      toast.error("Image upload failed");
    } finally {
      setIsUploading(false);
      if (fileRef.current) fileRef.current.value = "";
    }
  };

  const previewUrl = selectedImage ? getAssetUrl(selectedImage) : "";
  const isBusy = ["loading-model", "connecting"].includes(connectionState);

  return (
    <main className="min-h-screen overflow-x-hidden bg-[#080b10] text-white">
      <div className="pointer-events-none fixed inset-0 bg-[radial-gradient(circle_at_30%_0%,rgba(72,101,130,0.18),transparent_38%),radial-gradient(circle_at_85%_80%,rgba(181,111,68,0.12),transparent_36%)]" />

      <header className="relative z-10 flex h-16 items-center justify-between border-b border-white/10 px-5 lg:px-8">
        <div className="flex items-center gap-3">
          <div className="flex size-9 items-center justify-center rounded-xl bg-white text-black">
            <Sparkles className="size-4" />
          </div>
          <div>
            <div className="text-sm font-semibold tracking-wide">DAYDREAM</div>
            <div className="text-[10px] uppercase tracking-[0.24em] text-white/40">
              LingBot World Studio
            </div>
          </div>
        </div>
        <div className="flex items-center gap-2 rounded-full border border-white/10 bg-white/5 px-3 py-1.5 text-xs text-white/65">
          <span
            className={`size-2 rounded-full ${
              pipelineStatus?.status === "loaded"
                ? "bg-emerald-400 shadow-[0_0_10px_#34d399]"
                : "bg-amber-400"
            }`}
          />
          H200 ·{" "}
          {pipelineStatus?.status === "loaded" ? "model ready" : "checking"}
        </div>
      </header>

      <div className="relative z-10 mx-auto grid max-w-[1680px] gap-5 p-4 lg:h-[calc(100vh-4rem)] lg:grid-cols-[280px_minmax(0,1fr)_330px] lg:p-5">
        <aside className="order-2 flex min-h-0 flex-col gap-4 lg:order-1">
          <section className="rounded-2xl border border-white/10 bg-white/[0.045] p-4 backdrop-blur-xl">
            <div className="mb-3 flex items-center justify-between">
              <div>
                <p className="text-sm font-medium">Start image</p>
                <p className="text-xs text-white/40">
                  Identity and world anchor
                </p>
              </div>
              <ImagePlus className="size-4 text-white/40" />
            </div>
            <button
              className="group relative aspect-[13/10] w-full overflow-hidden rounded-xl border border-white/10 bg-black/40"
              onClick={() => setIsGalleryOpen(value => !value)}
              disabled={isStreaming}
            >
              {previewUrl ? (
                <img
                  src={previewUrl}
                  alt="Selected world seed"
                  className="size-full object-cover transition duration-500 group-hover:scale-[1.03]"
                />
              ) : (
                <div className="flex size-full items-center justify-center text-white/30">
                  Choose an image
                </div>
              )}
              {!isStreaming && (
                <div className="absolute inset-x-2 bottom-2 rounded-lg bg-black/70 px-3 py-2 text-xs backdrop-blur">
                  Click to choose another image
                </div>
              )}
            </button>

            {isGalleryOpen && !isStreaming && (
              <div className="mt-3 grid max-h-48 grid-cols-3 gap-2 overflow-y-auto pr-1">
                <button
                  className="flex aspect-square items-center justify-center rounded-lg border border-dashed border-white/20 bg-white/5 hover:bg-white/10"
                  onClick={() => fileRef.current?.click()}
                >
                  {isUploading ? (
                    <LoaderCircle className="size-5 animate-spin" />
                  ) : (
                    <Upload className="size-5" />
                  )}
                </button>
                {assets.map(asset => (
                  <button
                    key={asset.path}
                    onClick={() => {
                      setSelectedImage(asset.path);
                      setIsGalleryOpen(false);
                    }}
                    className={`aspect-square overflow-hidden rounded-lg border ${
                      asset.path === selectedImage
                        ? "border-white"
                        : "border-white/10"
                    }`}
                  >
                    <img
                      src={getAssetUrl(asset.path)}
                      alt={asset.name}
                      className="size-full object-cover"
                    />
                  </button>
                ))}
              </div>
            )}
            <input
              ref={fileRef}
              type="file"
              accept="image/png,image/jpeg,image/webp,image/bmp"
              className="hidden"
              onChange={event => void handleUpload(event.target.files?.[0])}
            />
          </section>

          <section className="flex min-h-0 flex-1 flex-col rounded-2xl border border-white/10 bg-white/[0.045] p-4 backdrop-blur-xl">
            <div className="mb-3">
              <p className="text-sm font-medium">World description</p>
              <p className="text-xs text-white/40">
                Keep identity, clothing and environment explicit
              </p>
            </div>
            <textarea
              value={prompt}
              onChange={event => setPrompt(event.target.value)}
              disabled={isStreaming}
              className="min-h-32 flex-1 resize-none rounded-xl border border-white/10 bg-black/25 p-3 text-sm leading-relaxed text-white/80 outline-none transition focus:border-white/30 disabled:opacity-60"
            />
          </section>
        </aside>

        <section className="order-1 flex min-h-[520px] flex-col overflow-hidden rounded-3xl border border-white/10 bg-black/45 shadow-2xl lg:order-2 lg:min-h-0">
          <div className="flex items-center justify-between border-b border-white/10 px-4 py-3">
            <div className="flex items-center gap-3">
              <div
                className={`size-2.5 rounded-full ${
                  connectionState === "live"
                    ? "bg-red-500 shadow-[0_0_12px_#ef4444]"
                    : isBusy || connectionState === "warming-up"
                      ? "animate-pulse bg-amber-400"
                      : "bg-white/20"
                }`}
              />
              <div>
                <p className="text-sm font-medium">
                  {STATE_LABELS[connectionState]}
                </p>
                <p className="text-xs text-white/40">{statusDetail}</p>
              </div>
            </div>
            {isStreaming ? (
              <Button
                variant="outline"
                size="sm"
                onClick={stopWorld}
                className="border-white/15 bg-white/5 text-white hover:bg-white/10 hover:text-white"
              >
                <CircleStop /> Stop
              </Button>
            ) : (
              <Button
                size="sm"
                onClick={() => void startWorld()}
                disabled={isBusy || !selectedImage}
                className="bg-white text-black hover:bg-white/90"
              >
                {isBusy ? <LoaderCircle className="animate-spin" /> : <Play />}
                Create world
              </Button>
            )}
          </div>

          <div
            ref={stageRef}
            className="group relative flex min-h-0 flex-1 items-center justify-center overflow-hidden bg-[#030405] outline-none"
            onClick={() => connectionState === "live" && requestPointerLock()}
          >
            {remoteStream ? (
              <video
                ref={videoRef}
                autoPlay
                muted
                playsInline
                onPlaying={() => {
                  setConnectionState("live");
                  setStatusDetail(
                    "Click the video for WASD + mouse camera control."
                  );
                }}
                className="size-full object-contain"
              />
            ) : (
              <div className="flex max-w-md flex-col items-center px-8 text-center">
                {isStreaming || isBusy ? (
                  <>
                    <div className="relative mb-6 flex size-20 items-center justify-center rounded-full border border-white/10 bg-white/5">
                      <div className="absolute inset-0 animate-ping rounded-full border border-white/10" />
                      <LoaderCircle className="size-8 animate-spin text-white/70" />
                    </div>
                    <h2 className="text-xl font-medium">Building your world</h2>
                    <p className="mt-2 text-sm leading-relaxed text-white/45">
                      The first frame takes longer while LingBot encodes the
                      image. Keep this tab open; video will appear
                      automatically.
                    </p>
                  </>
                ) : (
                  <>
                    <div className="mb-6 flex size-20 items-center justify-center rounded-full border border-white/10 bg-white/5">
                      <Video className="size-8 text-white/50" />
                    </div>
                    <h1 className="text-2xl font-medium">
                      Bring an image to life
                    </h1>
                    <p className="mt-2 text-sm leading-relaxed text-white/45">
                      One persistent character. A world you can direct with
                      natural actions and explore with the camera.
                    </p>
                    <Button
                      onClick={event => {
                        event.stopPropagation();
                        void startWorld();
                      }}
                      className="mt-6 bg-white text-black hover:bg-white/90"
                    >
                      Create world <ArrowRight />
                    </Button>
                  </>
                )}
              </div>
            )}

            {connectionState === "live" && (
              <div className="pointer-events-none absolute inset-x-0 bottom-0 flex items-end justify-between bg-gradient-to-t from-black/80 to-transparent p-4 pt-16 opacity-0 transition group-hover:opacity-100">
                <div className="flex items-center gap-2 rounded-full bg-black/55 px-3 py-2 text-xs text-white/70 backdrop-blur">
                  <MousePointer2 className="size-3.5" />
                  {isPointerLocked
                    ? "Camera active · ESC to release"
                    : "Click for camera control"}
                </div>
                {pressedKeys.size > 0 && (
                  <div className="rounded-full bg-white px-3 py-2 text-xs font-medium text-black">
                    {[...pressedKeys]
                      .map(key => key.replace("Key", ""))
                      .join(" + ")}
                  </div>
                )}
              </div>
            )}
          </div>
        </section>

        <aside className="order-3 flex min-h-0 flex-col gap-4">
          <section className="min-h-0 flex-1 rounded-2xl border border-white/10 bg-white/[0.045] p-4 backdrop-blur-xl">
            <div className="mb-4 flex items-start justify-between">
              <div>
                <p className="text-sm font-medium">Direct the character</p>
                <p className="text-xs text-white/40">
                  Actions apply on the next chunk
                </p>
              </div>
              <WandSparkles className="size-4 text-white/40" />
            </div>
            <div className="space-y-2">
              {EVENTS.map(item => {
                const Icon = item.icon;
                const active = activeEvent === item.key;
                return (
                  <button
                    key={item.key}
                    disabled={connectionState !== "live"}
                    onClick={() => triggerEvent(item.key, item.prompt)}
                    className={`group flex w-full items-center gap-3 rounded-xl border p-3 text-left transition disabled:cursor-not-allowed disabled:opacity-35 ${
                      active
                        ? "border-amber-300/50 bg-amber-300/10"
                        : "border-white/10 bg-black/20 hover:border-white/25 hover:bg-white/[0.06]"
                    }`}
                  >
                    <span
                      className={`flex size-9 shrink-0 items-center justify-center rounded-lg ${
                        active ? "bg-amber-300 text-black" : "bg-white/10"
                      }`}
                    >
                      <Icon className="size-4" />
                    </span>
                    <span className="min-w-0 flex-1">
                      <span className="block text-sm font-medium">
                        {item.label}
                      </span>
                      <span className="block truncate text-[11px] text-white/35">
                        {item.description}
                      </span>
                    </span>
                    <kbd className="rounded border border-white/10 bg-black/30 px-1.5 py-1 text-[10px] text-white/45">
                      {item.key}
                    </kbd>
                  </button>
                );
              })}
            </div>
          </section>

          <form
            onSubmit={submitCustomEvent}
            className="rounded-2xl border border-white/10 bg-white/[0.045] p-4 backdrop-blur-xl"
          >
            <label
              className="mb-2 block text-sm font-medium"
              htmlFor="custom-event"
            >
              Or describe an action
            </label>
            <div className="flex gap-2">
              <input
                id="custom-event"
                value={customEvent}
                onChange={event => setCustomEvent(event.target.value)}
                disabled={connectionState !== "live"}
                placeholder="Smile and wave…"
                className="min-w-0 flex-1 rounded-xl border border-white/10 bg-black/25 px-3 text-sm outline-none transition placeholder:text-white/25 focus:border-white/30"
              />
              <Button
                size="icon"
                type="submit"
                disabled={connectionState !== "live" || !customEvent.trim()}
                className="bg-white text-black hover:bg-white/90"
              >
                <Send />
              </Button>
            </div>
          </form>

          {connectionState === "error" && (
            <Button
              variant="outline"
              onClick={() => {
                stopWorld();
                void startWorld();
              }}
              className="border-red-400/30 bg-red-400/10 text-red-100 hover:bg-red-400/20 hover:text-white"
            >
              <RotateCcw /> Try again
            </Button>
          )}
        </aside>
      </div>
    </main>
  );
}
