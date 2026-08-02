"""External reference-image editing for the interactive persona demo.

This module deliberately has no SoulX, Flask, or CUDA imports.  The chat layer
asks it for a new, validated local image; the video layer remains responsible
for deciding how and when to restart a causal session with that image.
"""

from __future__ import annotations

import os
import re
import tempfile
import threading
import time
import urllib.parse
import urllib.request
from pathlib import Path

from PIL import Image

DEFAULT_MODEL = "fal-ai/nano-banana/edit"
MAX_DOWNLOAD_BYTES = 25 * 1024 * 1024
_CHANGE_COMMAND = re.compile(r"^/change(?:\s+(.*))?$", re.IGNORECASE | re.DOTALL)


class AppearanceError(RuntimeError):
    """A safe, user-displayable appearance-edit failure."""


class AppearanceRateLimit(AppearanceError):
    """The process-level paid-edit budget or cooldown was exceeded."""


def parse_change_command(message: str) -> str | None:
    """Return a `/change` instruction, or ``None`` for ordinary chat.

    An empty command returns an empty string so the caller can distinguish it
    from a non-command and provide a useful 400 response.
    """

    match = _CHANGE_COMMAND.fullmatch(message.strip())
    if not match:
        return None
    return (match.group(1) or "").strip()


def _edit_prompt(instruction: str) -> str:
    return (
        "Edit only the visible character as requested: "
        f"{instruction}. Preserve the same character identity, face, hairstyle "
        "unless explicitly requested, body proportions, pose, camera angle, "
        "framing, lighting, background, and image style. Keep exactly one "
        "front-facing character and do not add text, borders, or watermarks."
    )


class FalAppearanceEditor:
    """Edit a local reference image with fal and save a validated local PNG."""

    def __init__(self, output_dir: str | os.PathLike[str], model: str | None = None):
        self.output_dir = Path(output_dir)
        self.model = model or os.environ.get("SOULX_APPEARANCE_MODEL", DEFAULT_MODEL)
        self._lock = threading.Lock()
        self._attempts = 0
        self._last_attempt = 0.0

    @property
    def configured(self) -> bool:
        return bool(self._ensure_credentials())

    @staticmethod
    def _ensure_credentials() -> str:
        key = os.environ.get("FAL_KEY", "").strip()
        if key:
            return key
        # Match scope-soulx's documented server-only fallback without sourcing
        # arbitrary shell. This also keeps direct debug launches functional.
        secret_path = Path(
            os.environ.get(
                "SOULX_SECRETS_FILE", "/root/.config/soulx/secrets.env"
            )
        )
        try:
            for line in secret_path.read_text().splitlines():
                name, sep, value = line.partition("=")
                if sep and name.strip() == "FAL_KEY":
                    key = value.strip().strip('"\'')
                    if key:
                        os.environ["FAL_KEY"] = key
                        return key
        except OSError:
            pass
        return ""

    def edit(self, source_path: str | os.PathLike[str], instruction: str) -> str:
        source = Path(source_path)
        if not instruction.strip():
            raise AppearanceError("usage: /change <describe the new appearance>")
        if not source.is_file():
            raise AppearanceError("the current character image is unavailable")
        if not self.configured:
            raise AppearanceError("appearance changes are not configured")

        # One paid edit at a time. It also gives successive commands a stable
        # ordering when several browser tabs share this single demo session.
        with self._lock:
            try:
                max_edits = max(1, int(os.environ.get("SOULX_APPEARANCE_MAX_EDITS", "20")))
                cooldown = max(
                    0.0,
                    float(os.environ.get("SOULX_APPEARANCE_COOLDOWN_SECONDS", "20")),
                )
            except ValueError as exc:
                raise AppearanceError("invalid appearance rate-limit configuration") from exc
            now = time.monotonic()
            if self._attempts >= max_edits:
                raise AppearanceRateLimit("appearance edit limit reached for this demo run")
            remaining = cooldown - (now - self._last_attempt)
            if self._last_attempt and remaining > 0:
                raise AppearanceRateLimit(
                    f"wait {max(1, int(remaining + 0.999))}s before another appearance change"
                )
            # Count provider attempts, not successful downloads: rejected and
            # failed requests may still be billable.
            self._attempts += 1
            self._last_attempt = now
            try:
                import fal_client
            except ImportError as exc:
                raise AppearanceError("fal-client is not installed") from exc

            try:
                image_url = fal_client.upload_file(str(source))
                result = fal_client.subscribe(
                    self.model,
                    arguments={
                        "prompt": _edit_prompt(instruction.strip()),
                        "image_urls": [image_url],
                        "num_images": 1,
                        "output_format": "png",
                        "safety_tolerance": "5",
                        "limit_generations": True,
                    },
                )
                output_url = result["images"][0]["url"]
            except Exception as exc:
                # Provider exceptions may contain request metadata. Keep the
                # client-facing/logged error deliberately terse so credentials
                # can never be reflected through the chat API.
                raise AppearanceError("fal image edit request failed") from exc

            self.output_dir.mkdir(parents=True, exist_ok=True)
            final_path = self.output_dir / f"appearance_{time.time_ns()}.png"
            self._download_and_validate(output_url, final_path)
            return str(final_path)

    @staticmethod
    def _download_and_validate(url: str, final_path: Path) -> None:
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme != "https" or not parsed.netloc:
            raise AppearanceError("image provider returned an invalid URL")

        temp_path = None
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                content_type = response.headers.get_content_type()
                if not content_type.startswith("image/"):
                    raise AppearanceError("image provider returned a non-image response")
                body = response.read(MAX_DOWNLOAD_BYTES + 1)
            if len(body) > MAX_DOWNLOAD_BYTES:
                raise AppearanceError("generated image is too large")

            with tempfile.NamedTemporaryFile(
                prefix="appearance_", suffix=".tmp", dir=final_path.parent, delete=False
            ) as temp:
                temp.write(body)
                temp_path = Path(temp.name)

            # Decode and re-encode instead of trusting a content type or file
            # extension supplied by the remote service.
            with Image.open(temp_path) as generated:
                generated.load()
                if generated.width < 64 or generated.height < 64:
                    raise AppearanceError("generated image is too small")
                rgb = generated.convert("RGB")
                extrema = rgb.getextrema()
                dynamic_range = max(high for _low, high in extrema) - min(
                    low for low, _high in extrema
                )
                if dynamic_range < 8:
                    raise AppearanceError("image provider returned a blank image")
                rgb.save(final_path, format="PNG")
        except AppearanceError:
            raise
        except Exception as exc:
            raise AppearanceError("could not download the generated image") from exc
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)
