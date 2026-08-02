import io
import sys
from email.message import Message
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from appearance import (
    AppearanceError,
    AppearanceRateLimit,
    FalAppearanceEditor,
    parse_change_command,
)
from PIL import Image


def _png_bytes(size=(320, 480)):
    buf = io.BytesIO()
    Image.new("RGB", size, (20, 40, 60)).save(buf, "PNG")
    return buf.getvalue()


class _Response:
    def __init__(self, body, content_type="image/png"):
        self.body = body
        self.headers = Message()
        self.headers["Content-Type"] = content_type

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self, _limit):
        return self.body


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("hello", None),
        (" /CHANGE   your outfit to a wedding dress ", "your outfit to a wedding dress"),
        ("/change", ""),
        ("please /change the outfit", None),
    ],
)
def test_parse_change_command(message, expected):
    assert parse_change_command(message) == expected


def test_fal_editor_uploads_edits_downloads_and_validates(tmp_path):
    source = tmp_path / "source.png"
    source.write_bytes(_png_bytes())
    calls = {}

    def subscribe(model, arguments):
        calls["model"] = model
        calls["arguments"] = arguments
        return {"images": [{"url": "https://cdn.example/generated.png"}]}

    fake_client = SimpleNamespace(
        upload_file=lambda path: f"https://cdn.example/{path.split('/')[-1]}",
        subscribe=subscribe,
    )
    with (
        patch.dict("os.environ", {"FAL_KEY": "test-only"}, clear=False),
        patch.dict(sys.modules, {"fal_client": fake_client}),
        patch(
            "appearance.urllib.request.urlopen",
            return_value=_Response(_png_bytes((512, 768))),
        ),
    ):
        output = FalAppearanceEditor(tmp_path / "outputs").edit(
            source, "change the outfit to a wedding dress"
        )

    assert calls["model"] == "fal-ai/nano-banana/edit"
    assert calls["arguments"]["image_urls"][0].endswith("source.png")
    assert calls["arguments"]["safety_tolerance"] == "5"
    assert calls["arguments"]["limit_generations"] is True
    assert "wedding dress" in calls["arguments"]["prompt"]
    assert "Preserve the same character identity" in calls["arguments"]["prompt"]
    with Image.open(output) as generated:
        assert generated.size == (512, 768)
        assert generated.format == "PNG"


def test_fal_editor_requires_server_side_key(tmp_path):
    source = tmp_path / "source.png"
    source.write_bytes(_png_bytes())
    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(AppearanceError, match="not configured"):
            FalAppearanceEditor(tmp_path / "outputs").edit(source, "new outfit")


def test_fal_editor_loads_documented_secret_file(tmp_path):
    secret = tmp_path / "secrets.env"
    secret.write_text("FAL_KEY=test-from-file\n")
    editor = FalAppearanceEditor(tmp_path / "outputs")
    with patch.dict(
        "os.environ", {"SOULX_SECRETS_FILE": str(secret)}, clear=True
    ):
        assert editor.configured is True
        assert __import__("os").environ["FAL_KEY"] == "test-from-file"


def test_fal_editor_rejects_non_https_output(tmp_path):
    source = tmp_path / "source.png"
    source.write_bytes(_png_bytes())
    fake_client = SimpleNamespace(
        upload_file=lambda _path: "https://cdn.example/source.png",
        subscribe=lambda *_args, **_kwargs: {
            "images": [{"url": "file:///tmp/not-remote.png"}]
        },
    )
    with (
        patch.dict("os.environ", {"FAL_KEY": "test-only"}, clear=False),
        patch.dict(sys.modules, {"fal_client": fake_client}),
    ):
        with pytest.raises(AppearanceError, match="invalid URL"):
            FalAppearanceEditor(tmp_path / "outputs").edit(source, "new outfit")


def test_fal_editor_rejects_blank_image(tmp_path):
    source = tmp_path / "source.png"
    source.write_bytes(_png_bytes())
    fake_client = SimpleNamespace(
        upload_file=lambda _path: "https://cdn.example/source.png",
        subscribe=lambda *_args, **_kwargs: {
            "images": [{"url": "https://cdn.example/blank.png"}]
        },
    )
    blank = io.BytesIO()
    Image.new("RGB", (512, 768), "black").save(blank, "PNG")
    with (
        patch.dict("os.environ", {"FAL_KEY": "test-only"}, clear=False),
        patch.dict(sys.modules, {"fal_client": fake_client}),
        patch(
            "appearance.urllib.request.urlopen",
            return_value=_Response(blank.getvalue()),
        ),
    ):
        with pytest.raises(AppearanceError, match="blank image"):
            FalAppearanceEditor(tmp_path / "outputs").edit(source, "new outfit")


def test_fal_editor_bounds_paid_attempts(tmp_path):
    source = tmp_path / "source.png"
    source.write_bytes(_png_bytes())
    fake_client = SimpleNamespace(
        upload_file=lambda _path: "https://cdn.example/source.png",
        subscribe=lambda *_args, **_kwargs: {
            "images": [{"url": "https://cdn.example/generated.png"}]
        },
    )
    editor = FalAppearanceEditor(tmp_path / "outputs")
    with (
        patch.dict(
            "os.environ",
            {
                "FAL_KEY": "test-only",
                "SOULX_APPEARANCE_MAX_EDITS": "1",
                "SOULX_APPEARANCE_COOLDOWN_SECONDS": "0",
            },
            clear=False,
        ),
        patch.dict(sys.modules, {"fal_client": fake_client}),
        patch(
            "appearance.urllib.request.urlopen",
            return_value=_Response(_png_bytes((512, 768))),
        ),
    ):
        editor.edit(source, "first outfit")
        with pytest.raises(AppearanceRateLimit, match="limit reached"):
            editor.edit(source, "second outfit")
