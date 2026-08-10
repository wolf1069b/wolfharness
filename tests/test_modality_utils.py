"""Unit tests for modality_utils: describe_multimodal_content and classify_binary_content."""

from __future__ import annotations

from pydantic_ai import BinaryImage
from pydantic_ai.messages import (
    AudioUrl,
    BinaryContent,
    DocumentUrl,
    ImageUrl,
    VideoUrl,
)
import pytest

from wolfharness.capabilities.modality_utils import (
    classify_binary_content,
    describe_multimodal_content,
)


pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# describe_multimodal_content
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_describe_binary_image() -> None:
    """BinaryImage produces '[image/png]' placeholder."""
    img = BinaryImage(data=b"\x89PNG\r\n\x1a\n", media_type="image/png")
    assert describe_multimodal_content(img) == "[image/png]"


@pytest.mark.unit
def test_describe_binary_content_audio() -> None:
    """BinaryContent with audio media type produces '[audio/wav]'."""
    audio = BinaryContent(data=b"RIFF....", media_type="audio/wav")
    assert describe_multimodal_content(audio) == "[audio/wav]"


@pytest.mark.unit
def test_describe_binary_content_video() -> None:
    """BinaryContent with video media type produces '[video/mp4]'."""
    video = BinaryContent(data=b"\x00\x00\x00\x20ftyp", media_type="video/mp4")
    assert describe_multimodal_content(video) == "[video/mp4]"


@pytest.mark.unit
def test_describe_binary_content_document() -> None:
    """BinaryContent with document media type produces '[application/pdf]'."""
    doc = BinaryContent(data=b"%PDF-1.4", media_type="application/pdf")
    assert describe_multimodal_content(doc) == "[application/pdf]"


@pytest.mark.unit
def test_describe_image_url() -> None:
    """ImageUrl produces '[image: url]' placeholder."""
    url = ImageUrl(url="https://example.com/cat.png")
    assert describe_multimodal_content(url) == "[image: https://example.com/cat.png]"


@pytest.mark.unit
def test_describe_audio_url() -> None:
    """AudioUrl produces '[audio: url]' placeholder."""
    url = AudioUrl(url="https://example.com/sound.mp3")
    assert describe_multimodal_content(url) == "[audio: https://example.com/sound.mp3]"


@pytest.mark.unit
def test_describe_video_url() -> None:
    """VideoUrl produces '[video: url]' placeholder."""
    url = VideoUrl(url="https://example.com/clip.mp4")
    assert describe_multimodal_content(url) == "[video: https://example.com/clip.mp4]"


@pytest.mark.unit
def test_describe_document_url() -> None:
    """DocumentUrl produces '[document: url]' placeholder."""
    url = DocumentUrl(url="https://example.com/doc.pdf")
    assert describe_multimodal_content(url) == "[document: https://example.com/doc.pdf]"


# ---------------------------------------------------------------------------
# classify_binary_content
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_classify_image() -> None:
    """image/png classifies as 'image'."""
    content = BinaryContent(data=b"\x89PNG", media_type="image/png")
    assert classify_binary_content(content) == "image"


@pytest.mark.unit
def test_classify_audio() -> None:
    """audio/wav classifies as 'audio'."""
    content = BinaryContent(data=b"RIFF", media_type="audio/wav")
    assert classify_binary_content(content) == "audio"


@pytest.mark.unit
def test_classify_video() -> None:
    """video/mp4 classifies as 'video'."""
    content = BinaryContent(data=b"\x00ftyp", media_type="video/mp4")
    assert classify_binary_content(content) == "video"


@pytest.mark.unit
def test_classify_document_pdf() -> None:
    """application/pdf classifies as 'document'."""
    content = BinaryContent(data=b"%PDF-1.4", media_type="application/pdf")
    assert classify_binary_content(content) == "document"


@pytest.mark.unit
def test_classify_document_text_pdf() -> None:
    """text/pdf classifies as 'document'."""
    content = BinaryContent(data=b"%PDF-1.4", media_type="text/pdf")
    assert classify_binary_content(content) == "document"


@pytest.mark.unit
def test_classify_unknown_octet_stream() -> None:
    """application/octet-stream classifies as 'unknown'."""
    content = BinaryContent(data=b"\x00\x01", media_type="application/octet-stream")
    assert classify_binary_content(content) == "unknown"


@pytest.mark.unit
def test_classify_binary_image() -> None:
    """BinaryImage (subclass of BinaryContent) classifies correctly."""
    img = BinaryImage(data=b"\x89PNG\r\n\x1a\n", media_type="image/png")
    assert classify_binary_content(img) == "image"
