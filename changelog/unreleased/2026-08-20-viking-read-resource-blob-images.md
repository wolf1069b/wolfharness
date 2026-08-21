# viking read_resource serves image resources as blobs for vision models

Image resources (by extension, excluding SVG) read through the resources
capability's `read_resource` tool now come back as `BlobResourceContent`
with real bytes and MIME type, instead of a base64 text dump. Multimodal
models consume them as image parts directly; text-only path unchanged.
Mirrors the existing `viking_read` tool behavior.