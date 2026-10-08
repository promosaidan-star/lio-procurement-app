"""Text extraction for vendor-offer PDFs.

Two backends, tried in order:

1. ``pdfplumber`` — keeps table rows on one line and resolves ligatures, which
   is what the extraction model needs to pair a description with its price.
2. ``pypdf`` — fallback for files pdfplumber cannot open (it is more tolerant
   of broken cross-reference tables, e.g. the Apple quote in the fixtures).

Scanned PDFs (no text layer) raise ``NoTextLayerError`` so the API can tell the
user what is wrong instead of handing an empty string to the model.
"""
import io
import logging
import re
import unicodedata

logger = logging.getLogger(__name__)

# Below this many characters we assume there is no usable text layer.
MIN_TEXT_CHARS = 20

_LIGATURES = {
    "ﬀ": "ff",
    "ﬁ": "fi",
    "ﬂ": "fl",
    "ﬃ": "ffi",
    "ﬄ": "ffl",
}


class PdfTextError(Exception):
    """The PDF could not be turned into text."""


class NoTextLayerError(PdfTextError):
    """The PDF parsed but contains no selectable text (likely a scan)."""


def _normalize(text: str) -> str:
    for src, dst in _LIGATURES.items():
        text = text.replace(src, dst)
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Collapse runs of spaces/tabs but keep line breaks: layout matters to the model.
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _with_pdfplumber(pdf_bytes: bytes) -> str:
    import pdfplumber

    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        pages = [page.extract_text() or "" for page in pdf.pages]
    return "\n".join(pages)


def _with_pypdf(pdf_bytes: bytes) -> str:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(pdf_bytes))
    if reader.is_encrypted:
        # Most "encrypted" quotes just have an empty owner password.
        reader.decrypt("")
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


def extract_text(pdf_bytes: bytes) -> str:
    """Extract the text from a vendor-offer PDF.

    Raises ``NoTextLayerError`` when the file has no text layer and
    ``PdfTextError`` when neither backend can read it.
    """
    errors: list[str] = []
    best = ""
    for name, backend in (("pdfplumber", _with_pdfplumber), ("pypdf", _with_pypdf)):
        try:
            text = _normalize(backend(pdf_bytes))
        except Exception as exc:  # noqa: BLE001 — try the next backend
            logger.warning("PDF backend %s failed: %s", name, exc)
            errors.append(f"{name}: {exc}")
            continue
        if len(text) >= MIN_TEXT_CHARS:
            return text
        best = text if len(text) > len(best) else best

    if errors and len(errors) == 2:
        raise PdfTextError("; ".join(errors))
    raise NoTextLayerError(
        "This PDF has no selectable text (it is probably a scan or an image). "
        "Please upload a text-based PDF or fill the form in manually."
    )
