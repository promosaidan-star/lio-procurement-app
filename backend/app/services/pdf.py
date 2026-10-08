"""Text extraction for vendor-offer PDFs.

Two backends, tried in order:

1. ``pdfplumber`` — keeps table rows on one line and resolves ligatures, which
   is what the extraction model needs to pair a description with its price.
2. ``pypdf`` — fallback for files pdfplumber cannot open (it is more tolerant
   of broken cross-reference tables, e.g. the Apple quote in the fixtures).

Scanned PDFs (no text layer) raise ``NoTextLayerError`` so the API can tell the
user what is wrong instead of handing an empty string to the model.
"""
import io  # wraps the uploaded bytes in a file-like object the PDF libraries can read
import logging  # records which backend failed and why, without surfacing it to the user
import re  # whitespace normalisation of the extracted text
import unicodedata  # NFKC normalisation turns typographic variants into plain characters

logger = logging.getLogger(__name__)  # module-level logger named after this file

# Below this many characters we assume there is no usable text layer.
MIN_TEXT_CHARS = 20

# Unicode ligature glyphs that PDF fonts emit as a single character; the model
# (and string matching) wants the plain letters, e.g. "Conﬁguration" -> "Configuration".
_LIGATURES = {
    "ﬀ": "ff",  # ﬀ
    "ﬁ": "fi",  # ﬁ
    "ﬂ": "fl",  # ﬂ
    "ﬃ": "ffi",  # ﬃ
    "ﬄ": "ffl",  # ﬄ
}


class PdfTextError(Exception):
    """The PDF could not be turned into text."""


class NoTextLayerError(PdfTextError):
    """The PDF parsed but contains no selectable text (likely a scan)."""


def _normalize(text: str) -> str:
    for src, dst in _LIGATURES.items():  # walk the ligature table
        text = text.replace(src, dst)  # swap each ligature glyph for its letters
    text = unicodedata.normalize("NFKC", text)  # fold fancy quotes, full-width digits, etc.
    text = text.replace("\r\n", "\n").replace("\r", "\n")  # one line-ending style
    # Collapse runs of spaces/tabs but keep line breaks: layout matters to the model.
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)  # at most one blank line between blocks
    return text.strip()  # drop leading/trailing whitespace


def _with_pdfplumber(pdf_bytes: bytes) -> str:
    import pdfplumber  # imported lazily so the module loads even if the backend is missing

    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:  # open from memory, close on exit
        pages = [page.extract_text() or "" for page in pdf.pages]  # None -> "" for blank pages
    return "\n".join(pages)  # one string, pages separated by a newline


def _with_pypdf(pdf_bytes: bytes) -> str:
    from pypdf import PdfReader  # lazy import for the same reason as above

    reader = PdfReader(io.BytesIO(pdf_bytes))  # pypdf tolerates broken xref tables
    if reader.is_encrypted:  # some vendors "protect" quotes with an empty password
        # Most "encrypted" quotes just have an empty owner password.
        reader.decrypt("")
    pages = [page.extract_text() or "" for page in reader.pages]  # text per page
    return "\n".join(pages)  # join pages


def extract_text(pdf_bytes: bytes) -> str:
    """Extract the text from a vendor-offer PDF.

    Raises ``NoTextLayerError`` when the file has no text layer and
    ``PdfTextError`` when neither backend can read it.
    """
    errors: list[str] = []  # collects one message per backend that raised
    best = ""  # longest text seen so far, used only to decide "scan vs. corrupt"
    for name, backend in (("pdfplumber", _with_pdfplumber), ("pypdf", _with_pypdf)):  # in order of preference
        try:
            text = _normalize(backend(pdf_bytes))  # run the backend and clean its output
        except Exception as exc:  # noqa: BLE001 — try the next backend
            logger.warning("PDF backend %s failed: %s", name, exc)  # keep the reason in the logs
            errors.append(f"{name}: {exc}")  # and remember it for the final error
            continue  # move on to the fallback backend
        if len(text) >= MIN_TEXT_CHARS:  # enough text to be a real text layer
            return text  # first good result wins
        best = text if len(text) > len(best) else best  # otherwise keep the longer of the two

    if errors and len(errors) == 2:  # both backends raised: the file itself is unreadable
        raise PdfTextError("; ".join(errors))
    raise NoTextLayerError(  # at least one backend opened it but found (almost) no text
        "This PDF has no selectable text (it is probably a scan or an image). "
        "Please upload a text-based PDF or fill the form in manually."
    )
