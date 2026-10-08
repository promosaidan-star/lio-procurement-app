from fastapi import APIRouter, Depends, HTTPException, UploadFile, status  # routing, auth dependency, errors, uploads

from app.core.deps import get_current_user  # JWT check: only signed-in users may parse PDFs
from app.schemas.extraction import PdfParseResponse  # {"text": ...} response envelope
from app.services.pdf import NoTextLayerError, PdfTextError, extract_text  # the parser and its two error types

router = APIRouter(prefix="/pdf", tags=["pdf"])  # all routes here live under /pdf

MAX_PDF_BYTES = 10 * 1024 * 1024  # 10 MB, matches the upload UI's stated limit

PDF_CONTENT_TYPES = ("application/pdf", "application/x-pdf")  # MIME types browsers normally send


def _looks_like_pdf(file: UploadFile, head: bytes) -> bool:
    """Accept by content type, by extension, or by magic bytes.

    Browsers and OS integrations disagree on the MIME type they attach
    (``application/octet-stream`` from some mail clients, odd casing like
    ``.Pdf``), so we do not trust the content type alone.
    """
    if file.content_type in PDF_CONTENT_TYPES:  # the normal case
        return True
    if (file.filename or "").lower().endswith(".pdf"):  # "QuoteA0492_23.Pdf" with a generic MIME type
        return True
    return head.startswith(b"%PDF")  # last resort: every PDF starts with this signature


@router.post("/parse", response_model=PdfParseResponse, dependencies=[Depends(get_current_user)])  # POST /pdf/parse, auth required
async def parse_pdf(file: UploadFile) -> PdfParseResponse:
    pdf_bytes = await file.read()  # read the whole upload into memory (bounded below)
    if not _looks_like_pdf(file, pdf_bytes[:8]):  # only the first bytes are needed for the signature check
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Please upload a PDF file"
        )
    if len(pdf_bytes) > MAX_PDF_BYTES:  # enforce the size limit the UI advertises
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="PDF exceeds the 10MB limit",
        )

    try:
        text = extract_text(pdf_bytes)  # pdfplumber first, pypdf fallback
    except NoTextLayerError as exc:  # parsed fine but it is a scan: tell the user exactly that
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc
    except PdfTextError:  # both backends failed: the file is corrupt or locked
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Failed to parse PDF. The file may be corrupted or password protected.",
        )

    return PdfParseResponse(text=text)  # the frontend sends this text on to /extraction
