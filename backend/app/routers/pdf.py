from fastapi import APIRouter, Depends, HTTPException, UploadFile, status

from app.core.deps import get_current_user
from app.schemas.extraction import PdfParseResponse
from app.services.pdf import NoTextLayerError, PdfTextError, extract_text

router = APIRouter(prefix="/pdf", tags=["pdf"])

MAX_PDF_BYTES = 10 * 1024 * 1024  # 10 MB, matches the upload UI's stated limit

PDF_CONTENT_TYPES = ("application/pdf", "application/x-pdf")


def _looks_like_pdf(file: UploadFile, head: bytes) -> bool:
    """Accept by content type, by extension, or by magic bytes.

    Browsers and OS integrations disagree on the MIME type they attach
    (``application/octet-stream`` from some mail clients, odd casing like
    ``.Pdf``), so we do not trust the content type alone.
    """
    if file.content_type in PDF_CONTENT_TYPES:
        return True
    if (file.filename or "").lower().endswith(".pdf"):
        return True
    return head.startswith(b"%PDF")


@router.post("/parse", response_model=PdfParseResponse, dependencies=[Depends(get_current_user)])
async def parse_pdf(file: UploadFile) -> PdfParseResponse:
    pdf_bytes = await file.read()
    if not _looks_like_pdf(file, pdf_bytes[:8]):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Please upload a PDF file"
        )
    if len(pdf_bytes) > MAX_PDF_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="PDF exceeds the 10MB limit",
        )

    try:
        text = extract_text(pdf_bytes)
    except NoTextLayerError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc
    except PdfTextError:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Failed to parse PDF. The file may be corrupted or password protected.",
        )

    return PdfParseResponse(text=text)
