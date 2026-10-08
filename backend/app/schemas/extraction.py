from pydantic import BaseModel, ConfigDict, Field  # request/response models with camelCase aliases


class PdfParseResponse(BaseModel):
    text: str  # the plain text pulled out of the uploaded PDF


class ExtractionRequest(BaseModel):
    text: str  # that same text, sent back by the frontend for the AI step


# camelCase on the wire — matches the AI's JSON output format and what the
# frontend's extraction flow already consumes.
class OrderLineData(BaseModel):
    model_config = ConfigDict(populate_by_name=True)  # accept both snake_case and the alias when constructing

    position_description: str = Field(default="", alias="positionDescription")  # what is being bought
    unit_price: float = Field(default=0, alias="unitPrice")  # price per unit before discount
    amount: float = 0  # quantity, fractional allowed (13.78 sq ft)
    unit: str = ""  # "pieces", "licenses", "sq ft", ...
    total_price: float = Field(default=0, alias="totalPrice")  # line total after any discount


class ExtractedVendorData(BaseModel):
    model_config = ConfigDict(populate_by_name=True)  # same alias handling as above

    title: str = ""  # 2-5 word summary of the purchase
    vendor_name: str = Field(default="", alias="vendorName")  # the company issuing the quote
    vat_id: str = Field(default="", alias="vatId")  # vendor tax id (EIN), normalised to XX-XXXXXXX
    department: str = ""  # the customer / bill-to party named on the quote
    order_lines: list[OrderLineData] = Field(default_factory=list, alias="orderLines")  # chargeable items only
    total_cost: float = Field(default=0, alias="totalCost")  # grand total payable as printed
    commodity_group_id: int | None = Field(default=None, alias="commodityGroupId")  # resolved catalog row id, or None
    commodity_group_name: str | None = Field(default=None, alias="commodityGroupName")  # its display name


class ExtractionResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)  # alias handling

    success: bool  # False means "error" explains why; never an HTTP error, so the UI can show it
    data: ExtractedVendorData | None = None  # the extracted request when success is True
    error: str | None = None  # human-readable failure reason
    missing_fields: list[str] | None = Field(default=None, alias="missingFields")  # labels the user must fill in by hand
    # Checks that did not add up (line totals vs. subtotal, subtotal + tax +
    # shipping vs. grand total). The data is still returned; the user is asked
    # to double-check these before submitting.
    warnings: list[str] | None = None
