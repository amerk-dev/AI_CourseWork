from pydantic import BaseModel, Field


class ClientData(BaseModel):
    Age: int = Field(..., ge=18, le=100)
    Sex: str
    Job: int = Field(..., ge=0, le=3)
    Housing: str
    Saving_accounts: str
    Checking_account: str
    Duration: int = Field(..., ge=1)
    Purpose: str
