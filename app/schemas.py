from pydantic import BaseModel, Field

class CustomerDataInput(BaseModel):
    annual_income: float = Field(..., example=85.0, description="Annual Income in k$")
    spending_score: float = Field(..., example=90.0, description="Spending Score (1-100)")

class PredictionOutput(BaseModel):
    cluster_id: int
    cluster_label: str
    annual_income_k: float
    spending_score: float
