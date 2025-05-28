import asyncio
import sys
import traceback
from typing import List, Optional
from pydantic import BaseModel, Field
from beeai_framework.adapters.openai import OpenAIChatModel
from beeai_framework.backend import UserMessage, SystemMessage

class TicketClassifierOutput(BaseModel):
    """Structured payload returned by the LLM for a single ticket."""
    category: List[str] = Field(
        description="Options: Billing, Technical, Complaint, Account, Feedback, Other"
    )
    customer_name: Optional[str] = Field(
        default=None,
        description="Full customer name; null if not mentioned."
    )
    account_id: Optional[str] = Field(
        default=None,
        description="Exact account identifier as it appears in the ticket."
    )
    product: Optional[str] = Field(
        default=None,
        description="Product/SKU referenced in the ticket."
    )
    issue_summary: str = Field(
        description="concise plain-language summary of the problem, extracting key insights."
    )
    severity: str = Field(
        description='One of: "critical", "high", "medium", "low".'
    )
    sentiment: str = Field(
        description='One of: "negative", "neutral", "positive".'
    )
    incident_date: Optional[str] = Field(
        default=None,
        description="ISO-8601 date (YYYY-MM-DD) if provided."
    )

async def TicketClassifierAgent() -> None:
    llm = OpenAIChatModel("gpt-4.1-mini-2025-04-14")
    system_msg = SystemMessage(
             """
            You are “Support-Sensei,” an AI assistant that must:
            1. Choose the single best ticket category.
            2. Extract the required fields.
            """
        )
    user_message = "Hi there, this is Jane Doe. Ever since yesterday your ProPlan keeps throwing “Error 500” whenever I try to export reports. This is blocking my quarter-end close—please fix ASAP or refund the month.AccountNumber: 872-55-1234"
    response = await llm.create_structure(
            schema=TicketClassifierOutput,
            messages=[system_msg, UserMessage(user_message)],
        )
    print(response.object)

async def main() -> None:
    print("Starting TicketClassifierAgent...")
    await TicketClassifierAgent()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception:
        print("An error occurred:")
        traceback.print_exc()
        sys.exit(1)