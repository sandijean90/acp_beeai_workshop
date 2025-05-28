import asyncio
import sys
import traceback

from pydantic import BaseModel, Field
from beeai_framework.adapters.openai import OpenAIChatModel
from beeai_framework.backend import UserMessage, SystemMessage

class TicketClassifierOutput(BaseModel):
        answer: str = Field(description="your final answer")

async def TicketClassifierAgent() -> None:
    llm = OpenAIChatModel("gpt-4.1-mini-2025-04-14")
    system_message = SystemMessage("You are a helpful geography expert. Always provide accurate and concise answers.")
    user_message = UserMessage("How many islands make up the country of Cape Verde?")
    response = await llm.create_structure(
        schema=TicketClassifierOutput,
        messages=[system_message,user_message],
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