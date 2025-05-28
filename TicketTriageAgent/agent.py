import asyncio
import sys
import traceback
from acp_sdk import Message
from acp_sdk.models import MessagePart
from acp_sdk.server import Context, Server
from pydantic import BaseModel, Field
from beeai_framework.adapters.openai import OpenAIChatModel
from beeai_framework.backend import UserMessage, SystemMessage
from beeai_framework.adapters.acp.serve._utils import acp_msg_to_framework_msg
from beeai_framework.adapters.acp.serve.server import ACPServer, ACPServerConfig, to_acp_agent_metadata
from beeai_framework.adapters.acp.serve.agent import ACPServerAgent
from beeai_framework.agents.react import ReActAgent




#Create Agent using the Framework
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


# Create and start the server
server = ACPServer(config=ACPServerConfig(host="0.0.0.0", port=8000))
server.register(ACPServerAgent(
      fn=TicketClassifierAgent,
      name="ticket_classifier",
      description="Answers geography questions about Cape Verde"))

server.serve()
