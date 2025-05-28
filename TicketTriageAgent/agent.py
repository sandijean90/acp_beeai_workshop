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
#ACP SDK
from acp_sdk.models import Message, MessagePart
from acp_sdk.server import RunYield, RunYieldResume, Server
from collections.abc import AsyncGenerator
#Helpers
from helpers import package_response, flatten_messages

#load environment variables
from dotenv import load_dotenv
load_dotenv()

# Set up the ACP server
server = Server()


#Create Agent using the Framework
class TicketClassifierOutput(BaseModel):
        answer: str = Field(description="your final answer")

@server.agent()
async def ticket_triage_agent(input: list[Message]) -> AsyncGenerator[RunYield, RunYieldResume]:
        """ An agent that classifies customer support tickets.
        """
        user_prompt = flatten_messages(input[-1:])
        llm = OpenAIChatModel("gpt-4.1-mini-2025-04-14")
        system_msg = SystemMessage(
            "You are a helpful geography expert. Always provide accurate and concise answers."
        )
        response = await llm.create_structure(
            schema=TicketClassifierOutput,
            messages=[system_msg, UserMessage(user_prompt)],
        )
        yield package_response(response.object.get("answer", ""))

#Run the server
server.run()
