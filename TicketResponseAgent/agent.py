import sys
import os
#Pydantic Framework
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
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

#OLD VERSION NO HELPERS
# @server.agent()
# async def ticket_response_agent(input: list[Message]) -> AsyncGenerator[RunYield, RunYieldResume]:
#     """
#     An agent that responds to customer support tickets with a simple sentence.
#     """
#     user_prompt = str(input[0]).strip()

#     model = OpenAIModel('gpt-4.1-mini-2025-04-14', provider=OpenAIProvider(api_key=os.getenv('OPENAI_API_KEY')))
#     TicketResponseAgent = Agent( model=model,
#                             system_prompt=("You are a helpful assistant that responds to customer support tickets in one simple sentence."))
#     response = await TicketResponseAgent.run(user_prompt)
    
#     assistant_message = Message(parts=[MessagePart(content=response.output)])
#     yield {"messages": [assistant_message]}


@server.agent()
async def ticket_response_agent(input: list[Message]) -> AsyncGenerator[RunYield, RunYieldResume]:
    """
    An agent that responds to customer support tickets with a simple sentence.
    """
    user_prompt = flatten_messages(input)

    model = OpenAIModel('gpt-4.1-mini-2025-04-14', provider=OpenAIProvider(api_key=os.getenv('OPENAI_API_KEY')))
    TicketResponseAgent = Agent( model=model,
                            system_prompt=("You are a helpful assistant that responds to customer support tickets in one simple sentence."))
    response = await TicketResponseAgent.run(user_prompt)
    
    yield package_response(response.output)


#Run the server
server.run()