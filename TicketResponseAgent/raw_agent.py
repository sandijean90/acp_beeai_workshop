import sys
import os
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

#load environment variables
from dotenv import load_dotenv
load_dotenv()

# configure the OpenAI provider
model = OpenAIModel('gpt-4.1-mini-2025-04-14', provider=OpenAIProvider(api_key=os.getenv('OPENAI_API_KEY')))

#Create the agent using the Pydantic AI framework
TicketResponseAgent = Agent( model=model,
                            system_prompt=("You are a helpful assistant that responds to customer support tickets in one simple sentence."))


if __name__ == "__main__":
    try:
        result = TicketResponseAgent.run_sync("I have a problem with my order, it hasn't arrived yet. Can you help me?")
        print(result.output)
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)
 
