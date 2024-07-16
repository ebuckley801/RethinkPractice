from dotenv import load_dotenv
load_dotenv()

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader,Settings
from llama_index.llms.openai import OpenAI
from llama_index.core.tools import QueryEngineTool,FunctionTool,BaseTool
from llama_index.core.agent import ReActAgent
from llama_index.core.llms import ChatMessage,MessageRole

import os

OPENAI_API_KEY = os.getenv("OPEN_AI_KEY")

#helper functions
def multiply(a: float, b: float) -> float:
    return a * b

multiply_tool = FunctionTool.from_defaults( fn=multiply)

def add(a: float, b: float) -> float:
    return a + b

add_tool = FunctionTool.from_defaults( fn=add)

#Sets the LLM to use for the agent
Settings.llm = OpenAI(model="openai/gpt-4-turbo-preview",temperature=0,api_key=OPENAI_API_KEY)

documents = SimpleDirectoryReader("data").load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()

response = query_engine.query("What was the federal canadian budget in 2023?")

print(response)


#turn query engine into a tool
budget_tool = QueryEngineTool.from_defaults(
    query_engine,
    name="canadian_budget_tool",
    description="A RAG engine with some basic facts about the canadian budget"
)

#create an agent
agent = ReActAgent.from_tools(
    [multiply_tool, add_tool, budget_tool], verbose=True
)
