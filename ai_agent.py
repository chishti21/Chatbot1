
# setup api keys

from dotenv import load_dotenv
import os

load_dotenv()  # Load environment variables from .env file

GROQ_API_KEY = os.getenv('GROQ_API_KEY')
TAVILY_API_KEY = os.getenv('TAVILY_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Check if all API keys are set
if not all([GROQ_API_KEY, TAVILY_API_KEY, OPENAI_API_KEY]):
    raise ValueError("One or more API keys are not set. Please set the environment variables.")

# setup LLM tools
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages.ai import AIMessage


openai_llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY)
groq_llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=GROQ_API_KEY)
search_tool = TavilySearchResults(max_results=2, api_key=TAVILY_API_KEY)

# setup agent
from langgraph.prebuilt import create_react_agent

system_prompt = "Act as an AI chatbot who is smart and friendly"

def get_response_from_ai_agent(llm_id, query, allow_search, system_prompt, provider):
    if provider=="Groq":
        llm=ChatGroq(model=llm_id)
    elif provider=="OpenAI":
        llm=ChatOpenAI(model=llm_id)

    tools=[TavilySearchResults(max_results=2)] if allow_search else []
    agent=create_react_agent(
        model=llm,
        tools=tools,
        state_modifier=system_prompt
    )
    state={"messages": query}
    response=agent.invoke(state)
    messages=response.get("messages")
    ai_messages=[message.content for message in messages if isinstance(message, AIMessage)]
    return ai_messages[-1]


# agent = create_react_agent(
#     model=groq_llm,
#     tools=[search_tool],
#     state_modifier=system_prompt
# )

# query = "Tell me about trends in ecnomic crisis in pakistan and india and comapre them whre condition is good"
# state = {"messages": query}
# response = agent.invoke(state)
# messages=response.get("messages")
# ai_messages=[message.content for message in messages if isinstance(message, AIMessage)]
# print(ai_messages[-1])


