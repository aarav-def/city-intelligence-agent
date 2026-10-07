from dotenv import load_dotenv
import os
import requests

load_dotenv()

from langchain_mistralai import ChatMistralAI
from langchain.tools import tool
from langchain_core.messages import ToolMessage
from langchain_core.runnables import (
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from tavily import TavilyClient
from rich import print
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_tool_call

# =========================
# 🌦️ Weather Tool
# =========================

@tool
def get_weather(city: str) -> str:
    """Get current weather of a city"""
    api_key = os.getenv("OPENWEATHER_API_KEY")
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city},IN&appid={api_key}&units=metric"

    response = requests.get(url, timeout=10)
    response.raise_for_status()  # lets .with_retry() catch HTTP failures
    data = response.json()

    if str(data.get("cod")) != "200":
        return f"Error: {data.get('message', 'Could not fetch weather')}"

    temp = data["main"]["temp"]
    desc = data["weather"][0]["description"]
    return f"Weather in {city}: {desc}, {temp}°C"


# =========================
# 📰 News Tool (Tavily)
# =========================

tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

@tool
def get_news(city: str) -> str:
    """Get latest news about a city"""
    response = tavily_client.search(
        query=f"latest news in {city}",
        search_depth="basic",
        max_results=3,
    )

    results = response.get("results", [])
    if not results:
        return f"No news found for {city}"

    news_list = []
    for r in results:
        title = r.get("title", "No title")
        url = r.get("url", "")
        snippet = r.get("content", "")
        news_list.append(f"- {title}\n  🔗 {url}\n  📝 {snippet[:100]}...")

    return f"Latest news in {city}:\n\n" + "\n\n".join(news_list)


# =========================
# 🔗 Runnables (LCEL)
# =========================

# 1) Preprocess: normalise the city name
clean_city = RunnableLambda(lambda city: city.strip().title())

# 2) Wrap tools with retry (they're already Runnables)
weather_runnable = get_weather.with_retry(stop_after_attempt=2)
news_runnable = get_news.with_retry(stop_after_attempt=2)

# 3) Run weather + news in parallel on the same input
parallel_fetch = RunnableParallel(
    weather=weather_runnable,
    news=news_runnable,
)

# 4) Postprocess: merge into one report
def format_report(data: dict) -> str:
    return (
        "=== CITY REPORT ===\n\n"
        f"{data['weather']}\n\n"
        f"{data['news']}"
    )

format_runnable = RunnableLambda(format_report)

# 5) Full chain: clean -> parallel fetch -> format
city_report_chain = clean_city | parallel_fetch | format_runnable

# 6) Expose the chain to the agent as a tool
city_report = city_report_chain.as_tool(
    name="city_report",
    description="Get BOTH the current weather and latest news for a city in one call.",
    arg_types={"__root__": str},
)



# =========================
# 🧠 LLM Setup
# =========================

llm = ChatMistralAI(model="ministral-3b-2512")


@wrap_tool_call
def human_approval(request, handler):
    """Ask for human approval before every tool call."""
    tool_name = request.tool_call["name"]
    confirm = input(f"Agent wants to call '{tool_name}'. Approve? (yes/no): ")

    if confirm.lower() != "yes":
        return ToolMessage(
            content="Tool call denied by user.",
            tool_call_id=request.tool_call["id"],
        )
    return handler(request)


agent = create_agent(
    llm,
    tools=[get_weather, get_news, city_report],
    system_prompt=(
        "You are a helpful city assistant. "
        "If the user wants both weather and news, use the city_report tool."
    ),
    middleware=[human_approval],
)

# =========================
# 🔁 Wrap the agent itself in a Runnable pipeline
# =========================

to_agent_input = RunnableLambda(
    lambda text: {"messages": [{"role": "user", "content": text}]}
)
extract_answer = RunnableLambda(lambda result: result["messages"][-1].content)

# the agent is a Runnable, so it can sit inside a chain
bot = to_agent_input | agent | extract_answer

print("City Agent | type exit to quit")

while True:
    user_input = input("You : ")
    if user_input.lower() == "exit":
        break
    print("bot : ", bot.invoke(user_input))