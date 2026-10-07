# app.py  ->  run with:  streamlit run app.py
from dotenv import load_dotenv
import os
import requests
import streamlit as st

load_dotenv()

from langchain_mistralai import ChatMistralAI
from langchain.tools import tool
from langchain_core.messages import ToolMessage
from tavily import TavilyClient
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_tool_call

# =========================
# 🎨 Page config + styling
# =========================
st.set_page_config(page_title="City Intelligence Agent", page_icon="🌆", layout="wide")

st.markdown(
    """
<style>
.block-container {padding-top: 1.5rem; max-width: 900px;}
.hero {
    background: linear-gradient(135deg, #4f46e5 0%, #06b6d4 100%);
    padding: 1.6rem 2rem; border-radius: 18px; color: white;
    box-shadow: 0 10px 30px rgba(79,70,229,.25); margin-bottom: 1.2rem;
}
.hero h1 {margin: 0; font-size: 2rem; color: white;}
.hero p  {margin: .3rem 0 0; opacity: .9;}
.chip {
    display: inline-block; padding: 2px 12px; margin: 4px 6px 0 0;
    border-radius: 999px; font-size: .78rem; font-weight: 600;
    background: rgba(99,102,241,.15); color: #6366f1; border: 1px solid rgba(99,102,241,.35);
}
.chip.denied {background: rgba(239,68,68,.12); color: #ef4444; border-color: rgba(239,68,68,.35);}
[data-testid="stChatMessage"] {border-radius: 14px; padding: .8rem 1rem;}
div.stButton > button {
    border-radius: 12px; font-weight: 600; width: 100%;
    border: 1px solid rgba(99,102,241,.4);
}
div.stButton > button:hover {border-color: #6366f1; color: #6366f1;}
footer {visibility: hidden;}
</style>
""",
    unsafe_allow_html=True,
)

# =========================
# 🌦️ Weather Tool (same as yours)
# =========================
@tool
def get_weather(city: str) -> str:
    """Get current weather of a city"""

    api_key = os.getenv("OPENWEATHER_API_KEY")
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city},IN&appid={api_key}&units=metric"

    response = requests.get(url)
    data = response.json()

    if str(data.get("cod")) != "200":
        return f"Error: {data.get('message', 'Could not fetch weather')}"

    temp = data["main"]["temp"]
    desc = data["weather"][0]["description"]

    return f"Weather in {city}: {desc}, {temp}°C"


# =========================
# 📰 News Tool (same as yours)
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
# 🧠 Agent (cached across reruns)
# =========================
@st.cache_resource
def build_agent():
    # shared state the middleware reads/writes
    state = {"allow_tools": True, "log": []}

    llm = ChatMistralAI(model="ministral-3b-2512")

    @wrap_tool_call
    def human_approval(request, handler):
        """Browser version of your approval step: controlled by the sidebar switch."""
        tool_name = request.tool_call["name"]

        if not state["allow_tools"]:
            state["log"].append((tool_name, False))
            return ToolMessage(
                content="Tool call denied by user.",
                tool_call_id=request.tool_call["id"],
            )

        state["log"].append((tool_name, True))
        return handler(request)

    agent = create_agent(
        llm,
        tools=[get_weather, get_news],
        system_prompt="you are a helpful city assistant.",
        middleware=[human_approval],
    )
    return agent, state


agent, agent_state = build_agent()

# =========================
# 📚 Sidebar
# =========================
with st.sidebar:
    st.markdown("## ⚙️ Controls")
    agent_state["allow_tools"] = st.toggle(
        "Allow tool calls",
        value=True,
        help="Off = every tool call is denied (same as answering 'no' in the terminal).",
    )

    st.markdown("---")
    st.markdown("## 🏙️ Quick lookup")
    city = st.text_input("City", placeholder="e.g. Ghaziabad")

    if st.button("🌦️ Weather"):
        if city.strip():
            st.session_state.pending = f"What's the weather in {city.strip()}?"
    if st.button("📰 Latest news"):
        if city.strip():
            st.session_state.pending = f"Give me the latest news in {city.strip()}."
    if st.button("🌆 Weather + News"):
        if city.strip():
            st.session_state.pending = f"Tell me the weather and latest news in {city.strip()}."

    st.markdown("---")
    if st.button("🗑️ Clear chat"):
        st.session_state.messages = []
        st.rerun()

    st.caption("Powered by Mistral · OpenWeather · Tavily · LangChain")

# =========================
# 🏠 Header
# =========================
st.markdown(
    """
<div class="hero">
  <h1>🌆 City Intelligence Agent</h1>
  <p>Live weather and the latest news for any city, just ask.</p>
</div>
""",
    unsafe_allow_html=True,
)

# =========================
# 💬 Chat
# =========================
if "messages" not in st.session_state:
    st.session_state.messages = []

def render_chips(tools):
    if not tools:
        return
    html = "".join(
        f'<span class="chip {"" if ok else "denied"}">'
        f'{"🔧" if ok else "⛔"} {name}</span>'
        for name, ok in tools
    )
    st.markdown(html, unsafe_allow_html=True)

if not st.session_state.messages:
    st.info("👋 Ask something like **“Weather in Delhi”** or **“News in Mumbai”**, or use the sidebar.")

for m in st.session_state.messages:
    with st.chat_message(m["role"], avatar="🧑" if m["role"] == "user" else "🤖"):
        st.markdown(m["content"])
        if m["role"] == "assistant":
            render_chips(m.get("tools"))

prompt = st.chat_input("Ask about any city...") or st.session_state.pop("pending", None)

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🤖"):
        agent_state["log"].clear()
        try:
            with st.spinner("Fetching city intelligence..."):
                result = agent.invoke(
                    {
                        "messages": [
                            {"role": m["role"], "content": m["content"]}
                            for m in st.session_state.messages
                        ]
                    }
                )
            answer = result["messages"][-1].content
        except Exception as e:
            answer = f"⚠️ Something went wrong: `{e}`"

        tools_used = list(agent_state["log"])
        st.markdown(answer)
        render_chips(tools_used)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "tools": tools_used}
    )