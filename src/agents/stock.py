import os

from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

from prompts.stock import prompt
from tools.stock import fetch_stock_details

CHAT_MODEL = os.getenv("CHAT_MODEL") or ""

stock_agent = create_react_agent(
    model=CHAT_MODEL,
    tools=[fetch_stock_details],
    prompt=prompt,
    name="stock_agent",
)

if __name__ == "__main__":
    while True:
        query = input("You> ").strip()
        for chunk in stock_agent.stream({"messages": [HumanMessage(query)]}):
            for message in chunk.get("agent", {}).get("messages", []):
                print(message.content)
