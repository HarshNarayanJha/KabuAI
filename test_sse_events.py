import json
from pprint import pprint
from typing import Any, Literal

import requests
from pydantic import BaseModel, Field
from requests_sse import EventSource, InvalidContentTypeError, InvalidStatusCodeError


class Response(BaseModel):
    type: Literal["handoff", "tool", "chunk", "update"]
    # type = "handoff" | "tool"
    arguments: dict[str, Any] | None = Field(default=None)
    # type = "tool"
    name: str | None = Field(default=None)
    # type = "chunk"
    content: str | None = Field(default=None)
    # type = "update"
    state: dict[str, Any] | None = Field(default=None)


def test_chat():
    url = "http://localhost:8000/chat/"
    headers = {"Content-Type": "application/json"}
    data = {
        "state": {
            "messages": [
                {"type": "human", "content": "Latest price for Apple"},
            ],
            "ticker": None,
            "stock_data": None,
            "stock_summary": None,
        },
    }

    print(f"Sending request to {url} with data: {data}")

    with EventSource(url, method="POST", json=data, headers=headers, timeout=30) as event_source:
        try:
            for event in event_source:
                data = Response(**json.loads(event.data or "{}"))
                if data.type == "handoff":
                    print("Handoff".center(50, "="))
                    print(f"Arguments: {data.arguments}")
                elif data.type == "tool":
                    print("Tool Call".center(50, "="))
                    print(f"Name: {data.name}")
                    print(f"Arguments: {data.arguments}")
                elif data.type == "chunk":
                    print("Stream Chunk".center(50, "="))
                    print(f"Content: {data.content}")
                elif data.type == "update":
                    print("State Update".center(50, "="))
                    assert data.state is not None
                    print(f"State: {data.state.keys()}")

                    if data.state.get("next", None) == "__end__":
                        return

                print("=" * 50)
                print()
        except InvalidStatusCodeError:
            pass
        except InvalidContentTypeError:
            pass
        except requests.RequestException:
            pass


if __name__ == "__main__":
    test_chat()
