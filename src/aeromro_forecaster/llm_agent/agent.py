from __future__ import annotations

import os
import json
from urllib.error import URLError
from urllib.request import Request, urlopen
from typing import Annotated, TypedDict

from aeromro_forecaster.llm_agent.tools import get_forecast, query_demand, rag_search


class AgentUnavailable(RuntimeError):
    """Raised when optional LLM dependencies or Ollama are unavailable."""


def build_agent():
    try:
        from langchain_core.messages import HumanMessage
        from langchain_core.tools import tool
        from langchain_ollama import ChatOllama
        from langgraph.graph import END, StateGraph
        from langgraph.graph.message import add_messages
        from langgraph.prebuilt import ToolNode
    except ImportError as exc:
        raise AgentUnavailable("Install langchain, langgraph, and langchain-ollama to use the analyst agent.") from exc

    @tool
    def demand_history(sku_id: str, days: int = 30) -> str:
        """Fetch recent historical demand rows for a SKU."""
        return query_demand(sku_id, days)

    @tool
    def forecast_lookup(sku_id: str, model: str | None = None) -> str:
        """Fetch forecast rows for a SKU and optional model."""
        return get_forecast(sku_id, model)

    @tool
    def maintenance_context(query: str) -> str:
        """Search local maintenance documents for contextual evidence."""
        return rag_search(query)

    tools = [demand_history, forecast_lookup, maintenance_context]
    llm = ChatOllama(
        model=os.getenv("OLLAMA_CHAT_MODEL", "llama3.2"),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        temperature=0,
    ).bind_tools(tools)

    class State(TypedDict):
        messages: Annotated[list, add_messages]

    def oracle(state: State) -> State:
        system_hint = (
            "You are a demand forecasting analyst. Use tools for data, forecasts, "
            "and document context. Cite the tool names you used in the final answer."
        )
        messages = [HumanMessage(content=system_hint), *state["messages"]]
        return {"messages": [llm.invoke(messages)]}

    def should_continue(state: State):
        last = state["messages"][-1]
        return "tools" if getattr(last, "tool_calls", None) else END

    graph = StateGraph(State)
    graph.add_node("oracle", oracle)
    graph.add_node("tools", ToolNode(tools))
    graph.set_entry_point("oracle")
    graph.add_conditional_edges("oracle", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "oracle")
    return graph.compile()


def _first_known_sku() -> str | None:
    from aeromro_forecaster.config import DB_PATH

    if not DB_PATH.exists():
        return None
    import sqlite3

    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute("SELECT id FROM demand GROUP BY id ORDER BY SUM(demand) DESC LIMIT 1").fetchone()
    return row[0] if row else None


def _extract_sku(question: str) -> str | None:
    from aeromro_forecaster.config import DB_PATH

    if not DB_PATH.exists():
        return None
    import sqlite3

    lowered = question.lower()
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute("SELECT id FROM demand GROUP BY id ORDER BY SUM(demand) DESC LIMIT 250").fetchall()
    for (sku_id,) in rows:
        if sku_id.lower() in lowered:
            return sku_id
    return _first_known_sku()


def _ollama_chat(prompt: str) -> str:
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    model = os.getenv("OLLAMA_CHAT_MODEL", "llama3.2")
    body = {
        "model": model,
        "stream": False,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an aircraft MRO demand analyst. Answer from the supplied local data. "
                    "Be concise, call out missing data, and do not invent forecasts."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    }
    request = Request(
        f"{base_url}/api/chat",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=45) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, URLError) as exc:
        raise AgentUnavailable(f"Ollama is not reachable at {base_url}. Start Ollama and pull {model}.") from exc
    return payload.get("message", {}).get("content", "").strip() or "Ollama returned an empty response."


def ask_lightweight(question: str) -> str:
    sku_id = _extract_sku(question)
    history = query_demand(sku_id, days=30) if sku_id else "No demand database or SKU was available."
    forecast = get_forecast(sku_id) if sku_id else "No forecast lookup was possible."
    context = rag_search(question)
    prompt = (
        f"Question:\n{question}\n\n"
        f"Selected SKU:\n{sku_id or 'none'}\n\n"
        f"demand_history tool output:\n{history}\n\n"
        f"forecast_lookup tool output:\n{forecast}\n\n"
        f"maintenance_context tool output:\n{context}\n\n"
        "Answer using those tool outputs. Mention which local data was missing."
    )
    try:
        return _ollama_chat(prompt)
    except AgentUnavailable as exc:
        return (
            f"{exc}\n\n"
            "Local tool summary without LLM:\n"
            f"Selected SKU: {sku_id or 'none'}\n\n"
            f"Demand history:\n{history}\n\n"
            f"Forecast:\n{forecast}\n\n"
            f"Maintenance context:\n{context}"
        )


def ask(question: str) -> str:
    if os.getenv("AEROMRO_AGENT_BACKEND", "lightweight").lower() in {"lightweight", "lite", "ollama"}:
        return ask_lightweight(question)
    try:
        from langchain_core.messages import HumanMessage
    except ImportError as exc:
        return ask_lightweight(question)
    agent = build_agent()
    result = agent.invoke({"messages": [HumanMessage(content=question)]})
    return result["messages"][-1].content


if __name__ == "__main__":
    print(ask("Which tools are available, and what can you answer about demand?"))
