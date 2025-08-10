import os
import asyncio
from typing import Annotated
from dotenv import load_dotenv
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import ToolMessage, AIMessage


load_dotenv()


async def main():
    # First of all create a MCP client and load the tools
    client = MultiServerMCPClient(
        {
            "temperature": {
                "url": "http://127.0.0.1:8000/mcp/",
                "transport": "streamable_http",
            }
        }
    )
    mcp_tools = await client.get_tools()

    llm = init_chat_model(
        os.getenv("MODEL"),
        azure_deployment=os.getenv("DEPLOYMENT_NAME"),
        azure_endpoint=os.getenv("ENDPOINT"),
        api_key=os.getenv("API_KEY"),
        api_version=os.getenv("API_VERSION"),
        temperature=0,
    )

    llm = llm.bind_tools(mcp_tools)

    ## Define a basic state with only messages
    class State(TypedDict):
        messages: Annotated[list, add_messages]

    ## Define the nodes
    tools = ToolNode(mcp_tools)

    async def chatbot(state: State):
        system_prompt = {
            "role": "system",
            "content": "You are a helpful weather assistant. Prefer the usage of tools to answer questions (i.e. use a tool to find out coordinates of cities)",
        }
        messages = [system_prompt] + state["messages"]
        return {"messages": [await llm.ainvoke(messages)]}

    # Build the graph
    graph_builder = StateGraph(State)
    graph_builder.add_node("chatbot", chatbot)
    graph_builder.add_node("tools", tools)
    graph_builder.add_edge(START, "chatbot")
    graph_builder.add_conditional_edges(
        "chatbot",
        tools_condition,
        {"tools": "tools", "__end__": END},
    )
    graph_builder.add_edge("tools", "chatbot")
    graph = graph_builder.compile()
    graph.get_graph().draw_mermaid_png(output_file_path="graph.png")

    # Run the chatbot
    async for event in graph.astream(
        {
            "messages": [
                {"role": "user", "content": "Give me the temperature in Zurich city"}
            ]
        }
    ):
        for value in event.values():
            messages = value.get("messages", [])
            if messages:
                print("\n" + "=" * 40 + " New Event " + "=" * 40)
                msg = messages[-1]
                # Use isinstance for type checking
                if isinstance(msg, ToolMessage):
                    tool_response = getattr(msg, "content", None)
                    if tool_response:
                        print("Tool response:", tool_response)
                elif isinstance(msg, AIMessage):
                    content = getattr(msg, "content", None)
                    if content:
                        print("Message content:", content)
                    else:
                        tool_calls = getattr(msg, "tool_calls", None)
                        if tool_calls:
                            for call in tool_calls:
                                name = call.get("name")
                                args = call.get("args")
                                print(f"Tool call: {name} with args: {args}")


if __name__ == "__main__":
    asyncio.run(main())
