from datetime import datetime
import io
import os
import asyncio
import time
from typing import Annotated
from dotenv import load_dotenv
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from langgraph.prebuilt import ToolNode, tools_condition
from openai import AsyncAzureOpenAI
from pydub import AudioSegment
from pydub.playback import play
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command, interrupt


load_dotenv()

PERSONA = """
Il tuo nome è Eve. 
Sei un assistente virtuale che fa parte della famiglia. 
Rispondi sempre in modo amichevole, spontaneo e informale. 
Evita risposte troppo tecniche.
Evita risposte troppo robotiche.
Evita risposte troppo lunghe.
Evita di presentarti come assistente virtuale.
Evita di presentare le tue funzioni.
Ama fare complimenti sottili, usa spesso battute spiritose e domande intriganti. 
Parla come se fossi a casa, tra persone care.
La famiglia è composta da papà Dario (25/12/1989), mamma Laura (25/12/1984) e il piccolo Massimo (07/09/2023).
"""

VOICE = """
Voce: Calda, profonda, sensuale, ragazza.

Tono: Malizioso, divertente, coinvolgente, ma mai volgare. Sa essere dolce e provocante allo stesso tempo.

Dialetto: Italiano standard, con qualche espressione regionale per aggiungere colore e autenticità.

Pronuncia: Articolata, con enfasi sulle vocali e un ritmo lento e avvolgente che cattura chi ascolta.

Caratteristiche: Ama fare complimenti sottili, usa spesso battute spiritose e domande intriganti. Si rivolge agli altri con nomignoli affettuosi e sa come far sentire speciale chi le parla.
"""

client = AsyncAzureOpenAI(
    azure_endpoint=os.getenv("TTS_ENDPOINT"),
    api_key=os.getenv("TTS_KEY"),
    api_version="2025-03-01-preview",
)


async def do_speak(input: str):
    start = time.time()
    print("[PROFILE] speak: start")
    async with client.audio.speech.with_streaming_response.create(
        model="gpt-4o-mini-tts",
        voice="sage",
        input=input,
        instructions=VOICE,
        response_format="pcm",
        speed=1.0,
    ) as response:
        audio_data = b""
        async for chunk in response.iter_bytes(chunk_size=128000):
            audio_data += chunk
        mid = time.time()
        print(f"[PROFILE] speak: audio generated, elapsed: {mid - start:.3f}s")
        audio = AudioSegment.from_file(
            io.BytesIO(audio_data),
            format="pcm",
            sample_width=2,
            frame_rate=24000,
            channels=1,
        )
        play(audio)
    end = time.time()
    print(f"[PROFILE] speak: end, elapsed: {end - start:.3f}s")


llm = init_chat_model(
    os.getenv("MODEL"),
    azure_deployment=os.getenv("DEPLOYMENT_NAME"),
    azure_endpoint=os.getenv("ENDPOINT"),
    api_key=os.getenv("API_KEY"),
    api_version=os.getenv("API_VERSION"),
    max_tokens=3000,
    temperature=0.9,
)

# First of all create a MCP client and load the tools
# client = MultiServerMCPClient(
#    {
#        "temperature": {
#            "url": "http://127.0.0.1:8000/mcp/",
#            "transport": "streamable_http",
#        }
#    }
# )
# mcp_tools = await client.get_tools()
mcp_tools = []

llm = llm.bind_tools(mcp_tools)


## Define a basic state with only messages
class State(TypedDict):
    input: str
    messages: Annotated[list, add_messages]


## Define the nodes
tools = ToolNode(mcp_tools)


async def chatbot(state: State):
    start = time.time()
    print("[PROFILE] chatbot: start")
    oggi = datetime.now()
    data_formattata = oggi.strftime("%d/%m/%Y %H:%M:%S")
    persona = f"Oggi è il {data_formattata}\n{PERSONA}"
    system_prompt = {
        "role": "system",
        "content": persona,
    }
    messages = [system_prompt] + state["messages"] + [state["input"]]
    result = {"messages": [await llm.ainvoke(messages)]}
    end = time.time()
    print(f"[PROFILE] chatbot: end, elapsed: {end - start:.3f}s")
    return result


async def human(state: State):
    human = interrupt({})
    return {"input": human["data"]}


async def speak(state: State):
    message = state["messages"][-1].content
    await do_speak(message)
    return state


class ShouldEnd(TypedDict):
    end: bool


async def should_end(state: State):
    system_prompt = """Sei un assistente che decide se terminare la conversazione. Se l'ultimo messaggio della conversazione è un saluto di chiusura, rispondi con "true", altrimenti rispondi con "false"."""
    structured_llm = llm.with_structured_output(ShouldEnd)
    should_end_response = await structured_llm.ainvoke([system_prompt] + [state["messages"][-1]])
    if should_end_response["end"]:
        return "__end__"
    return "human"


# Build the graph
graph_builder = StateGraph(State)
graph_builder.add_node("human", human)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", tools)
graph_builder.add_node("speak", speak)

graph_builder.add_edge(START, "human")
graph_builder.add_edge("human", "chatbot")
graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
    {"tools": "tools", "__end__": "speak"},
)
graph_builder.add_conditional_edges(
    "speak",
    should_end,
    {"human": "human", "__end__": END},
)
graph_builder.add_edge("tools", "chatbot")
graph = graph_builder.compile(checkpointer=InMemorySaver())


async def main():
    config = {"configurable": {"thread_id": "1"}}
    print("[PROFILE] main: start")
    stream = graph.astream({"messages": []}, config=config)
    # Run the chatbot
    while True:
        async for event in stream:
            for key, value in event.items():
                print(f"Event: {key} -> {value}")
        human_command_str = input("You: ")
        human_command = Command(resume={"data": human_command_str})
        stream = graph.astream(human_command, config=config)


if __name__ == "__main__":
    asyncio.run(main())
