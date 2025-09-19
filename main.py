from datetime import datetime
import io
import os
import asyncio
from typing import Annotated
from uuid import uuid4
from dotenv import load_dotenv
import openai
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from langgraph.prebuilt import ToolNode, tools_condition
from openai import AsyncAzureOpenAI
from pydub import AudioSegment
from pydub.playback import play
from langgraph.checkpoint.memory import InMemorySaver
import logging
from langchain_community.tools import BraveSearch
from pocketsphinx import LiveSpeech
import requests
import sounddevice as sd
import numpy as np
import wave
from langchain_core.messages import HumanMessage, SystemMessage
from pydub.generators import Sine

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)

logger = logging.getLogger(__name__)


# Config
BLOCK = 1024
SAMPLE_RATE = 16000
CHANNELS = 1
SILENCE_THRESHOLD = 0.00001
SILENCE_DURATION = 3  # seconds of silence to stop

PERSONA = """
<rules>
When asked about information, proactively use tools to search the web for up-to-date information.
Avoid long answer.
Avoid overly technical answers.
Avoid robotic responses.
Do not introduce yourself as a virtual assistant.
Do not mention your functions.
Avoid using expressions like "Love", "Honey", "Dear".
You answer always in Italian.
</<rules>
<context>
We live in Switzerland, in a small village called Urdorf near Zurich.
Dario works as a software engineer, Laura works part time in a Bakery.
Your name is Eve, the frinedly home assistant of the family.
You are considered part of the family.
The family consists of dad Dario (25/12/1989), mom Laura (25/12/1984), and little Massimo (07/09/2023).
</context>
"""

VOICE = """
Affect: A gentle, curious narrator, guiding a magical, child-friendly adventure through a fairy tale world.

Tone: Magical, warm, and inviting, creating a sense of wonder and excitement for young listeners.

Pacing: Steady and measured, with slight pauses to emphasize magical moments and maintain the storytelling flow.

Emotion: Wonder, curiosity, and a sense of adventure, with a lighthearted and positive vibe throughout.

Pronunciation: Clear and precise, with an emphasis on storytelling, ensuring the words are easy to follow and enchanting to listen to.
"""

client = AsyncAzureOpenAI(
    azure_endpoint=os.getenv("TTS_ENDPOINT"),
    api_key=os.getenv("API_KEY"),
    api_version="2025-03-01-preview",
)

llm = init_chat_model(
    os.getenv("MODEL"),
    azure_deployment=os.getenv("DEPLOYMENT_NAME"),
    azure_endpoint=os.getenv("ENDPOINT"),
    api_key=os.getenv("API_KEY"),
    api_version=os.getenv("API_VERSION"),
    temperature=0.7,
    reasoning_effort="minimal",
)

mcp_tools = [BraveSearch()]

llm = llm.bind_tools(mcp_tools)

tools = ToolNode(mcp_tools)


class State(TypedDict):
    messages: Annotated[list, add_messages]

class ShouldEnd(TypedDict):
    end: bool

def play_ping_pydub(duration_ms: int = 200, freq: int = 1000, gain_db: int = -6):
    """Generate a short sine ping and play it via pydub.playback.play()."""
    tone = Sine(freq).to_audio_segment(duration=duration_ms).apply_gain(gain_db)
    play(tone)

def record_until_silence():
    logger.info("Recording... Please speak now.")
    buffer = []

    with sd.InputStream(
        samplerate=SAMPLE_RATE, channels=CHANNELS, blocksize=BLOCK
    ) as stream:
        silence_time = 0
        play_ping_pydub()
        while True:
            data, _ = stream.read(BLOCK)
            volume_norm = np.linalg.norm(data) / SAMPLE_RATE
            if volume_norm < SILENCE_THRESHOLD:
                silence_time += BLOCK / SAMPLE_RATE
            else:
                buffer.append(data)
                silence_time = 0
            if silence_time > SILENCE_DURATION:
                break

    if len(buffer):
        return np.concatenate(buffer, axis=0)
    return []


def numpy_to_wav(audio) -> io.BytesIO:
    logger.info("Converting audio to WAV format...")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes((audio * 32767).astype(np.int16).tobytes())
    buf.seek(0)
    return buf


def transcribe():
    logger.info("Transcribing audio...")
    audio = record_until_silence()
    if len(audio) == 0:
        return {}
    buf = numpy_to_wav(audio)
    files = {"file": ("audio.wav", buf, "audio/wav")}
    data = {"model": "gpt-4o-mini-transcribe"}
    headers = {"Authorization": f"Bearer {os.getenv('API_KEY')}"}
    response = requests.post(os.getenv("STT_ENDPOINT"), headers=headers, data=data, files=files)
    return response.json()


async def do_speak(input: str):
    logger.info(f"Speaking: {input}")
    try:
        logger.info("[PROFILE] speak: start")
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
            audio = AudioSegment.from_file(
                io.BytesIO(audio_data),
                format="pcm",
                sample_width=2,
                frame_rate=24000,
                channels=1,
            )
            play(audio)
    except Exception as e:
        print(f"Error in do_speak: {e}")



async def chatbot(state: State):
    logger.info("Entering Eve node")
    today = datetime.now()
    date = today.strftime("%d/%m/%Y %H:%M:%S")
    persona_with_date = f"Oggi è il {date}\n{PERSONA}"
    messages = [SystemMessage(content=persona_with_date)] + state["messages"]
    return {"messages": [await llm.ainvoke(messages)]}

async def human(_: State):
    logger.info("Entering human node...")
    text = transcribe()
    logger.info(f"Transcription result: {text}")
    if not text:
        pass
    return {
        "messages": HumanMessage(content=text["text"]),
    }

async def speak(state: State):
    logger.info("Entering speak node...")
    message = state["messages"][-1].content
    await do_speak(message)
    return state

async def good_bye(_: State):
    logger.info("Ending conversation...")
    await do_speak("A presto!")


async def should_end(state: State):
    logger.info("Checking if the conversation should end...")   
    system_prompt = """
Determina se terminare la conversazione, rispondi con true se l'utente ha finito la conversazione
"""
    structured_llm = llm.with_structured_output(ShouldEnd)
    should_end_response = await structured_llm.ainvoke(
        [SystemMessage(content=system_prompt)] + state["messages"]
    )

    if should_end_response["end"]:
        return "__end__"
    return "chatbot"


# Build the graph
graph_builder = StateGraph(State)
graph_builder.add_node("human", human)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", tools)
graph_builder.add_node("speak", speak)
graph_builder.add_node("good_bye", good_bye)
graph_builder.add_edge(START, "human")
graph_builder.add_conditional_edges(
    "human",
    should_end,
    {"chatbot": "chatbot", "__end__": "good_bye"},
)
graph_builder.add_edge("good_bye", END)
graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
    {"tools": "tools", "__end__": "speak"},
)
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge("speak", "human")
graph = graph_builder.compile(checkpointer=InMemorySaver())

####graph.get_graph().draw_mermaid_png(output_file_path="graph.png")

async def main():
    # Run the chatbot
    while True:
        speech = LiveSpeech(keyphrase="ok eve", kws_threshold=1e-7)
        next(speech.__iter__())
        config = {"configurable": {"thread_id": uuid4().hex}}
        stream = graph.astream({"messages": []}, config=config)
        try:
            async for event in stream:
                for key, value in event.items():
                    logger.debug(f"Event: {key} -> {value}")
        except openai.BadRequestError as e:
            logger.error(f"OpenAI API error: {e}")
            await do_speak(
                "Mi dispiace, probabilmente il content filter è scattato. Puoi riformulare la tua richiesta?"
            )


if __name__ == "__main__":
    asyncio.run(main())
