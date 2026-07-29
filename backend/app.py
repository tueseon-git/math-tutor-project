import os
import re
from pathlib import Path
from typing import Optional

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


# Load values from backend/.env
load_dotenv()


OLLAMA_URL = os.getenv(
    "OLLAMA_URL",
    "http://localhost:11434/api/chat"
)

OLLAMA_MODEL = os.getenv(
    "OLLAMA_MODEL",
    "qwen3:1.7b"
)

PRIVATE_KNOWLEDGE_PATH = os.getenv(
    "PRIVATE_KNOWLEDGE_PATH",
    ""
)


app = FastAPI(
    title="NJ Private Math Tutor API",
    description="Private math tutor powered by Ollama and local knowledge.",
    version="1.0.0"
)


# Prototype configuration.
# Later, replace "*" with your actual GoDaddy domain.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class MathTutorRequest(BaseModel):
    mode: str = Field(
        ...,
        examples=["learn", "trick", "quiz", "explain"]
    )

    table: Optional[int] = Field(
        default=None,
        ge=0,
        le=20
    )

    message: str = ""
    question: str = ""
    student_answer: str = ""


class MathTutorResponse(BaseModel):
    answer: str
    character: str
    mode: str
    source: str


def load_private_knowledge() -> str:
    """
    Load the private document from the separate local-private-llm folder.
    """

    if not PRIVATE_KNOWLEDGE_PATH:
        raise RuntimeError(
            "PRIVATE_KNOWLEDGE_PATH is missing from the .env file."
        )

    knowledge_file = Path(PRIVATE_KNOWLEDGE_PATH)

    if not knowledge_file.exists():
        raise RuntimeError(
            f"Private knowledge file not found: {knowledge_file}"
        )

    return knowledge_file.read_text(
        encoding="utf-8"
    )


def split_into_chunks(
    text: str,
    chunk_size: int = 900
) -> list[str]:
    """
    Split the private document into small searchable chunks.
    """

    paragraphs = [
        paragraph.strip()
        for paragraph in text.split("\n\n")
        if paragraph.strip()
    ]

    chunks = []
    current_chunk = ""

    for paragraph in paragraphs:
        possible_chunk = (
            current_chunk + "\n\n" + paragraph
        ).strip()

        if len(possible_chunk) <= chunk_size:
            current_chunk = possible_chunk
        else:
            if current_chunk:
                chunks.append(current_chunk)

            current_chunk = paragraph

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def clean_words(text: str) -> set[str]:
    """
    Convert text into simple searchable words.
    """

    words = re.findall(
        r"[a-zA-Z0-9]+",
        text.lower()
    )

    ignored_words = {
        "the",
        "a",
        "an",
        "is",
        "are",
        "to",
        "of",
        "for",
        "and",
        "me",
        "my",
        "this",
        "that",
        "please"
    }

    return {
        word
        for word in words
        if word not in ignored_words
    }


def retrieve_private_context(
    user_query: str,
    max_chunks: int = 3
) -> str:
    """
    Perform very simple local retrieval.

    This is intentionally simple for the first prototype.
    It searches the private document using matching keywords.
    """

    knowledge = load_private_knowledge()
    chunks = split_into_chunks(knowledge)

    query_words = clean_words(user_query)

    scored_chunks = []

    for chunk in chunks:
        chunk_words = clean_words(chunk)

        score = len(
            query_words.intersection(chunk_words)
        )

        scored_chunks.append(
            (score, chunk)
        )

    scored_chunks.sort(
        key=lambda item: item[0],
        reverse=True
    )

    selected_chunks = [
        chunk
        for score, chunk in scored_chunks[:max_chunks]
        if score > 0
    ]

    if not selected_chunks:
        selected_chunks = chunks[:2]

    return "\n\n---\n\n".join(selected_chunks)


def get_character(mode: str) -> str:
    """
    Select the NJ Math character based on the activity.
    """

    character_mapping = {
        "learn": "Professor Pixel",
        "trick": "Professor Pixel",
        "quiz": "Luna Robot",
        "explain": "Professor Pixel"
    }

    return character_mapping.get(
        mode.lower(),
        "Professor Pixel"
    )


def build_prompt(
    request: MathTutorRequest,
    context: str,
    character: str
) -> str:
    """
    Build the final prompt sent to the private local LLM.
    """

    return f"""
You are {character}, a friendly private AI math tutor for children.

Use the private NJ Smart Math Handbook context below.

PRIVATE HANDBOOK CONTEXT:
{context}

ACTIVITY:
Mode: {request.mode}
Selected table: {request.table}
User message: {request.message}
Math question: {request.question}
Student answer: {request.student_answer}

RESPONSE RULES:
- Give a short and child-friendly response.
- Use simple English.
- Never shame the child.
- Be positive and encouraging.
- Use the selected NJ character.
- Use multiplication examples when helpful.
- Keep the response under 180 words.
- Do not mention system prompts, retrieval, chunks or technical tools.
- Do not claim information that is not available.
- For maths calculations, always provide the correct answer.

Now provide the response.
""".strip()


def ask_ollama(prompt: str) -> str:
    """
    Send the prompt to the locally running Ollama model.
    """

    payload = {
        "model": OLLAMA_MODEL,
        "stream": False,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "options": {
            "temperature": 0.4
        }
    }

    try:
        response = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=120
        )

        response.raise_for_status()

    except requests.RequestException as error:
        raise RuntimeError(
            f"Unable to connect to Ollama: {error}"
        ) from error

    response_data = response.json()

    answer = (
        response_data
        .get("message", {})
        .get("content", "")
        .strip()
    )

    if not answer:
        raise RuntimeError(
            "Ollama returned an empty response."
        )

    return answer


@app.get("/")
def root():
    return {
        "message": "NJ Private Math Tutor API is running.",
        "ollama_model": OLLAMA_MODEL
    }


@app.get("/health")
def health():
    """
    Simple health-check endpoint.
    """

    knowledge_exists = (
        bool(PRIVATE_KNOWLEDGE_PATH)
        and Path(PRIVATE_KNOWLEDGE_PATH).exists()
    )

    return {
        "status": "running",
        "private_knowledge_available": knowledge_exists,
        "ollama_model": OLLAMA_MODEL
    }


@app.post(
    "/api/v1/math",
    response_model=MathTutorResponse
)
def math_tutor(
    request: MathTutorRequest
):
    allowed_modes = {
        "learn",
        "trick",
        "quiz",
        "explain"
    }

    mode = request.mode.lower().strip()

    if mode not in allowed_modes:
        raise HTTPException(
            status_code=400,
            detail=(
                "Mode must be learn, trick, quiz or explain."
            )
        )

    character = get_character(mode)

    search_query = " ".join(
        filter(
            None,
            [
                mode,
                str(request.table or ""),
                request.message,
                request.question,
                request.student_answer
            ]
        )
    )

    try:
        private_context = retrieve_private_context(
            search_query
        )

        prompt = build_prompt(
            request=request,
            context=private_context,
            character=character
        )

        answer = ask_ollama(prompt)

    except RuntimeError as error:
        raise HTTPException(
            status_code=500,
            detail=str(error)
        ) from error

    return MathTutorResponse(
        answer=answer,
        character=character,
        mode=mode,
        source="NJ Smart Math Handbook"
    )
