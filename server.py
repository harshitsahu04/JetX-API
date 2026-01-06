import os
from fastapi import FastAPI
from pydantic import BaseModel
from groq import Groq
from typing import Dict, List

client = Groq(api_key=os.environ["GROQ_API_KEY"])
MODEL_NAME = "llama-3.1-8b-instant"

JETX_SYSTEM_PROMPT = """<YOUR FINAL PROMPT HERE>"""

app = FastAPI()

# =========================================================
# SESSION STORE (IN-MEMORY)
# =========================================================
SESSION_STORE: Dict[str, List[dict]] = {}

# =========================================================
# REQUEST MODEL
# =========================================================
class ChatRequest(BaseModel):
    session_id: str
    message: str

# =========================================================
# CHAT ENDPOINT
# =========================================================
@app.post("/chat")
def chat(req: ChatRequest):
    # Initialize session if not exists
    if req.session_id not in SESSION_STORE:
        SESSION_STORE[req.session_id] = []

    history = SESSION_STORE[req.session_id]

    messages = [
        {"role": "system", "content": JETX_SYSTEM_PROMPT},
        *history,
        {"role": "user", "content": req.message}
    ]

    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0.4
    )

    reply = completion.choices[0].message.content

    # Persist conversation
    history.append({"role": "user", "content": req.message})
    history.append({"role": "assistant", "content": reply})

    return {"reply": reply}
