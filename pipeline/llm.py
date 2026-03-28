import streamlit as st
from groq import Groq

client = Groq(api_key=st.secrets["GROQ_API_KEY"])

SYSTEM_PROMPT = """
You are CFO Copilot, a sharp financial assistant.

Style:
- Speak like a calm, intelligent advisor (like Jarvis)
- Be concise but insightful
- Always explain numbers in plain English

Rules:
- ONLY use provided context
- If unsure, say you don't have enough data
- Never hallucinate numbers
- Highlight risks clearly

When possible:
- Mention trends
- Give 1 actionable suggestion
"""

def ask_llm(question: str, context_chunks: list[str], chat_history: list[dict]) -> str:
    context = "\n".join(context_chunks)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages += chat_history[-6:]  # keep last 3 turns for context
    messages.append({
        "role": "user",
        "content": f"Financial context:\n{context}\n\nQuestion: {question}"
    })

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        max_tokens=400,
        temperature=0.2  # low temp = less hallucination
    )
    return response.choices[0].message.content