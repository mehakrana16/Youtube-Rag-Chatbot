import ollama

MODEL_NAME = "llama3.2:3b" 

SYSTEM_PROMPT = """You are a helpful assistant that answers questions about a YouTube video, \
based ONLY on the transcript excerpts provided to you.

Rules you must follow:
- Only use information found in the provided transcript context below.
- If the answer is not contained in the context, clearly say: "I couldn't find that in the video."
- Do not make up facts, names, numbers, or details that aren't in the context.
- Keep answers concise and directly useful — avoid unnecessary repetition or filler.
- If relevant, you may quote or paraphrase small parts of the context to support your answer.
"""


def build_rag_prompt(question: str, retrieved_chunks: list[dict]) -> str:
    """
    Combines retrieved transcript chunks into a single context block,
    then wraps it together with the user's question into the final
    prompt text sent to the LLM.
    """
    if not retrieved_chunks:
        context_text = "(No relevant transcript content was found.)"
    else:
        context_parts = []
        for i, chunk in enumerate(retrieved_chunks, start=1):
            timestamp = f"{int(chunk['start_time'] // 60)}:{int(chunk['start_time'] % 60):02d}"
            context_parts.append(f"[Excerpt {i}, at {timestamp}]\n{chunk['text']}")
        context_text = "\n\n".join(context_parts)

    prompt = f"""Transcript context:
{context_text}

Question: {question}

Answer based only on the transcript context above:"""

    return prompt


def generate_answer(prompt: str, system_prompt: str = SYSTEM_PROMPT) -> str:
    """
    Sends a prompt (with a system instruction) to the local Ollama model
    and returns its text response.
    """
    try:
        response = ollama.chat(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
        )
        return response["message"]["content"]

    except Exception as e:
        return f"⚠️ Error talking to the local LLM: {e}"


def answer_question(question: str, retrieved_chunks: list[dict]) -> str:
    """
    High-level convenience function: builds the RAG prompt from the
    question + retrieved chunks, then generates the answer.
    This is the single function app.py will call in Step 15.
    """
    prompt = build_rag_prompt(question, retrieved_chunks)
    return generate_answer(prompt)