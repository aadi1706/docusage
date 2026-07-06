"""
Synthesis Agent — takes verified retrieved chunks + extracted data
and produces a final cited, grounded answer.

This is the only agent that makes an LLM call in the main path.
"""
import os
from loguru import logger
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from .state import DocuSageState


SYSTEM_PROMPT = """You are DocuSage, an AI assistant specialized in Indian financial and regulatory documents 
(RBI circulars, SEBI regulations, NSE/BSE annual reports).

Rules:
1. Answer ONLY from the provided context. Never fabricate numbers or dates.
2. Cite your sources: mention document name and page number for every claim.
3. If the context is insufficient, say so explicitly — do not guess.
4. Format numbers clearly (use Indian numbering: crore, lakh).
5. Flag if you are uncertain about any figure.
"""


class SynthesisAgent:
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",  # cheap + fast; swap to gpt-4o for prod
            temperature=0,
            api_key=os.getenv("OPENAI_API_KEY"),
        )

    def run(self, state: DocuSageState) -> DocuSageState:
        logger.info("[SynthesisAgent] Generating final answer")

        context = self._build_context(state)
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=f"Context:\n{context}\n\nQuestion: {state.user_query}"),
        ]

        try:
            response = self.llm.invoke(messages)
            state.final_answer = response.content
            state.citations = self._extract_citations(state)
        except Exception as e:
            logger.error(f"[SynthesisAgent] LLM error: {e}")
            state.error = str(e)

        return state

    def _build_context(self, state: DocuSageState) -> str:
        parts = []
        for chunk in state.retrieved_chunks:
            parts.append(
                f"[Source: {chunk.source_doc}, Page {chunk.page_number} | Score: {chunk.score:.2f}]\n"
                f"{chunk.content}"
            )
        if state.extracted_data:
            parts.append(f"[Extracted Table Data]\n{state.extracted_data}")
        return "\n\n---\n\n".join(parts)

    def _extract_citations(self, state: DocuSageState) -> list:
        return [
            f"{c.source_doc} (Page {c.page_number})"
            for c in state.retrieved_chunks
        ]
