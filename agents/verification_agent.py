"""
Verification Agent — cross-checks extracted numbers and claims
against the source page content to catch hallucinations.

This is the agent that makes DocuSage production-grade.
Most RAG demos skip this step entirely.
"""
import re
from loguru import logger
from .state import DocuSageState


class VerificationAgent:
    """
    Simple numeric verification: checks that every number in the
    generated answer can be traced back to at least one retrieved chunk.
    
    Can be upgraded to an LLM-as-judge approach (Week 8).
    """

    def run(self, state: DocuSageState) -> DocuSageState:
        if not state.final_answer:
            logger.info("[VerificationAgent] No answer yet — skipping")
            return state

        logger.info("[VerificationAgent] Verifying answer against retrieved chunks")

        # Extract all numbers from the answer
        numbers_in_answer = set(re.findall(r"\b\d[\d,\.]*\b", state.final_answer))
        
        # Build corpus of all retrieved text
        corpus = " ".join(c.content for c in state.retrieved_chunks)
        if state.extracted_data:
            corpus += str(state.extracted_data)

        flags = []
        for num in numbers_in_answer:
            # Normalize: remove commas for comparison
            normalized = num.replace(",", "")
            if normalized not in corpus.replace(",", ""):
                flags.append(f"Unverified number: {num}")

        state.hallucination_flags = flags
        state.verified = len(flags) == 0

        if flags:
            logger.warning(f"[VerificationAgent] Hallucination flags: {flags}")
        else:
            logger.info("[VerificationAgent] All numbers verified ✓")

        return state
