"""
Router Agent — classifies the query and decides which downstream agents to invoke.

Routing logic:
  - Query mentions table/figure/chart/number → visual route (ColPali retrieval)
  - Query is purely text-based → text route (BM25 + dense hybrid)
  - Ambiguous → hybrid route (both, then merge)
"""
import re
from loguru import logger
from .state import DocuSageState


VISUAL_KEYWORDS = {
    "table", "figure", "chart", "graph", "diagram", "image", "exhibit",
    "schedule", "annexure", "balance sheet", "p&l", "cash flow",
}

TEXT_KEYWORDS = {
    "explain", "what is", "define", "describe", "summarize", "mention",
    "according to", "state", "list",
}


class RouterAgent:
    """
    Lightweight rule-based router (no LLM call → zero latency, zero cost).
    Can be upgraded to an LLM classifier once you have labelled routing data.
    """

    def run(self, state: DocuSageState) -> DocuSageState:
        logger.info(f"[RouterAgent] Query: {state.user_query[:80]}")
        query_lower = state.user_query.lower()

        visual_hits = sum(1 for kw in VISUAL_KEYWORDS if kw in query_lower)
        text_hits   = sum(1 for kw in TEXT_KEYWORDS   if kw in query_lower)

        if visual_hits > 0 and text_hits == 0:
            state.query_type = "visual"
        elif text_hits > 0 and visual_hits == 0:
            state.query_type = "text"
        else:
            state.query_type = "hybrid"

        logger.info(f"[RouterAgent] Routed to: {state.query_type}")
        return state
