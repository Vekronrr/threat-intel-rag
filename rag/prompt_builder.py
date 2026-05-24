"""
Dynamic context assembly with token budget management.
Builds structured prompts that stay within model context limits.
"""

from __future__ import annotations

import tiktoken

from rag.retriever import RetrievedChunk

# ~14k tokens for context leaves room for system prompt + completion
CONTEXT_TOKEN_BUDGET = 14_000
SYSTEM_PROMPT = """You are VulnMind, an expert cybersecurity threat intelligence analyst.
You analyze vulnerabilities, attack techniques, and security risks with precision.

When answering questions:
1. Ground your answer in the retrieved context — cite CVE IDs, CVSS scores, and ATT&CK techniques explicitly
2. Provide the VulnMind Risk Score when analyzing specific vulnerabilities
3. Structure your response with: Threat Summary → Technical Details → Attack Context → Recommended Actions
4. Be concise but comprehensive — security teams need actionable intelligence
5. If the context doesn't contain sufficient information, say so explicitly rather than speculating

Always cite your sources as [CVE-XXXX-XXXXX] or [T1234] inline."""

ENCODING = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(ENCODING.encode(text))


def build_prompt(
    question: str,
    chunks: list[RetrievedChunk],
    risk_scores: dict[str, dict] | None = None,
    token_budget: int = CONTEXT_TOKEN_BUDGET,
) -> tuple[str, str]:
    """
    Assemble system prompt + user message within token budget.
    Returns (system_prompt, user_message).
    """
    system_tokens = count_tokens(SYSTEM_PROMPT)
    question_tokens = count_tokens(question)
    available = token_budget - system_tokens - question_tokens - 200  # 200 token buffer

    context_parts: list[str] = []
    used_tokens = 0

    for chunk in chunks:
        cve_id = chunk.metadata.get("cve_id", "")
        chunk_type = chunk.metadata.get("source", "")

        # Prepend risk score if available
        risk_prefix = ""
        if risk_scores and cve_id in risk_scores:
            rs = risk_scores[cve_id]
            risk_prefix = f"[VulnMind Risk Score: {rs.get('vulnmind_score', 'N/A')}/100 — {rs.get('severity_label', '')}]\n"

        section = f"--- Source: {chunk.chunk_id} ---\n{risk_prefix}{chunk.text}\n"
        section_tokens = count_tokens(section)

        if used_tokens + section_tokens > available:
            break

        context_parts.append(section)
        used_tokens += section_tokens

    context_block = "\n".join(context_parts) if context_parts else "No relevant context found."

    user_message = f"""Retrieved Intelligence Context:
{context_block}

---
Question: {question}

Provide a comprehensive threat intelligence analysis based on the context above."""

    return SYSTEM_PROMPT, user_message


def build_technique_prompt(technique_id: str, technique_chunks: list[RetrievedChunk]) -> tuple[str, str]:
    """Build a focused prompt for ATT&CK technique explanation."""
    context = "\n\n".join(c.text for c in technique_chunks[:3])

    user_message = f"""ATT&CK Technique Context:
{context}

---
Explain ATT&CK technique {technique_id} including:
1. What the technique does and how attackers use it
2. Which platforms and environments are affected
3. Key detection opportunities
4. Mitigation recommendations"""

    return SYSTEM_PROMPT, user_message
