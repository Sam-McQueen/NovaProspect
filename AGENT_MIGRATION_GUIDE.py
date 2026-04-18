"""
AGENT MIGRATION GUIDE — Responses API
======================================
How to update each agent to use core/llm.py instead of raw requests.post().

This is a reference doc, not executable code. Apply these patterns to each agent.

IMPORTANT: The only agent that should use llm_orchestrate() is the top-level
reasoning agent. Every other agent uses llm_call() or llm_vision_call().
Only the reasoning agent knows the prospecting purpose — all other agents
write unbiased geological summaries.
"""

# ═══════════════════════════════════════════════════════════════════════════
# PATTERN 1: structural_agent.py (text-only → llm_call)
# ═══════════════════════════════════════════════════════════════════════════

# BEFORE (agents/structural_agent.py lines 162-190):
#
#   import requests
#   from config.settings import GROK_API_KEY, GROK_API_BASE, ACTIVE_CONFIG
#
#   def call_grok_structural(summary, cell):
#       if not GROK_API_KEY:
#           log.error("GROK_API_KEY not set")
#           return None
#       resp = requests.post(
#           f"{GROK_API_BASE}/chat/completions",
#           headers={"Authorization": f"Bearer {GROK_API_KEY}", ...},
#           json={"model": ACTIVE_CONFIG["model"], "messages": [...], ...},
#           timeout=90,
#       )
#       return resp.json()["choices"][0]["message"]["content"]

# AFTER:
#
#   from core.llm import llm_call
#
#   def call_grok_structural(summary, cell):
#       system = "You are a structural geologist..."   # same system prompt
#       user = summary                                  # same user content
#       return llm_call(system=system, user=user)
#
# That's it. No requests import, no headers, no JSON parsing, no error handling.
# llm_call() handles retries, budget, logging, and the Responses API format.


# ═══════════════════════════════════════════════════════════════════════════
# PATTERN 2: vision_agent.py (images → llm_vision_call)
# ═══════════════════════════════════════════════════════════════════════════

# BEFORE (agents/vision_agent.py lines 239-340):
#
#   def call_grok_vision(images, cell):
#       if not GROK_API_KEY:
#           ...
#       content = [{"type": "text", "text": prompt}]
#       for img in images:
#           content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img}"}})
#       resp = requests.post(
#           f"{GROK_API_BASE}/chat/completions",
#           headers={...},
#           json={"model": ..., "messages": [{"role": "user", "content": content}], ...},
#       )

# AFTER:
#
#   from core.llm import llm_vision_call
#
#   def call_grok_vision(images_b64, cell):
#       system = "You are a remote sensing geologist..."
#       user = f"Analyze these images for cell {cell.tile_id}..."
#       images = [{"label": f"view_{i}", "b64": b} for i, b in enumerate(images_b64)]
#       return llm_vision_call(system=system, user=user, images=images)


# ═══════════════════════════════════════════════════════════════════════════
# PATTERN 3: reasoning agent (orchestrator → llm_orchestrate)
# ═══════════════════════════════════════════════════════════════════════════

# The reasoning agent is special. It's the ONLY agent that:
#   1. Knows the prospecting purpose (all others write unbiased geology)
#   2. Uses the multi-agent model (grok-4.20-multi-agent-beta-0309)
#   3. Synthesizes summaries from all other agents into rankings
#
# Use llm_orchestrate() here — it routes to the multi-agent model automatically.

# EXAMPLE:
#
#   from core.llm import llm_orchestrate
#
#   def rank_cells(level, summaries):
#       system = "You are a senior exploration geologist..."
#       user = f"Given these geological summaries for {len(summaries)} cells..."
#       return llm_orchestrate(system=system, user=user)


# ═══════════════════════════════════════════════════════════════════════════
# FULL LIST OF AGENTS TO UPDATE
# ═══════════════════════════════════════════════════════════════════════════

# Agent                     | LLM call?  | Function to use      | Priority
# --------------------------|------------|----------------------|----------
# terrain_agent.py          | Maybe      | llm_call()           | Check first
# hyperspectral_agent.py    | Maybe      | llm_call()           | Check first
# point_data_agent.py       | Maybe      | llm_call()           | Check first
# structural_agent.py       | YES        | llm_call()           | HIGH
# vision_agent.py           | YES        | llm_vision_call()    | HIGH
# lidar_agent.py            | Likely     | llm_vision_call()    | MEDIUM
# geochemistry_agent.py     | Maybe      | llm_call()           | Check first
# spectral_agent.py         | Maybe      | llm_call()           | Check first
# textual_agent.py          | Maybe      | llm_call()           | Check first
# history_agent.py          | Maybe      | llm_call()           | Check first
# vector_agent.py           | Maybe      | llm_call()           | Check first
# (reasoning/orchestrator)  | YES        | llm_orchestrate()    | HIGH

# To check which agents actually make API calls:
#   grep -rn "requests.post\|GROK_API\|chat/completions" agents/
