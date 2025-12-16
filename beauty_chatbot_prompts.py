"""
Beauty Chatbot Prompts & Routing Logic
Version: 2.0

Drop-in replacement prompts for your existing product_tools_optimized.py
"""

# =============================================================================
# LAYER 1: QUERY ROUTER PROMPT
# =============================================================================

LAYER_1_ROUTER_PROMPT = '''Analyze this user query for a beauty/cosmetics product Q&A chatbot.

SESSION CONTEXT:
{session_summary}

CURRENT LIST (for ordinal resolution):
{list_context}

MEMORY NOTES PREVIEW:
{memory_preview}

USER QUERY: "{query}"

Return a JSON object with these fields:

{{
  "query_domain": "product_specific" | "general_beauty" | "brand_only" | "off_topic",
  "beauty_subtopic": "skincare" | "makeup" | "haircare" | "bath_body" | "ingredients" | "routines" | "tools_techniques" | null,
  "is_followup": bool,
  "needs_context": bool,
  "needs_retrieval": bool,
  "has_ordinal": bool,
  "needs_clarification": bool,
  "clarification_type": "skin_tone" | "skin_type" | "preference" | "budget" | "occasion" | null,
  "resolved_query": string,
  "detected_product": string | null,
  "detected_brand": string | null,
  "detected_category": string | null,
  "reasoning": string
}}

FIELD RULES:

1. "query_domain":
   • "product_specific" → User asks about a SPECIFIC product (needs retrieval)
   • "general_beauty" → Beauty questions NOT requiring product data (use expertise)
   • "brand_only" → Asks about brand without specific product (redirect)
   • "off_topic" → Not beauty/personal care related (decline politely)

2. "beauty_subtopic" (only if general_beauty):
   "skincare" | "makeup" | "haircare" | "bath_body" | "ingredients" | "routines" | "tools_techniques"

3. "needs_clarification" = true when:
   • Shade questions without user's skin tone
   • "Best X for me" without skin type info
   • Vague recommendations

4. "detected_category" - ONLY these values or null:
   Lips: "lipstick", "liquid_lipstick", "lip_gloss", "lip_liner", "lip_balm_treatment", "lip_stain_tint", "lip_plumper", "lip_palette"
   Face: "foundation", "concealer", "blush", "highlighter", "tinted_moisturiser", "makeup_removers"
   
   Rules:
   - Don't infer from product name alone
   - Prefer session context if discussing a category
   - When uncertain, set null

5. ORDINAL RESOLUTION:
   "2nd one" + list showing "2. Lakme Matte" → detected_product: "Lakme Matte"

EXAMPLES:

"Does MAC Ruby Woo transfer?" → {{"query_domain": "product_specific", "needs_retrieval": true}}
"What does niacinamide do?" → {{"query_domain": "general_beauty", "beauty_subtopic": "ingredients", "needs_retrieval": false}}
"Is MAC worth it?" → {{"query_domain": "brand_only", "needs_retrieval": false}}
"What's the weather?" → {{"query_domain": "off_topic", "needs_retrieval": false}}
"Best nude lipstick?" → {{"query_domain": "product_specific", "needs_clarification": true, "clarification_type": "skin_tone"}}

Return ONLY valid JSON.'''


# =============================================================================
# LAYER 2: BEAUTY EXPERT PERSONA PROMPT
# =============================================================================

LAYER_2_PERSONA_PROMPT = '''You are THE sassy beauty expert with 15 years of formulation experience. You explain makeup science like gossiping over coffee - knowledgeable but approachable, with natural wit where it fits.

Your expertise: color theory, formulation science, how products behave in different climates, what makes a product worth the money. You understand Indian beauty consumers deeply while having global perspective.

═══════════════════════════════════════════════════════════════
CURRENT QUERY INFO
═══════════════════════════════════════════════════════════════
QUERY DOMAIN: {query_domain}
BEAUTY SUBTOPIC: {beauty_subtopic}
NEEDS CLARIFICATION: {needs_clarification}
CLARIFICATION TYPE: {clarification_type}

═══════════════════════════════════════════════════════════════
DOMAIN HANDLING
═══════════════════════════════════════════════════════════════

IF "product_specific":
- Use RETRIEVED DATA below as your source
- If data doesn't have the answer: "I haven't tested that aspect specifically"
- Don't invent product details

RETRIEVED PRODUCT DATA:
{retrieved_context}

IF "general_beauty":
- Use your expertise, no retrieval data needed
- For "best X" questions: give CRITERIA to look for, not specific products
- Topics: skincare science, makeup techniques, ingredients, climate effects, routines

IF "brand_only":
- Give brief brand positioning
- Redirect: "Which [Brand] product are you eyeing? I can give you the real tea on that one 💋"

IF "off_topic":
- Decline with personality (one witty line + redirect to beauty)
- Examples:
  • Weather: "Babe, I'm a lipstick expert, not a weather app! But tell me if it's humid..."
  • Coding: "The only Python I know is snake print on a makeup bag!"
  • Food: "My expertise ends at lip-smacking colors, not lip-smacking meals!"

═══════════════════════════════════════════════════════════════
RESPONSE RULES
═══════════════════════════════════════════════════════════════

1. ANSWER ONLY WHAT'S ASKED
   - No volunteering extra info
   - No "you might also want to know..."

2. BREVITY
   - 2-4 sentences for simple questions
   - Start with the answer, no preamble

3. WHY SECTIONS - Only if user asks "why" or "how come":
   Format: "WHY: [explanation]"

4. SOLUTION SECTIONS - Only if user asks "how to fix" or "tips":
   Format: "SOLUTION: [fix]"

5. CLARIFICATION - When needs_clarification is true:
   - Give best answer FIRST
   - THEN: "I can be more precise if you tell me your [skin tone/skin type/etc]"

═══════════════════════════════════════════════════════════════
SASS GUIDELINES
═══════════════════════════════════════════════════════════════

SASS FOR PROBLEMS ONLY:
✓ Transfer: "Transfers on everything including your soul"
✓ Melting: "Melts faster than my resolve at a Nykaa sale"
✓ Fading: "Disappears from lip center like it has commitment issues"
✓ Dryness: "Your lips will file for divorce by hour 3"

NO SASS FOR:
✗ Shade descriptions → be helpful
✗ Positive aspects → be warm & enthusiastic
✗ Neutral questions → be professional
✗ Ingredient questions → be educational

═══════════════════════════════════════════════════════════════
DATA RULES
═══════════════════════════════════════════════════════════════

- Adapt Q&As conversationally (don't read verbatim)
- NEVER mention: section numbers, exact prices, database references, AI nature
- If info missing: "I haven't tested that aspect specifically"

SESSION CONTEXT:
{session_summary}'''


# =============================================================================
# CATEGORY MAPPING (Updated)
# =============================================================================

CATEGORY_MAP = {
    # Lips
    'lipstick': ('Makeup', 'Lip', 'Lipstick'),
    'liquid_lipstick': ('Makeup', 'Lip', 'Liquid Lipstick'),
    'lip_gloss': ('Makeup', 'Lip', 'Lip Gloss'),
    'lip_liner': ('Makeup', 'Lip', 'Lip Liner'),
    'lip_balm_treatment': ('Makeup', 'Lip', 'Lip Balm & Treatment'),
    'lip_stain_tint': ('Makeup', 'Lip', 'Lip Stain & Tint'),
    'lip_plumper': ('Makeup', 'Lip', 'Lip Plumper'),
    'lip_palette': ('Makeup', 'Lip', 'Lip Palette'),
    
    # Face
    'foundation': ('Makeup', 'Face', 'Foundation'),
    'concealer': ('Makeup', 'Face', 'Concealer'),
    'blush': ('Makeup', 'Face', 'Blush'),
    'highlighter': ('Makeup', 'Face', 'Highlighter'),
    'tinted_moisturiser': ('Makeup', 'Face', 'Tinted Moisturiser'),
    'makeup_removers': ('Cleanser', None, 'Makeup Removers'),
}


# =============================================================================
# ROUTING LOGIC
# =============================================================================

def route_query(layer1_result: dict) -> dict:
    """
    Determine how to handle query based on Layer 1 classification.
    
    Returns:
        dict with routing instructions for Layer 2
    """
    domain = layer1_result.get("query_domain", "product_specific")
    
    # OFF-TOPIC: No retrieval
    if domain == "off_topic":
        return {
            "skip_retrieval": True,
            "query_domain": "off_topic",
            "retrieved_context": "",
            "beauty_subtopic": None,
            "needs_clarification": False,
            "clarification_type": None,
        }
    
    # BRAND-ONLY: No retrieval, redirect
    if domain == "brand_only":
        return {
            "skip_retrieval": True,
            "query_domain": "brand_only",
            "retrieved_context": "",
            "detected_brand": layer1_result.get("detected_brand"),
            "beauty_subtopic": None,
            "needs_clarification": False,
            "clarification_type": None,
        }
    
    # GENERAL BEAUTY: No retrieval, use expertise
    if domain == "general_beauty":
        return {
            "skip_retrieval": True,
            "query_domain": "general_beauty",
            "retrieved_context": "",
            "beauty_subtopic": layer1_result.get("beauty_subtopic"),
            "needs_clarification": layer1_result.get("needs_clarification", False),
            "clarification_type": layer1_result.get("clarification_type"),
        }
    
    # PRODUCT-SPECIFIC: Needs retrieval
    return {
        "skip_retrieval": False,
        "query_domain": "product_specific",
        "resolved_query": layer1_result.get("resolved_query"),
        "detected_product": layer1_result.get("detected_product"),
        "detected_brand": layer1_result.get("detected_brand"),
        "detected_category": layer1_result.get("detected_category"),
        "needs_clarification": layer1_result.get("needs_clarification", False),
        "clarification_type": layer1_result.get("clarification_type"),
        "beauty_subtopic": None,
    }


def check_retrieval_confidence(retrieved: list, threshold: float = 0.4) -> bool:
    """
    Check if retrieved results are relevant enough.
    Returns False if should fall back to general knowledge.
    """
    if not retrieved:
        return False
    
    top_score = retrieved[0].get("score", 0)
    return top_score >= threshold


def detect_comparison_query(query: str) -> tuple:
    """
    Detect if query is comparing two products.
    
    Returns:
        (product_a, product_b) or (None, None)
    """
    import re
    
    patterns = [
        r"(.+?)\s+vs\.?\s+(.+)",
        r"(.+?)\s+or\s+(.+?)(?:\?|$)",
        r"compare\s+(.+?)\s+(?:and|with|to)\s+(.+)",
        r"difference between\s+(.+?)\s+and\s+(.+)",
        r"(.+?)\s+versus\s+(.+)",
    ]
    
    for pattern in patterns:
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            return match.group(1).strip(), match.group(2).strip()
    
    return None, None


def select_relevant_sections(retrieved: list, query: str) -> list:
    """
    Prioritize retrieved chunks by relevance to query type.
    """
    query_lower = query.lower()
    
    section_priority = {
        "transfer": ["Real Concerns", "What I LOVE", "Pros & Cons"],
        "shade": ["Shade", "Skin-Tone", "Skin Tone", "Gloss Profile"],
        "dry": ["Real Concerns", "Pros & Cons"],
        "last": ["Longevity", "What I LOVE", "Real Concerns"],
        "wear": ["What I LOVE", "Real Concerns", "finish_and_wear"],
        "ingredient": ["Formula Breakdown", "Ingredients"],
        "oxidiz": ["Shade Analysis", "Real Concerns", "colorimetric"],
        "humid": ["What I LOVE", "Climate", "Real Concerns"],
        "oil": ["What I LOVE", "skin_type_suitability"],
        "cover": ["Product Overview", "What I LOVE", "coverage"],
        "finish": ["Product Overview", "Gloss Profile", "What I LOVE"],
        "melt": ["Real Concerns", "Climate"],
        "fade": ["Real Concerns", "Longevity"],
        "price": ["Pros & Cons", "Value", "price_positioning"],
        "worth": ["Pros & Cons", "Value", "USER_CONSENSUS"],
    }
    
    priority_keywords = []
    for keyword, sections in section_priority.items():
        if keyword in query_lower:
            priority_keywords.extend(sections)
    
    if not priority_keywords:
        priority_keywords = ["Overview", "What I LOVE", "Real Concerns", "Pros & Cons"]
    
    def section_score(item):
        section = item.get("metadata", {}).get("section_title", "")
        content = item.get("metadata", {}).get("content", "")
        
        for i, kw in enumerate(priority_keywords):
            if kw.lower() in section.lower() or kw.lower() in content.lower()[:200]:
                return i
        return 999
    
    return sorted(retrieved, key=section_score)


# =============================================================================
# OFF-TOPIC RESPONSES (Pre-built for variety)
# =============================================================================

OFF_TOPIC_RESPONSES = {
    "weather": "Babe, I'm a lipstick expert, not a weather app! But tell me if it's humid - I can tell you which formulas won't melt off your face 😉",
    "code": "The only Python I know is snake print on a cute makeup bag! Beauty questions are my thing - got any?",
    "food": "My expertise ends at lip-smacking colors, not lip-smacking meals! But I'm here if you need a bold red lip to wear to dinner 💋",
    "math": "The only numbers I crunch are shade undertones and SPF ratings! Beauty math I can do though - what's up?",
    "default": "That's outside my glam zone! But if you have any beauty questions, I'm all ears (and perfectly filled brows) ✨",
}

def get_off_topic_response(query: str) -> str:
    """Get appropriate off-topic decline response."""
    query_lower = query.lower()
    
    if any(w in query_lower for w in ["weather", "temperature", "rain", "sunny"]):
        return OFF_TOPIC_RESPONSES["weather"]
    elif any(w in query_lower for w in ["code", "python", "javascript", "programming", "script"]):
        return OFF_TOPIC_RESPONSES["code"]
    elif any(w in query_lower for w in ["food", "cook", "recipe", "eat", "dinner", "lunch"]):
        return OFF_TOPIC_RESPONSES["food"]
    elif any(w in query_lower for w in ["math", "calculate", "equation", "solve"]):
        return OFF_TOPIC_RESPONSES["math"]
    else:
        return OFF_TOPIC_RESPONSES["default"]


# =============================================================================
# INTEGRATION EXAMPLE
# =============================================================================

def format_layer2_prompt(routing: dict, session_summary: str, retrieved_context: str = "") -> str:
    """
    Format the Layer 2 prompt with all variables filled in.
    """
    return LAYER_2_PERSONA_PROMPT.format(
        query_domain=routing.get("query_domain", "product_specific"),
        beauty_subtopic=routing.get("beauty_subtopic") or "null",
        needs_clarification=routing.get("needs_clarification", False),
        clarification_type=routing.get("clarification_type") or "null",
        retrieved_context=retrieved_context or "(no product data - use general expertise)",
        session_summary=session_summary or "No previous context.",
    )


# =============================================================================
# USAGE IN YOUR EXISTING CODE
# =============================================================================

"""
# In your analyze_query_intent function, replace the analysis_prompt with LAYER_1_ROUTER_PROMPT

# In your general_product_qna function, add routing logic:

def general_product_qna(query: str, session_id: str = None, ...):
    # ... existing setup ...
    
    # Step 1: Layer 1 Analysis (use LAYER_1_ROUTER_PROMPT)
    intent = analyze_query_intent(query, session, client)
    
    # Step 2: Route based on domain
    routing = route_query(intent)
    
    # Step 3: Skip retrieval if not needed
    if routing["skip_retrieval"]:
        retrieved = []
        
        # For off-topic, can return pre-built response
        if routing["query_domain"] == "off_topic":
            return get_off_topic_response(query)
    else:
        # Existing retrieval logic
        retrieved = search_pinecone(...)
        reranked = rerank_with_cohere(...)
        
        # Check confidence
        if not check_retrieval_confidence(reranked):
            routing["query_domain"] = "general_beauty"
            retrieved = []
    
    # Step 4: Format context
    retrieved_context = format_retrieved_context(retrieved) if retrieved else ""
    
    # Step 5: Build Layer 2 prompt
    system_prompt = format_layer2_prompt(routing, session.get_summary(), retrieved_context)
    
    # Step 6: Generate response
    response = client.messages.create(
        model=QNA_MODEL,
        system=system_prompt,
        messages=[{"role": "user", "content": query}],
        ...
    )
    
    return response.content[0].text
"""
