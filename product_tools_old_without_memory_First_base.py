

import os
import json
from typing import Optional, List, Dict, Any
from pathlib import Path
from anthropic import Anthropic
from openai import OpenAI
try:
    # Optional decorator; not required for runtime since main.py defines tool schemas explicitly
    from anthropic import beta_tool  # type: ignore
except Exception:
    def beta_tool(*args, **kwargs):  # no-op fallback
        def _decorator(func):
            return func
        return _decorator
from pinecone import Pinecone
import requests

# Initialize Pinecone using flexible env var names
_pc_api_key = os.getenv("PINECONE_API_KEY")
_pc_env = os.getenv("PINECONE_ENV") or os.getenv("PINECONE_ENVIRONMENT")
_pc_index_name = os.getenv("PINECONE_INDEX") or os.getenv("PINECONE_INDEX_NAME")
_pc_namespace = os.getenv("PINECONE_NAMESPACE") or None
_pc_dim_env = os.getenv("PINECONE_DIMENSION")
_pc_expected_dim = int(_pc_dim_env) if _pc_dim_env and _pc_dim_env.isdigit() else None

# Create Pinecone client and target index (assumes index already exists)
pc = Pinecone(api_key=_pc_api_key)
index = pc.Index(_pc_index_name)

# LLM client for in-tool disambiguation
_anthropic_client: Optional[Anthropic] = None
try:
    _anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
except Exception:
    _anthropic_client = None

def _load_prompt_text(filename: str) -> Optional[str]:
    """Load prompt text from the project root.

    tools/product_tools.py is inside tools/, so prompts are expected one level up.
    """
    try:
        root = Path(__file__).parent.parent
        path = root / filename
        if path.exists():
            return path.read_text().strip()
    except Exception:
        return None
    return None


def embed_text(text: str) -> List[float]:
    """Embed text using OpenAI's text-embedding-3-large model.

    Requires OPENAI_API_KEY in environment. Returns a list[float] suitable for Pinecone.
    """
    if not text:
        return []
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set; cannot call text-embedding-3-large")

    model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")
    client = OpenAI(api_key=api_key)
    
    try:
        resp = client.embeddings.create(model=model, input=text)
        vec = [float(v) for v in resp.data[0].embedding]
        
        # Optional dimension validation
        if _pc_expected_dim is not None and len(vec) != _pc_expected_dim:
            raise RuntimeError(
                f"Embedding dimension {len(vec)} != PINECONE_DIMENSION={_pc_expected_dim}. "
                "Make sure your Pinecone index dimension matches the embedding model."
            )
        return vec
    except Exception as e:
        raise RuntimeError(f"OpenAI Embedding API request failed: {e}")


def _matches_to_output(matches: List) -> List[Dict]:
    out = []
    for m in matches or []:
        if isinstance(m, dict):
            pid = m.get("id")
            score = m.get("score")
            meta = m.get("metadata")
        else:
            pid = getattr(m, "id", None)
            score = getattr(m, "score", None)
            meta = getattr(m, "metadata", None)
        out.append({"product_id": pid, "score": score, "metadata": meta})
    return out


@beta_tool(
    name="general_product_qna",
    description=(
        "Answers general product questions directly from Pinecone results without calling resolve_product. "
        "Use for simple QnA like ingredients, finishes, brand details, or quick facts."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "User's natural-language question (e.g., 'What ingredient use in Dr. PawPaw Lip & Eye Balm?')"},
            "category": {"type": "string", "enum": ["lipstick", "lip_balm_treatment", "lip_liner", "lip_stain_tint", "lip_gloss"], "description": "Optional category to filter search"},
            "top_k": {"type": "integer", "minimum": 5, "description": "How many candidates to fetch from Pinecone before answering"},
            "session_id": {"type": "string", "description": "Stable identifier to scope Anthropic memory across requests"},
            "memory_seed": {"type": "string", "description": "Optional facts/instructions to store in memory before answering"},
            "suppress_memory_notice": {"type": "boolean", "description": "If true, instruct model to not announce memory checks", "default": True}
        },
        "required": ["query"],
        "additionalProperties": False
    }
)
def general_product_qna(query: str,
                        category: Optional[str] = None,
                        top_k: int = 5,
                        session_id: Optional[str] = None,
                        memory_seed: Optional[str] = None,
                        suppress_memory_notice: bool = True) -> str:
    """
    General QnA over product corpus. Does NOT call resolve_product.
    Flow: embed query -> Pinecone -> prompt1.txt to compose the answer -> return plain text.
    """
    try:
        # Optional category filter aligned to Pinecone schema
        category_mapping = {
            'lip_balm_treatment': {
                'category': 'Makeup',
                'sub_category': 'Lip',
                'leaf_level_category': 'Lip Balm & Treatment'
            },
            'lipstick': {
                'category': 'Makeup',
                'sub_category': 'Lip',
                'leaf_level_category': 'Lipstick'
            },
            'lip_liner': {
                'category': 'Makeup',
                'sub_category': 'Lip',
                'leaf_level_category': 'Lip Liner'
            },
            'lip_stain_tint': {
                'category': 'Makeup',
                'sub_category': 'Lip',
                'leaf_level_category': 'Lip Stain & Tint'
            },
            'lip_gloss': {
                'category': 'Makeup',
                'sub_category': 'Lip',
                'leaf_level_category': 'Lip Gloss'
            }
        }

        # Embed the raw user query (no resolve step)
        vec = embed_text(query)
        if not vec:
            return "Sorry, I couldn't process your question right now. Please try again."

        query_params = {
            "vector": vec,
            "top_k": top_k,
            "include_values": False,
            "include_metadata": True,
            "namespace": _pc_namespace
        }
        if category and category.lower() in category_mapping:
            cat_info = category_mapping[category.lower()]
            query_params["filter"] = {
                "$and": [
                    {"category": {"$eq": cat_info['category']}},
                    {"sub_category": {"$eq": cat_info['sub_category']}},
                    {"leaf_level_category": {"$eq": cat_info['leaf_level_category']}}
                ]
            }

        try:
            results = index.query(**query_params)
            matches = getattr(results, "matches", []) or results.get("matches", [])
            retrieved = _matches_to_output(matches)
        except Exception as e:
            return f"Unable to retrieve data right now: {e}"

        # Compose final answer using prompt text and Anthropic with memory tool
        prompt_text = _load_prompt_text("/Users/ptah/Documents/QnA_Tools_Chatbot/Chatbot system message prompt.txt")
        if _anthropic_client is not None and prompt_text:
            try:
                model = os.getenv("LLM_MODEL_QNA", os.getenv("LLM_MODEL_ROUTER", "claude-haiku-4-5-20251001"))
                input_payload = {
                    "query": query,
                    "category": category,
                    "retrieved_items": retrieved,
                }
                # Use Anthropic prompt caching for the static prompt text
                system_header = prompt_text
                if suppress_memory_notice:
                    system_header += "\n\nBe concise. If you check memory, do not explicitly announce it unless the user asks."
                system_header += "\nAlways retrieve any relevant stored facts before answering and store stable, reusable facts after answering when appropriate."
                system_header += (
                    "\n\nAnswer-Only Policy:\n"
                    "- Start directly with the answer.\n"
                    "- Do not include prefaces like 'Let me check my memory', 'Checking memory', or any mention of internal steps.\n"
                    "- Do not describe tool or memory usage.\n"
                )
                system_header += (
                    "\nTool/Memory Use Policy:\n"
                    "- Silently use tools and memory when helpful.\n"
                    "- Never announce that you are reading/writing memory.\n"
                )
                # Include a stable namespace hint to help the model organize memory implicitly
                sid = session_id or os.getenv("MEMORY_SESSION_ID") or "global"
                system_header += f"\nMemory namespace: {sid}"
                system_blocks = [
                    {
                        "type": "text",
                        "text": system_header,
                        "cache_control": {"type": "ephemeral"},
                    }
                ]
                instruction = (
                    "Use the inputs below to answer the user's question using only the retrieved items when possible.\n"
                    "If asking about ingredients or product facts, extract from metadata fields.\n"
                    "Inputs (JSON):\n"
                    f"{json.dumps(input_payload, ensure_ascii=False)}\n\n"
                    "Return only the final answer (no code fences).\n"
                    "Do not include meta statements (e.g., 'Let me check my memory', 'Checking memory') or any mention of memory/tool usage."
                )
                # Mandatory: Use Anthropic beta messages API with memory tool enabled
                beta_iface = getattr(_anthropic_client, "beta", None)
                if beta_iface is None or getattr(beta_iface, "messages", None) is None:
                    raise RuntimeError(
                        "Anthropic beta messages with memory tool is not available; please upgrade SDK or enable beta access."
                    )
                # Optional: seed memory if provided
                if memory_seed:
                    try:
                        seed_instruction = (
                            "Store the following facts as long-term memory that can help with future QnA. "
                            "Focus on durable mappings, product attributes, and reusable guidance.\n\n"
                            f"{memory_seed}"
                        )
                        beta_iface.messages.create(
                            model=model,
                            max_tokens=1280,
                            temperature=0.0,
                            system=system_blocks,
                            messages=[{"role": "user", "content": seed_instruction}],
                            tools=[{"type": "memory_20250818", "name": "memory"}],
                            betas=["context-management-2025-06-27"],
                        )
                    except Exception as e:
                        pass
                msg = beta_iface.messages.create(
                    model=model,
                    max_tokens=20000,
                    temperature=0.2,
                    system=system_blocks,
                    messages=[{"role": "user", "content": instruction}],
                    tools=[{"type": "memory_20250818", "name": "memory"}],
                    betas=["context-management-2025-06-27"],
                )
                text_blocks = [getattr(b, "text", "") for b in msg.content if getattr(b, "type", None) == "text"]
                llm_out = ("\n".join(text_blocks)).strip()
                if llm_out.startswith("```"):
                    llm_out = llm_out.strip().lstrip("`")
                    llm_out = "\n".join(llm_out.splitlines()[1:]) if "\n" in llm_out else llm_out
                    if llm_out.endswith("```"):
                        llm_out = llm_out[:-3].strip()
                return llm_out
            except Exception as e:
                pass

        # Fallback: minimal textual summary from top match
        try:
            top = retrieved[0] if retrieved else None
            if top and isinstance(top, dict):
                meta = top.get("metadata", {}) or {}
                name = meta.get("product_name") or meta.get("title") or meta.get("name") or "the product"
                return f"Here's what I found about {name}: {json.dumps(meta, ensure_ascii=False)[:800]}..."
        except Exception:
            pass
        return "I couldn't find a confident answer right now. Please try rephrasing your question."
    except Exception as e:
        return f"Error: {e}"
