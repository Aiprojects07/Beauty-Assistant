"""
Product Tools v3.2 - Production-Ready Beauty Chatbot Backend

══════════════════════════════════════════════════════════════════════════════
CHANGES FROM CURRENT CODE (Comparison Table)
══════════════════════════════════════════════════════════════════════════════

┌─────────────────────────────────┬─────────────────────────────────────────┐
│ CURRENT CODE                    │ v3.2                                    │
├─────────────────────────────────┼─────────────────────────────────────────┤
│ Scattered os.getenv() calls     │ Centralized Config dataclass            │
│ top_k=80, [:40], [:10] inline   │ Named constants with rationale          │
│ print() statements              │ logging module                          │
│ Hardcoded META_MARKERS tuple    │ Regex-based META_MARKER_PATTERN         │
│ except: pass                    │ Proper error logging                    │
│ Always prints debug             │ Conditional DEBUG_* flags               │
│ No input validation             │ validate_query() function               │
│ 120-line run_with_memory_tool   │ Split into 4 focused functions          │
│ query_domain + query_type       │ Single "intent" field                   │
│ No deduplication                │ Smart product-level deduplication       │
│ Shade-level metrics             │ Averaged product-level metrics          │
│ No web validation               │ Claude Haiku web_search tool            │
│ No brand query handling         │ is_brand_query flag with search         │
│ No ingredient query handling    │ is_ingredient_query flag                │
│ No price query handling         │ is_price_query flag (graceful decline)  │
│ No negative query handling      │ is_negative_query + exclude_attributes  │
└─────────────────────────────────┴─────────────────────────────────────────┘

ARCHITECTURE:
1. Input Validation
2. Intent Analysis (Layer 1) - New flattened structure
3. Pinecone Search (100 results for diversity)
4. Cohere Rerank (top 50)
5. Smart Product Deduplication (metadata-first, regex fallback)
6. Metrics Averaging (product-level)
7. Web Validation (Claude Haiku web_search for recommend intent)
8. Layer 2 Response Generation
"""

import os
import re
import json
import time
import logging
from typing import Optional, List, Dict, Any, Tuple, Callable, Set
from pathlib import Path
from dataclasses import dataclass, field

# =============================================================================
# LOGGING SETUP
# =============================================================================
# CHANGED: From print() to proper logging module

logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(asctime)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION (CENTRALIZED)
# =============================================================================
# CHANGED: From scattered os.getenv() to centralized Config dataclass

@dataclass
class Config:
    """
    Centralized configuration with validation.
    
    CHANGE: Previously, API keys and settings were scattered:
        _pc_api_key = os.getenv("PINECONE_API_KEY")
        _pc_index_name = os.getenv("PINECONE_INDEX")
        ...repeated in multiple places
    
    NOW: All config in one place with validation method.
    """
    
    # API Keys
    PINECONE_API_KEY: str = field(default_factory=lambda: os.getenv("PINECONE_API_KEY", ""))
    ANTHROPIC_API_KEY: str = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""))
    OPENAI_API_KEY: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    # COHERE_API_KEY: str = field(default_factory=lambda: os.getenv("COHERE_API_KEY", ""))
    
    # Pinecone
    PINECONE_INDEX: str = field(default_factory=lambda: os.getenv("PINECONE_INDEX") or os.getenv("PINECONE_INDEX_NAME", ""))
    PINECONE_NAMESPACE: Optional[str] = field(default_factory=lambda: os.getenv("PINECONE_NAMESPACE"))
    
    # Models
    ROUTER_MODEL: str = field(default_factory=lambda: os.getenv("LLM_MODEL_ROUTER", "claude-haiku-4-5-20251001"))
    QNA_MODEL: str = field(default_factory=lambda: os.getenv("LLM_MODEL_QNA", "claude-haiku-4-5-20251001"))
    EMBEDDING_MODEL: str = field(default_factory=lambda: os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large"))
    # COHERE_RERANK_MODEL: str = field(default_factory=lambda: os.getenv("COHERE_RERANK_MODEL", "rerank-v3.5"))
    
    # Paths
    LAYER1_PROMPT_PATH: str = field(default_factory=lambda: os.getenv("LAYER1_PROMPT_PATH", "Layer_1_prompt.txt"))
    LAYER2_PROMPT_PATH: str = field(default_factory=lambda: os.getenv("QNA_PROMPT_PATH", "Layer_2_prompt.txt"))
    MEMORY_DIR: Path = field(default_factory=lambda: Path(os.getenv("MEMORY_DIR", "./memories")))
    
    # Feature Flags
    ENABLE_WEB_SEARCH: bool = field(default_factory=lambda: os.getenv("ENABLE_WEB_SEARCH", "true").lower() == "true")
    DEBUG_INTENT_STREAM: bool = field(default_factory=lambda: os.getenv("DEBUG_INTENT_STREAM", "false").lower() == "true")
    DEBUG_MODE: bool = field(default_factory=lambda: os.getenv("DEBUG_MODE", "false").lower() == "true")
    STREAM_FINAL_ONLY: bool = field(default_factory=lambda: os.getenv("STREAM_FINAL_ONLY", "false").lower() == "true")
    COMPARE_TOP_K_PER_ENTITY: int = field(default_factory=lambda: int(os.getenv("COMPARE_TOP_K_PER_ENTITY", 20)))
    
    # Limits
    MAX_MEMORY_FILE_SIZE: int = field(default_factory=lambda: int(os.getenv("MAX_MEMORY_FILE_SIZE", 1024 * 100)))
    MAX_TOOL_ITERATIONS: int = field(default_factory=lambda: int(os.getenv("MAX_TOOL_ITERATIONS", 8)))
    
    def validate(self) -> List[str]:
        """Return list of missing required configs."""
        missing = []
        if not self.PINECONE_API_KEY:
            missing.append("PINECONE_API_KEY")
        if not self.ANTHROPIC_API_KEY:
            missing.append("ANTHROPIC_API_KEY")
        if not self.OPENAI_API_KEY:
            missing.append("OPENAI_API_KEY")
        if not self.PINECONE_INDEX:
            missing.append("PINECONE_INDEX")
        return missing


# Initialize config
config = Config()


# =============================================================================
# NAMED CONSTANTS (WITH RATIONALE)
# =============================================================================
# CHANGED: From magic numbers to documented constants

# ┌─────────────────────────────────────────────────────────────────────────────
# │ RETRIEVAL PIPELINE CONSTANTS
# └─────────────────────────────────────────────────────────────────────────────

# Pinecone top_k: Cast wide net for maximum brand diversity
# CHANGED: From 80 to 100
# RATIONALE: Higher top_k ensures we don't miss products from underrepresented brands
#            Claude Haiku can handle the larger context efficiently
PINECONE_TOP_K = 30

# Cohere rerank top_n: Keep most relevant after semantic ranking
# CHANGED: From implicit 10 to explicit 50
# RATIONALE: 50 gives good diversity while staying within Cohere API limits
#            More candidates for deduplication to work with
# COHERE_TOP_N = 50

# Max chunks per unique product during deduplication
# NEW in v3.2
# RATIONALE: 3 chunks usually cover different aspects (performance, ingredients, value)
#            Prevents single product from dominating results
MAX_CHUNKS_PER_PRODUCT = 3

# Final products sent to Layer 2
# NEW in v3.2
# RATIONALE: ~20 products with full metadata fits well in Haiku's context
#            Provides good variety without overwhelming the response
MAX_PRODUCTS_FOR_LLM = 20

# Max products for web validation
# NEW in v3.2
# RATIONALE: 10 products keeps web search focused and fast
MAX_PRODUCTS_FOR_WEB_VALIDATION = 10

# ┌─────────────────────────────────────────────────────────────────────────────
# │ META-MARKER DETECTION CONSTANTS
# └─────────────────────────────────────────────────────────────────────────────

# Threshold for meta-only text detection
# RATIONALE: Memory announcements are typically short
META_TEXT_THRESHOLD = 160

# Session files
SESSION_STATE_FILE = "session_state.json"
LIST_INDEX_FILE = "list_index.json"


# =============================================================================
# META-MARKER DETECTION (IMPROVED)
# =============================================================================
# CHANGED: From hardcoded tuple to regex pattern

"""
BEFORE (fragile - broke if Claude phrased differently):
META_MARKERS = (
    "let me update the memory",
    "updating the memory",
    "let me check the memory",
    ...
)

AFTER (robust - catches variations):
META_MARKER_PATTERN = re.compile(r"(?i)^(i('ll|...)...")
"""

# Regex pattern for detecting memory-related preamble
META_MARKER_PATTERN = re.compile(
    r"(?i)^(i('ll|'ve| will| have)|let me|saving|updating|checking|recording|noting)"
    r".*?(memory|note|save|record|update|check)",
    re.IGNORECASE
)

# Additional patterns for stripping preamble from start of response
PREAMBLE_STRIP_PATTERNS = [
    re.compile(r"^I'll save this.*?(?:and then |then |\.)\s*", re.IGNORECASE),
    re.compile(r"^(?:now\s+)?let me (?:save|update|check|note).*?(?:first|and then|\.)\s*", re.IGNORECASE),
    re.compile(r"^I'm saving.*?(?:\.)\s*", re.IGNORECASE),
    re.compile(r"^Saving.*?(?:\.)\s*", re.IGNORECASE),
    re.compile(r"^I'll update.*?(?:and |\.)\s*", re.IGNORECASE),
    re.compile(r"^I've (?:saved|updated|noted).*?(?:\.)\s*", re.IGNORECASE),
    re.compile(r"^(?:Memory|Notes?) (?:updated|saved).*?(?:\.)\s*", re.IGNORECASE),
    re.compile(r"^I need to.*?(?:first|\.)\s*", re.IGNORECASE),
    # New: Catch connectors that leave dangling phrases like ', then give you the comparison.'
    re.compile(r"^[\s,;:\-]*(?:then|and\s+then)\s+(?:give|provide)\s+you\s+(?:the\s+)?(?:recommendations?|comparison|answer)\.?\s*", re.IGNORECASE),
    re.compile(r"^[\s,;:\-]*(?:give|let\s+me\s+give|i'?ll\s+give)\s+you\s+(?:the\s+)?(?:recommendations?|comparison|answer)\.?\s*", re.IGNORECASE),
]


def is_meta_only_text(text: str) -> bool:
    """Check if text is only memory-related meta commentary."""
    if not text or not text.strip():
        return True
    text = text.strip()
    if len(text) <= META_TEXT_THRESHOLD and META_MARKER_PATTERN.search(text):
        return True
    return False


def strip_memory_preamble(text: str) -> str:
    """Remove memory-related preamble from response start."""
    if not text:
        return ""
    result = text.strip()
    for pattern in PREAMBLE_STRIP_PATTERNS:
        result = pattern.sub("", result)
    # Clean leftover connectors/punctuation from the start
    # Remove leading commas/semicolons/dashes/spaces
    result = re.sub(r"^[\s,;:\-]+", "", result)
    # Remove residual leading connectors (e.g., 'then', 'and then')
    result = re.sub(r"^(?:then|and\s+then|and)\s+", "", result, flags=re.IGNORECASE)
    # If a stray leading comma remains (e.g., ', then ...'), strip again
    result = re.sub(r"^[\s,;:\-]+", "", result)
    return result.strip()


def extract_meaningful_text(text: str) -> str:
    """Extract only meaningful (non-meta) sentences from text."""
    if not text:
        return ""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    meaningful = [s for s in sentences if not META_MARKER_PATTERN.search(s)]
    return " ".join(meaningful).strip()


# =============================================================================
# INPUT VALIDATION (NEW)
# =============================================================================

def validate_query(query: str) -> Tuple[bool, str]:
    """
    Validate user query input.
    NEW in v3.2: Previously no validation existed.
    """
    if not query:
        return False, "Query cannot be empty"
    if not isinstance(query, str):
        return False, "Query must be a string"
    
    query = query.strip()
    
    if len(query) < 2:
        return False, "Query too short (minimum 2 characters)"
    if len(query) > 5000:
        return False, "Query too long (maximum 5000 characters)"
    
    return True, query


# =============================================================================
# CLIENT INITIALIZATION
# =============================================================================

from anthropic import Anthropic
from openai import OpenAI
from pinecone import Pinecone

# Cohere disabled
# try:
#     import cohere
# except ImportError:
#     cohere = None
#     logger.warning("Cohere not installed - reranking will be skipped")

_anthropic_client: Optional[Anthropic] = None
_pinecone_index = None


def get_anthropic_client() -> Anthropic:
    """Get or initialize Anthropic client."""
    global _anthropic_client
    if _anthropic_client is None:
        if not config.ANTHROPIC_API_KEY:
            raise RuntimeError("ANTHROPIC_API_KEY not configured")
        _anthropic_client = Anthropic(api_key=config.ANTHROPIC_API_KEY)
    return _anthropic_client


def get_pinecone_index():
    """Get or initialize Pinecone index."""
    global _pinecone_index
    if _pinecone_index is None:
        if not config.PINECONE_API_KEY or not config.PINECONE_INDEX:
            raise RuntimeError("Pinecone not configured")
        pc = Pinecone(api_key=config.PINECONE_API_KEY)
        _pinecone_index = pc.Index(config.PINECONE_INDEX)
    return _pinecone_index


# =============================================================================
# SESSION STATE MANAGER (With proper logging)
# =============================================================================

class SessionState:
    """Manages session state separately from LLM memory files."""
    
    def __init__(self, session_id: str = "global"):
        self.session_id = session_id
        self.memory_dir = config.MEMORY_DIR / session_id if session_id != "global" else config.MEMORY_DIR
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self._cache: Optional[Dict] = None
    
    def _state_path(self) -> Path:
        return self.memory_dir / SESSION_STATE_FILE
    
    def load(self) -> Dict[str, Any]:
        if self._cache is not None:
            return self._cache
        
        path = self._state_path()
        if path.exists():
            try:
                self._cache = json.loads(path.read_text(encoding="utf-8"))
                return self._cache
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Failed to load session state: {e}")
        
        self._cache = {
            "current_product": None,
            "current_brand": None,
            "current_category": None,
            "last_query": None,
            "last_answer_preview": None,
            "last_list_file": None,
            "turn_count": 0,
            "conversation_history": [],
        }
        return self._cache

    def get_note_items(self) -> Tuple[Optional[str], List[str]]:
        """Parse session_note.md and extract (topic, items)."""
        note_path = self.memory_dir / "session_note.md"
        if not note_path.exists():
            return None, []
        try:
            text = note_path.read_text(encoding="utf-8")
        except IOError as e:
            logger.warning(f"Failed to read session note: {e}")
            return None, []
        
        topic = None
        items: List[str] = []
        for line in text.splitlines():
            if not topic and line.strip().lower().startswith("topic:"):
                topic = line.split(":", 1)[1].strip() or None
            stripped = line.strip()
            if stripped and stripped[0].isdigit():
                name_part = re.sub(r"^\d+[\.)]\s*", "", stripped)
                name_part = name_part.replace("**", "").split(" - ", 1)[0].strip()
                if name_part:
                    items.append(name_part)
        return topic, items

    def get_latest_note_items(self) -> Tuple[Optional[str], List[str], Optional[str]]:
        """Scan memory dir for most recent .md note."""
        latest_path: Optional[Path] = None
        latest_mtime = -1.0
        try:
            for f in self.memory_dir.iterdir():
                if f.is_file() and f.suffix == ".md" and not f.name.startswith("."):
                    mtime = f.stat().st_mtime
                    if mtime > latest_mtime:
                        latest_mtime = mtime
                        latest_path = f
        except OSError as e:
            logger.warning(f"Failed to list memory dir: {e}")
            return None, [], None
        
        if not latest_path:
            return None, [], None
        
        try:
            text = latest_path.read_text(encoding="utf-8")
        except IOError as e:
            logger.warning(f"Failed to read note file: {e}")
            return None, [], latest_path.name
        
        topic = None
        items: List[str] = []
        for line in text.splitlines():
            if not topic and line.strip().lower().startswith("topic:"):
                topic = line.split(":", 1)[1].strip() or None
            stripped = line.strip()
            if stripped and stripped[0].isdigit():
                name_part = re.sub(r"^\d+[\.)]\s*", "", stripped)
                name_part = name_part.replace("**", "").split(" - ", 1)[0].strip()
                if name_part:
                    items.append(name_part)
        return topic, items, latest_path.name
    
    def save(self) -> None:
        if self._cache is None:
            return
        try:
            self._state_path().write_text(
                json.dumps(self._cache, indent=2, ensure_ascii=False),
                encoding="utf-8"
            )
        except IOError as e:
            logger.error(f"Failed to save session state: {e}")
    
    def update(self, **kwargs) -> None:
        state = self.load()
        entry: Dict[str, Any] = {}
        if "last_query" in kwargs and kwargs["last_query"]:
            entry["query"] = kwargs["last_query"]
        if "current_category" in kwargs and kwargs["current_category"]:
            entry["topic"] = kwargs["current_category"]
        if "current_product" in kwargs and kwargs["current_product"]:
            entry["product"] = kwargs["current_product"]
        if entry:
            entry["turn"] = state.get("turn_count", 0) + 1
            state.setdefault("conversation_history", []).append(entry)
        state.update(kwargs)
        state["turn_count"] = state.get("turn_count", 0) + 1
        self._cache = state
        self.save()

    def _list_index_path(self) -> Path:
        return self.memory_dir / LIST_INDEX_FILE

    def get_list_index(self) -> Dict[str, Any]:
        path = self._list_index_path()
        if not path.exists():
            return {"lists": [], "current_list_id": None}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            data.setdefault("lists", [])
            data.setdefault("current_list_id", None)
            return data
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Failed to load list index: {e}")
            return {"lists": [], "current_list_id": None}

    def get_current_list(self) -> Optional[Dict[str, Any]]:
        idx = self.get_list_index()
        cur = idx.get("current_list_id")
        if not cur:
            topic, items = self.get_note_items()
            if not items:
                topic, items, fname = self.get_latest_note_items()
            else:
                fname = "session_note.md"
            if items:
                return {"id": "note_current", "topic": topic or "recent list", "items": items, "source_file": fname or "(unknown)"}
            return None
        
        for item in idx.get("lists", []):
            if isinstance(item, dict) and item.get("id") == cur:
                return item
        return None

    def save_list_index(self, items: List[str], topic: str, source_file: Optional[str] = None) -> None:
        idx = self.get_list_index()
        lists = idx.get("lists", [])
        new_id = f"list_{len(lists)+1}"
        entry = {"id": new_id, "topic": topic, "items": items, "source_file": source_file, "created_at": time.strftime("%Y-%m-%d %H:%M:%S")}
        lists.append(entry)
        idx["lists"] = lists
        idx["current_list_id"] = new_id
        try:
            self._list_index_path().write_text(json.dumps(idx, indent=2, ensure_ascii=False), encoding="utf-8")
            logger.info(f"Saved list_index with {len(items)} items (topic='{topic}')")
        except IOError as e:
            logger.error(f"Failed to save list index: {e}")
    
    def get_summary(self) -> str:
        state = self.load()
        parts = []
        if state.get("current_product"):
            parts.append(f"Current product: {state['current_product']}")
        if state.get("current_brand"):
            parts.append(f"Current brand: {state['current_brand']}")
        if state.get("current_category"):
            parts.append(f"Current category: {state['current_category']}")
        if state.get("last_query"):
            parts.append(f"Last question: {state['last_query']}")
        hist = state.get("conversation_history") or []
        if hist:
            tail = hist[-2:] if len(hist) >= 2 else hist
            htxt = "; ".join([f"#{h.get('turn')}: {h.get('query')}" for h in tail if h.get('query')])
            if htxt:
                parts.append(f"Recent turns: {htxt}")
        return "\n".join(parts) if parts else "No previous context available."
    
    def clear(self) -> None:
        self._cache = {"current_product": None, "current_brand": None, "current_category": None, "last_query": None, "last_answer_preview": None, "last_list_file": None, "turn_count": 0, "conversation_history": []}
        self.save()
    
    def get_memory_files_content(self) -> str:
        contents = []
        try:
            for f in sorted(self.memory_dir.iterdir()):
                if f.suffix == ".md" and not f.name.startswith("."):
                    try:
                        text = f.read_text(encoding="utf-8")
                        if len(text) > 2000:
                            text = text[:2000] + "\n... (truncated)"
                        contents.append(f"=== {f.name} ===\n{text}")
                    except IOError as e:
                        logger.warning(f"Failed to read memory file {f.name}: {e}")
        except OSError as e:
            logger.warning(f"Failed to list memory files: {e}")
        return "\n\n".join(contents) if contents else ""


# =============================================================================
# OFF-TOPIC RESPONSE HANDLER
# =============================================================================

OFF_TOPIC_RESPONSES = {
    "weather": "I'm a lipstick expert, not a weather app! But if it's humid, I can tell you which formulas won't melt 😉",
    "code": "The only Python I know is snake print on a cute makeup bag! Beauty questions are my thing - got any?",
    "food": "My expertise ends at lip-smacking colors, not lip-smacking meals! But I'm here if you need a bold red lip 💋",
    "math": "The only numbers I crunch are shade undertones and SPF ratings! Beauty math I can help with though.",
    "default": "That's outside my glam zone! But if you have beauty questions, I'm all ears (and perfectly filled brows) ✨",
}


def get_off_topic_response(query: str) -> str:
    """Return appropriate off-topic response based on query content."""
    q = query.lower()
    if any(w in q for w in ["weather", "temperature", "rain", "sunny", "cold"]):
        return OFF_TOPIC_RESPONSES["weather"]
    elif any(w in q for w in ["code", "python", "javascript", "programming"]):
        return OFF_TOPIC_RESPONSES["code"]
    elif any(w in q for w in ["food", "cook", "recipe", "eat", "dinner"]):
        return OFF_TOPIC_RESPONSES["food"]
    elif any(w in q for w in ["math", "calculate", "equation", "solve"]):
        return OFF_TOPIC_RESPONSES["math"]
    return OFF_TOPIC_RESPONSES["default"]


# =============================================================================
# SMART PRODUCT DEDUPLICATION (NEW in v3.2)
# =============================================================================
"""
┌─────────────────────────────────────────────────────────────────────────────
│ DEDUPLICATION STRATEGY - 3 TIER APPROACH
├─────────────────────────────────────────────────────────────────────────────
│
│ PROBLEM: Pinecone stores data at SHADE level:
│   - Top 5 results = 5 shades of same product (Focallure NU03, NU05, PK01...)
│   - Limited brand diversity
│
│ SOLUTION: Smart 3-tier grouping
│
│   TIER 1 (Best): Use metadata fields directly
│     - If "product_line" exists → use brand + product_line
│     - If "sku_family" exists → use that
│
│   TIER 2 (Good): Regex extraction from product_name
│     - Strip shade codes/names using patterns
│     - "Focallure Lasting Matte NU03 Maple" → "Focallure Lasting Matte"
│
│   TIER 3 (Fallback): Brand + Category
│     - Group by brand + leaf_level_category
│     - Less precise but still better than no deduplication
│
└─────────────────────────────────────────────────────────────────────────────
"""

# Shade extraction regex patterns (applied in order, first match wins)
# UPDATED: Added more patterns based on actual Pinecone data analysis
SHADE_EXTRACTION_PATTERNS = [
    # Pattern: "Product Name 70 Amazonian" → "Product Name"
    re.compile(r'\s+\d{1,3}\s+[A-Z][a-zA-Z\s]+$'),
    
    # Pattern: "Product Name NU03 Maple Nude" → "Product Name"
    re.compile(r'\s+[A-Z]{1,3}\d{1,3}\s+[A-Z][a-zA-Z\s]+$'),
    
    # Pattern: "Product Name #Nu02" or "Product Name #5 Red" → "Product Name"
    re.compile(r'\s+#[A-Za-z]*\d+\s*[A-Za-z\s]*$'),
    
    # Pattern: "Product Name - 01 Rose" → "Product Name"
    re.compile(r'\s+-\s*\d+\s+[A-Za-z\s]+$'),
    
    # Pattern: "Product Name Merry Berry - 004" → tries to extract
    # NEW: Handles "Shade Name - Code" format at end
    re.compile(r'\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+-\s*\d{2,4}$'),
    
    # Pattern: "Product Name (01)" → "Product Name"
    re.compile(r'\s+\(\d+\)\s*$'),
    
    # Pattern: "Product Name Shade 1" → "Product Name"
    re.compile(r'\s+Shade\s+\d+.*$', re.IGNORECASE),
    
    # Pattern: "Product Name No. 5" → "Product Name"
    re.compile(r'\s+No\.?\s*\d+.*$', re.IGNORECASE),
    
    # Pattern: "Product Name - Nude Pink" (color name only after dash)
    re.compile(r'\s+-\s+[A-Z][a-z]+\s+[A-Z][a-z]+$'),
    
    # Pattern: "Product Name Barely Brown 29" → "Product Name" (shade name + number)
    # NEW: Handles "Shade Name Number" format
    re.compile(r'\s+[A-Z][a-z]+\s+[A-Z][a-z]+\s+\d{1,3}$'),
    
    # Pattern: "Product Name 225 Delicate" → "Product Name" (number + shade name)
    re.compile(r'\s+\d{1,3}\s+[A-Z][a-z]+$'),
]

# Patterns that should NOT be stripped (part of product name, not shade)
KEEP_PATTERNS = ["9to5", "24H", "16H", "2in1", "3in1"]


def get_product_grouping_key(metadata: Dict) -> str:
    """
    Get unique key for grouping shades of same product.
    Uses 3-tier approach: metadata → clean product_name → regex → brand+category
    
    UPDATED: Better handling of actual Pinecone data structure
    """
    brand = metadata.get("brand", "").strip()
    
    # TIER 1: Direct metadata fields (most reliable)
    # product_line is the best source when available
    if metadata.get("product_line"):
        return f"{brand}|{metadata['product_line']}"
    if metadata.get("sku_family"):
        return f"{brand}|{metadata['sku_family']}"
    if metadata.get("product_family"):
        return f"{brand}|{metadata['product_family']}"
    
    # TIER 2: Check if product_name is already clean (shade in separate field)
    # attrs::N chunks typically have clean product_name
    product_name = metadata.get("product_name") or metadata.get("title") or ""
    shade = metadata.get("shade", "")
    
    # If shade exists AND product_name doesn't contain the shade, product_name is clean
    if shade and shade not in product_name:
        return f"{brand}|{product_name}"
    
    # TIER 3: Try regex extraction if product_name contains shade
    base_name = extract_product_base_name(product_name)
    if base_name and base_name != product_name:
        return f"{brand}|{base_name}"
    
    # TIER 4: Brand + Category fallback
    category = metadata.get("leaf_level_category") or metadata.get("sub_category") or metadata.get("category") or ""
    if brand and category:
        return f"{brand}|{category}|{product_name}"
    
    return f"{brand}|{product_name}"


def get_clean_product_name(metadata: Dict) -> str:
    """
    Get clean product name for display (without shade).
    
    PRIORITY:
    1. brand + product_line (most reliable)
    2. brand + product_name (if product_name is clean)
    3. Regex extraction (fallback)
    """
    brand = metadata.get("brand", "").strip()
    product_line = metadata.get("product_line", "").strip()
    product_name = metadata.get("product_name") or metadata.get("title") or ""
    shade = metadata.get("shade", "")
    
    # BEST: Use product_line if available
    if product_line:
        if brand and brand.lower() not in product_line.lower():
            return f"{brand} {product_line}"
        return product_line
    
    # GOOD: If product_name is clean (shade is separate field)
    if shade and shade not in product_name:
        if brand and brand.lower() not in product_name.lower():
            return f"{brand} {product_name}"
        return product_name
    
    # FALLBACK: Try regex extraction
    base_name = extract_product_base_name(product_name)
    if brand and brand.lower() not in base_name.lower():
        return f"{brand} {base_name}"
    return base_name


def get_section_key(metadata: Dict) -> str:
    """
    Extract section identifier from chunk for diversity tracking.
    
    Returns section_key like 'attrs::5', 'attrs::11', 'product', 'sec-4'
    """
    # Try section_key field first
    if metadata.get("section_key"):
        return metadata["section_key"]
    
    # Try to extract from parent_id (e.g., "SKU::attrs::5" or "SKU::product")
    parent_id = metadata.get("parent_id", "")
    if "::" in parent_id:
        parts = parent_id.split("::")
        if len(parts) >= 2:
            # Return everything after the SKU
            return "::".join(parts[1:])
    
    # Fallback to section_index
    section_idx = metadata.get("section_index")
    if section_idx is not None:
        return f"section_{int(section_idx)}"
    
    return "unknown"


def extract_product_base_name(full_name: str) -> str:
    """
    Extract base product name without shade/color variants using regex.
    
    FALLBACK method - only used when metadata fields don't provide clean name.
    
    Examples:
        "Maybelline SuperStay Matte Ink 70 Amazonian" → "Maybelline SuperStay Matte Ink"
        "Focallure Airy Velvet Lipcream #Nu02" → "Focallure Airy Velvet Lipcream"
        "Daily Life FOREVER52 Sensational Lip Merry Berry - 004" → tries to extract
    """
    if not full_name:
        return "Unknown Product"
    
    result = full_name.strip()
    
    # Check for patterns we should keep (avoid false positives)
    has_keep_pattern = any(keep.lower() in result.lower() for keep in KEEP_PATTERNS)
    
    if not has_keep_pattern:
        # Apply shade extraction patterns
        for pattern in SHADE_EXTRACTION_PATTERNS:
            new_result = pattern.sub('', result)
            if new_result != result and len(new_result) > 5:
                result = new_result.strip()
                break
    
    # Clean trailing punctuation
    result = re.sub(r'[\s\-]+$', '', result)
    
    return result if result else full_name


def dedupe_by_product(retrieved: List[Dict], max_chunks_per_product: int = MAX_CHUNKS_PER_PRODUCT) -> List[Dict]:
    """
    Deduplicate retrieved chunks by product, keeping top N per unique product.
    
    UPDATED: Now prioritizes SECTION DIVERSITY to get different types of data
    (e.g., attrs::5 for ingredients, attrs::11 for issues, attrs::2 for performance)
    
    Strategy:
    1. Group all chunks by product
    2. For each product, try to get diverse section types first
    3. Fill remaining slots with highest-scored chunks
    """
    if not retrieved:
        return []
    
    # First pass: group by product
    product_chunks: Dict[str, List[Dict]] = {}
    
    for item in retrieved:
        metadata = item.get("metadata", {})
        grouping_key = get_product_grouping_key(metadata)
        base_name = get_clean_product_name(metadata)
        section_key = get_section_key(metadata)
        
        # Add computed fields to metadata
        item["metadata"]["product_grouping_key"] = grouping_key
        item["metadata"]["product_base_name"] = base_name
        item["metadata"]["_section_key"] = section_key
        
        if grouping_key not in product_chunks:
            product_chunks[grouping_key] = []
        product_chunks[grouping_key].append(item)
    
    # Second pass: select diverse chunks for each product
    result: List[Dict] = []
    
    for grouping_key, chunks in product_chunks.items():
        selected = _select_diverse_chunks(chunks, max_chunks_per_product)
        result.extend(selected)
    
    # Sort by original score to maintain relevance order
    result = sorted(result, key=lambda x: x.get("score", 0), reverse=True)
    
    logger.info(f"Dedupe: {len(retrieved)} → {len(result)} ({len(product_chunks)} unique products)")
    return result


def _select_diverse_chunks(chunks: List[Dict], max_chunks: int) -> List[Dict]:
    """
    Select chunks prioritizing section diversity.
    
    Priority sections (in order of importance for recommendations):
    1. 'product' or 'sec-' - Product overview
    2. 'attrs::2' or 'section_04' - Performance metrics
    3. 'attrs::11' or 'section_13' - Issue flags
    4. 'attrs::5' or 'section_07' - Formula/ingredients
    5. 'attrs::1' or 'section_03' - Finish description
    """
    if len(chunks) <= max_chunks:
        return chunks
    
    # Sort by score first
    sorted_chunks = sorted(chunks, key=lambda x: x.get("score", 0), reverse=True)
    
    # Priority section patterns
    priority_patterns = [
        ["product", "sec-"],           # Product overview
        ["attrs::2", "section_04"],     # Performance
        ["attrs::11", "section_13"],    # Issue flags
        ["attrs::5", "section_07"],     # Formula/ingredients
        ["attrs::1", "section_03"],     # Finish
    ]
    
    selected: List[Dict] = []
    used_indices: Set[int] = set()
    
    # First: try to get one chunk from each priority section
    for patterns in priority_patterns:
        if len(selected) >= max_chunks:
            break
        for idx, chunk in enumerate(sorted_chunks):
            if idx in used_indices:
                continue
            section = chunk.get("metadata", {}).get("_section_key", "")
            if any(p in section for p in patterns):
                selected.append(chunk)
                used_indices.add(idx)
                break
    
    # Second: fill remaining slots with highest-scored chunks we haven't used
    for idx, chunk in enumerate(sorted_chunks):
        if len(selected) >= max_chunks:
            break
        if idx not in used_indices:
            selected.append(chunk)
            used_indices.add(idx)
    
    return selected


# =============================================================================
# PRODUCT AGGREGATION (SIMPLIFIED - v3.2)
# =============================================================================
"""
┌─────────────────────────────────────────────────────────────────────────────
│ PRODUCT AGGREGATION - NO HARDCODED FIELD LISTS
├─────────────────────────────────────────────────────────────────────────────
│
│ KEY INSIGHT: Your metrics are in the `content` text field, not as separate
│ metadata keys. So we pass ALL metadata/content to Layer 2 for the LLM to parse.
│
│ APPROACH:
│   1. Use `shade` field directly from metadata (not extracted from product_name)
│   2. Pass ALL metadata to Layer 2 without filtering
│   3. Let LLM parse the `content` text for metrics
│   4. Collect ANY numeric/boolean/text fields dynamically (no hardcoded lists)
│
└─────────────────────────────────────────────────────────────────────────────
"""

# Fields to EXCLUDE from dynamic aggregation (internal/structural fields)
EXCLUDE_FROM_AGGREGATION = {
    "content", "content_len", "chunk_index", "total_chunks", "parent_id",
    "section_index", "section_key", "section_title", "language",
    "product_grouping_key", "product_base_name", "_section_key",
    "sku", "product_id", "unique_code",
}


def aggregate_products_for_display(retrieved: List[Dict]) -> List[Dict]:
    """
    Aggregate multiple shade entries into single product entries.
    
    SIMPLIFIED v3.2:
    - Uses `shade` field directly from metadata
    - Dynamically collects ALL metadata fields (no hardcoded lists)
    - Passes full content to Layer 2 for LLM parsing
    """
    if not retrieved:
        return []
    
    products: Dict[str, Dict] = {}
    
    for item in retrieved:
        metadata = item.get("metadata", {})
        grouping_key = metadata.get("product_grouping_key") or get_product_grouping_key(metadata)
        base_name = metadata.get("product_base_name") or get_clean_product_name(metadata)
        
        if grouping_key not in products:
            products[grouping_key] = {
                "product_base_name": base_name,
                "brand": metadata.get("brand"),
                "category": metadata.get("leaf_level_category") or metadata.get("sub_category") or metadata.get("category"),
                "product_line": metadata.get("product_line"),
                "product_type": metadata.get("product_type"),
                "shades_seen": [],
                "skus_seen": [],
                "sections_seen": [],
                "best_score": item.get("score", 0),
                "best_item": item,
                "all_chunks": [],  # Keep full chunks for Layer 2
                "_dynamic_values": {},  # Dynamically collected metrics
            }
        
        product = products[grouping_key]
        
        # Track shade using the `shade` field directly (not extracted from product_name)
        shade = metadata.get("shade", "")
        if shade and shade not in product["shades_seen"]:
            product["shades_seen"].append(shade)
        
        # Track SKUs for reference
        sku = metadata.get("sku", "")
        if sku and sku not in product["skus_seen"]:
            product["skus_seen"].append(sku)
        
        # Track sections for diversity info
        section_key = metadata.get("_section_key") or metadata.get("section_key") or ""
        section_title = metadata.get("section_title", "")
        if section_title and section_title not in product["sections_seen"]:
            product["sections_seen"].append(section_title)
        
        # Track best score
        score = item.get("score", 0)
        if score > product["best_score"]:
            product["best_score"] = score
            product["best_item"] = item
        
        # Store full chunk (with content) for Layer 2
        product["all_chunks"].append({
            "score": score,
            "section": section_title,
            "content": metadata.get("content", ""),
            "shade": shade,
        })
        
        # Dynamically collect ALL metadata fields
        _collect_dynamic_metrics(product, metadata)
    
    # Build final list
    result = [_build_aggregated_product(p) for p in products.values()]
    result = sorted(result, key=lambda x: x["relevance_score"], reverse=True)
    
    logger.info(f"Aggregated: {len(result)} unique products from {len(retrieved)} entries")
    return result


def _collect_dynamic_metrics(product: Dict, metadata: Dict) -> None:
    """
    Dynamically collect ALL metadata fields without hardcoded lists.
    Automatically detects numeric, boolean, and text values.
    """
    for field, value in metadata.items():
        # Skip excluded fields
        if field in EXCLUDE_FROM_AGGREGATION:
            continue
        
        # Skip None values
        if value is None:
            continue
        
        # Initialize field storage if needed
        if field not in product["_dynamic_values"]:
            product["_dynamic_values"][field] = {"values": [], "type": None}
        
        field_data = product["_dynamic_values"][field]
        
        # Detect and store value based on type
        if isinstance(value, bool):
            field_data["type"] = "boolean"
            field_data["values"].append(value)
        elif isinstance(value, (int, float)):
            field_data["type"] = "numeric"
            field_data["values"].append(float(value))
        elif isinstance(value, str):
            # Try to detect if string is actually numeric or boolean
            stripped = value.strip().lower()
            if stripped in ("true", "yes", "1"):
                field_data["type"] = "boolean"
                field_data["values"].append(True)
            elif stripped in ("false", "no", "0"):
                field_data["type"] = "boolean"
                field_data["values"].append(False)
            else:
                # Try numeric conversion
                try:
                    num_val = float(value)
                    field_data["type"] = "numeric"
                    field_data["values"].append(num_val)
                except (ValueError, TypeError):
                    # Keep as text
                    field_data["type"] = "text"
                    if value not in field_data["values"]:
                        field_data["values"].append(value)


def _build_aggregated_product(product: Dict) -> Dict:
    """Build final aggregated product with computed metrics."""
    
    # Compute aggregated values from dynamic collection
    aggregated_metrics = {}
    
    for field, data in product.get("_dynamic_values", {}).items():
        values = data.get("values", [])
        value_type = data.get("type")
        
        if not values:
            continue
        
        if value_type == "numeric":
            # Average numeric values
            avg = sum(values) / len(values)
            # Round based on field name hints
            if any(kw in field.lower() for kw in ["hour", "time", "duration", "wear"]):
                aggregated_metrics[field] = round(avg, 1)
            elif any(kw in field.lower() for kw in ["score", "rating", "level"]):
                aggregated_metrics[field] = round(avg, 1)
            elif any(kw in field.lower() for kw in ["price", "mrp", "cost"]):
                aggregated_metrics[field] = round(avg, 0)
            else:
                aggregated_metrics[field] = round(avg, 2) if avg != int(avg) else int(avg)
                
        elif value_type == "boolean":
            # Majority vote
            true_count = sum(1 for v in values if v)
            aggregated_metrics[field] = true_count > len(values) / 2
            
        elif value_type == "text":
            # Combine unique text values
            unique_values = list(dict.fromkeys(values))  # Preserve order, remove dupes
            if len(unique_values) == 1:
                aggregated_metrics[field] = unique_values[0]
            else:
                aggregated_metrics[field] = unique_values  # Keep as list for multiple values
    
    # Get representative metadata from best item
    best_metadata = product["best_item"].get("metadata", {}) if product.get("best_item") else {}
    
    return {
        "product": product["product_base_name"],
        "brand": product["brand"],
        "category": product["category"],
        "product_line": product.get("product_line"),
        "product_type": product.get("product_type"),
        "shades_available": product["shades_seen"],
        "shades_count": len(product["shades_seen"]),
        "skus": product["skus_seen"],
        "sections_covered": product["sections_seen"],
        "relevance_score": round(product["best_score"], 4),
        "aggregated_metrics": aggregated_metrics,
        "detailed_data": {
            "full_name": best_metadata.get("full_name") or best_metadata.get("product_name"),
            "sku": best_metadata.get("sku"),
            "content_preview": (best_metadata.get("content") or "")[:500],  # Preview for context
        },
        "all_chunks": product["all_chunks"],  # Full content for Layer 2
    }


def get_unique_product_names(aggregated_products: List[Dict]) -> List[str]:
    """Extract list of unique product names for web search validation."""
    return [p.get("product", "") for p in aggregated_products 
            if p.get("product") and p.get("product") != "Unknown Product"]


# =============================================================================
# DISABLE GROUPING/DEDUP/AGGREGATION (OVERRIDES)
# =============================================================================
# Per request: bypass shade/product grouping, deduplication, and dynamic aggregation.
# Keep the original implementations above intact but override their usage here to
# avoid complex processing. This ensures the pipeline continues to run.
DISABLE_GROUPING_AND_AGGREGATION = True

if DISABLE_GROUPING_AND_AGGREGATION:
    def dedupe_by_product(retrieved: List[Dict], max_chunks_per_product: int = MAX_CHUNKS_PER_PRODUCT) -> List[Dict]:
        """Bypass deduplication and return the retrieved list as-is."""
        return retrieved or []

    def aggregate_products_for_display(retrieved: List[Dict]) -> List[Dict]:
        """Minimal passthrough aggregation to keep Layer 2 input shape stable."""
        if not retrieved:
            return []
        products: List[Dict] = []
        for item in retrieved:
            md = item.get("metadata", {}) or {}
            products.append({
                "product_base_name": md.get("product_name") or md.get("full_name") or md.get("title") or "Unknown Product",
                "brand": md.get("brand"),
                "category": md.get("leaf_level_category") or md.get("sub_category") or md.get("category"),
                "shades_seen": [md.get("shade")] if md.get("shade") else [],
                "shades_count": 1 if md.get("shade") else 0,
                "best_score": item.get("score", 0) or 0.0,
                "aggregated_metrics": {},
                "representative_metadata": md,
            })
        return products


# =============================================================================
# WEB SEARCH VALIDATION (NEW in v3.2) - DISABLED
# =============================================================================
"""
PURPOSE: Validate OUR products against market popularity (not discover new ones)
WHEN: Only for "recommend" intent queries
HOW: Use Claude Haiku's web_search tool
"""


# def perform_web_search_validation(user_query: str, product_names: List[str], client: Anthropic) -> str:
#     """Use Claude Haiku's web_search tool to validate product popularity."""
#     if not config.ENABLE_WEB_SEARCH:
#         logger.info("Web search disabled")
#         return ""
#     
#     if not product_names:
#         return ""
#     
#     products_to_validate = product_names[:MAX_PRODUCTS_FOR_WEB_VALIDATION]
#     product_list = ", ".join(products_to_validate)
#     
#     search_prompt = f"""Search for reviews and popularity of these beauty products, especially in India:
# 
# User's need: "{user_query}"
# 
# Products to evaluate: {product_list}
# 
# Find which are most recommended and any standout reviews."""
# 
#     logger.info(f"Web validation: {len(products_to_validate)} products...")
#     
#     try:
#         _start = time.perf_counter()
#         
#         response = client.messages.create(
#             model=config.ROUTER_MODEL,
#             max_tokens=1500,
#             messages=[{"role": "user", "content": search_prompt}],
#             tools=[{"type": "web_search_20250305", "name": "web_search"}]
#         )
#         
#         result_parts = [block.text for block in response.content if hasattr(block, "text")]
#         result = "\n".join(result_parts).strip()
#         
#         logger.info(f"Web validation completed in {time.perf_counter() - _start:.2f}s")
#         return result if result else ""
#         
#     except Exception as e:
#         logger.warning(f"Web search failed: {e}")
#         return ""


# =============================================================================
# INTENT ANALYSIS (Layer 1) - UPDATED STRUCTURE
# =============================================================================
"""
CHANGED: query_domain + query_type → single "intent" field + special flags
"""


def analyze_query_intent(query: str, session: SessionState, client: Anthropic) -> Dict[str, Any]:
    """Use LLM to analyze query with new flattened intent structure."""
    session_summary = session.get_summary()
    
    list_context = ""
    current_list = session.get_current_list()
    if current_list and current_list.get("items"):
        topic = current_list.get("topic") or "recent list"
        list_context = f"\nCurrent list (topic: {topic}):\n"
        for i, item in enumerate(current_list.get("items", []), 1):
            list_context += f"  {i}. {item}\n"
    else:
        note_topic, note_items = session.get_note_items()
        if note_items:
            list_context = f"\nIndexed list (topic: {note_topic or 'recent list'}):\n"
            for i, item in enumerate(note_items, 1):
                list_context += f"  {i}. {item}\n"
    
    memory_preview = session.get_memory_files_content()
    if memory_preview and len(memory_preview) > 2500:
        memory_preview = memory_preview[:2500] + "\n... (truncated)"

    try:
        tpl_path = Path(config.LAYER1_PROMPT_PATH)
        tpl_text = tpl_path.read_text(encoding="utf-8").strip()
        
        if (tpl_text.startswith('"""') and tpl_text.endswith('"""')) or \
           (tpl_text.startswith("'''") and tpl_text.endswith("'''")):
            tpl_text = tpl_text[3:-3].strip()
        
        analysis_prompt = tpl_text.format(
            session_summary=session_summary,
            list_context=list_context or "(none)",
            memory_preview=memory_preview or "(none)",
            query=query,
        )
    except FileNotFoundError:
        logger.error("Layer 1 prompt not found at %s", config.LAYER1_PROMPT_PATH)
        raise
    except Exception as e:
        logger.error(f"Failed to load Layer 1 prompt: {e}")
        return _fallback_intent(query)

    try:
        _start = time.perf_counter()
        
        with client.messages.stream(
            model=config.ROUTER_MODEL,
            max_tokens=2000,
            temperature=0.0,
            messages=[{"role": "user", "content": analysis_prompt}],
        ) as stream:
            streamed_parts = []
            for chunk in stream.text_stream:
                print(chunk, end="", flush=True)
                streamed_parts.append(chunk)
            response = stream.get_final_message()
        
        print()
        
        logger.info(f"Intent analysis: {time.perf_counter() - _start:.2f}s")
        
        result_text = response.content[0].text.strip() if response.content else "".join(streamed_parts).strip()
        
        if result_text.startswith("```"):
            result_text = result_text.split("```")[1]
            if result_text.startswith("json"):
                result_text = result_text[4:]
        if result_text.endswith("```"):
            result_text = result_text[:-3]
        
        result = json.loads(result_text.strip())
        
        return {
            "intent": result.get("intent", "recommend"),
            "requires_retrieval": result.get("requires_retrieval", True),
            "requires_web_validation": result.get("requires_web_validation", False),
            # NEW: Memory-based answering passthrough
            "answer_in_memory": result.get("answer_in_memory", False),
            "memory_answer": result.get("memory_answer"),
            "is_brand_query": result.get("is_brand_query", False),
            "is_ingredient_query": result.get("is_ingredient_query", False),
            "is_price_query": result.get("is_price_query", False),
            "is_negative_query": result.get("is_negative_query", False),
            "exclude_attributes": result.get("exclude_attributes"),
            "is_followup": result.get("is_followup", False),
            "has_ordinal": result.get("has_ordinal", False),
            "needs_clarification": result.get("needs_clarification", False),
            "clarification_type": result.get("clarification_type"),
            "resolved_query": result.get("resolved_query", query),
            "detected_product": result.get("detected_product"),
            "detected_brand": result.get("detected_brand"),
            "detected_category": result.get("detected_category"),
            "detected_ingredients": result.get("detected_ingredients"),
            "comparison_entities": result.get("comparison_entities"),
            "comparison_attribute": result.get("comparison_attribute"),
            "reasoning": result.get("reasoning", ""),
        }
        
    except json.JSONDecodeError as e:
        logger.warning(f"Intent JSON parse failed: {e}")
        return _fallback_intent(query)
    except Exception as e:
        logger.warning(f"Intent analysis failed: {e}")
        return _fallback_intent(query)


def _fallback_intent(query: str) -> Dict[str, Any]:
    """Fallback intent when analysis fails."""
    return {
        "intent": "recommend", "requires_retrieval": True, "requires_web_validation": False,
        "is_brand_query": False, "is_ingredient_query": False, "is_price_query": False,
        "is_negative_query": False, "exclude_attributes": None,
        "is_followup": False, "has_ordinal": False, "needs_clarification": False,
        "clarification_type": None, "resolved_query": query,
        "detected_product": None, "detected_brand": None, "detected_category": None,
        "detected_ingredients": None, "reasoning": "Fallback",
    }



# =============================================================================
# MEMORY TOOL HANDLER
# =============================================================================

class MemoryToolHandler:
    """Handles memory tool operations."""
    
    def __init__(self, memory_dir: Path):
        self.memory_dir = memory_dir.resolve()
        self.memory_dir.mkdir(parents=True, exist_ok=True)
    
    def _validate_path(self, path_str: str) -> Tuple[bool, Path, str]:
        if not path_str:
            return False, Path(), "Path cannot be empty"
        
        if path_str.startswith("/memories"):
            relative_part = path_str[len("/memories"):].lstrip("/")
        elif path_str.startswith("memories"):
            relative_part = path_str[len("memories"):].lstrip("/")
        else:
            relative_part = path_str.lstrip("/")
        
        if ".." in path_str:
            return False, Path(), "Path traversal not allowed"
        
        local_path = (self.memory_dir / relative_part).resolve() if relative_part else self.memory_dir
        
        try:
            local_path.relative_to(self.memory_dir)
        except ValueError:
            return False, Path(), "Path escapes memory directory"
        
        return True, local_path, ""
    
    def handle(self, tool_input: Dict[str, Any]) -> str:
        command = tool_input.get("command", "").lower()
        handlers = {"view": self._view, "create": self._create, "str_replace": self._str_replace, "delete": self._delete}
        handler = handlers.get(command)
        return handler(tool_input) if handler else f"Unknown command: {command}"
    
    def _view(self, tool_input: Dict[str, Any]) -> str:
        path_str = tool_input.get("path", "/memories")
        is_valid, local_path, error = self._validate_path(path_str)
        if not is_valid:
            return f"Error: {error}"
        if not local_path.exists():
            return f"Path not found: {path_str}"
        if local_path.is_dir():
            items = [f.name for f in sorted(local_path.iterdir()) if not f.name.startswith(".")]
            return f"Files: {', '.join(items)}" if items else "Directory empty"
        try:
            content = local_path.read_text(encoding="utf-8")
            return content[:3000] + "\n... (truncated)" if len(content) > 3000 else content
        except IOError as e:
            return f"Error: {e}"
    
    def _create(self, tool_input: Dict[str, Any]) -> str:
        path_str = tool_input.get("path", "")
        file_text = tool_input.get("file_text", "")
        is_valid, local_path, error = self._validate_path(path_str)
        if not is_valid:
            return f"Error: {error}"
        if len(file_text.encode("utf-8")) > config.MAX_MEMORY_FILE_SIZE:
            return "Error: File too large"
        try:
            local_path.parent.mkdir(parents=True, exist_ok=True)
            # If file exists, auto-rename to avoid overwriting previous notes
            target_path = local_path
            if target_path.exists():
                stem = target_path.stem
                suffix = target_path.suffix
                i = 1
                while True:
                    candidate = target_path.with_name(f"{stem}_{i}{suffix}")
                    if not candidate.exists():
                        target_path = candidate
                        break
                    i += 1
            target_path.write_text(file_text, encoding="utf-8")
            # Return the actual created path as a normalized memory path
            rel = str(target_path.relative_to(self.memory_dir)) if target_path.is_relative_to(self.memory_dir) else target_path.name
            return f"Created: /memories/{rel}"
        except IOError as e:
            return f"Error: {e}"
    
    def _str_replace(self, tool_input: Dict[str, Any]) -> str:
        path_str = tool_input.get("path", "")
        old_str = tool_input.get("old_str", "")
        new_str = tool_input.get("new_str", "")
        is_valid, local_path, error = self._validate_path(path_str)
        if not is_valid:
            return f"Error: {error}"
        if not local_path.exists():
            return "File not found"
        try:
            content = local_path.read_text(encoding="utf-8")
            if old_str not in content:
                return "String not found"
            updated = content.replace(old_str, new_str, 1)
            # Version instead of overwriting: create a new file alongside original
            stem = local_path.stem
            suffix = local_path.suffix
            i = 1
            while True:
                candidate = local_path.with_name(f"{stem}_{i}{suffix}")
                if not candidate.exists():
                    break
                i += 1
            candidate.write_text(updated, encoding="utf-8")
            # Return normalized path under /memories/
            rel = str(candidate.relative_to(self.memory_dir)) if candidate.is_relative_to(self.memory_dir) else candidate.name
            return f"Created: /memories/{rel}"
        except IOError as e:
            return f"Error: {e}"
    
    def _delete(self, tool_input: Dict[str, Any]) -> str:
        path_str = tool_input.get("path", "")
        is_valid, local_path, error = self._validate_path(path_str)
        if not is_valid:
            return f"Error: {error}"
        if local_path == self.memory_dir:
            return "Cannot delete root"
        if not local_path.exists():
            return "Not found"
        try:
            if local_path.is_file():
                local_path.unlink()
                return f"Deleted: {path_str}"
            return "Not a file"
        except OSError as e:
            return f"Error: {e}"


# =============================================================================
# AGENTIC LOOP (REFACTORED INTO 4 FUNCTIONS)
# =============================================================================


def _stream_with_filtering(stream, stream_callback: Optional[Callable[[str], None]]) -> Tuple[str, bool]:
    """Stream response while filtering meta markers."""
    streamed_parts, preview_buffer = [], []
    printing_enabled, skip_initial_meta = False, True
     
    for chunk in stream.text_stream:
        streamed_parts.append(chunk)
        preview_buffer.append(chunk)
         
        if isinstance(chunk, str) and chunk:
            if not printing_enabled:
                buf_txt = "".join(preview_buffer).strip()
                if skip_initial_meta and META_MARKER_PATTERN.search(buf_txt):
                    continue
                skip_initial_meta = False
                if buf_txt and (len(buf_txt) > META_TEXT_THRESHOLD or not META_MARKER_PATTERN.search(buf_txt)):
                    clean_buf = strip_memory_preamble(buf_txt)
                    if clean_buf:
                        if stream_callback:
                            stream_callback(clean_buf)
                        printing_enabled = True
                    preview_buffer.clear()
            else:
                if stream_callback:
                    stream_callback(chunk)
     
    return "".join(streamed_parts).strip(), printing_enabled


def _handle_tool_calls(tool_uses: List, memory_handler: MemoryToolHandler) -> List[Dict]:
    """Execute tool calls and return results."""
    tool_results = []
    for tb in tool_uses:
        tool_input = getattr(tb, "input", {})
        cmd = tool_input.get("command", "")
        result = memory_handler.handle(tool_input)
        if config.DEBUG_MODE:
            logger.debug(f"Tool: {cmd} -> {result[:80]}...")
        tool_results.append({"type": "tool_result", "tool_use_id": getattr(tb, "id", ""), "content": result})
    return tool_results


def _assemble_final_response(all_texts: List[str], last_text: str) -> str:
    """Assemble final response from collected meaningful texts."""
    if all_texts:
        final = all_texts[0]
        for additional in all_texts[1:]:
            if additional not in final and len(additional) > 100:
                sentences = additional.split('. ')[:2]
                if not any(s in final for s in sentences if len(s) > 20):
                    final += "\n\n" + additional
        return strip_memory_preamble(final.strip())
    return strip_memory_preamble(extract_meaningful_text(last_text))


def run_with_memory_tool(client: Anthropic, model: str, system_prompt: str, user_message: str,
                         memory_handler: MemoryToolHandler, max_iterations: int = None,
                         temperature: float = 0.2, max_tokens: int = 8000,
                         stream_callback: Optional[Callable[[str], None]] = None) -> str:
    """Run agentic loop with memory tool."""
    # Suppress incremental streaming in UI if final-only mode is enabled
    if config.STREAM_FINAL_ONLY:
        stream_callback = None
    max_iterations = max_iterations or config.MAX_TOOL_ITERATIONS
    
    beta_iface = getattr(client, "beta", None)
    if not beta_iface or not getattr(beta_iface, "messages", None):
        raise RuntimeError("Anthropic beta API not available")

    system_blocks = [{"type": "text", "text": system_prompt, "cache_control": {"type": "ephemeral"}}]
    messages = [{"role": "user", "content": user_message}]

    all_meaningful_texts: List[str] = []
    last_step_text = ""
    _total_start = time.perf_counter()

    for iter_idx in range(max_iterations):
        _api_start = time.perf_counter()
        
        with beta_iface.messages.stream(
            model=model, max_tokens=max_tokens, temperature=temperature,
            system=system_blocks, messages=messages,
            tools=[{"type": "memory_20250818", "name": "memory"}],
            betas=["context-management-2025-06-27"],
        ) as stream:
            step_text, _ = _stream_with_filtering(stream, stream_callback)
            response = stream.get_final_message()
        
        step_text = strip_memory_preamble(step_text)
        last_step_text = step_text
        if step_text and not is_meta_only_text(step_text):
            all_meaningful_texts.append(step_text)

        tool_uses = [b for b in response.content if getattr(b, "type", None) == "tool_use"]
        logger.info(f"Agent step {iter_idx + 1}: API={time.perf_counter() - _api_start:.2f}s, tools={len(tool_uses)}")

        if not tool_uses:
            logger.info(f"Agent total: {time.perf_counter() - _total_start:.2f}s")
            return _assemble_final_response(all_meaningful_texts, last_step_text)

        assistant_content = []
        for block in response.content:
            btype = getattr(block, "type", None)
            if btype == "text":
                assistant_content.append({"type": "text", "text": getattr(block, "text", "")})
            elif btype == "tool_use":
                assistant_content.append({"type": "tool_use", "id": getattr(block, "id", ""), "name": getattr(block, "name", ""), "input": getattr(block, "input", {})})
        messages.append({"role": "assistant", "content": assistant_content})
        
        tool_results = _handle_tool_calls(tool_uses, memory_handler)
        messages.append({"role": "user", "content": tool_results})

    logger.info(f"Agent total: {time.perf_counter() - _total_start:.2f}s (max iterations)")
    return _assemble_final_response(all_meaningful_texts, last_step_text) or "Max iterations reached."


# =============================================================================
# EMBEDDING & PINECONE
# =============================================================================


def embed_text(text: str) -> List[float]:
    """Generate embedding for text using OpenAI."""
    if not text:
        return []
    if not config.OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY not configured")
    
    client = OpenAI(api_key=config.OPENAI_API_KEY)
    resp = client.embeddings.create(model=config.EMBEDDING_MODEL, input=text)
    return [float(v) for v in resp.data[0].embedding]


def search_pinecone(query: str, top_k: int = PINECONE_TOP_K) -> List[Dict]:
    """Search Pinecone with high top_k for maximum coverage."""
    vec = embed_text(query)
    if not vec:
        return []
    
    try:
        idx = get_pinecone_index()
        results = idx.query(vector=vec, top_k=top_k, include_values=False, include_metadata=True, namespace=config.PINECONE_NAMESPACE)
        matches = getattr(results, "matches", []) or results.get("matches", [])
        return [{"product_id": getattr(m, "id", None) or m.get("id"), "score": getattr(m, "score", None) or m.get("score"), "metadata": getattr(m, "metadata", None) or m.get("metadata")} for m in matches]
    except Exception as e:
        logger.error(f"Pinecone search failed: {e}")
        return []


def search_for_comparison(entities: List[str], base_query: str, top_k_per_entity: int = 15) -> List[Dict]:
    """
    Run separate searches for each comparison entity and merge results.
    Works for any entities - brands, products, categories, etc.
    """
    all_results: List[Dict] = []
    seen_ids: set = set()
    base_query = (base_query or "").strip()
    
    for entity in entities:
        entity = (entity or "").strip()
        if not entity:
            continue
        # Combine entity with any comparison attribute context
        search_query = f"{entity} {base_query}".strip()
        results = search_pinecone(search_query, top_k=top_k_per_entity)
        
        # Add results avoiding duplicates
        for item in results:
            pid = item.get("product_id")
            if pid not in seen_ids:
                item["_searched_entity"] = entity  # Track which search found it
                all_results.append(item)
                seen_ids.add(pid)
    
    # Sort by score
    all_results.sort(key=lambda x: x.get("score", 0) or 0.0, reverse=True)
    return all_results


def parse_comparison_entities(text: str) -> List[str]:
    """
    DEPRECATED: We now rely on the LLM intent to provide `comparison_entities`.
    This function is kept only for reference and is no longer used.
    """
    return []


# =============================================================================
# COHERE RERANKER
# =============================================================================

# def rerank_with_cohere(query: str, retrieved: List[Dict], top_n: int = COHERE_TOP_N) -> List[Dict]:
#     """Use Cohere Rerank to get most relevant results."""
#     if not retrieved:
#         return retrieved
#     if cohere is None or not config.COHERE_API_KEY:
#         logger.warning("Cohere not available")
#         return retrieved[:top_n]

#     try:
#         co = cohere.ClientV2(api_key=config.COHERE_API_KEY)
#         docs = [" | ".join([str(m.get("metadata", {}).get(f) or "") for f in ["product_name", "brand", "leaf_level_category"]]) for m in retrieved]
        
#         resp = co.rerank(model=config.COHERE_RERANK_MODEL, query=query, documents=docs, top_n=min(len(docs), top_n))
        
#         seen: Set[int] = set()
#         new_list: List[Dict] = []
#         for r in getattr(resp, "results", []) or []:
#             idx_val = getattr(r, "index", None)
#             if idx_val is None:
#                 idx_val = getattr(getattr(r, "document", None), "index", None)
#             if isinstance(idx_val, int) and 0 <= idx_val < len(retrieved) and idx_val not in seen:
#                 new_list.append(retrieved[idx_val])
#                 seen.add(idx_val)
        
#         logger.info(f"Cohere rerank: {len(new_list)} results")
#         return new_list
        
#     except Exception as e:
#         logger.warning(f"Cohere rerank failed: {e}")
#         return retrieved[:top_n]


# =============================================================================
# MAIN PRODUCT QNA FUNCTION
# =============================================================================

def general_product_qna(query: str, category: Optional[str] = None, session_id: Optional[str] = None,
                        stream_callback: Optional[Callable[[str], None]] = None) -> str:
    """
    Main entry point for product Q&A.
    
    v3.2 FLOW: Validate → Intent → Pinecone(100) → Cohere(50) → Dedupe → Aggregate → Web → Layer 2
    """
    
    # STEP 1: Validate input
    is_valid, clean_query = validate_query(query)
    if not is_valid:
        return f"Invalid query: {clean_query}"
    query = clean_query
    
    # STEP 2: Initialize
    client = get_anthropic_client()
    total_start = time.perf_counter()
    
    sid = session_id or os.getenv("MEMORY_SESSION_ID") or "global"
    session = SessionState(sid)
    memory_handler = MemoryToolHandler(session.memory_dir)
    
    logger.info(f"Query: {query}")
    
    # STEP 3: Analyze Intent
    logger.info("Analyzing intent...")
    intent = analyze_query_intent(query, session, client)
    
    intent_type = intent.get("intent", "recommend")
    requires_retrieval = intent.get("requires_retrieval", True)
    requires_web_validation = intent.get("requires_web_validation", False)
    
    logger.info(f"Intent: {intent_type}, retrieval={requires_retrieval}, web={requires_web_validation}")

    # NEW: Memory-based answering short-circuit
    answer_in_memory = intent.get("answer_in_memory", False)
    memory_answer = intent.get("memory_answer")
    if answer_in_memory and memory_answer:
        logger.info("✓ Answer found in memory - skipping retrieval and Layer 2")
        # Update session state
        session.update(
            current_product=intent.get("detected_product"),
            current_brand=intent.get("detected_brand"),
            current_category=intent.get("detected_category") or category,
            last_query=query,
            last_answer_preview=(memory_answer or "")[:200],
        )
        # Stream the answer if callback provided
        if stream_callback:
            try:
                stream_callback(memory_answer)
            except Exception:
                pass
        logger.info(f"Total: {time.perf_counter() - total_start:.2f}s (from memory)")
        return memory_answer
    
    # STEP 4: Handle Off-Topic
    if intent_type == "off_topic":
        response = get_off_topic_response(query)
        session.update(last_query=query, last_answer_preview=response[:200])
        return response
    
    # STEP 5: Retrieval Pipeline
    retrieved, aggregated_products = [], []
    
    if requires_retrieval:
        search_query = intent.get("resolved_query", query)
        
        if intent.get("is_brand_query") and intent.get("detected_brand"):
            brand = intent["detected_brand"]
            if brand.lower() not in search_query.lower():
                search_query = f"{brand} {search_query}"
        
        # 5a. Pinecone Search
        comparison_entities = intent.get("comparison_entities") or []
        comparison_attribute = intent.get("comparison_attribute", "")
        if intent_type == "compare" and len(comparison_entities) >= 2:
            per_k = max(1, config.COMPARE_TOP_K_PER_ENTITY)
            logger.info(f"Pinecone (compare): entities={comparison_entities}, attr='{comparison_attribute}', per_k={per_k}")
            retrieved = search_for_comparison(comparison_entities, comparison_attribute, top_k_per_entity=per_k)
            logger.info(f"Pinecone (compare): {len(retrieved)} merged results")
        else:
            logger.info(f"Pinecone: '{search_query}' (top {PINECONE_TOP_K})")
            retrieved = search_pinecone(search_query, top_k=PINECONE_TOP_K)
            logger.info(f"Pinecone: {len(retrieved)} results")
        
        # 5b-5d. Bypass rerank/dedupe/aggregation — pass raw Pinecone docs to Layer 2
        # retrieved = rerank_with_cohere(search_query, retrieved, top_n=COHERE_TOP_N)
        # retrieved = dedupe_by_product(retrieved, max_chunks_per_product=MAX_CHUNKS_PER_PRODUCT)
        # aggregated_products = aggregate_products_for_display(retrieved)[:MAX_PRODUCTS_FOR_LLM]
        aggregated_products = retrieved
    
    # STEP 6: Web Validation
    web_validation_context = ""
    # if requires_web_validation and aggregated_products and intent_type == "recommend":
    #     product_names = get_unique_product_names(aggregated_products)
    #     web_validation_context = perform_web_search_validation(query, product_names, client)
    
    # STEP 7: Check for empty results
    if requires_retrieval and not aggregated_products:
        return "I couldn't find relevant products. Could you try rephrasing?"
    
    # STEP 8: Build Layer 2 Prompt
    try:
        base_prompt = Path(config.LAYER2_PROMPT_PATH).read_text().strip()
    except FileNotFoundError:
        logger.error("Layer 2 prompt not found at %s", config.LAYER2_PROMPT_PATH)
        raise
    except Exception as e:
        logger.error(f"Failed to load Layer 2 prompt: {e}")
        raise
    
    if aggregated_products:
        # Send raw Pinecone results directly to Layer 2
        retrieved_context = json.dumps(aggregated_products, indent=2, ensure_ascii=False)
    else:
        retrieved_context = "(no products)"
    
    try:
        base_prompt = base_prompt.format(
            intent=intent_type, requires_retrieval=requires_retrieval, requires_web_validation=requires_web_validation,
            is_brand_query=intent.get("is_brand_query", False), is_ingredient_query=intent.get("is_ingredient_query", False),
            is_price_query=intent.get("is_price_query", False), is_negative_query=intent.get("is_negative_query", False),
            needs_clarification=intent.get("needs_clarification", False),
            clarification_type=intent.get("clarification_type"),
            retrieved_context=retrieved_context, web_search_results=web_validation_context or "(none)",
            session_summary=session.get_summary(),
        )
    except KeyError as e:
        logger.warning(f"Missing placeholder: {e}")
    
    turn_count = session.load().get("turn_count", 0) + 1
    system_prompt = f"""{base_prompt}

CRITICAL: Start with actual answer. No memory announcements.

PRE-LOADED MEMORY: {session.get_memory_files_content() or "(empty)"}
SESSION: {session.get_summary()}
Turn: {turn_count}"""

    user_msg = json.dumps({
        "user_question": query, "resolved_query": intent.get("resolved_query", query),
        "detected_product": intent.get("detected_product"), "detected_brand": intent.get("detected_brand"),
        "is_followup": intent.get("is_followup", False), "intent": intent_type,
        "is_price_query": intent.get("is_price_query", False), "is_ingredient_query": intent.get("is_ingredient_query", False),
        "is_negative_query": intent.get("is_negative_query", False),
        "exclude_attributes": intent.get("exclude_attributes"),
        "needs_clarification": intent.get("needs_clarification", False),
        "unique_products_found": len(aggregated_products),
    }, indent=2)

    # STEP 9: Generate Answer
    logger.info("Generating answer...")
    answer = run_with_memory_tool(client=client, model=config.QNA_MODEL, system_prompt=system_prompt,
                                   user_message=user_msg, memory_handler=memory_handler, stream_callback=stream_callback)
    
    answer = strip_memory_preamble(answer.strip())
    
    if not answer:
        answer = "I found products but couldn't formulate a clear answer. Please try rephrasing."
    
    # STEP 10: Update Session
    product_name = intent.get("detected_product")
    brand_name = intent.get("detected_brand")
    
    if not product_name and aggregated_products:
        if len(aggregated_products) > 1:
            product_name = f"List: {len(aggregated_products)} products"
        else:
            # Robust handling for raw Pinecone items
            first = aggregated_products[0]
            if isinstance(first, dict):
                # Try aggregated keys first, then raw metadata
                product_name = first.get("product_base_name")
                brand_name = brand_name or first.get("brand")
                if not product_name:
                    md = first.get("metadata", {}) or {}
                    product_name = md.get("product_name") or md.get("full_name") or md.get("title")
                    brand_name = brand_name or md.get("brand")
    
    session.update(current_product=product_name, current_brand=brand_name,
                   current_category=intent.get("detected_category") or category,
                   last_query=query, last_answer_preview=answer[:200])
    
    logger.info(f"Total: {time.perf_counter() - total_start:.2f}s")
    
    return answer


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Beauty Expert QnA v3.2")
    print("Commands: exit, reset, context, debug")
    print("=" * 60)
    
    missing = config.validate()
    if missing:
        print(f"⚠️  Missing: {', '.join(missing)}")
    
    session = SessionState("cli_session")
    
    while True:
        try:
            user_input = input("\n📝 You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        
        if not user_input:
            continue
        
        cmd = user_input.lower()
        
        if cmd in ("exit", "quit", "q"):
            break
        
        if cmd == "reset":
            session.clear()
            for f in session.memory_dir.iterdir():
                if f.is_file() and not f.name.startswith("."):
                    try:
                        f.unlink()
                    except:
                        pass
            print("✔ Cleared.")
            continue
        
        if cmd == "context":
            print(f"\n{session.get_summary()}")
            continue
        
        if cmd == "debug":
            config.DEBUG_MODE = not config.DEBUG_MODE
            config.DEBUG_INTENT_STREAM = config.DEBUG_MODE
            print(f"Debug: {'ON' if config.DEBUG_MODE else 'OFF'}")
            continue
        
        try:
            def on_chunk(c):
                print(c, end="", flush=True)
            
            print("\n🤖 Assistant: ", end="", flush=True)
            response = general_product_qna(query=user_input, session_id="cli_session", stream_callback=on_chunk)
            print()
        except Exception as e:
            logger.error(f"Error: {e}")
            if config.DEBUG_MODE:
                import traceback
                traceback.print_exc()