"""
Product Tools with LLM-Based Context Resolution (Optimized v2)

FIXES from previous version:
1. Separated Python context (session_state.json) from LLM memory files
2. Added ordinal resolution BEFORE Pinecone search
3. Reduced tool iterations with better prompting
4. Pre-inject memory content into system prompt to avoid view calls
5. Fixed product extraction to use discussed products, not random Pinecone results

Key principle: Let the LLM make decisions, not regex patterns.
"""

import os
import json
import time
from typing import Optional, List, Dict, Any, Tuple, Callable
from pathlib import Path
from anthropic import Anthropic
from openai import OpenAI
from pinecone import Pinecone
try:
    import cohere  # Optional: for reranking
except Exception:
    cohere = None

# =============================================================================
# Configuration
# =============================================================================

_pc_api_key = os.getenv("PINECONE_API_KEY")
_pc_index_name = os.getenv("PINECONE_INDEX") or os.getenv("PINECONE_INDEX_NAME")
_pc_namespace = os.getenv("PINECONE_NAMESPACE") or None
_pc_dim_env = os.getenv("PINECONE_DIMENSION")
_pc_expected_dim = int(_pc_dim_env) if _pc_dim_env and _pc_dim_env.isdigit() else None

pc = Pinecone(api_key=_pc_api_key)
index = pc.Index(_pc_index_name)

_anthropic_client: Optional[Anthropic] = None
try:
    _anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
except Exception:
    _anthropic_client = None

MEMORY_DIR = Path(os.getenv("MEMORY_DIR", "./memories"))
MEMORY_DIR.mkdir(parents=True, exist_ok=True)

MAX_MEMORY_FILE_SIZE = int(os.getenv("MAX_MEMORY_FILE_SIZE", 1024 * 100))
MAX_TOOL_ITERATIONS = int(os.getenv("MAX_TOOL_ITERATIONS", 8))

# SEPARATE FILES: Python manages session_state.json, LLM manages everything else
SESSION_STATE_FILE = "session_state.json"  # Python-only
LIST_INDEX_FILE = "list_index.json"  # LLM-managed for ordinal resolution

# Models
ROUTER_MODEL = os.getenv("LLM_MODEL_ROUTER", "claude-haiku-4-5-20251001")
QNA_MODEL = os.getenv("LLM_MODEL_QNA", "claude-haiku-4-5-20251001")


# =============================================================================
# Session State Manager (Python-only, separate from LLM memory)
# =============================================================================

class SessionState:
    """
    Manages session state separately from LLM memory files.
    This prevents overwriting conflicts.
    """
    
    def __init__(self, session_id: str = "global"):
        self.session_id = session_id
        self.memory_dir = MEMORY_DIR / session_id if session_id != "global" else MEMORY_DIR
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self._cache: Optional[Dict] = None
    
    def _state_path(self) -> Path:
        return self.memory_dir / SESSION_STATE_FILE
    
    def load(self) -> Dict[str, Any]:
        """Load state from disk."""
        if self._cache is not None:
            return self._cache
        
        path = self._state_path()
        if path.exists():
            try:
                self._cache = json.loads(path.read_text(encoding="utf-8"))
                return self._cache
            except (json.JSONDecodeError, IOError):
                pass
        
        self._cache = {
            "current_product": None,
            "current_brand": None,
            "current_category": None,
            "last_query": None,
            "last_answer_preview": None,
            "last_list_file": None,  # Track which list file was last created
            "turn_count": 0,
            "conversation_history": [],
        }
        return self._cache

    def get_note_items(self) -> Tuple[Optional[str], List[str]]:
        """Parse /memories/session_note.md and extract (topic, items[1..N])."""
        note_path = self.memory_dir / "session_note.md"
        if not note_path.exists():
            return None, []
        try:
            text = note_path.read_text(encoding="utf-8")
        except Exception:
            return None, []
        topic = None
        items: List[str] = []
        # Try to read a front-matter topic: 'topic: ...'
        for line in text.splitlines():
            if not topic and line.strip().lower().startswith("topic:"):
                topic = line.split(":", 1)[1].strip() or None
            # Extract numbered list like '1. Name' or '1) Name' (optionally bolded)
            stripped = line.strip()
            if stripped and (stripped[0].isdigit()):
                # Remove leading markdown bold and number pattern
                # e.g., '**1. Name** - desc' -> 'Name'
                name_part = stripped
                # Drop leading ** if present
                if name_part.startswith("**"):
                    name_part = name_part[2:]
                # Remove leading number markers
                import re as _re
                name_part = _re.sub(r"^\d+[\.)]\s*", "", name_part)
                # Strip trailing bold and description after dash
                name_part = name_part.replace("**", "")
                name_part = name_part.split(" - ", 1)[0].strip()
                if name_part:
                    items.append(name_part)
        return topic, items

    def get_latest_note_items(self) -> Tuple[Optional[str], List[str], Optional[str]]:
        """Scan memory dir for the most recent .md note and extract (topic, items, filename)."""
        latest_path: Optional[Path] = None
        latest_mtime = -1.0
        try:
            for f in self.memory_dir.iterdir():
                if f.is_file() and f.suffix == ".md" and not f.name.startswith("."):
                    mtime = f.stat().st_mtime
                    if mtime > latest_mtime:
                        latest_mtime = mtime
                        latest_path = f
        except Exception:
            pass
        if not latest_path:
            return None, [], None
        try:
            text = latest_path.read_text(encoding="utf-8")
        except Exception:
            return None, [], latest_path.name
        topic = None
        items: List[str] = []
        for line in text.splitlines():
            if not topic and line.strip().lower().startswith("topic:"):
                topic = line.split(":", 1)[1].strip() or None
            stripped = line.strip()
            if stripped and (stripped[0].isdigit()):
                name_part = stripped
                if name_part.startswith("**"):
                    name_part = name_part[2:]
                import re as _re
                name_part = _re.sub(r"^\d+[\.)]\s*", "", name_part)
                name_part = name_part.replace("**", "")
                name_part = name_part.split(" - ", 1)[0].strip()
                if name_part:
                    items.append(name_part)
        return topic, items, latest_path.name
    
    def save(self) -> None:
        """Persist state to disk."""
        if self._cache is None:
            return
        try:
            self._state_path().write_text(
                json.dumps(self._cache, indent=2, ensure_ascii=False),
                encoding="utf-8"
            )
        except IOError:
            pass
    
    def update(self, **kwargs) -> None:
        """Update state fields, append to conversation_history, and persist."""
        state = self.load()
        # Build a history entry opportunistically
        entry: Dict[str, Any] = {}
        if "last_query" in kwargs and kwargs["last_query"]:
            entry["query"] = kwargs["last_query"]
        if "current_category" in kwargs and kwargs["current_category"]:
            entry["topic"] = kwargs["current_category"]
        if "current_product" in kwargs and kwargs["current_product"]:
            entry["product"] = kwargs["current_product"]
        if "last_list_file" in kwargs and kwargs["last_list_file"]:
            entry["list_file"] = kwargs["last_list_file"]
        if entry:
            entry["turn"] = state.get("turn_count", 0) + 1
            state.setdefault("conversation_history", []).append(entry)
        # Merge fields and persist
        state.update(kwargs)
        state["turn_count"] = state.get("turn_count", 0) + 1
        
        self._cache = state
        self.save()

    # ----- Multi-list index helpers -----
    def _list_index_path(self) -> Path:
        return self.memory_dir / LIST_INDEX_FILE

    def get_list_index(self) -> Dict[str, Any]:
        path = self._list_index_path()
        if not path.exists():
            return {"lists": [], "current_list_id": None}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            # Normalize legacy shape
            if isinstance(data, dict) and "items" in data:
                lst_id = f"list_{int(time.time())}"
                data = {
                    "lists": [{"id": lst_id, "topic": data.get("topic"), "items": data.get("items", []), "source_file": data.get("source_file"), "created_at": time.strftime("%Y-%m-%d %H:%M:%S")}],
                    "current_list_id": lst_id,
                }
            # Ensure keys
            data.setdefault("lists", [])
            data.setdefault("current_list_id", None)
            return data
        except Exception:
            return {"lists": [], "current_list_id": None}

    def get_current_list(self) -> Optional[Dict[str, Any]]:
        idx = self.get_list_index()
        if not isinstance(idx, dict):
            return None
        cur = idx.get("current_list_id")
        if not cur:
            # Fallback: try to synthesize current list from the latest note
            topic, items = self.get_note_items()
            if not items:
                topic, items, fname = self.get_latest_note_items()
            else:
                fname = "session_note.md"
            if items:
                return {
                    "id": "note_current",
                    "topic": topic or "recent list",
                    "items": items,
                    "source_file": fname or "(unknown)",
                    "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
            return None
        lists = idx.get("lists", [])
        if not isinstance(lists, list):
            return None
        for item in lists:
            if isinstance(item, dict) and item.get("id") == cur:
                return item
        return None

    def save_list_index(self, items: List[str], topic: str, source_file: Optional[str] = None) -> None:
        idx = self.get_list_index()
        lists = idx.get("lists", [])
        new_id = f"list_{len(lists)+1}"
        entry = {
            "id": new_id,
            "topic": topic,
            "items": items,
            "source_file": source_file,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        lists.append(entry)
        idx["lists"] = lists
        idx["current_list_id"] = new_id
        try:
            self._list_index_path().write_text(json.dumps(idx, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"[INDEX] Saved list_index with {len(items)} items (topic='{topic}')")
        except IOError as e:
            print(f"[ERROR] Writing list_index.json failed: {e}")
    
    def get_summary(self) -> str:
        """Get a richer summary for LLM context (brief, single-paragraph)."""
        state = self.load()
        parts = []
        if state.get("current_product"):
            parts.append(f"Current product: {state['current_product']}")
        if state.get("current_brand"):
            parts.append(f"Current brand: {state['current_brand']}")
        if state.get("current_category"):
            parts.append(f"Current category: {state['current_category']}")
        if state.get("last_list_file"):
            parts.append(f"Last list file: {state['last_list_file']}")
        if state.get("last_query"):
            parts.append(f"Last question: {state['last_query']}")
        # Include a brief history tail
        hist = state.get("conversation_history") or []
        if hist:
            try:
                tail = hist[-2:] if len(hist) >= 2 else hist
                htxt = "; ".join([f"#{h.get('turn')}: {h.get('query')}" for h in tail if h.get('query')])
                if htxt:
                    parts.append(f"Recent turns: {htxt}")
            except Exception:
                pass
        return "\n".join(parts) if parts else "No previous context available."
    
    def clear(self) -> None:
        """Clear all state."""
        self._cache = {
            "current_product": None,
            "current_brand": None,
            "current_category": None,
            "last_query": None,
            "last_answer_preview": None,
            "last_list_file": None,
            "turn_count": 0,
        }
        self.save()
    
    def get_list_index(self) -> Optional[Dict]:
        """Read the LLM-managed list index file."""
        list_path = self.memory_dir / LIST_INDEX_FILE
        if list_path.exists():
            try:
                return json.loads(list_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, IOError):
                pass
        return None
    
    def get_memory_files_content(self) -> str:
        """Read all memory .md files for pre-injection into system prompt."""
        contents = []
        try:
            for f in sorted(self.memory_dir.iterdir()):
                if f.suffix == ".md" and not f.name.startswith("."):
                    try:
                        text = f.read_text(encoding="utf-8")
                        # Truncate large files
                        if len(text) > 2000:
                            text = text[:2000] + "\n... (truncated)"
                        contents.append(f"=== {f.name} ===\n{text}")
                    except:
                        pass
        except:
            pass
        return "\n\n".join(contents) if contents else ""


# =============================================================================
# Ordinal Resolution (NEW - Critical for "2nd one" type queries)
# =============================================================================

def resolve_ordinal_reference(
    query: str,
    session: SessionState,
    client: Anthropic,
) -> Tuple[Optional[str], Optional[str]]:
    """
    If query contains ordinal reference ("2nd one", "the third"), resolve it
    to a specific product name using the last list created.
    
    Returns: (resolved_product_name, list_file_used) or (None, None)
    """
    # Check if we have a list index
    list_index = session.get_list_index()
    if not list_index:
        return None, None
    
    items = list_index.get("items", [])
    if not items:
        return None, None
    
    # Use LLM to resolve the ordinal
    resolve_prompt = f"""Given this list of items:
{json.dumps(items, indent=2)}

And this user query: "{query}"

If the user is referring to a specific item by position (like "2nd one", "the first", "third item", "last one"), 
return ONLY a JSON object with:
{{"resolved_item": "exact item name from list", "position": number}}

If the query doesn't reference a specific position, return:
{{"resolved_item": null, "position": null}}

Return ONLY valid JSON, nothing else."""

    try:
        response = client.messages.create(
            model=ROUTER_MODEL,
            max_tokens=200,
            temperature=0.0,
            messages=[{"role": "user", "content": resolve_prompt}],
        )
        
        result_text = response.content[0].text.strip()
        if result_text.startswith("```"):
            result_text = result_text.split("```")[1]
            if result_text.startswith("json"):
                result_text = result_text[4:]
        if result_text.endswith("```"):
            result_text = result_text[:-3]
        
        result = json.loads(result_text.strip())
        resolved = result.get("resolved_item")
        
        if resolved:
            return resolved, list_index.get("source_file")
        
    except Exception as e:
        print(f"[WARN] Ordinal resolution failed: {e}")
    
    return None, None


# =============================================================================
# LLM-Based Intent Analysis
# =============================================================================

def analyze_query_intent(
    query: str,
    session: SessionState,
    client: Anthropic,
) -> Dict[str, Any]:
    """
    Use LLM to analyze the query and make intelligent decisions.
    """
    session_summary = session.get_summary()
    # Prefer current list from list_index.json for ordinal context
    list_context = ""
    current_list = session.get_current_list()
    if current_list and current_list.get("items"):
        src = current_list.get("source_file") or "(no file)"
        topic = current_list.get("topic") or "recent list"
        list_context = f"\nCurrent list from list_index.json (topic: {topic}, source: {src}):\n"
        for i, item in enumerate(current_list.get("items", []), 1):
            list_context += f"  {i}. {item}\n"
    else:
        # Fallback: parse most recent note for 1..N items
        note_topic, note_items = session.get_note_items()
        if note_items:
            list_context = f"\nIndexed list from session_note.md (topic: {note_topic or 'recent list'}):\n"
            for i, item in enumerate(note_items, 1):
                list_context += f"  {i}. {item}\n"
    
    # Include a compact preview of memory notes so the router has enough context
    memory_preview = session.get_memory_files_content()
    if memory_preview and len(memory_preview) > 2500:
        memory_preview = memory_preview[:2500] + "\n... (truncated)"

    # Build analysis prompt using external template file (Layer_1_prompt.txt)
    try:
        _tpl_path = Path("/Users/ptah/Documents/QnA_Tools_Chatbot/Layer_1_prompt.txt")
        if not _tpl_path.exists():
            _tpl_path = Path("Layer_1_prompt.txt")  # fallback to CWD
        _tpl_text = _tpl_path.read_text(encoding="utf-8").strip()
        # Strip surrounding triple quotes if present in the template file
        if (_tpl_text.startswith('"""') and _tpl_text.endswith('"""')) or (_tpl_text.startswith("'''") and _tpl_text.endswith("'''")):
            _tpl_text = _tpl_text[3:-3].strip()
        analysis_prompt = _tpl_text.format(
            session_summary=session_summary,
            list_context=(list_context or "(none)"),
            memory_preview=(memory_preview or "(none)"),
            query=query,
        )
    except Exception as _e:
        # Strict mode: do not fallback; ensure the external template is used
        raise RuntimeError(f"Failed to load Layer_1_prompt.txt: {_e}")

    try:
        _start = time.perf_counter()
        # Stream the intent analysis response
        with client.messages.stream(
            model=ROUTER_MODEL,
            max_tokens=10000,
            temperature=0.0,
            messages=[{"role": "user", "content": analysis_prompt}],
        ) as stream:
            streamed_parts = []
            for chunk in stream.text_stream:
                print(chunk, end="", flush=True)
                streamed_parts.append(chunk)
            response = stream.get_final_message()
        print(f"\n[TIMING] Intent analysis: {time.perf_counter() - _start:.2f}s")
        
        result_text = response.content[0].text.strip() if getattr(response, "content", None) else ("".join(streamed_parts).strip())
        
        # Clean markdown
        if result_text.startswith("```"):
            result_text = result_text.split("```")[1]
            if result_text.startswith("json"):
                result_text = result_text[4:]
        if result_text.endswith("```"):
            result_text = result_text[:-3]
        
        result = json.loads(result_text.strip())
        
        return {
            "is_followup": result.get("is_followup", False),
            "needs_context": result.get("needs_context", False),
            "needs_retrieval": result.get("needs_retrieval", True),
            "has_ordinal": result.get("has_ordinal", False),
            "resolved_query": result.get("resolved_query", query),
            "detected_product": result.get("detected_product"),
            "detected_brand": result.get("detected_brand"),
            "detected_category": result.get("detected_category"),
            "reasoning": result.get("reasoning", ""),
        }
        
    except Exception as e:
        print(f"[WARN] Intent analysis failed: {e}")
        return {
            "is_followup": False,
            "needs_context": False,
            "needs_retrieval": True,
            "has_ordinal": False,
            "resolved_query": query,
            "detected_product": None,
            "detected_brand": None,
            "detected_category": None,
            "reasoning": f"Fallback: {e}",
        }


# =============================================================================
# Memory Tool Handler
# =============================================================================

class MemoryToolHandler:
    """Handles memory tool operations."""
    
    def __init__(self, memory_dir: Path):
        self.memory_dir = memory_dir.resolve()
        self.memory_dir.mkdir(parents=True, exist_ok=True)
    
    def _validate_path(self, path_str: str) -> Tuple[bool, Path, str]:
        if not path_str:
            return False, Path(), "Path cannot be empty"
        
        # Normalize
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
        
        if command == "view":
            return self._view(tool_input)
        elif command == "create":
            return self._create(tool_input)
        elif command == "str_replace":
            return self._str_replace(tool_input)
        elif command == "delete":
            return self._delete(tool_input)
        else:
            return f"Unknown command: {command}"
    
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
        else:
            try:
                content = local_path.read_text(encoding="utf-8")
                if len(content) > 3000:
                    content = content[:3000] + "\n... (truncated)"
                return content
            except Exception as e:
                return f"Error: {e}"
    
    def _create(self, tool_input: Dict[str, Any]) -> str:
        path_str = tool_input.get("path", "")
        file_text = tool_input.get("file_text", "")
        
        is_valid, local_path, error = self._validate_path(path_str)
        if not is_valid:
            return f"Error: {error}"
        
        if len(file_text.encode("utf-8")) > MAX_MEMORY_FILE_SIZE:
            return f"Error: File too large"
        
        try:
            local_path.parent.mkdir(parents=True, exist_ok=True)
            local_path.write_text(file_text, encoding="utf-8")
            return f"Created: {path_str}"
        except Exception as e:
            return f"Error: {e}"
    
    def _str_replace(self, tool_input: Dict[str, Any]) -> str:
        path_str = tool_input.get("path", "")
        old_str = tool_input.get("old_str", "")
        new_str = tool_input.get("new_str", "")
        
        is_valid, local_path, error = self._validate_path(path_str)
        if not is_valid:
            return f"Error: {error}"
        
        if not local_path.exists():
            return f"File not found"
        
        try:
            content = local_path.read_text(encoding="utf-8")
            if old_str not in content:
                return "String not found"
            local_path.write_text(content.replace(old_str, new_str, 1), encoding="utf-8")
            return f"Updated: {path_str}"
        except Exception as e:
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
        except Exception as e:
            return f"Error: {e}"


# =============================================================================
# Optimized Agentic Loop (Reduced Iterations)
# =============================================================================

def run_with_memory_tool(
    client: Anthropic,
    model: str,
    system_prompt: str,
    user_message: str,
    memory_handler: MemoryToolHandler,
    max_iterations: int = MAX_TOOL_ITERATIONS,
    temperature: float = 0.2,
    max_tokens: int = 10000,
    stream_callback: Optional[Callable[[str], None]] = None,
) -> str:
    """Optimized agentic loop with reduced iterations."""
    
    beta_iface = getattr(client, "beta", None)
    if not beta_iface or not getattr(beta_iface, "messages", None):
        raise RuntimeError("Anthropic beta API not available")
    
    system_blocks = [{"type": "text", "text": system_prompt, "cache_control": {"type": "ephemeral"}}]
    messages = [{"role": "user", "content": user_message}]
    last_text = ""
    last_meaningful_text = ""
    
    total_start = time.perf_counter()
    
    for iter_idx in range(max_iterations):
        api_start = time.perf_counter()
        # Streaming response: print text chunks as they arrive
        with beta_iface.messages.stream(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_blocks,
            messages=messages,
            tools=[{"type": "memory_20250818", "name": "memory"}],
            betas=["context-management-2025-06-27"],
        ) as stream:
            streamed_text_parts = []
            # Buffer to suppress meta-only prefaces from reaching UI
            preview_buffer = []
            printing_enabled = False
            meta_markers = (
                "let me update the memory",
                "updating the memory",
                "let me check the memory",
                "checking memory",
                "i will update the memory",
                "i'll update the memory",
                "memory is already up to date",
                "no additional tool calls needed",
                "no tool calls needed",
                "tools are not required",
            )
            for chunk in stream.text_stream:
                # Collect streamed assistant text incrementally
                streamed_text_parts.append(chunk)
                preview_buffer.append(chunk)
                # Forward the chunk to optional UI callback (UI/CLI handles display)
                try:
                    if stream_callback and isinstance(chunk, str) and chunk:
                        if not printing_enabled:
                            # Decide whether the buffered text is meaningful enough to show
                            buf_txt = "".join(preview_buffer).strip()
                            lower = buf_txt.lower()
                            if buf_txt and (len(buf_txt) > 160 or not any(m in lower for m in meta_markers)):
                                # Flush buffer once and enable streaming
                                stream_callback(buf_txt)
                                printing_enabled = True
                                preview_buffer.clear()
                        else:
                            stream_callback(chunk)
                except Exception:
                    # Never let UI callback failures break the agent loop
                    pass
            # finalize full message for tool blocks
            response = stream.get_final_message()
        api_elapsed = time.perf_counter() - api_start
        
        # Capture streamed text as the assistant text for this step
        if streamed_text_parts:
            current_text = "".join(streamed_text_parts).strip()
            if current_text:
                last_text = current_text
                lower = current_text.lower()
                meta_markers = (
                    "let me update the memory",
                    "updating the memory",
                    "let me check the memory",
                    "checking memory",
                    "i will update the memory",
                    "i'll update the memory",
                    "memory is already up to date",
                    "no additional tool calls needed",
                    "no tool calls needed",
                    "tools are not required",
                )
                if not any(m in lower for m in meta_markers) or len(current_text) > 160:
                    last_meaningful_text = current_text
                else:
                    print("\n[DEBUG] Ignoring meta-only assistant text for final output:", repr(current_text[:120]))
        
        tool_uses = [b for b in response.content if getattr(b, "type", None) == "tool_use"]
        
        print(f"[AGENT] Step {iter_idx + 1}: API={api_elapsed:.2f}s, tools={len(tool_uses)}")
        # DEBUG: print compact info about tool uses
        if tool_uses:
            try:
                for idx, tb in enumerate(tool_uses, start=1):
                    tname = getattr(tb, "name", "")
                    tid = getattr(tb, "id", "")
                    tinp = getattr(tb, "input", {}) or {}
                    tcmd = tinp.get("command")
                    print(f"  - ToolUse[{idx}] id={tid} name={tname} command={tcmd}")
            except Exception:
                pass
        
        if not tool_uses:
            print(f"[TIMING] Agent total: {time.perf_counter() - total_start:.2f}s")
            # Prefer the last meaningful text over a meta-only preface
            return (last_meaningful_text or last_text or "")
        
        # Build assistant content
        assistant_content = []
        for block in response.content:
            btype = getattr(block, "type", None)
            if btype == "text":
                assistant_content.append({"type": "text", "text": getattr(block, "text", "")})
            elif btype == "tool_use":
                assistant_content.append({
                    "type": "tool_use",
                    "id": getattr(block, "id", ""),
                    "name": getattr(block, "name", ""),
                    "input": getattr(block, "input", {})
                })
        messages.append({"role": "assistant", "content": assistant_content})
        
        # Execute tools
        tool_results = []
        for tb in tool_uses:
            tool_input = getattr(tb, "input", {})
            cmd = tool_input.get("command", "")
            result = memory_handler.handle(tool_input)
            print(f"  - Tool: {cmd} -> {result[:100]}...")
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": getattr(tb, "id", ""),
                "content": result
            })
        messages.append({"role": "user", "content": tool_results})
    
    print(f"[TIMING] Agent total: {time.perf_counter() - total_start:.2f}s (max iterations)")
    return last_text or "Max iterations reached."


# =============================================================================
# Embedding & Pinecone
# =============================================================================

def embed_text(text: str) -> List[float]:
    if not text:
        return []
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")
    
    model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")
    client = OpenAI(api_key=api_key)
    
    resp = client.embeddings.create(model=model, input=text)
    return [float(v) for v in resp.data[0].embedding]


def search_pinecone(query: str, top_k: int = 30, category: Optional[str] = None) -> List[Dict]:
    vec = embed_text(query)
    if not vec:
        return []
    
    query_params = {
        "vector": vec,
        "top_k": top_k,
        "include_values": False,
        "include_metadata": True,
        "namespace": _pc_namespace
    }
    
    category_map = {
        'lip_balm_treatment': ('Makeup', 'Lip', 'Lip Balm & Treatment'),
        'lipstick': ('Makeup', 'Lip', 'Lipstick'),
        'lip_liner': ('Makeup', 'Lip', 'Lip Liner'),
        'lip_stain_tint': ('Makeup', 'Lip', 'Lip Stain & Tint'),
        'lip_gloss': ('Makeup', 'Lip', 'Lip Gloss'),
        'liquid_lipstick': ('Makeup', 'Lip', 'Liquid Lipstick'),
        'lip_plumper': ('Makeup', 'Lip', 'Lip Plumper'),
    }
    
    # [DISABLED] Metadata filter based on category (kept for reference; do not remove)
    # if category and category.lower() in category_map:
    #     cat, sub, leaf = category_map[category.lower()]
    #     query_params["filter"] = {
    #         "$and": [
    #             {"category": {"$eq": cat}},
    #             {"sub_category": {"$eq": sub}},
    #             {"leaf_level_category": {"$eq": leaf}}
    #         ]
    #     }
    
    try:
        results = index.query(**query_params)
        matches = getattr(results, "matches", []) or results.get("matches", [])
        return [
            {
                "product_id": getattr(m, "id", None) or m.get("id"),
                "score": getattr(m, "score", None) or m.get("score"),
                "metadata": getattr(m, "metadata", None) or m.get("metadata"),
            }
            for m in matches
        ]
    except Exception as e:
        print(f"[ERROR] Pinecone: {e}")
        return []


# =============================================================================
# Cohere Reranker (Optional)
# =============================================================================

def rerank_with_cohere(query: str, retrieved: List[Dict]) -> List[Dict]:
    """Use Cohere Rerank v3.5 to reorder Pinecone results.

    Falls back to the original order if cohere package or API key is missing
    or if any error occurs.
    """
    if not retrieved:
        return retrieved
    if cohere is None:
        return retrieved
    api_key = os.getenv("COHERE_API_KEY")
    if not api_key:
        return retrieved

    try:
        co = cohere.ClientV2(api_key=api_key)
        # Build lightweight doc strings from metadata
        docs: List[str] = []
        for m in retrieved:
            md = m.get("metadata", {}) or {}
            parts = [
                str(md.get("product_name") or md.get("title") or ""),
                str(md.get("brand") or ""),
                str(md.get("leaf_level_category") or md.get("category") or ""),
                str(md.get("description") or md.get("ingredients") or md.get("text") or ""),
            ]
            # join non-empty parts
            docs.append(" | ".join([p for p in parts if p]))

        topn = min(len(docs), 10)
        resp = co.rerank(
            model=os.getenv("COHERE_RERANK_MODEL", "rerank-v3.5"),
            query=query,
            documents=docs,
            top_n=topn,
        )

        # resp.results is expected to include .index of original doc
        # Construct new order by those indices first, then append any missing
        seen = set()
        new_list: List[Dict] = []
        for r in getattr(resp, "results", []) or []:
            idx = getattr(r, "index", None)
            if idx is None:
                # Some SDKs use r.document.index or r.document['index']
                idx = getattr(getattr(r, "document", None), "index", None)
            if isinstance(idx, int) and 0 <= idx < len(retrieved) and idx not in seen:
                new_list.append(retrieved[idx])
                seen.add(idx)
        # Append any not seen to preserve full list
        for i, item in enumerate(retrieved):
            if i not in seen:
                new_list.append(item)
        return new_list
    except Exception as e:
        print(f"[WARN] Cohere rerank skipped: {e}")
        return retrieved


# =============================================================================
# Main Product QnA Function (Optimized)
# =============================================================================

def general_product_qna(
    query: str,
    category: Optional[str] = None,
    top_k: int = 30,
    session_id: Optional[str] = None,
    suppress_memory_notice: bool = True,
    prefer_memory: bool = False,
    stream_callback: Optional[Callable[[str], None]] = None,
) -> str:
    """
    Optimized QnA with proper ordinal resolution and reduced iterations.
    """
    
    if _anthropic_client is None:
        return "Error: Anthropic client not initialized"
    
    total_start = time.perf_counter()
    
    # Initialize session
    sid = session_id or os.getenv("MEMORY_SESSION_ID") or "global"
    session = SessionState(sid)
    memory_handler = MemoryToolHandler(session.memory_dir)
    
    print(f"\n{'='*60}")
    print(f"[QUERY] {query}")
    print(f"[SESSION] {sid}")
    
    # Step 1: Analyze intent (includes ordinal resolution now)
    print("[STEP] Analyzing intent...")
    intent = analyze_query_intent(query, session, _anthropic_client)
    
    print(f"[INTENT] followup={intent['is_followup']}, ordinal={intent.get('has_ordinal', False)}")
    print(f"[INTENT] resolved: {intent['resolved_query']}")
    print(f"[INTENT] product: {intent['detected_product']}, brand: {intent['detected_brand']}")
    
    # Step 2: Search Pinecone with RESOLVED query (ordinals already resolved)
    search_query = intent["resolved_query"]
    search_category = intent["detected_category"] or category
    
    print(f"[STEP] Searching Pinecone: '{search_query}'")
    _pc_start = time.perf_counter()
    retrieved = search_pinecone(search_query, top_k=top_k, category=search_category)
    print(f"[PINECONE] {len(retrieved)} results in {time.perf_counter() - _pc_start:.2f}s")
    
    # Optional: rerank with Cohere before sending to the LLM
    _rr_start = time.perf_counter()
    reranked = rerank_with_cohere(search_query, retrieved)
    if reranked is not retrieved:
        print(f"[RERANK] Applied Cohere reranker in {time.perf_counter() - _rr_start:.2f}s")
        # Keep top 10 in reranked order
        retrieved = reranked[:10]
    else:
        # No rerank: sort by Pinecone score desc and keep top 10
        try:
            retrieved = sorted(
                retrieved,
                key=lambda m: (m.get("score") if isinstance(m, dict) else getattr(m, "score", 0)) or 0,
                reverse=True,
            )[:10]
        except Exception:
            retrieved = retrieved[:10]
    print(f"[FILTER] Passing top {len(retrieved)} documents to LLM")
    
    if not retrieved:
        return "I couldn't find relevant information. Please try rephrasing."
    
    # Step 3: Pre-load memory content (avoid view calls)
    existing_memory = session.get_memory_files_content()
    
    # Step 4: Build optimized system prompt
    prompt_path = os.getenv("QNA_PROMPT_PATH", "Layer_2_prompt.txt")
    try:
        base_prompt = Path(prompt_path).read_text().strip()
    except Exception as _e:
        # Strict mode: do not fallback; ensure the external template is used
        raise RuntimeError(f"Failed to load Layer_2_prompt.txt: {_e}")
    
    turn_for_filename = session.load().get("turn_count", 0) + 1
    system_prompt = f"""{base_prompt}

EFFICIENCY RULES (CRITICAL):
1. Do NOT say "Let me check memory" or announce tool usage
2. Start your answer IMMEDIATELY with the content
3. At most ONE tool call (CREATE) if needed
4. Memory is pre-loaded below - avoid view calls unless absolutely necessary

PRE-LOADED MEMORY:
{existing_memory if existing_memory else "(empty)"}

SESSION STATE (preview):
{session.get_summary()}

LIST MEMORY POLICY (MULTI-LIST):
- For list answers, create a UNIQUE markdown note under /memories with this pattern:
  /memories/{{topic_snake}}_{turn_for_filename:02d}.md
- Do NOT overwrite existing files; always create a new one per list.
- The note MUST start with:
  ---
  topic: <brief topic>
  items:
    - Product 1 Name
    - Product 2 Name
    - Product 3 Name
    - Product 4 Name
    - Product 5 Name
  updated: <YYYY-MM-DD HH:MM>
  ---
  Then include a numbered list with short reasoning per item.
- After creating the note, APPEND a new list entry to /memories/{LIST_INDEX_FILE} with shape:
  {{"id": "list_N", "topic": "<topic>", "items": [...], "source_file": "<the md filename>", "created_at": "<timestamp>"}}
- Also set "current_list_id" to this new list id.
- Do NOT create or modify session_state.json (Python manages it).
- Avoid str_replace; use CREATE to write entire files.

PRODUCT NOTE POLICY (SINGLE PRODUCT):
- When the answer is a single-product recommendation (not a list), create a concise markdown note under /memories with this pattern:
  /memories/product_{{topic_snake}}_{turn_for_filename:02d}.md
- Compute topic_snake from the product name by:
  lowercasing, replacing spaces and punctuation with underscores, and keeping only a–z, 0–9, and underscores (collapse repeats).
  Example: "Dior Forever Skin Glow" -> "dior_forever_skin_glow".
- Use exactly ONE tool call (CREATE) to write the file.
- The note MUST start with this YAML front matter:
  ---
  product: <exact product name>
  brand: <brand if known, else "unknown">
  attributes:
    - <key attribute 1>
    - <key attribute 2>
    - <key attribute 3>
  updated: <YYYY-MM-DD HH:MM>
  ---
  Then add a short summary (3–6 sentences) explaining why it fits the user's criteria.
- Do NOT modify list_index.json for single-product notes.
- Do NOT create or modify session_state.json (Python manages it).
- Avoid str_replace; use CREATE to write entire files.

FOR FOLLOW-UP QUESTIONS:
- Product resolved: {intent.get('detected_product') or 'none'}
- Answer directly using retrieved data and the single note if needed
"""

    # Build instruction
    instruction = {
        "user_question": query,
        "resolved_query": intent["resolved_query"],
        "detected_product": intent.get("detected_product"),
        "is_followup": intent["is_followup"],
        "retrieved_products": retrieved, 
        "turn_count": turn_for_filename,
    }
    
    user_msg = f"""Answer this question using the retrieved data.

{json.dumps(instruction, indent=2, ensure_ascii=False)}

Remember: Start with your answer immediately. Maximum 2 tool calls."""

    # Step 5: Generate answer
    print("[STEP] Generating answer...")
    answer = run_with_memory_tool(
        client=_anthropic_client,
        model=QNA_MODEL,
        system_prompt=system_prompt,
        user_message=user_msg,
        memory_handler=memory_handler,
        temperature=0.2,
        max_tokens=8000,
        stream_callback=stream_callback,
    )
    
    # Clean response
    answer = answer.strip()
    if answer.startswith("```"):
        lines = answer.split("\n")
        answer = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    
    # Fallback
    if not answer:
        top = retrieved[0].get("metadata", {}) if retrieved else {}
        name = top.get("product_name") or top.get("title") or "the product"
        answer = f"Based on my search, I found {name}. Please try a more specific question."
    
    # Step 6: Update session state (Python-managed, won't conflict)
    product_name = intent.get("detected_product")
    brand_name = intent.get("detected_brand")
    
    # If we didn't detect a product but have retrieved results, check if it's a list
    if not product_name and retrieved and len(retrieved) > 1:
        # Check if this was a list query
        list_index = session.get_list_index()
        if list_index and list_index.get("items"):
            product_name = f"List: {list_index.get('topic', 'multiple products')}"
    elif not product_name and retrieved:
        meta = retrieved[0].get("metadata", {})
        product_name = meta.get("product_name") or meta.get("title")
        brand_name = brand_name or meta.get("brand")
    
    session.update(
        current_product=product_name,
        current_brand=brand_name,
        current_category=intent.get("detected_category") or category,
        last_query=query,
        last_answer_preview=answer[:200],
    )
    
    print(f"[SESSION] Updated: product={product_name}")
    print(f"[TIMING] Total: {time.perf_counter() - total_start:.2f}s")
    print(f"{'='*60}\n")
    
    return answer


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    print("="*60)
    print("Optimized Product QnA v2")
    print("Commands: exit, reset, context")
    print("="*60)
    
    session = SessionState("cli_session")
    
    while True:
        try:
            user_input = input("\n📝 You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        
        if not user_input:
            continue
        
        if user_input.lower() in ("exit", "quit", "q"):
            break
        
        if user_input.lower() == "reset":
            session.clear()
            # Also clear memory files
            for f in session.memory_dir.iterdir():
                if f.is_file() and not f.name.startswith("."):
                    f.unlink()
            print("✓ Session and memory cleared.")
            continue
        
        if user_input.lower() == "context":
            print(f"\n{session.get_summary()}")
            list_idx = session.get_list_index()
            if list_idx:
                print(f"\nList index: {json.dumps(list_idx, indent=2)}")
            continue
        
        try:
            response = general_product_qna(query=user_input, session_id="cli_session")
            print(f"\n🤖 Assistant: {response}")
        except Exception as e:
            print(f"[Error] {e}")
            import traceback
            traceback.print_exc()
