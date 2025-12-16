"""
Optimized QnA Chatbot Main

Simple wrapper around the optimized product_tools.
"""

import os
import time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent / ".env", override=False)

from product_tools_optimized import general_product_qna, SessionState


def main():
    print("="*60)
    print("🤖 Optimized Product QnA v2")
    print("="*60)
    print("\nCommands:")
    print("  'exit' - Leave")
    print("  'reset' - Clear session & memory")
    print("  'context' - Show current context")
    print("="*60)
    
    session_id = os.getenv("MEMORY_SESSION_ID", "cli_session")
    session = SessionState(session_id)
    
    while True:
        try:
            user_input = input("\n📝 You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nGoodbye! 👋")
            break
        
        if not user_input:
            continue
        
        cmd = user_input.lower()
        
        if cmd in ("exit", "quit", "q"):
            print("Goodbye! 👋")
            break
        
        if cmd == "reset":
            session.clear()
            # Clear memory files too
            import json
            for f in session.memory_dir.iterdir():
                if f.is_file() and not f.name.startswith("."):
                    try:
                        f.unlink()
                    except:
                        pass
            print("✓ Session and memory cleared.")
            continue
        
        if cmd == "context":
            print("\n📋 Session State:")
            print("-" * 40)
            print(session.get_summary())
            
            list_idx = session.get_list_index()
            if list_idx:
                lists = list_idx.get("lists", [])
                current_id = list_idx.get("current_list_id")
                if lists:
                    print("\n📋 Lists (multi-list index):")
                    for lst in lists:
                        mark = "*" if lst.get("id") == current_id else " "
                        print(f" {mark} {lst.get('id')} | {lst.get('topic')} | file: {lst.get('source_file')}")
                    # Show items for current list
                    cur = next((l for l in lists if l.get("id") == current_id), None)
                    if cur:
                        print("\n📋 Current List Items:")
                        for i, item in enumerate(cur.get("items", []), 1):
                            print(f"  {i}. {item}")
            print("-" * 40)
            continue
        
        # Process query
        start = time.perf_counter()
        # Stream chunks immediately; first chunk prints the assistant prompt line
        streamed_any = {"v": False}
        def on_chunk(chunk: str):
            if not chunk:
                return
            if not streamed_any["v"]:
                print("\n🤖 Assistant: ", end="", flush=True)
                streamed_any["v"] = True
            print(chunk, end="", flush=True)
        response = general_product_qna(query=user_input, session_id=session_id, stream_callback=on_chunk)
        elapsed = time.perf_counter() - start
        
        # If nothing was streamed (e.g., tool disabled or short response), print the full answer now
        if not streamed_any["v"]:
            print(f"\n🤖 Assistant: {response}")
        else:
            # Ensure trailing newline after streamed output
            print()
        print(f"\n[Total time: {elapsed:.1f}s]")


if __name__ == "__main__":
    main()
