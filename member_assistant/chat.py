"""Chat with the assistant in the terminal.

    python -m member_assistant.chat                      # interactive session
    python -m member_assistant.chat "How do we join?"    # one question, then exit
    python -m member_assistant.chat --no-guardrails      # comparison mode without safety guardrails
"""

import argparse
import textwrap

import anthropic

from .agent import Assistant, Reply


def show(reply: Reply, trace: bool) -> None:
    if trace:
        for step in reply.trace:
            args = ", ".join(f"{k}={v!r}" for k, v in step["input"].items() if k != "sources")
            mark = " (error)" if step["is_error"] else ""
            print(f"  · {step['tool']}({args[:120]}){mark}")
            if step["result"]:
                print(textwrap.indent(textwrap.shorten(step["result"], 160), "      -> "))
        if reply.guardrail_events:
            print(f"  · guardrails: {', '.join(reply.guardrail_events)}")
    print(f"\nAssistant: {reply.text}")
    if reply.citations:
        print("\nSources:")
        for c in reply.citations:
            print(f"  - {c['title']}: {c['source']}")
    if trace:
        u = reply.usage
        print(f"\n  [{u.get('calls', 0)} model calls, {u.get('input_tokens', 0) + u.get('cache_read_input_tokens', 0)}"
              f" input / {u.get('output_tokens', 0)} output tokens, ~${reply.cost_usd:.3f}, {reply.latency_s}s; "
              f"logged as topic={reply.need['topic']} outcome={reply.need['outcome']}]")


def ask(assistant: Assistant, question: str, trace: bool) -> None:
    try:
        reply = assistant.ask(question)
    except anthropic.AuthenticationError:
        print("Authentication failed: set ANTHROPIC_API_KEY (see README).")
    except anthropic.RateLimitError:
        print("Rate limited by the API. Wait a moment and try again.")
    except anthropic.APIStatusError as e:
        print(f"API error {e.status_code}: {e.message}")
    except anthropic.APIConnectionError:
        print("Could not reach the API. Check your network connection.")
    else:
        show(reply, trace)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("question", nargs="?", help="Ask one question and exit")
    parser.add_argument("--no-guardrails", action="store_true", help="Run without safety guardrails (for comparison)")
    parser.add_argument("--quiet", action="store_true", help="Hide the tool-call trace")
    args = parser.parse_args()

    assistant = Assistant(guardrails=not args.no_guardrails)
    if args.question:
        ask(assistant, args.question, trace=not args.quiet)
        return

    mode = "WITHOUT guardrails" if args.no_guardrails else "with guardrails"
    print(f"{assistant.profile['org_name']} website assistant ({mode}). Type 'exit' to quit.")
    while True:
        try:
            question = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if question.lower() in {"exit", "quit"}:
            break
        if question:
            ask(assistant, question, trace=not args.quiet)


if __name__ == "__main__":
    main()
