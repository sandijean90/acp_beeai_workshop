from typing import List, Dict
from datetime import datetime, timezone
from acp_sdk.models import Message, MessagePart


def flatten_messages(messages: List[Message]) -> str:
    """
    Collapse the *text/plain* content of an entire Message list into one string.

    * If you only care about the first/last message you can slice before calling.
    * Non-plain parts (images, files, etc.) are ignored, mirroring Message.__str__.
    """
    if not messages:
        return ""

    # join in chronological order
    return "".join(
        part.content
        for m in messages
        for part in m.parts
        if part.content_type == "text/plain" and part.content is not None
    ).strip()

# assume `msgs` is the full List[Message] that comes into your agent
#  Only the very first message (index 0)
#user_prompt = flatten_messages(msgs[:1])
#  Only the most-recent message
#user_prompt = flatten_messages(msgs[-1:])   # same as msgs[len(msgs)-1:]
#  The last three messages (good for short-context agents)
#user_prompt = flatten_messages(msgs[-3:])

def package_response(text: str) -> Dict[str, List[Message]]:
    """
    Wrap a plain-text model output into the structure expected by the ACP runner:

        {"messages": [Message(parts=[MessagePart(content=<text>)])]}
    """
    assistant_message = Message(
        parts=[MessagePart(content=text)],
        created_at=datetime.now(timezone.utc),
        completed_at=datetime.now(timezone.utc),
    )
    return {"messages": [assistant_message]}