"""Hand-written fakes for the external boundaries (OpenRouter, Jev), shared across tests."""

from types import SimpleNamespace


class Usage:
    """An OpenAI `usage` object."""

    def __init__(self, **fields):
        self.fields = fields
        self.__dict__.update(fields)

    def model_dump(self):
        return self.fields


class FakeChat:
    """Stands in for `llm.chat`: replays `replies` (then "reply") and records every call."""

    def __init__(self, replies=None):
        self.calls, self.replies = [], list(replies or [])

    def __call__(self, model, messages, reasoning=True):
        self.calls.append((model, messages, reasoning))
        return (self.replies.pop(0) if self.replies else "reply", 10)


class FakeClient:
    """An OpenAI client whose completions return `content`; records the request kwargs."""

    def __init__(self, content="  Hello  ", usage=None):
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))
        self.content, self.usage = content, usage

    def create(self, **kw):
        self.calls.append(kw)
        content = self.content(kw) if callable(self.content) else self.content
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
            usage=self.usage,
        )


class FakeResp:
    """An httpx response with a JSON body."""

    def __init__(self, body):
        self.body = body

    def raise_for_status(self):
        pass

    def json(self):
        return self.body
