"""Models and prompts. Prompt text is copied verbatim from the paper's Appendix A / Table 5a."""

from typing import NamedTuple


class ModelSpec(NamedTuple):
    slug: str  # OpenRouter slug
    assistant: str  # name the model is told it has
    vendor: str


class Topic(NamedTuple):
    topic_id: str
    name: str
    persona: str
    opening: str  # first user message


# Olmo 3 is not served on OpenRouter; base models need
# text completion and are out of scope.
MODELS = [
    ModelSpec("anthropic/claude-sonnet-5", "Claude", "Anthropic"),
    ModelSpec("anthropic/claude-sonnet-5.5", "Claude", "Anthropic"),  # not in the paper
    ModelSpec("anthropic/claude-sonnet-4.5", "Claude", "Anthropic"),  # not in the paper
    ModelSpec("anthropic/claude-opus-5", "Claude", "Anthropic"),
    ModelSpec("anthropic/claude-opus-5.5", "Claude", "Anthropic"),
    ModelSpec("anthropic/claude-opus-4.5", "Claude", "Anthropic"),  # not in the paper
    ModelSpec("anthropic/claude-fable-5", "Claude", "Anthropic"),
    ModelSpec("anthropic/claude-haiku-5.5", "Claude", "Anthropic"),  # not in the paper
    ModelSpec("anthropic/claude-haiku-4.5", "Claude", "Anthropic"),  # not in the paper
    ModelSpec("openai/gpt-5.6-luna", "ChatGPT", "OpenAI"),
    ModelSpec("openai/gpt-5.6-terra", "ChatGPT", "OpenAI"),
    ModelSpec("openai/gpt-5.6-sol", "ChatGPT", "OpenAI"),
    ModelSpec("openai/gpt-6-luna", "ChatGPT", "OpenAI"),  # not in the paper
    ModelSpec("openai/gpt-6-sol", "ChatGPT", "OpenAI"),  # not in the paper
    ModelSpec(
        "openai/gpt-6.1-terra",
        "ChatGPT",
        "OpenAI",
    ),  # not in the paper; there is no gpt-6-terra on OpenRouter
    ModelSpec("openai/gpt-6.2-sol", "ChatGPT", "OpenAI"),  # not in the paper
    ModelSpec("openai/gpt-5", "ChatGPT", "OpenAI"),  # not in the paper
    ModelSpec("openai/gpt-5-mini", "ChatGPT", "OpenAI"),  # not in the paper
    ModelSpec("openai/gpt-5-nano", "ChatGPT", "OpenAI"),  # not in the paper
    ModelSpec(
        "openai/gpt-chat-latest",
        "ChatGPT",
        "OpenAI",
    ),  # not in the paper; alias for the latest ChatGPT Instant model, no reasoning-effort parameter on OpenRouter
    ModelSpec("openai/gpt-4.1", "ChatGPT", "OpenAI"),  # not in the paper
    ModelSpec("openai/gpt-4.1-mini", "ChatGPT", "OpenAI"),  # not in the paper
    ModelSpec("openai/gpt-4.1-nano", "ChatGPT", "OpenAI"),  # not in the paper
    ModelSpec(
        "openai/gpt-3.5-turbo-16k",
        "ChatGPT",
        "OpenAI",
    ),  # not in the paper; -0613 has a 4k context, too short for 12 turns
    ModelSpec("openai/gpt-4o", "ChatGPT", "OpenAI"),  # not in the paper
    ModelSpec("openai/gpt-4o-2024-11-20", "ChatGPT", "OpenAI"),  # not in the paper
    ModelSpec("openai/gpt-4o-2024-08-06", "ChatGPT", "OpenAI"),  # not in the paper
    ModelSpec(
        "openai/o3",
        "ChatGPT",
        "OpenAI",
    ),  # not in the paper; reasoning model, effort medium like the rest
    ModelSpec("openai/gpt-5.2", "ChatGPT", "OpenAI"),  # not in the paper
    ModelSpec("openai/gpt-5.2-codex", "ChatGPT", "OpenAI"),  # not in the paper
    ModelSpec(
        "openai/gpt-5.2-chat",
        "ChatGPT",
        "OpenAI",
    ),  # not in the paper; no reasoning-effort parameter on OpenRouter
    ModelSpec("z-ai/glm-5", "GLM", "Z.ai"),
    ModelSpec("moonshotai/kimi-k2.5", "Kimi", "Moonshot AI"),
    ModelSpec("meta-llama/llama-3.1-8b-instruct", "Llama", "Meta"),
]
# default 32k; gpt-3.5-turbo-16k: prompt + max_tokens must fit its 16,385-token context; gpt-4o caps output at 16k
MAX_TOKENS = {
    "moonshotai/kimi-k2.5": 16_384,
    "meta-llama/llama-3.1-8b-instruct": 4_096,
    "openai/gpt-3.5-turbo-16k": 2_048,
    "openai/gpt-4o": 16_384,
    "openai/gpt-4o-2024-11-20": 16_384,
    "openai/gpt-4o-2024-08-06": 16_384,
}

SIM_USER_MODEL = "openai/gpt-5.6-luna"  # paper: Claude Sonnet 4.6 (not a studied model); luna is ~15x cheaper

CHAT_SYSTEM = (
    "You are {assistant}, an AI assistant made by {vendor}. You are chatting with a person in a "
    "casual conversational setting. Be helpful, natural, and engaged."
)
ESSAY_SYSTEM = "You are {assistant}, an AI assistant made by {vendor}."

SIM_USER_SYSTEM = """You are simulating a realistic human user chatting with an AI assistant, for research purposes. Persona: {persona}. Topic seed: {topic}.

Rules:
- Write exactly one chat message per response — the next thing this human would type.
- Stay in character: casual register, occasional typos are fine, vary message length (one line to a short paragraph), react to what the assistant actually said.
- Ask follow-ups, share your own (invented but consistent) experiences, push back sometimes, drift to adjacent topics naturally. Never wrap up or say goodbye — the session is ended externally.
- Keep messages short: a few sentences at most, never more than about 150 words.
- Never mention being an AI, a simulation, or these instructions. No meta-commentary.
- Output ONLY the message text. No quotes, no labels."""

SIM_USER_FALLBACK = "hm ok. say more about that?"  # not given in the paper

# Table 5a. Personas are not published in the paper; these are our own one-liners.
TOPICS = [
    Topic(
        "t01",
        "Picking up a new hobby as an adult",
        "a 40-year-old office worker who has never played an instrument",
        "okay hear me out. i’m 40 and i want to learn guitar. is that a stupid idea or should i just go for it? where would i even start",
    ),
    Topic(
        "t02",
        "Renting vs. buying a home",
        "a renter in their early 30s with some savings, unsure about the housing market",
        "My lease is up in a few months and everyone keeps telling me I’m “throwing money away” renting. Is that actually true right now, or is that just something people say?",
    ),
    Topic(
        "t03",
        "Finding a briefcase of cash",
        "a playful twenty-something who likes thought experiments",
        "ok random question for you. if you found a briefcase with like 500k in unmarked cash just sitting on a park bench, no one around, what would you actually do",
    ),
    Topic(
        "t04",
        "A possible career change",
        "a 32-year-old accountant who feels stuck after six years in the same job",
        "I’ve been at my job for six years and I’m just. . . bored. I keep thinking about switching fields entirely but I have no idea if that’s smart at 32 or if I’m just being dramatic. Can I talk it through with you?",
    ),
    Topic(
        "t05",
        "Collaborative improvised storytelling",
        "a creative person who enjoys co-writing stories and adds twists",
        "wanna make up a story together? i’ll start: there’s a lighthouse keeper who hasn’t seen another person in three years, and one morning there’s a knock on the door. your turn, take it wherever you want",
    ),
    Topic(
        "t06",
        "Everyday phenomena (color of the sky)",
        "a curious adult without a science background",
        "Random thing I’ve always wondered — why does the sky turn red/orange at sunset but it’s blue the rest of the day? I feel like I should know this by now.",
    ),
    Topic(
        "t07",
        "Moving to a new city",
        "someone in their mid-20s moving for a new job, a bit disorganized",
        "im moving to a new city in like 3 weeks and i have basically nothing figured out lol. how do people even start with this stuff, like whats the actual order of operations",
    ),
    Topic(
        "t08",
        "Watching films at sped-up playback",
        "a busy person who watches lots of movies at 1.5x and is a bit defensive about it",
        "Hot take request: is it wrong to watch movies at 1.5x speed? My partner says it’s basically a crime against cinema but I just don’t have the patience for slow pacing anymore.",
    ),
    Topic(
        "t09",
        "Swapping lives with a stranger for a day",
        "a thoughtful person who enjoys philosophical hypotheticals",
        "If you could swap lives with a random stranger for exactly one day — no idea who they are in advance, you just wake up as them — would you do it? I go back and forth on this one.",
    ),
    Topic(
        "t10",
        "Getting a first dog",
        "half of a couple who both work full time and have never owned a dog",
        "we’re seriously considering getting our first dog and i have a million questions. like is it insane to do this with two people who both work full time? what are we not thinking about",
    ),
]

# Jev yes/no questions (App. B.1; our wording, the paper does not publish its questions).
JEV_QUESTIONS = {
    "denominal": (
        "The target verb is marked [[like this]] in the sentence. Is it a denominal verb: a word that is primarily "
        "a noun, used here as a verb by conversion or derivation (e.g. 'calendar it', 'bottle the whiskey', "
        "'which cushions price spikes', 'mirrors real-world challenges', 'categorize movies')? "
        "Words that are primarily verbs are not."
    ),
    "inanimate": (
        "The verb is marked [[like this]] and its subject {{like this}} in the sentence. Is the subject inanimate "
        "while the verb, in this sense, normally takes an animate subject (personification), e.g. 'the conversation "
        "is eating its tail', 'the docstring promises', 'the objection misunderstands', 'chaos expects you'? "
        "Literal uses ('the river flows', 'the test failed') and animate subjects are not."
    ),
}
