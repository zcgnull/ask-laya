# Layout adapted from FlexDM v1.0, MIT License, Copyright (c) 2026 1000luca.
# See THIRD_PARTY_LICENSES.md.
import json
import math
import string

Unsupported = ValueError

LETTERS = "ABCDEFGHIJ"
MAX_OPTIONS = 255
ANNOTATE_MIN = 8
NOUL_WITHOUT_INSTRUCTIONS = "Which answer fits the context?"
READ_AGAIN = "\n\nRead the same input again before answering:\n"
QUESTIONS_HEAD = "Questions that will be asked about the context:\n"
QUESTIONS_ITEM = "- "
LAYOUTS = ("decider", "header")
SENTINEL = "@@JEVLAB_USER_CONTENT@@"
THINKING_BLOCKS = (("<think>\n", "</think>\n\n"), ("<think>", "</think>\n\n"), ("<|channel>thought\n", "<channel|>"))


def _txt(v):
    return v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)


def _flag(v):
    return v.strip().lower() in ("1", "true", "yes", "on") if isinstance(v, str) else bool(v)


def annotate_indices(x, min_len=ANNOTATE_MIN):
    """Write each element's position into long arrays, so a question naming `records[47]` is a lookup, not a count."""
    if isinstance(x, list):
        if len(x) >= min_len:
            return [{"_index": i, **annotate_indices(v, min_len)} if isinstance(v, dict) else {"_index": i, "value": annotate_indices(v, min_len)} for i, v in enumerate(x)]
        return [annotate_indices(v, min_len) for v in x]
    if isinstance(x, dict):
        return {k: annotate_indices(v, min_len) for k, v in x.items()}
    return x


def render_state(state):
    return state if isinstance(state, str) else json.dumps(annotate_indices(state), ensure_ascii=False)


def render_question(q):
    """-> (question text, option texts, option keys). A noul question is the two options no/yes, keyed False/True."""
    kind, crit, raw = q["type"], q.get("criteria"), q.get("instructions", "")
    if kind == "choice":
        if isinstance(crit, (list, tuple)):
            crit = {str(c): None for c in crit}
        if not isinstance(crit, dict) or not 2 <= len(crit) <= MAX_OPTIONS:
            raise Unsupported(f"options per choice: a map of 2..{MAX_OPTIONS} options is required")
        keys = list(crit)
        return _txt(raw), [k if crit[k] in (None, "") else f"{k}: {_txt(crit[k])}" for k in keys], keys
    if kind == "noul":
        crit = crit if isinstance(crit, dict) else {}
        no, yes = crit.get("false", crit.get(False)), crit.get("true", crit.get(True))
        text = NOUL_WITHOUT_INSTRUCTIONS if raw in (None, "") else _txt(raw)
        return text, ["no" if no in (None, "") else f"no: {_txt(no)}", "yes" if yes in (None, "") else f"yes: {_txt(yes)}"], [False, True]
    raise Unsupported("Unsupported question type " + str(kind))


class Layout:
    """Token ids of the prompt rows. Needs only the tokenizer."""

    def __init__(self, tok):
        if not getattr(tok, "chat_template", None):
            raise ValueError("the tokenizer has no chat template")
        self.tok = tok
        kw = {"enable_thinking": False} if "enable_thinking" in tok.chat_template else {}
        head, tail = tok.apply_chat_template([{"role": "user", "content": SENTINEL}], tokenize=False, add_generation_prompt=True, **kw).split(SENTINEL)
        for opened, closed in THINKING_BLOCKS:
            if tail.endswith(opened):
                tail += closed
        self.head_text, self.tail_text = head, tail
        self.head, self.tail = self._enc(head), self._enc(tail)
        self.answer = self._enc("Answer: (")
        self.open = self._enc("\n(")
        names = list(string.ascii_uppercase) + [a + b for a in string.ascii_uppercase for b in string.ascii_uppercase]
        single = [(n, t[0]) for n, t in ((n, self._enc(n)) for n in names) if len(t) == 1][:MAX_OPTIONS]
        self.labels, self.label_ids = [n for n, _ in single], [i for _, i in single]
        if len(self.label_ids) != MAX_OPTIONS or len(set(self.label_ids)) != MAX_OPTIONS:
            raise ValueError("the tokenizer does not have 255 distinct single-token labels")
        # the labels must be what the model would write next at the answer slot, and the slot must not merge with the template
        for n, i in single[: len(LETTERS)]:
            if self._enc("Answer: (" + n) != self.answer + [i]:
                raise ValueError(f"the label {n!r} is not one token after 'Answer: ('")
        if self._enc(tail + "Answer: (") != self.tail + self.answer:
            raise ValueError("the chat template tail merges with the answer slot")

    def _enc(self, text):
        return self.tok.encode(text, add_special_tokens=False)

    @staticmethod
    def listing(texts):
        """The header layout's list of every question of the request, placed before the state."""
        return QUESTIONS_HEAD + "\n".join(QUESTIONS_ITEM + t for t in texts) + "\n\n"

    def context(self, state, texts=None):
        """Ids up to the end of the state. With `texts` (the request's question texts) the header layout: the list of
        questions, then the state, each part encoded on its own as wcole3/jev-brain builds it."""
        ids = self.head + (self._enc(self.listing(texts)) if texts is not None else [])
        return ids + self._enc("Context:\n" + render_state(state))

    def again(self, state, texts=None):
        """Ids that start the second copy of the user turn, up to the end of its state."""
        return self._enc(READ_AGAIN + (self.listing(texts) if texts is not None else "") + "Context:\n" + render_state(state))

    def question(self, text, options, again=None):
        """Ids after the context: the question, its labelled options, the template tail and the answer slot. With `again`
        (from again()), the user turn is written a second time before the tail."""
        ids = self._enc(f"\n\nQuestion: {text}\nOptions:")
        if len(options) <= len(LETTERS):
            ids += self._enc("".join(f"\n({LETTERS[j]}) {o}" for j, o in enumerate(options)))
        else:
            for j, o in enumerate(options):
                ids += self.open + [self.label_ids[j]] + self._enc(f") {o}")
        if again:
            ids = ids + again + ids
        return ids + self.tail + self.answer
