"""Tests for problem (priority)."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest

from cs329z_hw1.llm import ScriptedLM
from cs329z_hw1 import adapters
from tests.conftest import eval_slice
from tests.helpers import load_fixture
from tests.thresholds import PRIORITY_MIN_ACCURACY

CATEGORIES = ("urgent", "normal", "ignore")

EMAILS = [
    {
        "id": "em-90001",
        "thread_id": "th-90001",
        "reply_to": None,
        "from": "dana.ortiz@cardinal.example",
        "to": ["sam.lee@cardinal.example"],
        "date": "2001-05-14T15:02:00Z",
        "subject": "Need your sign-off on the Q2 forecast by 5pm today",
        "body": "Sam,\n\nFinance closes the book at 5pm. Please reply with your approval.\n\nDana",
    },
    {
        "id": "em-90002",
        "thread_id": "th-90002",
        "reply_to": None,
        "from": "deals@officesupplies.example",
        "to": ["sam.lee@cardinal.example"],
        "date": "2001-05-14T16:40:00Z",
        "subject": "",
        "body": "Here is our forecast\n\n ",
    },
]

# The scripted model replies with these whatever your prompt asks for. None
# of them is written in your format, except by coincidence.
REPLIES = {
    "plain_word": "urgent",
    "upper_case": "IGNORE",
    "mixed_case_sentence": "I would say this one is Normal.",
    "two_categories": "This is either urgent or normal, it is hard to say.",
    "all_three": "urgent? normal? ignore? I cannot decide.",
    "garbage": "%%% <<>> lorem ipsum 12345 {]",
    "empty": "",
    "whitespace": "  \n\n\t ",
    "refusal": "I'm sorry, I can't help with that.",
    "json_wrong_label": '{"label": "high", "why": "deadline today"}',
    "json_like": '{"category": "urgent", "reason": "deadline today"}',
    "long_multiline": "Let me think.\n\n" + "The sender wants something. " * 80 + "\n\nFinal: normal",
}


def check_contract(result, reply: str) -> None:
    shown = f"The scripted model replied {reply[:80]!r}; run_priority returned {result!r}."
    assert isinstance(result, dict), f"run_priority should return a dict. {shown}"
    assert "category" in result and "reason" in result, (
        f"The result needs the keys 'category' and 'reason'. {shown}"
    )
    assert result["category"] in CATEGORIES, (
        f"'category' must be one of {CATEGORIES} (lower case) whatever the model replies. {shown}"
    )
    assert isinstance(result["reason"], str) and result["reason"].strip(), (
        f"'reason' must be a non-empty string whatever the model replies. {shown}"
    )


@pytest.mark.parametrize("name", list(REPLIES))
@pytest.mark.parametrize("email", EMAILS, ids=["with_subject", "blank_subject"])
def test_contract_holds_for_any_reply(name, email):
    """run_priority returns a valid result, and does not raise, for replies
    that follow no particular format."""
    reply = REPLIES[name]
    lm = ScriptedLM([reply], repeat_last=True)
    result = adapters.run_priority(email, lm)
    check_contract(result, reply)
    assert len(lm.calls) >= 1, "run_priority should call the model at least once; it made no calls."


def test_email_is_not_modified():
    email = dict(EMAILS[0], to=list(EMAILS[0]["to"]))
    before = dict(email, to=list(email["to"]))
    adapters.run_priority(email, ScriptedLM(["normal"], repeat_last=True))
    assert email == before, "run_priority should not change the email dict it is given."


# ------------------------------------------------------------------ live ---


@pytest.mark.live
def test_live_accuracy(live_lm):
    """Agreement with the gold labels in tests/fixtures/priority_gold.json."""
    gold = eval_slice(load_fixture("priority_gold.json"))
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda item: adapters.run_priority(item["email"], live_lm), gold))

    wrong = []
    for item, result in zip(gold, results):
        check_contract(result, "(live model)")
        if result["category"] != item["category"]:
            wrong.append(
                f"  {item['email']['id']} gold={item['category']} yours={result['category']} "
                f"subject={item['email']['subject'][:50]!r}"
            )
    correct = len(gold) - len(wrong)
    print(f"priority accuracy: {correct} of {len(gold)}")
    assert correct / len(gold) >= PRIORITY_MIN_ACCURACY, (
        f"{correct} of {len(gold)} emails match the gold label "
        f"({correct / len(gold):.0%}); at least {PRIORITY_MIN_ACCURACY:.0%} is required. "
        "Disagreements:\n" + "\n".join(wrong)
    )
