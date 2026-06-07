"""LLM worker turn loop: prompt, strict-JSON parse, one-shot repair, wait fallback,
message sanitization, and card/memory consumption (§4.2-4.6). No live API — the
LLM is always mocked (live pattern lives in tests/test_llm.py, gated by ORCA_LIVE_LLM).
"""

from __future__ import annotations

from agents.memory import looks_seed_specific
from agents.prompts import build_worker_prompt
from agents.worker import LLMWorker
from contracts import BehaviorCard, ExecutionMemory, Heuristic, Message
from contracts.enums import ActionName, MessageType, Role
from obs_guard import scan_for_leaks
from tests.fixtures import sample_behavior_card, sample_execution_memory, sample_observation


def _msg(frm, to, content, rnd=0, type=MessageType.REPORT):
    return Message(**{"from": frm}, to=to, type=type, content=content, urgency=0.3, round=rnd)


class ScriptedLLM:
    """Mock LLM that returns queued strings and records the prompts it received."""

    def __init__(self, responses: list[str], default: str = "{}") -> None:
        self.responses = list(responses)
        self.default = default
        self.prompts: list[str] = []
        self.calls = 0

    def complete(self, prompt: str, schema=None, **kwargs) -> str:
        self.prompts.append(prompt)
        self.calls += 1
        return self.responses.pop(0) if self.responses else self.default


def _worker(responses, *, card=None, memory=None, agent_id="agent_2"):
    return LLMWorker(
        agent_id,
        ScriptedLLM(responses),
        card or sample_behavior_card(),
        memory or sample_execution_memory(),
    )


# --------------------------------------------------------------------------- #
# A1/A2 — parse + validate + invalid handling
# --------------------------------------------------------------------------- #
def test_valid_output_parses_to_action():
    w = _worker(['{"reasoning":"r","action":{"name":"scout"},"messages":[]}'])
    action = w.act(sample_observation())
    assert action.name == ActionName.SCOUT
    assert w.parse_failures == 0
    assert w.llm.calls == 1


def test_malformed_then_repair_succeeds():
    # First response is unparseable; the single repair retry returns valid JSON.
    w = _worker(["this is not json", '{"action":{"name":"gather","args":{"resource":"wood"}}}'])
    action = w.act(sample_observation())
    assert action.name == ActionName.GATHER
    assert action.args == {"resource": "wood"}
    assert w.llm.calls == 2  # original + exactly one repair
    assert w.parse_failures == 0


def test_invalid_action_name_triggers_repair_against_contract():
    # 'teleport' is not in ActionName -> Action validation fails -> repair.
    w = _worker(['{"action":{"name":"teleport"}}', '{"action":{"name":"wait"}}'])
    action = w.act(sample_observation())
    assert action.name == ActionName.WAIT
    assert w.llm.calls == 2


def test_malformed_repair_fails_falls_back_to_wait():
    w = _worker(["garbage", "still garbage"])
    action = w.act(sample_observation())
    assert action.name == ActionName.WAIT  # safe_default
    assert w.parse_failures == 1
    assert w.pending_messages == []
    assert w.llm.calls == 2  # one original + one repair, then give up — no crash


def test_no_llm_degrades_to_wait():
    w = LLMWorker("agent_2", llm=None, card=sample_behavior_card(), memory=ExecutionMemory(agent_id="agent_2"))
    action = w.act(sample_observation())
    assert action.name == ActionName.WAIT
    assert w.parse_failures == 1


def test_safe_default_is_wait():
    assert LLMWorker.safe_default().name == ActionName.WAIT


# --------------------------------------------------------------------------- #
# A4 — message validation: fill from/round, default to, sanitize
# --------------------------------------------------------------------------- #
def test_messages_fill_from_round_and_default_to_team():
    out = '{"action":{"name":"wait"},"messages":[{"type":"share_finding","content":"caves to the N"}]}'
    w = _worker([out], agent_id="agent_3")
    obs = sample_observation()  # round == 42
    w.act(obs)
    assert len(w.pending_messages) == 1
    msg = w.pending_messages[0]
    assert msg.from_agent == "agent_3"          # injected, never trusted from model
    assert msg.round == obs.round == 42          # injected current round
    assert msg.to == "team"                      # defaulted
    assert msg.type == MessageType.SHARE_FINDING


def test_bad_message_type_coerced_not_crashing():
    out = '{"action":{"name":"wait"},"messages":[{"type":"gossip","content":"hi"}]}'
    w = _worker([out])
    w.act(sample_observation())
    assert w.pending_messages[0].type == MessageType.REPORT  # unknown type -> report


def test_coord_laden_message_is_sanitized():
    out = '{"action":{"name":"wait"},"messages":[{"to":"team","type":"report","content":"iron at 12.0, 3.5 near r_07"}]}'
    w = _worker([out])
    w.act(sample_observation())
    assert len(w.pending_messages) == 1
    content = w.pending_messages[0].content
    assert not looks_seed_specific(content)  # coords/region-id scrubbed
    assert w.messages_dropped == 1


def test_pure_coord_message_is_dropped():
    out = '{"action":{"name":"wait"},"messages":[{"content":"12.0, 3.5"}]}'
    w = _worker([out])
    w.act(sample_observation())
    assert w.pending_messages == []  # nothing left after scrubbing -> dropped


def test_empty_content_message_dropped():
    out = '{"action":{"name":"wait"},"messages":[{"type":"ack","content":"   "}]}'
    w = _worker([out])
    w.act(sample_observation())
    assert w.pending_messages == []


def test_messages_capped_per_round():
    msgs = ",".join('{"type":"report","content":"m%d"}' % i for i in range(5))
    out = '{"action":{"name":"wait"},"messages":[%s]}' % msgs
    w = _worker([out])  # default max_messages_per_round == 2
    w.act(sample_observation())
    assert len(w.pending_messages) == 2


def test_report_action_content_scrubbed():
    out = '{"action":{"name":"report","args":{"content":"iron at 12.0, 3.5","urgency":0.5}}}'
    w = _worker([out])
    action = w.act(sample_observation())
    assert action.name == ActionName.REPORT
    assert not looks_seed_specific(action.args["content"])  # env-bound message stays clean


# --------------------------------------------------------------------------- #
# A4/§3.2 — the message RECIPIENT (`to`) is a leak vector too: validate it.
# --------------------------------------------------------------------------- #
def test_draft_message_region_id_recipient_defaults_to_team():
    # A model emits a clean content but a leaky region-id recipient.
    out = '{"action":{"name":"wait"},"messages":[{"to":"r_07","type":"report","content":"clean"}]}'
    w = _worker([out])
    w.act(sample_observation())
    assert len(w.pending_messages) == 1
    msg = w.pending_messages[0]
    assert msg.to == "team"            # leaky recipient downgraded, never reaches Message.to
    assert msg.content == "clean"      # content preserved
    assert scan_for_leaks(msg.model_dump(by_alias=True)) == []


def test_draft_message_coordinate_recipient_defaults_to_team():
    out = '{"action":{"name":"wait"},"messages":[{"to":"12, 3","content":"hi"}]}'
    w = _worker([out])
    w.act(sample_observation())
    assert w.pending_messages[0].to == "team"


def test_draft_message_valid_agent_recipient_preserved():
    out = '{"action":{"name":"wait"},"messages":[{"to":"agent_3","content":"for you"}]}'
    w = _worker([out])
    w.act(sample_observation())
    assert w.pending_messages[0].to == "agent_3"   # legitimate agent id kept


def test_report_action_leaky_recipient_downgraded_not_nuked():
    # A leaky `to` on a report action downgrades to team rather than nuking the
    # whole report to wait (the content is still useful coordination).
    out = '{"action":{"name":"report","args":{"content":"need help","to":"r_07"}}}'
    w = _worker([out])
    action = w.act(sample_observation())
    assert action.name == ActionName.REPORT
    assert action.args["to"] == "team"
    assert scan_for_leaks(action.model_dump()) == []


# --------------------------------------------------------------------------- #
# A4/§5.2 — older messages are summarized into history; recent stays full.
# --------------------------------------------------------------------------- #
def _wait_worker(agent_id="agent_2"):
    w = LLMWorker(agent_id, ScriptedLLM([], default='{"action":{"name":"wait"}}'),
                  sample_behavior_card(), sample_execution_memory())
    return w


def test_older_messages_appear_in_history_summary():
    w = _wait_worker()
    base = sample_observation()
    msg_a = _msg("agent_9", "team", "found caves to the N", rnd=0)
    obs1 = base.model_copy(update={"round": 1, "recent_messages": [msg_a]})
    obs2 = base.model_copy(update={"round": 2, "recent_messages": []})  # msg_a scrolled out
    w.act(obs1)
    w.act(obs2)
    assert "messages:" in w.history_summary
    assert "agent_9" in w.history_summary   # older message activity represented


def test_live_message_not_yet_summarized_no_double_count():
    # While a message is still live in the window it is NOT also folded into the
    # history summary (recent + summarized stay disjoint).
    w = _wait_worker()
    base = sample_observation()
    msg_a = _msg("agent_9", "team", "still here", rnd=0)
    obs1 = base.model_copy(update={"round": 1, "recent_messages": [msg_a]})
    obs2 = base.model_copy(update={"round": 2, "recent_messages": [msg_a]})  # still live
    w.act(obs1)
    w.act(obs2)
    assert "messages:" not in w.history_summary  # never summarized while live


def test_history_summary_is_leak_free():
    w = _wait_worker()
    base = sample_observation()
    leaky = _msg("agent_9", "team", "cache at 12, 3 near r_07", rnd=0)
    obs1 = base.model_copy(update={"round": 1, "recent_messages": [leaky]})
    obs2 = base.model_copy(update={"round": 2, "recent_messages": []})
    w.act(obs1)
    w.act(obs2)
    assert not looks_seed_specific(w.history_summary)
    assert "r_07" not in w.history_summary


# --------------------------------------------------------------------------- #
# A6/§4.1 — the no-card seam preserves the roster role.
# --------------------------------------------------------------------------- #
def test_role_kwarg_sets_default_card_role():
    for role in (Role.EXPLORER, Role.MINER, Role.TINKERER, Role.SUPPORT):
        w = LLMWorker("agent_1", ScriptedLLM([]), role=role)  # no card supplied
        assert w.role == role
        prompt = build_worker_prompt(sample_observation(), w.card, w.memory)
        assert role.value.capitalize() in prompt  # role primer matches


# --------------------------------------------------------------------------- #
# A5/§4.5 — episode-end memory write driven by the (coach) learning_signal.
# --------------------------------------------------------------------------- #
def test_neutral_signal_skips_memory_llm_call():
    mem = ExecutionMemory(agent_id="agent_2", heuristics=[])
    w = LLMWorker("agent_2", ScriptedLLM([]), sample_behavior_card(), mem)
    w.end_episode_update("digest", learning_signal=0.0)
    assert w.llm.calls == 0          # ~0 signal -> no memory-proposal LLM call
    assert w.memory.heuristics == []  # and no change


def test_negative_signal_weakens_existing_heuristic():
    # A negative coach signal weakens the agent's OWN current heuristics — without
    # an LLM call (it doesn't rely on the 'propose good how-to' LLM re-surfacing the
    # rule to drop). This is the reachable, effective negative path.
    existing = Heuristic(condition="mine without scouting", action="dig straight down", confidence=0.9)
    mem = ExecutionMemory(agent_id="agent_2", heuristics=[existing])
    w = LLMWorker("agent_2", ScriptedLLM([]), sample_behavior_card(), mem)
    w.end_episode_update("digest", learning_signal=-0.5)
    assert len(w.memory.heuristics) == 1
    assert w.memory.heuristics[0].confidence < 0.9   # weakened
    assert w.llm.calls == 0                           # negative path needs no LLM


def test_strongly_negative_signal_removes_existing_heuristic():
    existing = Heuristic(condition="rush the nether", action="skip armor", confidence=0.3)
    mem = ExecutionMemory(agent_id="agent_2", heuristics=[existing])
    w = LLMWorker("agent_2", ScriptedLLM([]), sample_behavior_card(), mem)
    w.end_episode_update("digest", learning_signal=-1.0)
    assert w.memory.heuristics == []   # weakened below the floor -> removed
    assert w.llm.calls == 0


# --------------------------------------------------------------------------- #
# A2/A6 — recursive action-arg sanitization: NO action arg may leak coords/seeds
# --------------------------------------------------------------------------- #
def test_leaky_gather_resource_is_scrubbed():
    out = '{"action":{"name":"gather","args":{"resource":"iron at 12.0, 3.5 near r_07"}}}'
    w = _worker([out])
    action = w.act(sample_observation())
    assert action.name == ActionName.GATHER          # partial leak -> scrubbed, kept
    assert not looks_seed_specific(action.args["resource"])
    assert scan_for_leaks(action.model_dump()) == []  # nothing leaks into the ActionRecord


def test_leaky_move_direction_falls_back_to_wait():
    # A direction that is *entirely* a coordinate scrubs to empty -> unusable -> wait.
    out = '{"action":{"name":"move","args":{"direction":"12.0, 3.5"}}}'
    w = _worker([out])
    action = w.act(sample_observation())
    assert action.name == ActionName.WAIT
    assert scan_for_leaks(action.model_dump()) == []


def test_nested_dict_and_list_args_scrubbed():
    out = (
        '{"action":{"name":"give_item","args":{'
        '"agent":"agent_2",'
        '"items":[{"name":"iron at 12.0, 3.5"},{"name":"coal"}],'
        '"note":"stash near r_07, walk 40 blocks N"}}}'
    )
    w = _worker([out])
    action = w.act(sample_observation())
    assert action.name == ActionName.GIVE_ITEM
    assert action.args["agent"] == "agent_2"               # enum-ish id preserved
    assert action.args["items"][1]["name"] == "coal"       # clean nested value preserved
    assert scan_for_leaks(action.model_dump()) == []        # recursive scrub leaves no leak


def test_legitimate_enum_args_are_preserved_unchanged():
    for out, key, val in [
        ('{"action":{"name":"gather","args":{"resource":"iron_ore"}}}', "resource", "iron_ore"),
        ('{"action":{"name":"move","args":{"direction":"N"}}}', "direction", "N"),
        ('{"action":{"name":"give_item","args":{"agent":"agent_2","item":"coal","n":3}}}', "agent", "agent_2"),
    ]:
        w = _worker([out])
        action = w.act(sample_observation())
        assert action.args[key] == val
        assert scan_for_leaks(action.model_dump()) == []


# --- non-string leak vectors: numeric coordinate pairs + leaky dict keys ----- #
def test_numeric_coordinate_pair_arg_becomes_wait():
    out = '{"action":{"name":"move","args":{"target":[12.0,3.5]}}}'
    w = _worker([out])
    action = w.act(sample_observation())
    assert action.name == ActionName.WAIT          # numeric pair is unusable
    assert scan_for_leaks(action.model_dump()) == []


def test_integer_coordinate_pair_arg_becomes_wait():
    out = '{"action":{"name":"move","args":{"cell":[3,7]}}}'
    w = _worker([out])
    action = w.act(sample_observation())
    assert action.name == ActionName.WAIT          # 2-int pair is also coordinate-shaped
    assert scan_for_leaks(action.model_dump()) == []


def test_leaky_dict_key_is_dropped():
    out = '{"action":{"name":"report","args":{"r_07":"found iron","content":"ok"}}}'
    w = _worker([out])
    action = w.act(sample_observation())
    assert action.name == ActionName.REPORT
    assert "r_07" not in action.args               # region-id key removed
    assert action.args.get("content") == "ok"      # clean sibling key preserved
    assert scan_for_leaks(action.model_dump()) == []


def test_pos_dict_key_is_dropped():
    out = '{"action":{"name":"move","args":{"pos":"N","direction":"N"}}}'
    w = _worker([out])
    action = w.act(sample_observation())
    assert "pos" not in action.args
    assert action.args.get("direction") == "N"
    assert scan_for_leaks(action.model_dump()) == []


def test_nested_numeric_pair_becomes_wait():
    out = '{"action":{"name":"give_item","args":{"agent":"agent_2","route":{"waypoint":[1.0,2.0]}}}}'
    w = _worker([out])
    action = w.act(sample_observation())
    assert action.name == ActionName.WAIT          # pair found deep in the args tree
    assert scan_for_leaks(action.model_dump()) == []


def test_clean_nested_args_preserved_and_leak_free():
    out = (
        '{"action":{"name":"give_item","args":'
        '{"agent":"agent_2","item":"coal","n":3,"tags":["mining","iron_ore"]}}}'
    )
    w = _worker([out])
    action = w.act(sample_observation())
    assert action.name == ActionName.GIVE_ITEM
    assert action.args == {"agent": "agent_2", "item": "coal", "n": 3, "tags": ["mining", "iron_ore"]}
    assert scan_for_leaks(action.model_dump()) == []


def test_prompt_action_arg_specs_match_build_spec():
    prompt = build_worker_prompt(sample_observation(), sample_behavior_card(), sample_execution_memory())
    assert 'place: args: {"item": "<block>"}' in prompt
    assert 'give_item: args: {"agent": "agent_k", "item": "<name>", "n": <int>}' in prompt
    assert 'regroup: args: {"agent": "agent_k"}' in prompt
    assert 'place: args: {"structure": "<name>"}' not in prompt  # old, wrong place spec gone


# --------------------------------------------------------------------------- #
# A6 — prompt consumes card directives + execution memory; card change matters
# --------------------------------------------------------------------------- #
def test_prompt_includes_card_and_memory():
    obs = sample_observation()
    card = sample_behavior_card()
    memory = sample_execution_memory()
    prompt = build_worker_prompt(obs, card, memory)
    # behavior card content
    assert card.assignment in prompt
    assert "craft a stone pickaxe before mining iron" in prompt  # directive
    assert "iron tooling" in prompt                              # priority
    assert "don't mine without the right pickaxe" in prompt      # dont
    # execution memory heuristic
    assert "craft stone pickaxe first" in prompt
    # action menu + no-leak rules present
    assert "gather" in prompt and "wait" in prompt
    assert "NO-LEAK" in prompt


def test_card_change_changes_prompt():
    obs = sample_observation()
    memory = ExecutionMemory(agent_id="agent_2")
    card_a = BehaviorCard(agent_id="agent_2", role=Role.MINER, assignment="Mine iron fast", directives=["alpha-directive"])
    card_b = BehaviorCard(agent_id="agent_2", role=Role.EXPLORER, assignment="Scout the north", directives=["beta-directive"])
    pa = build_worker_prompt(obs, card_a, memory)
    pb = build_worker_prompt(obs, card_b, memory)
    assert pa != pb
    assert "Mine iron fast" in pa and "alpha-directive" in pa
    assert "Scout the north" in pb and "beta-directive" in pb
    # role primer also switches with the card's role
    assert "Explorer" in pb


def test_card_change_changes_llm_observed_input():
    obs = sample_observation()
    memory = ExecutionMemory(agent_id="agent_2")
    resp = '{"action":{"name":"wait"}}'
    card_a = BehaviorCard(agent_id="agent_2", role=Role.MINER, assignment="ASSIGNMENT-ALPHA")
    card_b = BehaviorCard(agent_id="agent_2", role=Role.MINER, assignment="ASSIGNMENT-BETA")
    wa = LLMWorker("agent_2", ScriptedLLM([resp]), card_a, memory)
    wb = LLMWorker("agent_2", ScriptedLLM([resp]), card_b, memory)
    wa.act(obs)
    wb.act(obs)
    assert "ASSIGNMENT-ALPHA" in wa.llm.prompts[-1]
    assert "ASSIGNMENT-BETA" in wb.llm.prompts[-1]


def test_history_summary_stays_bounded():
    obs = sample_observation()
    w = _worker(['{"action":{"name":"scout"}}'] * 0)  # responses irrelevant; reuse default
    w.llm = ScriptedLLM([], default='{"action":{"name":"scout"}}')
    for _ in range(20):
        w.act(obs)
    # history is compacted: detailed tail + a "(+N earlier rounds)" prefix
    assert "earlier rounds" in w.history_summary
    assert len(w.history_summary) < 400
