# functional_graph_rules_mdl_api.py
#
#  - 4 comparable modes:
#       A_topk:    disambiguate=False, rules=False
#       B_rules:   disambiguate=False, rules=True  (your rules/MDL)
#       C_cg3:     disambiguate=True,  rules=False (VISL CG3 baseline)
#       D_cg3rules disambiguate=True,  rules=True  (optional)
#  - TopK is preserved always (Decision.topk).
#  - Rules "count" ONLY when they CHANGE top1:
#       - explanation "rule:..." only if changed
#       - ResolvedByRule exception only if changed
#       - MDL reduced_ambiguity == changed
#  - R_PREF_NONNEG fires ONLY if there is at least one NEG and one non-NEG candidate.
#  - Stable ordering: analyze -> base decision -> learn rules -> patch -> exceptions -> (optional sem) -> persist -> eval
#
# Requirements:
#   pip install uniparser-udmurt
#   Optional: ruwordnet
#
# NOTE about CG3/VISLCG3:
#   uniparser_morph calls binary named "cg3".
#   If your system has vislcg3, make a symlink:  ln -s $(which vislcg3) /usr/local/bin/cg3

import re
import json
import csv
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional, Tuple, Callable, Iterable, Iterator
from collections import Counter, defaultdict

from uniparser_udmurt import UdmurtAnalyzer

try:
    from ruwordnet import RuWordNet
except Exception:
    RuWordNet = None

# Types / Artifacts

@dataclass
class Token:
    token_id: str
    surface: str
    sent_id: str = "S1"
    idx: int = 0
    left: List[str] = field(default_factory=list)
    right: List[str] = field(default_factory=list)
    prev_token_id: Optional[str] = None
    next_token_id: Optional[str] = None


@dataclass
class Cand:
    cand_id: str
    analysis: str
    tag_str: str
    features: Dict[str, Any]
    score: float = 0.0
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Decision:
    token_id: str
    surface: str
    top1: Optional[str]
    topk: List[str]
    explanation: str


@dataclass
class ExceptionCase:
    token_id: str
    surface: str
    reason: str
    cand_count: int
    cands: List[Dict[str, Any]]
    notes: Optional[str] = None


@dataclass
class SenseCand:
    synset_id: str
    resource: str = "ruwordnet"
    score: float = 0.0
    evidence: List[str] = field(default_factory=list)


@dataclass
class SenseDecision:
    token_id: str
    surface: str
    ru_hint: Optional[str]
    top1: Optional[str]
    topk: List[str]
    explanation: List[str]


@dataclass
class TraceEvent:
    token_id: str
    node: str
    inputs: List[str]
    outputs: List[str]
    note: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RunContext:
    analyzer_mode: str = "strict"
    analyzer_disambiguate: bool = False
    ruwordnet_db_path: Optional[str] = None
    sem_enabled: bool = False
    topk_limit: int = 5

    # Rules / MDL
    rules_enabled: bool = True
    accept_threshold: float = 0.0
    lambda_cost: float = 1.0
    max_rules_apply: int = 50

    # iterative learning
    rule_learn_max_iters: int = 2
    rule_learn_min_gain: float = 0.0
    rule_learn_stop_if_no_change: bool = True

    _analyzer: Any = None
    _rwn: Any = None

    def analyzer(self) -> UdmurtAnalyzer:
        if self._analyzer is None:
            self._analyzer = UdmurtAnalyzer(mode=self.analyzer_mode)
        return self._analyzer

    def rwn(self):
        if not self.sem_enabled:
            return None
        if RuWordNet is None:
            return None
        if self._rwn is None and self.ruwordnet_db_path:
            try:
                self._rwn = RuWordNet(self.ruwordnet_db_path)
            except Exception as e:
                print(f"[SEM disabled] RuWordNet init failed: {e}")
                self._rwn = None
        return self._rwn


# Utilities

RULE_ID_RE = re.compile(r"^rule:([A-Z0-9_]+)\s*->")


def decision_rule_id(decision: Optional[Decision]) -> Optional[str]:
    if not decision or not decision.explanation:
        return None
    m = RULE_ID_RE.match(decision.explanation)
    return m.group(1) if m else None


def _norm(x: Any) -> Any:
    return "" if x is None else x


# Helpers (morph/sem)

def gramm_to_features(gramm: List[str], lemma: Optional[str]) -> Dict[str, Any]:
    g = [x.lower() for x in (gramm or [])]
    feats: Dict[str, Any] = {}
    if lemma:
        feats["lemma"] = lemma
    if gramm:
        feats["pos"] = gramm[0]

    case_map = {
        "nom": "NOM", "acc": "ACC", "gen": "GEN", "dat": "DAT",
        "ill": "ILL", "loc": "LOC", "abl": "ABL", "el": "EL",
        "ins": "INS", "prol": "PROL",
    }
    for x in g:
        if x in case_map:
            feats["case"] = case_map[x]

    if "sg" in g:
        feats["number"] = "SG"
    if "pl" in g:
        feats["number"] = "PL"

    for p in ("1", "2", "3"):
        if p in g:
            feats["person"] = int(p)

    if "pass" in g:
        feats["voice"] = "PASS"
    if "neg" in g:
        feats["polarity"] = "NEG"
    if "prs" in g:
        feats["tense"] = "PRS"
    if "pst" in g:
        feats["tense"] = "PST"
    if "fut" in g:
        feats["tense"] = "FUT"

    if "tr" in g:
        feats["transitivity"] = "TR"
    if "intr" in g:
        feats["transitivity"] = "INTR"

    return feats


def build_cands(surface: str, analyses_for_token: List[Dict[str, Any]]) -> List[Cand]:
    out: List[Cand] = []
    for i, an in enumerate(analyses_for_token or [], start=1):
        out.append(Cand(
            cand_id=f"K{i}",
            analysis=an.get("wfGlossed") or an.get("wf") or surface,
            tag_str=an.get("gloss") or "",
            features=gramm_to_features(an.get("gramm", []), an.get("lemma")),
            score=0.0,
            meta={"raw": an}
        ))
    return out


def fallback_choose_top1(cands: List[Cand]) -> Tuple[Optional[str], str]:
    """
    Top-1 baseline (clean):
      - no heuristics
      - deterministic: first candidate as returned by analyzer
    """
    if not cands:
        return None, "no_candidates"
    if len(cands) == 1:
        return cands[0].cand_id, "single_candidate"
    return cands[0].cand_id, "first_candidate"


def build_ru_hint(chosen: Optional[Cand]) -> Optional[str]:
    raw = (chosen.meta.get("raw", {}) if chosen else {}) or {}
    return raw.get("trans_ru")


def ruwordnet_candidates(rwn: Any, ru_hint: str) -> List[SenseCand]:
    ru = (ru_hint or "").strip().lower()
    if not ru or rwn is None:
        return []
    out: List[SenseCand] = []

    try:
        senses = rwn.get_senses(ru)
        for s in senses:
            synset = getattr(s, "synset", None)
            synset_id = getattr(synset, "id", None) if synset else None
            if synset_id:
                out.append(SenseCand(synset_id=str(synset_id), evidence=[f"ru_hint={ru}"]))
        if out:
            return out
    except Exception:
        pass

    try:
        synsets = rwn.get_synsets(ru)
        for syn in synsets:
            synset_id = getattr(syn, "id", None)
            if synset_id:
                out.append(SenseCand(synset_id=str(synset_id), evidence=[f"ru_hint={ru}"]))
        return out
    except Exception:
        return []

    return out


def choose_synset(token_id: str, surface: str, ru_hint: Optional[str], cands: List[SenseCand], topk_limit: int = 5) -> SenseDecision:
    if not cands:
        return SenseDecision(token_id=token_id, surface=surface, ru_hint=ru_hint, top1=None, topk=[], explanation=["no_synsets"])
    topk = [c.synset_id for c in cands][:topk_limit]
    return SenseDecision(token_id=token_id, surface=surface, ru_hint=ru_hint, top1=topk[0], topk=topk, explanation=["ru_hint_lookup"])


# Store

class ArtifactStore:
    def __init__(self):
        self._data: Dict[Tuple[str, str], Any] = {}

    def put(self, token_id: str, name: str, value: Any) -> None:
        self._data[(token_id, name)] = value

    def get(self, token_id: str, name: str) -> Any:
        return self._data.get((token_id, name))

    def has(self, token_id: str, name: str) -> bool:
        return (token_id, name) in self._data


# Rule system + MDL scoring

@dataclass
class Rule:
    rule_id: str
    description: str
    predicate: Callable[[Token, List[Cand]], bool]
    chooser: Callable[[Token, List[Cand]], Optional[str]]
    cost: float = 1.0


@dataclass
class RuleEval:
    rule_id: str
    matched: int
    changed: int
    reduced_ambiguity: int
    gain: float
    details: Dict[str, Any] = field(default_factory=dict)


def neighbor_best_cand(store: ArtifactStore, token_id: Optional[str], decision_key: str) -> Optional[Cand]:
    if not token_id:
        return None
    d: Optional[Decision] = store.get(token_id, decision_key)
    if not d or not d.top1:
        return None
    cands: List[Cand] = store.get(token_id, "morph_cands") or []
    return next((c for c in cands if c.cand_id == d.top1), None)


def neighbor_pos(store: ArtifactStore, token_id: Optional[str], decision_key: str) -> Optional[str]:
    c = neighbor_best_cand(store, token_id, decision_key)
    return (c.features.get("pos") if c else None)


def has_pos(cands: List[Cand], pos: str) -> bool:
    pos_u = pos.upper()
    return any(((c.features.get("pos") or "").upper() == pos_u) for c in cands)


def pick_first(cands: List[Cand], pred) -> Optional[str]:
    for c in cands:
        if pred(c):
            return c.cand_id
    return None


def apply_rules_once(
    tok: Token,
    cands: List[Cand],
    base_top1: Optional[str],
    rules: List[Rule],
) -> Tuple[Optional[str], Optional[str], bool]:
    """
    Returns: (patched_top1, rule_id_if_changed_else_None, changed)
    IMPORTANT: rule_id is returned ONLY when changed=True.
    """
    for r in rules:
        if not r.predicate(tok, cands):
            continue
        cid = r.chooser(tok, cands)
        if cid and any(c.cand_id == cid for c in cands):
            if cid != base_top1:
                return cid, r.rule_id, True
            # matched but didn't change -> ignore
            return base_top1, None, False
    return base_top1, None, False


def evaluate_rule_mdl(tokens: List[Token], store: ArtifactStore, rule: Rule, lambda_cost: float) -> RuleEval:
    matched = 0
    changed = 0

    for tok in tokens:
        cands: List[Cand] = store.get(tok.token_id, "morph_cands") or []
        if len(cands) <= 1:
            continue
        if not rule.predicate(tok, cands):
            continue

        chosen = rule.chooser(tok, cands)
        if not chosen or not any(c.cand_id == chosen for c in cands):
            continue

        matched += 1

        base: Optional[Decision] = store.get(tok.token_id, "morph_decision_base")
        base_top1 = base.top1 if base else None
        if chosen != base_top1:
            changed += 1

    reduced = changed  # only actual changes count as ambiguity reduction
    gain = float(reduced) + 0.2 * float(changed) - lambda_cost * float(rule.cost)

    return RuleEval(
        rule_id=rule.rule_id,
        matched=matched,
        changed=changed,
        reduced_ambiguity=reduced,
        gain=gain,
        details={"cost": rule.cost, "lambda": lambda_cost, "reduced": reduced}
    )


# Rule proposal

def rule_pref_nonneg() -> Rule:
    """
    Applies ONLY if there exists at least one NEG candidate and at least one non-NEG candidate.
    """
    def pred(tok: Token, cands: List[Cand]) -> bool:
        if len(cands) <= 1:
            return False
        has_neg = any(c.features.get("polarity") == "NEG" for c in cands)
        has_nonneg = any(c.features.get("polarity") != "NEG" for c in cands)
        return has_neg and has_nonneg

    def choose(tok: Token, cands: List[Cand]) -> Optional[str]:
        non_neg = [c for c in cands if c.features.get("polarity") != "NEG"]
        return non_neg[0].cand_id if non_neg else None

    return Rule(
        rule_id="R_PREF_NONNEG",
        description="If both NEG and non-NEG are among candidates, prefer non-NEG",
        predicate=pred,
        chooser=choose,
        cost=1.0
    )


def handcrafted_compressing_rules(store: ArtifactStore, decision_key_for_neighbors: str) -> List[Rule]:
    def is_v(pos: Optional[str]) -> bool:
        return (pos or "").upper() == "V"

    def is_n(pos: Optional[str]) -> bool:
        return (pos or "").upper() == "N"

    def is_adj(pos: Optional[str]) -> bool:
        return (pos or "").upper() == "ADJ"

    def is_pro(pos: Optional[str]) -> bool:
        return (pos or "").upper() in {"PRO", "PRON"}

    def is_content(pos: Optional[str]) -> bool:
        return (pos or "").upper() in {"N", "V", "ADJ", "ADV", "PRO", "PRON", "NUM"}

    rules: List[Rule] = [rule_pref_nonneg()]

    def r_adj_before_n():
        rid = "R_ADJ_BEFORE_N"
        def pred(tok: Token, cands: List[Cand]) -> bool:
            if len(cands) <= 1:
                return False
            if not (has_pos(cands, "ADJ") and has_pos(cands, "N")):
                return False
            nxt = neighbor_pos(store, tok.next_token_id, decision_key_for_neighbors)
            return is_n(nxt)
        def choose(tok: Token, cands: List[Cand]) -> Optional[str]:
            return pick_first(cands, lambda c: is_adj(c.features.get("pos")))
        return Rule(rid, "ADJ vs N: if next is N -> ADJ", pred, choose, cost=1.5)

    def r_n_if_not_followed_by_n():
        rid = "R_N_IF_NOT_FOLLOWED_BY_N"
        def pred(tok: Token, cands: List[Cand]) -> bool:
            if len(cands) <= 1:
                return False
            if not (has_pos(cands, "ADJ") and has_pos(cands, "N")):
                return False
            nxt = neighbor_pos(store, tok.next_token_id, decision_key_for_neighbors)
            return (nxt is None) or (not is_n(nxt))
        def choose(tok: Token, cands: List[Cand]) -> Optional[str]:
            return pick_first(cands, lambda c: (c.features.get("pos") or "").upper() == "N")
        return Rule(rid, "ADJ vs N: if next not N -> N", pred, choose, cost=1.5)

    def r_cnj_between_content():
        rid = "R_CNJ_BETWEEN_CONTENT"
        def pred(tok: Token, cands: List[Cand]) -> bool:
            if len(cands) <= 1:
                return False
            if not (has_pos(cands, "CNJ") and has_pos(cands, "PART")):
                return False
            prv = neighbor_pos(store, tok.prev_token_id, decision_key_for_neighbors)
            nxt = neighbor_pos(store, tok.next_token_id, decision_key_for_neighbors)
            return is_content(prv) and is_content(nxt)
        def choose(tok: Token, cands: List[Cand]) -> Optional[str]:
            return pick_first(cands, lambda c: (c.features.get("pos") or "").upper() == "CNJ")
        return Rule(rid, "CNJ vs PART: content-content -> CNJ", pred, choose, cost=1.6)

    def r_part_sentence_initial():
        rid = "R_PART_SENTENCE_INITIAL"
        def pred(tok: Token, cands: List[Cand]) -> bool:
            return tok.idx == 0 and len(cands) > 1 and has_pos(cands, "CNJ") and has_pos(cands, "PART")
        def choose(tok: Token, cands: List[Cand]) -> Optional[str]:
            return pick_first(cands, lambda c: (c.features.get("pos") or "").upper() == "PART")
        return Rule(rid, "CNJ vs PART: idx==0 -> PART", pred, choose, cost=1.6)

    def r_nom_before_verb():
        rid = "R_NOM_BEFORE_VERB"
        def pred(tok: Token, cands: List[Cand]) -> bool:
            if len(cands) <= 1:
                return False
            has_nom = any((c.features.get("pos") == "N" and c.features.get("case") == "NOM") for c in cands)
            has_acc = any((c.features.get("pos") == "N" and c.features.get("case") == "ACC") for c in cands)
            if not (has_nom and has_acc):
                return False
            nxt = neighbor_pos(store, tok.next_token_id, decision_key_for_neighbors)
            return is_v(nxt)
        def choose(tok: Token, cands: List[Cand]) -> Optional[str]:
            return pick_first(cands, lambda c: (c.features.get("pos") == "N" and c.features.get("case") == "NOM"))
        return Rule(rid, "NOM vs ACC: next V -> NOM", pred, choose, cost=1.7)

    def r_pro_before_verb():
        rid = "R_PRO_BEFORE_VERB"
        def pred(tok: Token, cands: List[Cand]) -> bool:
            if len(cands) <= 1:
                return False
            if not (has_pos(cands, "PRO") and has_pos(cands, "N")):
                return False
            nxt = neighbor_pos(store, tok.next_token_id, decision_key_for_neighbors)
            return is_v(nxt)
        def choose(tok: Token, cands: List[Cand]) -> Optional[str]:
            return pick_first(cands, lambda c: is_pro(c.features.get("pos")))
        return Rule(rid, "PRO vs N: next V -> PRO", pred, choose, cost=1.7)

    def r_nom_sentence_initial():
        rid = "R_NOM_SENTENCE_INITIAL"
        def pred(tok: Token, cands: List[Cand]) -> bool:
            if tok.idx != 0 or len(cands) <= 1:
                return False
            has_nom = any((c.features.get("pos") == "N" and c.features.get("case") == "NOM") for c in cands)
            has_acc_ill = any((c.features.get("pos") == "N" and c.features.get("case") in {"ACC", "ILL"}) for c in cands)
            return has_nom and has_acc_ill
        def choose(tok: Token, cands: List[Cand]) -> Optional[str]:
            return pick_first(cands, lambda c: (c.features.get("pos") == "N" and c.features.get("case") == "NOM"))
        return Rule(rid, "idx==0: NOM over ACC/ILL", pred, choose, cost=1.6)

    rules.extend([
        r_adj_before_n(),
        r_n_if_not_followed_by_n(),
        r_cnj_between_content(),
        r_part_sentence_initial(),
        r_nom_before_verb(),
        r_pro_before_verb(),
        r_nom_sentence_initial(),
    ])
    return rules


def cand_type(c: Cand) -> Tuple:
    f = c.features or {}
    return (
        _norm((f.get("pos") or "").upper()),
        _norm(f.get("case")),
        _norm(f.get("number")),
        _norm(f.get("tense")),
        _norm(f.get("polarity")),
        _norm(f.get("voice")),
        _norm(f.get("person")),
        _norm(f.get("transitivity")),
    )


def _alt_sig_two(cands: List[Cand]) -> Optional[Tuple[Tuple, Tuple]]:
    types = [cand_type(c) for c in cands]
    uniq = list(dict.fromkeys(types))
    if len(uniq) != 2:
        return None
    uniq_sorted = sorted(uniq, key=lambda t: tuple(str(x) for x in t))
    return (uniq_sorted[0], uniq_sorted[1])


def propose_rules_auto(ctx: RunContext, tokens: List[Token], store: ArtifactStore, decision_key_for_neighbors: str) -> List[Rule]:
    """
    Optional: auto rules mined from current decisions (MDL-ish).
    If you want ONLY handcrafted rules, set min_support huge or just return [] here.
    """
    min_support = 8
    min_conf = 0.85
    bucket: Dict[Tuple[Any, Any, Any, Any], List[Tuple]] = defaultdict(list)

    for tok in tokens:
        cands: List[Cand] = store.get(tok.token_id, "morph_cands") or []
        if len(cands) <= 1:
            continue

        alt = _alt_sig_two(cands)
        if not alt:
            continue

        dec: Optional[Decision] = store.get(tok.token_id, decision_key_for_neighbors)
        if not dec or not dec.top1:
            continue

        chosen = next((c for c in cands if c.cand_id == dec.top1), None)
        if not chosen:
            continue

        prev_pos = neighbor_pos(store, tok.prev_token_id, decision_key_for_neighbors)
        next_pos = neighbor_pos(store, tok.next_token_id, decision_key_for_neighbors)
        idx0 = (tok.idx == 0)

        pattern = (alt, prev_pos, next_pos, idx0)
        bucket[pattern].append(cand_type(chosen))

    auto_rules: List[Rule] = []
    for pattern, choices in bucket.items():
        support = len(choices)
        if support < min_support:
            continue

        cnt = Counter(choices)
        best_choice, best_n = cnt.most_common(1)[0]
        conf = best_n / support
        if conf < min_conf:
            continue

        alt, prev_pos, next_pos, idx0 = pattern
        cost = 1.0 + (0.5 if prev_pos is not None else 0.0) + (0.5 if next_pos is not None else 0.0) + (0.5 if idx0 else 0.0)
        rid = f"R_AUTO_{len(auto_rules) + 1:03d}"

        def make_pred(alt_, prev_pos_, next_pos_, idx0_):
            def pred(tok: Token, cands: List[Cand]) -> bool:
                if len(cands) <= 1:
                    return False
                alt2 = _alt_sig_two(cands)
                if not alt2 or alt2 != alt_:
                    return False
                if idx0_ and tok.idx != 0:
                    return False
                if prev_pos_ is not None and neighbor_pos(store, tok.prev_token_id, decision_key_for_neighbors) != prev_pos_:
                    return False
                if next_pos_ is not None and neighbor_pos(store, tok.next_token_id, decision_key_for_neighbors) != next_pos_:
                    return False
                return True
            return pred

        def make_choose(best_choice_):
            def choose(tok: Token, cands: List[Cand]) -> Optional[str]:
                for c in cands:
                    if cand_type(c) == best_choice_:
                        return c.cand_id
                return None
            return choose

        auto_rules.append(Rule(
            rule_id=rid,
            description=f"auto support={support} conf={conf:.2f} prev={prev_pos} next={next_pos} idx0={idx0}",
            predicate=make_pred(alt, prev_pos, next_pos, idx0),
            chooser=make_choose(best_choice),
            cost=cost
        ))

    return auto_rules


def propose_rule_pool(ctx: RunContext, tokens: List[Token], store: ArtifactStore, decision_key_for_neighbors: str) -> List[Rule]:
    pool = handcrafted_compressing_rules(store, decision_key_for_neighbors)
    pool.extend(propose_rules_auto(ctx, tokens, store, decision_key_for_neighbors))
    return pool


def learn_rules(ctx: RunContext, tokens: List[Token], store: ArtifactStore, decision_key_for_neighbors: str) -> Tuple[List[Rule], List[RuleEval]]:
    pool = propose_rule_pool(ctx, tokens, store, decision_key_for_neighbors)
    evals: List[RuleEval] = [evaluate_rule_mdl(tokens, store, r, ctx.lambda_cost) for r in pool]

    accepted: List[Rule] = []
    for r, ev in zip(pool, evals):
        if ev.matched <= 0:
            continue
        if ev.changed <= 0:
            continue
        if ev.gain <= ctx.accept_threshold:
            continue
        if ev.gain < ctx.rule_learn_min_gain:
            continue
        accepted.append(r)

    return accepted, evals


# Pipeline nodes

def node_rule_patch(
    ctx: RunContext,
    tok: Token,
    store: ArtifactStore,
    trace: List[TraceEvent],
    rules: List[Rule],
    base_decision_key: str = "morph_decision_base",
    out_decision_key: str = "morph_decision"
) -> None:
    base: Optional[Decision] = store.get(tok.token_id, base_decision_key)
    cands: List[Cand] = store.get(tok.token_id, "morph_cands") or []

    if not base:
        d = Decision(tok.token_id, tok.surface, None, [], "no_base_decision")
        store.put(tok.token_id, out_decision_key, d)
        store.put(tok.token_id, "morph_chosen", None)
        trace.append(TraceEvent(tok.token_id, "RulePatch", [base_decision_key, "morph_cands"], [out_decision_key], note="no_base"))
        return

    # if not ambiguous or no rules -> copy base
    if (not ctx.rules_enabled) or (not rules) or len(cands) <= 1:
        store.put(tok.token_id, out_decision_key, base)
        store.put(tok.token_id, "morph_chosen", store.get(tok.token_id, "morph_chosen_base"))
        trace.append(TraceEvent(tok.token_id, "RulePatch", [base_decision_key], [out_decision_key], note="no_rules_or_not_ambiguous"))
        return

    patched_top1, changed_rule_id, changed = apply_rules_once(tok, cands, base.top1, rules[: ctx.max_rules_apply])

    expl = base.explanation
    if changed_rule_id and changed:
        expl = f"rule:{changed_rule_id} -> {base.explanation}"

    d = Decision(tok.token_id, tok.surface, patched_top1, base.topk, expl)
    store.put(tok.token_id, out_decision_key, d)

    chosen = next((c for c in cands if c.cand_id == patched_top1), None) if patched_top1 else None
    store.put(tok.token_id, "morph_chosen", chosen)

    trace.append(TraceEvent(
        tok.token_id, "RulePatch",
        [base_decision_key, "morph_cands"],
        [out_decision_key, "morph_chosen"],
        note=("changed" if changed else "no_change"),
        meta={"changed_rule_id": changed_rule_id, "changed": changed, "base_top1": base.top1, "patched_top1": patched_top1}
    ))


def node_morph_exceptions(ctx: RunContext, tok: Token, store: ArtifactStore, trace: List[TraceEvent], decision_key: str = "morph_decision") -> None:
    raw_cands: List[Cand] = store.get(tok.token_id, "morph_cands") or []
    d: Optional[Decision] = store.get(tok.token_id, decision_key)
    rid = decision_rule_id(d)

    ex: List[ExceptionCase] = []

    if not raw_cands:
        ex.append(ExceptionCase(tok.token_id, tok.surface, "CoverageException", 0, []))
        store.put(tok.token_id, "exceptions", ex)
        trace.append(TraceEvent(tok.token_id, "MorphExceptions", ["morph_cands", decision_key], ["exceptions"], meta={"reasons": [e.reason for e in ex]}))
        return

    if len(raw_cands) > 1:
        # ResolvedByRule ONLY if rule actually changed (rid is only present when changed)
        if rid:
            ex.append(ExceptionCase(
                token_id=tok.token_id,
                surface=tok.surface,
                reason="ResolvedByRule",
                cand_count=len(raw_cands),
                cands=[asdict(c) for c in raw_cands],
                notes=f"rule_id={rid}"
            ))
        else:
            ex.append(ExceptionCase(
                token_id=tok.token_id,
                surface=tok.surface,
                reason="AmbiguityException",
                cand_count=len(raw_cands),
                cands=[asdict(c) for c in raw_cands]
            ))

        analyses = {c.analysis for c in raw_cands}
        glosses = {c.tag_str for c in raw_cands}
        feat_strings = {json.dumps(c.features, sort_keys=True, ensure_ascii=False) for c in raw_cands}

        if len(analyses) > 1:
            ex.append(ExceptionCase(tok.token_id, tok.surface, "SegmentationConflictException", len(raw_cands),
                                   [asdict(c) for c in raw_cands], notes="разные wfGlossed => разные морфемные разрезы"))
        if len(glosses) > 1:
            ex.append(ExceptionCase(tok.token_id, tok.surface, "TagConflictException", len(raw_cands),
                                   [asdict(c) for c in raw_cands], notes="разные gloss при анализе"))
        if len(feat_strings) > 1 and len(glosses) == 1 and len(analyses) == 1:
            ex.append(ExceptionCase(tok.token_id, tok.surface, "FeatureMismatchException", len(raw_cands),
                                   [asdict(c) for c in raw_cands], notes="одинаковые analysis+gloss, но разные gramm/features"))

    store.put(tok.token_id, "exceptions", ex)
    trace.append(TraceEvent(
        tok.token_id, "MorphExceptions",
        ["morph_cands", decision_key], ["exceptions"],
        meta={"rule_id": rid, "raw_cand_count": len(raw_cands), "reasons": [e.reason for e in ex]}
    ))


def node_build_ru_hint(ctx: RunContext, tok: Token, store: ArtifactStore, trace: List[TraceEvent]) -> None:
    chosen: Optional[Cand] = store.get(tok.token_id, "morph_chosen")
    ru_hint = build_ru_hint(chosen)
    store.put(tok.token_id, "ru_hint", ru_hint)
    trace.append(TraceEvent(tok.token_id, "BuildRuHint", ["morph_chosen"], ["ru_hint"], meta={"ru_hint": ru_hint}))


def node_sense_candidates(ctx: RunContext, tok: Token, store: ArtifactStore, trace: List[TraceEvent]) -> None:
    ru_hint: Optional[str] = store.get(tok.token_id, "ru_hint")
    rwn = ctx.rwn()
    cands = ruwordnet_candidates(rwn, ru_hint) if (ru_hint and rwn is not None) else []
    store.put(tok.token_id, "sense_cands", cands)
    trace.append(TraceEvent(tok.token_id, "SenseCandidates", ["ru_hint"], ["sense_cands"], meta={"sense_cand_count": len(cands)}))


def node_sense_choose(ctx: RunContext, tok: Token, store: ArtifactStore, trace: List[TraceEvent]) -> None:
    ru_hint: Optional[str] = store.get(tok.token_id, "ru_hint")
    scands: List[SenseCand] = store.get(tok.token_id, "sense_cands") or []
    if ctx.sem_enabled and ru_hint:
        sdec = choose_synset(tok.token_id, tok.surface, ru_hint, scands, topk_limit=ctx.topk_limit)
    else:
        sdec = SenseDecision(tok.token_id, tok.surface, ru_hint, None, [], ["sem_disabled_or_no_ru_hint"])
    store.put(tok.token_id, "sense_decision", sdec)
    trace.append(TraceEvent(tok.token_id, "SenseChoose", ["sense_cands", "ru_hint"], ["sense_decision"], meta={"top1": sdec.top1}))


# Dataset IO

BRACE_TOKEN_RE = re.compile(r"\{\{(.*?)\}\}")


def extract_tokens_from_braces(text: str) -> List[str]:
    toks = [t.strip() for t in BRACE_TOKEN_RE.findall(text or "")]
    return [t for t in toks if t]


def iter_dataset_sentences(csv_path: str, delimiter: str = ";") -> Iterator[Tuple[str, List[str]]]:
    """
    Reads CSV and finds first cell containing {{...}} tokens.
    """
    with open(csv_path, "r", encoding="utf8", newline="") as f:
        reader = csv.reader(f, delimiter=delimiter, quotechar='"', doublequote=True)

        sent_no = 0
        for row in reader:
            cell = None
            for c in row:
                if c and "{{" in c and "}}" in c:
                    cell = c
                    break
            if not cell:
                continue

            tokens = extract_tokens_from_braces(cell)
            if not tokens:
                continue

            sent_no += 1
            yield f"S{sent_no}", tokens


def make_tokens_for_sentence(words: List[str], sent_id: str, window: int = 2) -> List[Token]:
    toks: List[Token] = []
    for i, w in enumerate(words):
        left = words[max(0, i - window): i]
        right = words[i + 1: i + 1 + window]
        prev_id = f"{sent_id}_T{i}" if i > 0 else None
        next_id = f"{sent_id}_T{i + 2}" if i + 1 < len(words) else None
        toks.append(Token(
            token_id=f"{sent_id}_T{i + 1}",
            surface=w,
            sent_id=sent_id,
            idx=i,
            left=left,
            right=right,
            prev_token_id=prev_id,
            next_token_id=next_id
        ))
    return toks


def phaseA_morph_sentence_batched(
    ctx: RunContext,
    store: ArtifactStore,
    trace: List[TraceEvent],
    csv_path: str,
    window: int = 2,
    limit_sentences: Optional[int] = None,
    delimiter: str = ";",
) -> List[Token]:
    a = ctx.analyzer()
    all_tokens: List[Token] = []
    sent_count = 0

    for sent_id, words in iter_dataset_sentences(csv_path, delimiter=delimiter):
        sent_count += 1
        if limit_sentences is not None and sent_count > limit_sentences:
            break

        toks = make_tokens_for_sentence(words, sent_id=sent_id, window=window)
        all_tokens.extend(toks)

        analyses = a.analyze_words(words, disambiguate=ctx.analyzer_disambiguate, format="json")

        for tok, token_analyses in zip(toks, analyses):
            cands = build_cands(tok.surface, token_analyses)
            store.put(tok.token_id, "morph_cands", cands)
            trace.append(TraceEvent(tok.token_id, "MorphAnalyzeSentenceBatch", ["__sentence__"], ["morph_cands"],
                                    meta={"sent_id": sent_id, "cand_count": len(cands)}))

            top1, expl = fallback_choose_top1(cands)
            d = Decision(tok.token_id, tok.surface, top1, [c.cand_id for c in cands], expl)
            store.put(tok.token_id, "morph_decision_base", d)
            store.put(tok.token_id, "morph_chosen_base", next((c for c in cands if c.cand_id == top1), None) if top1 else None)

    print(f"Loaded sentences: {sent_count}, tokens: {len(all_tokens)}")
    return all_tokens


# Iterative learning loop

def run_rule_learning_iterations(ctx: RunContext, tokens: List[Token], store: ArtifactStore, trace: List[TraceEvent]) -> None:
    """
    Learns a set of rules (accepted_all) and writes:
      __GLOBAL__/rules_accepted, __GLOBAL__/rules_evals
    """
    accepted_all: List[Rule] = []
    evals_all: List[RuleEval] = []
    neighbor_key = "morph_decision_base"

    for it in range(ctx.rule_learn_max_iters):
        pool = propose_rule_pool(ctx, tokens, store, decision_key_for_neighbors=neighbor_key)
        evals = [evaluate_rule_mdl(tokens, store, r, ctx.lambda_cost) for r in pool]
        evals_all.extend(evals)

        accepted = [r for r, ev in zip(pool, evals)
                    if ev.matched > 0 and ev.changed > 0 and ev.gain > ctx.accept_threshold and ev.gain >= ctx.rule_learn_min_gain]

        already = {r.rule_id for r in accepted_all}
        accepted_new = [r for r in accepted if r.rule_id not in already]

        trace.append(TraceEvent(
            token_id="__GLOBAL__",
            node="RuleLearnIter",
            inputs=["tokens", "morph_cands", neighbor_key],
            outputs=["__GLOBAL__.rules_accepted_iter"],
            note=f"iter={it} accepted_new={len(accepted_new)}",
            meta={"iter": it, "accepted_new": [r.rule_id for r in accepted_new]}
        ))

        if not accepted_new:
            break

        accepted_all.extend(accepted_new)

        # apply to create new neighbor decisions for next iteration
        changed_tokens = 0
        for tok in tokens:
            base: Optional[Decision] = store.get(tok.token_id, neighbor_key)
            before = base.top1 if base else None

            node_rule_patch(ctx, tok, store, trace, rules=accepted_all,
                            base_decision_key=neighbor_key,
                            out_decision_key="morph_decision")

            after = store.get(tok.token_id, "morph_decision").top1 if store.get(tok.token_id, "morph_decision") else None
            if before != after:
                changed_tokens += 1

        trace.append(TraceEvent(
            token_id="__GLOBAL__",
            node="RuleApplyIter",
            inputs=["__GLOBAL__.rules_accepted_all"],
            outputs=["morph_decision"],
            note=f"iter={it} changed_tokens={changed_tokens}",
            meta={"iter": it, "changed_tokens": changed_tokens, "rules_total": len(accepted_all)}
        ))

        neighbor_key = "morph_decision"
        if ctx.rule_learn_stop_if_no_change and changed_tokens == 0:
            break

    store.put("__GLOBAL__", "rules_accepted", accepted_all)
    store.put("__GLOBAL__", "rules_evals", evals_all)

# Persist

def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def persist_dataset(tokens: List[Token], store: ArtifactStore, trace: List[TraceEvent], prefix: str = "") -> None:
    morph_candidates = []
    decisions = []
    decisions_base = []
    exceptions = []
    sense_candidates = []
    sense_decisions = []

    for t in tokens:
        c = store.get(t.token_id, "morph_cands") or []
        morph_candidates.append({
            "sent_id": t.sent_id, "idx": t.idx,
            "token_id": t.token_id, "surface": t.surface,
            "cands": [asdict(x) for x in c]
        })

        db = store.get(t.token_id, "morph_decision_base")
        if db:
            r = asdict(db); r.update({"sent_id": t.sent_id, "idx": t.idx})
            decisions_base.append(r)

        d = store.get(t.token_id, "morph_decision")
        if d:
            r = asdict(d); r.update({"sent_id": t.sent_id, "idx": t.idx})
            decisions.append(r)

        exs: List[ExceptionCase] = store.get(t.token_id, "exceptions") or []
        for ex in exs:
            r = asdict(ex); r.update({"sent_id": t.sent_id, "idx": t.idx})
            exceptions.append(r)

        # semantic artifacts (optional)
        sense_candidates.append({
            "sent_id": t.sent_id, "idx": t.idx,
            "token_id": t.token_id,
            "surface": t.surface,
            "ru_hint": store.get(t.token_id, "ru_hint"),
            "cands": [asdict(x) for x in (store.get(t.token_id, "sense_cands") or [])]
        })

        sd = store.get(t.token_id, "sense_decision")
        if sd:
            r = asdict(sd); r.update({"sent_id": t.sent_id, "idx": t.idx})
            sense_decisions.append(r)

    evals: List[RuleEval] = store.get("__GLOBAL__", "rules_evals") or []
    rules_report = [asdict(e) for e in evals]
    rules_accepted: List[Rule] = store.get("__GLOBAL__", "rules_accepted") or []

    write_jsonl(prefix + "morph_candidates.jsonl", morph_candidates)
    write_jsonl(prefix + "decisions_base.jsonl", decisions_base)
    write_jsonl(prefix + "decisions.jsonl", decisions)
    write_jsonl(prefix + "exceptions.jsonl", exceptions)
    write_jsonl(prefix + "sense_candidates.jsonl", sense_candidates)
    write_jsonl(prefix + "sense_decisions.jsonl", sense_decisions)
    write_jsonl(prefix + "rules_evals.jsonl", rules_report)
    write_jsonl(prefix + "rules_accepted.jsonl", [{"rule_id": r.rule_id, "description": r.description, "cost": r.cost} for r in rules_accepted])
    write_jsonl(prefix + "trace.jsonl", [asdict(t) for t in trace])

    print(f"Wrote {prefix}morph_candidates.jsonl ({len(morph_candidates)})")
    print(f"Wrote {prefix}decisions_base.jsonl ({len(decisions_base)})")
    print(f"Wrote {prefix}decisions.jsonl ({len(decisions)})")
    print(f"Wrote {prefix}exceptions.jsonl ({len(exceptions)})")
    print(f"Wrote {prefix}rules_evals.jsonl ({len(rules_report)})")
    print(f"Wrote {prefix}rules_accepted.jsonl ({len(rules_accepted)})")
    print(f"Wrote {prefix}trace.jsonl ({len(trace)})")

# Run dataset (single entrypoint)

def run_dataset(
    csv_path: str,
    *,
    delimiter: str = ";",
    ruwordnet_db_path: Optional[str] = None,
    analyzer_disambiguate: bool = False,
    rules_enabled: bool = True,
    sem_enabled: bool = False,
    out_prefix: str = "ds_",
    window: int = 2,
    limit_sentences: Optional[int] = None,
    rule_learn_max_iters: int = 2,
) -> None:
    ctx = RunContext(
        analyzer_mode="strict",
        analyzer_disambiguate=analyzer_disambiguate,
        rules_enabled=rules_enabled,
        ruwordnet_db_path=ruwordnet_db_path,
        sem_enabled=sem_enabled,
        accept_threshold=0.0,
        lambda_cost=1.0,
        rule_learn_max_iters=rule_learn_max_iters,
    )

    store = ArtifactStore()
    trace: List[TraceEvent] = []

    # Phase A: analyze + base decision
    tokens = phaseA_morph_sentence_batched(
        ctx, store, trace,
        csv_path,
        window=window,
        limit_sentences=limit_sentences,
        delimiter=delimiter,
    )

    # Learn rules (optional)
    if ctx.rules_enabled and ctx.rule_learn_max_iters > 0:
        run_rule_learning_iterations(ctx, tokens, store, trace)
    else:
        store.put("__GLOBAL__", "rules_accepted", [])
        store.put("__GLOBAL__", "rules_evals", [])

    # Apply final patch
    rules = store.get("__GLOBAL__", "rules_accepted") or []
    for tok in tokens:
        node_rule_patch(ctx, tok, store, trace, rules=rules,
                        base_decision_key="morph_decision_base",
                        out_decision_key="morph_decision")

    # Exceptions (after patch, stable ordering)
    for tok in tokens:
        node_morph_exceptions(ctx, tok, store, trace, decision_key="morph_decision")

    # Semantics (optional)
    if ctx.sem_enabled:
        for tok in tokens:
            node_build_ru_hint(ctx, tok, store, trace)
            node_sense_candidates(ctx, tok, store, trace)
            node_sense_choose(ctx, tok, store, trace)

    persist_dataset(tokens, store, trace, prefix=out_prefix)


# Gold reader + eval

def _split_ws(s: Optional[str]) -> List[str]:
    return [x for x in str(s or "").strip().split() if x]


def _find_first_cell_with_braces(row: Dict[str, str]) -> Optional[str]:
    for v in row.values():
        if v and "{{" in v and "}}" in v:
            return v
    return None


def read_gold_by_sentence(
    csv_path: str,
    *,
    tokens_col: str = "tokens_braced",
    gold_analysis_col: str = "gold_analysis_line",
    gold_tag_col: str = "gold_tag_line",
    delimiter: str = ";",
) -> Dict[str, Dict[str, Any]]:
    gold_map: Dict[str, Dict[str, Any]] = {}

    with open(csv_path, "r", encoding="utf8", newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter, quotechar='"', doublequote=True)

        sent_no = 0
        for row in reader:
            tokens_cell = row.get(tokens_col) or _find_first_cell_with_braces(row)
            if not tokens_cell:
                continue

            tokens = extract_tokens_from_braces(tokens_cell)
            if not tokens:
                continue

            sent_no += 1
            sent_id = f"S{sent_no}"

            ga = row.get(gold_analysis_col, "")
            gt = row.get(gold_tag_col, "")

            gold_map[sent_id] = {
                "tokens": tokens,
                "gold_analysis": _split_ws(ga),
                "gold_tag": _split_ws(gt),
            }

    return gold_map


def load_jsonl_indexed(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def index_morph_candidates(path_candidates_jsonl: str) -> Dict[Tuple[str, int], Dict[str, Any]]:
    out: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for r in load_jsonl_indexed(path_candidates_jsonl):
        key = (r["sent_id"], int(r["idx"]))
        out[key] = r
    return out


def index_decisions(path_decisions_jsonl: str) -> Dict[Tuple[str, int], Dict[str, Any]]:
    out: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for r in load_jsonl_indexed(path_decisions_jsonl):
        key = (r["sent_id"], int(r["idx"]))
        out[key] = r
    return out


def chosen_cand_from_decision(cands_row: Dict[str, Any], decision_row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    top1 = decision_row.get("top1")
    if not top1:
        return None
    for c in cands_row.get("cands", []) or []:
        if c.get("cand_id") == top1:
            return c
    return None


def eval_against_gold(
    csv_path: str,
    *,
    prefix: str,
    tokens_col: str = "tokens_braced",
    gold_analysis_col: str = "gold_analysis_line",
    gold_tag_col: str = "gold_tag_line",
    delimiter: str = ";",
    debug_examples: int = 10,
) -> None:
    gold_map = read_gold_by_sentence(
        csv_path,
        tokens_col=tokens_col,
        gold_analysis_col=gold_analysis_col,
        gold_tag_col=gold_tag_col,
        delimiter=delimiter
    )

    cand_idx = index_morph_candidates(prefix + "morph_candidates.jsonl")
    dec_base = index_decisions(prefix + "decisions_base.jsonl")
    dec_rules = index_decisions(prefix + "decisions.jsonl")

    def _init_stats():
        return {"n": 0, "correct_analysis": 0, "correct_pair": 0}

    stats_all_base = _init_stats()
    stats_all_rules = _init_stats()
    stats_amb_base = _init_stats()
    stats_amb_rules = _init_stats()

    changed_total = 0
    changed_help_analysis = 0
    changed_hurt_analysis = 0

    skipped_missing_gold = 0
    skipped_mismatch_len = 0

    printed = 0
    mismatch_printed = 0

    for (sent_id, idx), cands_row in cand_idx.items():
        g = gold_map.get(sent_id)
        if not g:
            continue

        tokens = g["tokens"]
        gold_a = g["gold_analysis"]
        gold_t = g["gold_tag"]

        if len(gold_a) != len(tokens) or len(gold_t) != len(tokens):
            skipped_mismatch_len += 1
            if mismatch_printed < 10:
                print("MISMATCH", sent_id)
                print("  len(tokens) =", len(tokens))
                print("  len(gold_a) =", len(gold_a))
                print("  len(gold_t) =", len(gold_t))
                print("  tokens:", tokens)
                print("  gold_a:", gold_a)
                print("  gold_t:", gold_t)
                print("---")
                mismatch_printed += 1
            continue

        if idx < 0 or idx >= len(tokens):
            continue

        ga = gold_a[idx]
        gt = gold_t[idx]
        if not ga or not gt:
            skipped_missing_gold += 1
            continue

        key = (sent_id, idx)
        db = dec_base.get(key)
        dr = dec_rules.get(key)
        if not db or not dr:
            continue

        cb = chosen_cand_from_decision(cands_row, db)
        cr = chosen_cand_from_decision(cands_row, dr)
        if not cb or not cr:
            continue

        if printed < debug_examples:
            print("TOK", sent_id, idx, tokens[idx])
            print("  gold_analysis:", ga)
            print("  pred_analysis_base:", cb.get("analysis"))
            print("  pred_analysis_rules:", cr.get("analysis"))
            print("  gold_tag:", gt)
            print("  pred_tag_base:", cb.get("tag_str"))
            print("  pred_tag_rules:", cr.get("tag_str"))
            print("  cand_count:", len(cands_row.get("cands", []) or []))
            print("---")
            printed += 1

        is_amb = (len(cands_row.get("cands", []) or []) > 1)

        # base
        stats_all_base["n"] += 1
        if cb.get("analysis") == ga:
            stats_all_base["correct_analysis"] += 1
        if (cb.get("analysis") == ga) and (cb.get("tag_str") == gt):
            stats_all_base["correct_pair"] += 1

        # rules/cg3/etc (whatever is in decisions.jsonl)
        stats_all_rules["n"] += 1
        if cr.get("analysis") == ga:
            stats_all_rules["correct_analysis"] += 1
        if (cr.get("analysis") == ga) and (cr.get("tag_str") == gt):
            stats_all_rules["correct_pair"] += 1

        if is_amb:
            stats_amb_base["n"] += 1
            if cb.get("analysis") == ga:
                stats_amb_base["correct_analysis"] += 1
            if (cb.get("analysis") == ga) and (cb.get("tag_str") == gt):
                stats_amb_base["correct_pair"] += 1

            stats_amb_rules["n"] += 1
            if cr.get("analysis") == ga:
                stats_amb_rules["correct_analysis"] += 1
            if (cr.get("analysis") == ga) and (cr.get("tag_str") == gt):
                stats_amb_rules["correct_pair"] += 1

        # change impact (analysis only)
        if db.get("top1") != dr.get("top1"):
            changed_total += 1
            base_ok = (cb.get("analysis") == ga)
            rules_ok = (cr.get("analysis") == ga)
            if (not base_ok) and rules_ok:
                changed_help_analysis += 1
            elif base_ok and (not rules_ok):
                changed_hurt_analysis += 1

    def _acc(correct: int, n: int) -> float:
        return (correct / n) if n else 0.0

    print("\nEVAL (analysis only)")
    print(f"ALL   base : {stats_all_base['correct_analysis']}/{stats_all_base['n']}  acc={_acc(stats_all_base['correct_analysis'], stats_all_base['n']):.4f}")
    print(f"ALL   run  : {stats_all_rules['correct_analysis']}/{stats_all_rules['n']}  acc={_acc(stats_all_rules['correct_analysis'], stats_all_rules['n']):.4f}")
    print(f"AMB   base : {stats_amb_base['correct_analysis']}/{stats_amb_base['n']}  acc={_acc(stats_amb_base['correct_analysis'], stats_amb_base['n']):.4f}")
    print(f"AMB   run  : {stats_amb_rules['correct_analysis']}/{stats_amb_rules['n']}  acc={_acc(stats_amb_rules['correct_analysis'], stats_amb_rules['n']):.4f}")

    print("\nEVAL (analysis + tag_str exact match)")
    print(f"ALL   base : {stats_all_base['correct_pair']}/{stats_all_base['n']}  acc={_acc(stats_all_base['correct_pair'], stats_all_base['n']):.4f}")
    print(f"ALL   run  : {stats_all_rules['correct_pair']}/{stats_all_rules['n']}  acc={_acc(stats_all_rules['correct_pair'], stats_all_rules['n']):.4f}")
    print(f"AMB   base : {stats_amb_base['correct_pair']}/{stats_amb_base['n']}  acc={_acc(stats_amb_base['correct_pair'], stats_amb_base['n']):.4f}")
    print(f"AMB   run  : {stats_amb_rules['correct_pair']}/{stats_amb_rules['n']}  acc={_acc(stats_amb_rules['correct_pair'], stats_amb_rules['n']):.4f}")

    print("\nRULE CHANGE IMPACT (analysis)")
    print(f"changed_total={changed_total}")
    print(f"changed_help_analysis={changed_help_analysis}")
    print(f"changed_hurt_analysis={changed_hurt_analysis}")

    print("\nSKIPS")
    print(f"skipped_mismatch_len(sent gold != tokens)={skipped_mismatch_len}")
    print(f"skipped_missing_gold(token gold empty)={skipped_missing_gold}")


# 4-way comparison runner

def run_and_eval_modes(DATA: str) -> None:
    def one(name: str, analyzer_disambiguate: bool, rules_enabled: bool, rule_learn_max_iters: int):
        out_prefix = name + "_"
        print("\n" + "=" * 80)
        print(f"RUN {name}: disambiguate={analyzer_disambiguate} rules_enabled={rules_enabled} iters={rule_learn_max_iters}")
        print("=" * 80)

        run_dataset(
            DATA,
            delimiter=";",
            ruwordnet_db_path=None,
            sem_enabled=False,
            out_prefix=out_prefix,
            window=2,
            limit_sentences=None,
            rule_learn_max_iters=rule_learn_max_iters,
            analyzer_disambiguate=analyzer_disambiguate,
            rules_enabled=rules_enabled
        )

        eval_against_gold(
            DATA,
            prefix=out_prefix,
            tokens_col="tokens_braced",
            gold_analysis_col="gold_analysis_line",
            gold_tag_col="gold_tag_line",
            delimiter=";",
            debug_examples=5
        )

    # A) TopK baseline (no CG3, no rules)
    one("A_topk", analyzer_disambiguate=False, rules_enabled=False, rule_learn_max_iters=0)

    # B) Your rules over TopK (no CG3)
    one("B_rules", analyzer_disambiguate=False, rules_enabled=True, rule_learn_max_iters=2)

    # C) CG3 baseline (CG3 on, rules off)
    one("C_cg3", analyzer_disambiguate=True, rules_enabled=False, rule_learn_max_iters=0)

    # D) CG3 + rules (optional but very useful)
    one("D_cg3rules", analyzer_disambiguate=True, rules_enabled=True, rule_learn_max_iters=2)

# --- ADD BELOW into functional_graph_rules_mdl.py ---

from typing import Dict

def run_sentence(
    text_or_tokens: str,
    *,
    disambiguate: bool,
    rules_enabled: bool,
    rule_learn_max_iters: int = 2,
    window: int = 2,
) -> Dict[str, Any]:
    """
    Input:
      - either plain text (split by whitespace) OR string with {{token}} markup
    Output:
      dict with tokens + store artifacts serialized for UI
    """
    # parse tokens
    if "{{" in (text_or_tokens or "") and "}}" in (text_or_tokens or ""):
        words = extract_tokens_from_braces(text_or_tokens)
    else:
        words = [w for w in (text_or_tokens or "").strip().split() if w]

    ctx = RunContext(
        analyzer_mode="strict",
        analyzer_disambiguate=disambiguate,
        rules_enabled=rules_enabled,
        sem_enabled=False,
        accept_threshold=0.0,
        lambda_cost=1.0,
        rule_learn_max_iters=rule_learn_max_iters,
    )

    store = ArtifactStore()
    trace: List[TraceEvent] = []

    sent_id = "S1"
    toks = make_tokens_for_sentence(words, sent_id=sent_id, window=window)

    # analyze
    a = ctx.analyzer()
    analyses = a.analyze_words(words, disambiguate=ctx.analyzer_disambiguate, format="json")

    for tok, token_analyses in zip(toks, analyses):
        cands = build_cands(tok.surface, token_analyses)
        store.put(tok.token_id, "morph_cands", cands)

        top1, expl = fallback_choose_top1(cands)
        d = Decision(tok.token_id, tok.surface, top1, [c.cand_id for c in cands], expl)
        store.put(tok.token_id, "morph_decision_base", d)
        store.put(tok.token_id, "morph_chosen_base", next((c for c in cands if c.cand_id == top1), None) if top1 else None)

    # learn rules (optional)
    if ctx.rules_enabled and ctx.rule_learn_max_iters > 0:
        run_rule_learning_iterations(ctx, toks, store, trace)
    else:
        store.put("__GLOBAL__", "rules_accepted", [])
        store.put("__GLOBAL__", "rules_evals", [])

    # final patch
    rules = store.get("__GLOBAL__", "rules_accepted") or []
    for tok in toks:
        node_rule_patch(ctx, tok, store, trace, rules=rules,
                        base_decision_key="morph_decision_base",
                        out_decision_key="morph_decision")

    # exceptions
    for tok in toks:
        node_morph_exceptions(ctx, tok, store, trace, decision_key="morph_decision")

    # serialize for UI
    out_tokens = []
    for tok in toks:
        cands: List[Cand] = store.get(tok.token_id, "morph_cands") or []
        db: Optional[Decision] = store.get(tok.token_id, "morph_decision_base")
        d: Optional[Decision] = store.get(tok.token_id, "morph_decision")
        exs: List[ExceptionCase] = store.get(tok.token_id, "exceptions") or []

        out_tokens.append({
            "token_id": tok.token_id,
            "idx": tok.idx,
            "surface": tok.surface,
            "cand_count": len(cands),
            "cands": [asdict(c) for c in cands],
            "base": asdict(db) if db else None,
            "final": asdict(d) if d else None,
            "exceptions": [asdict(e) for e in exs],
        })

    return {
        "words": words,
        "tokens": out_tokens,
        "rules_accepted": [
            {"rule_id": r.rule_id, "description": r.description, "cost": r.cost}
            for r in (store.get("__GLOBAL__", "rules_accepted") or [])
        ],
        "rules_evals": [asdict(e) for e in (store.get("__GLOBAL__", "rules_evals") or [])],
    }


def run_4_modes_for_sentence(text_or_tokens: str) -> Dict[str, Any]:
    """
    Exactly the 4 modes you described.
    """
    return {
        "A_topk": run_sentence(text_or_tokens, disambiguate=False, rules_enabled=False, rule_learn_max_iters=0),
        "B_rules": run_sentence(text_or_tokens, disambiguate=False, rules_enabled=True,  rule_learn_max_iters=2),
        "C_cg3":   run_sentence(text_or_tokens, disambiguate=True,  rules_enabled=False, rule_learn_max_iters=0),
        "D_cg3rules": run_sentence(text_or_tokens, disambiguate=True, rules_enabled=True, rule_learn_max_iters=2),
    }
