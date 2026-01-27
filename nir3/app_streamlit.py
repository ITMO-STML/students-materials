import streamlit as st
import pandas as pd

from functional_graph_rules_mdl_api import run_4_modes_for_sentence

st.set_page_config(page_title="CG3 vs Rules vs Nothing", layout="wide")
st.title("Сравнение режимов: без CG3 и правил / без CG3 с правилами / CG3 и правила / CG3 и правила")

st.markdown("""
Вводи:
- либо обычный текст: `Кот любит молоко`
- либо разметку токенов: `{{Кот}} {{любит}} {{молоко}}`
""")

text = st.text_area("Текст / токены", height=120, value="")

colA, colB = st.columns([1, 1])
with colA:
    show_candidates = st.checkbox("Показывать кандидатов по токенам", value=False)
with colB:
    show_exceptions = st.checkbox("Показывать исключения", value=True)

run_btn = st.button("Запустить 4 режима", type="primary")

def mode_table(mode_name: str, mode_data: dict):
    rows = []
    for t in mode_data["tokens"]:
        rows.append({
            "idx": t["idx"],
            "surface": t["surface"],
            "cand_count": t["cand_count"],
            "base_top1": (t["base"]["top1"] if t["base"] else None),
            "base_expl": (t["base"]["explanation"] if t["base"] else None),
            "final_top1": (t["final"]["top1"] if t["final"] else None),
            "final_expl": (t["final"]["explanation"] if t["final"] else None),
        })
    df = pd.DataFrame(rows).sort_values("idx")
    st.subheader(mode_name)
    st.dataframe(df, use_container_width=True, hide_index=True)

    if show_candidates:
        st.caption("Кандидаты по токенам")
        for t in mode_data["tokens"]:
            with st.expander(f"[{t['idx']}] {t['surface']}  (cand_count={t['cand_count']})"):
                cdf = pd.DataFrame([{
                    "cand_id": c["cand_id"],
                    "analysis": c["analysis"],
                    "tag_str": c["tag_str"],
                    "pos": (c.get("features") or {}).get("pos"),
                    "case": (c.get("features") or {}).get("case"),
                    "number": (c.get("features") or {}).get("number"),
                    "tense": (c.get("features") or {}).get("tense"),
                    "polarity": (c.get("features") or {}).get("polarity"),
                    "lemma": (c.get("features") or {}).get("lemma"),
                } for c in t["cands"]])
                st.dataframe(cdf, use_container_width=True, hide_index=True)

    if show_exceptions:
        st.caption("Исключения")
        for t in mode_data["tokens"]:
            if not t["exceptions"]:
                continue
            with st.expander(f"Exceptions: [{t['idx']}] {t['surface']} ({len(t['exceptions'])})"):
                edf = pd.DataFrame([{
                    "reason": e["reason"],
                    "cand_count": e["cand_count"],
                    "notes": e.get("notes"),
                } for e in t["exceptions"]])
                st.dataframe(edf, use_container_width=True, hide_index=True)

    # rules accepted
    if mode_data.get("rules_accepted"):
        st.caption("Принятые правила (в этом запуске)")
        st.dataframe(pd.DataFrame(mode_data["rules_accepted"]), use_container_width=True, hide_index=True)

if run_btn:
    if not text.strip():
        st.warning("Введи текст.")
        st.stop()

    try:
        res = run_4_modes_for_sentence(text)
    except Exception as e:
        st.error(f"Ошибка запуска: {e}")
        st.stop()

    tabs = st.tabs(["A_topk", "B_rules", "C_cg3", "D_cg3rules"])

    with tabs[0]:
        mode_table("A_topk (no CG3, no rules)", res["A_topk"])
    with tabs[1]:
        mode_table("B_rules (no CG3, with rules)", res["B_rules"])
    with tabs[2]:
        mode_table("C_cg3 (CG3, no rules)", res["C_cg3"])
    with tabs[3]:
        mode_table("D_cg3rules (CG3, with rules)", res["D_cg3rules"])
