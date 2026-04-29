from __future__ import annotations

import html
import json
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import streamlit as st
import streamlit.components.v1 as components



PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from semgraph.pipeline import SemanticKGSystem
except Exception as exc:  # pragma: no cover
    st.error(
        "Не удалось импортировать модули проекта LEMON. "
        "Проверьте, что app.py лежит в корне проекта рядом с папкой src."
    )
    st.exception(exc)
    st.stop()


st.set_page_config(
    page_title="LEMON Demo",
    page_icon="🍋",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner=False)
def get_system() -> SemanticKGSystem:
    """Создаёт объект системы один раз за сессию Streamlit."""
    return SemanticKGSystem()


def safe_metric_value(data: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = data.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def graph_stats(graph: dict[str, Any]) -> tuple[int, int]:
    return len(graph.get("nodes", [])), len(graph.get("edges", []))


def compact_label(text: str | None, max_len: int = 42) -> str:
    if not text:
        return ""
    text = str(text).replace("\n", " ").strip()
    return text if len(text) <= max_len else text[: max_len - 1] + "…"


def relation_to_russian(relation: str) -> str:
    mapping = {
        "has_goal": "цель",
        "uses_method": "метод",
        "studies_condition": "изучает",
        "patient_count": "пациенты",
        "eye_count": "глаза",
        "mean_age": "средний возраст",
        "age_range": "возрастной диапазон",
        "female_share": "женщины",
        "male_share": "мужчины",
        "baseline_se": "SE",
        "baseline_sph": "Sph",
        "baseline_cyl": "Cyl",
        "myopia_min": "миопия >",
        "astigmatism_min": "астигматизм >",
        "uses_device": "устройство",
        "followup_duration": "наблюдение",
        "max_corrected_visual_acuity": "МКОЗ",
        "quality_of_life_score": "качество жизни",
        "emmetropia_rate": "эмметропия",
        "ucva_gain": "прирост НКОЗ",
        "complaint_glare": "блики",
        "complaint_halo": "гало",
        "complaint_night_driving": "ночное вождение",
        "conclusion_stability": "стабильность",
        "conclusion_effectiveness": "эффективность",
    }
    return mapping.get(relation, relation)


def graph_to_vis_html(graph: dict[str, Any], height: int = 620) -> str:
    """Создаёт HTML-визуализацию графа через vis-network CDN."""
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])

    color_by_type = {
        "study": "#ffe08a",
        "goal": "#d7ecff",
        "method": "#d5f5d5",
        "condition": "#f8d7da",
        "population_metric": "#e5d7ff",
        "baseline_metric": "#d7fff4",
        "criteria": "#fce4d6",
        "device": "#e2e3e5",
        "followup": "#fff3cd",
        "outcome": "#d4edda",
        "complaint": "#f8d7da",
        "conclusion": "#d1ecf1",
        "fact": "#eeeeee",
    }

    vis_nodes = []
    for node in nodes:
        node_id = node.get("id", "")
        node_type = node.get("node_type", "fact")
        label = compact_label(node.get("label") or node.get("canonical") or node_id)
        title = html.escape(json.dumps(node, ensure_ascii=False, indent=2))
        vis_nodes.append(
            {
                "id": node_id,
                "label": label,
                "title": title,
                "shape": "box" if node_type == "study" else "ellipse",
                "color": {
                    "background": color_by_type.get(node_type, "#eeeeee"),
                    "border": "#333333",
                },
                "font": {"size": 18 if node_type == "study" else 14},
            }
        )

    vis_edges = []
    for edge in edges:
        relation = edge.get("relation", "")
        vis_edges.append(
            {
                "from": edge.get("source"),
                "to": edge.get("target"),
                "label": relation_to_russian(relation),
                "arrows": "to",
                "font": {"align": "middle", "size": 12},
                "color": {"color": "#555555"},
                "title": html.escape(json.dumps(edge, ensure_ascii=False, indent=2)),
            }
        )

    nodes_json = json.dumps(vis_nodes, ensure_ascii=False)
    edges_json = json.dumps(vis_edges, ensure_ascii=False)

    return f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <script src="https://unpkg.com/vis-network@9.1.9/standalone/umd/vis-network.min.js"></script>
  <style>
    body {{ margin: 0; font-family: Arial, sans-serif; background: #ffffff; }}
    #network {{ width: 100%; height: {height}px; border: 1px solid #ddd; border-radius: 12px; }}
  </style>
</head>
<body>
  <div id="network"></div>
  <script>
    const nodes = new vis.DataSet({nodes_json});
    const edges = new vis.DataSet({edges_json});
    const container = document.getElementById("network");
    const data = {{ nodes: nodes, edges: edges }};
    const options = {{
      layout: {{ improvedLayout: true }},
      physics: {{
        enabled: true,
        solver: "forceAtlas2Based",
        forceAtlas2Based: {{
          gravitationalConstant: -60,
          centralGravity: 0.015,
          springLength: 150,
          springConstant: 0.08
        }},
        stabilization: {{ iterations: 250 }}
      }},
      interaction: {{ hover: true, tooltipDelay: 120 }},
      edges: {{
        smooth: {{ type: "dynamic" }},
        width: 1.5
      }},
      nodes: {{
        margin: 10,
        borderWidth: 1,
        shadow: true
      }}
    }};
    new vis.Network(container, data, options);
  </script>
</body>
</html>
"""


def escape_cypher_string(value: Any) -> str:
    text = "" if value is None else str(value)
    return text.replace("\\", "\\\\").replace("'", "\\'")


def make_safe_label(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in value)
    cleaned = cleaned.strip("_")
    return cleaned or "Node"


def graph_to_cypher(graph: dict[str, Any]) -> str:
    """Генерирует Cypher-скрипт для импорта графа в Neo4j."""
    lines: list[str] = [
        "// LEMON graph export",
        "// Запуск в Neo4j Browser: вставьте весь скрипт и выполните.",
        "",
        "CREATE CONSTRAINT lemon_node_id IF NOT EXISTS",
        "FOR (n:LEMON_Node) REQUIRE n.id IS UNIQUE;",
        "",
    ]

    for node in graph.get("nodes", []):
        node_type = make_safe_label(str(node.get("node_type", "Node")))
        props = {
            "id": node.get("id", ""),
            "label": node.get("label", ""),
            "node_type": node.get("node_type", ""),
            "canonical": node.get("canonical", ""),
        }
        attrs = node.get("attributes") or {}
        for key in ("slot", "value", "unit", "fact_type", "section"):
            if key in attrs:
                props[key] = attrs.get(key)

        props_text = ", ".join(
            f"{key}: '{escape_cypher_string(value)}'" for key, value in props.items()
        )
        lines.append(f"MERGE (n:LEMON_Node:{node_type} {{id: '{escape_cypher_string(props['id'])}'}})")
        lines.append(f"SET n += {{{props_text}}};")

    lines.append("")
    for edge in graph.get("edges", []):
        relation = make_safe_label(str(edge.get("relation", "RELATED_TO"))).upper()
        source = escape_cypher_string(edge.get("source", ""))
        target = escape_cypher_string(edge.get("target", ""))
        evidence = escape_cypher_string(edge.get("evidence", ""))
        weight = edge.get("weight", 1.0)
        try:
            weight_float = float(weight)
        except (TypeError, ValueError):
            weight_float = 1.0

        lines.append(
            f"MATCH (a:LEMON_Node {{id: '{source}'}}), (b:LEMON_Node {{id: '{target}'}})"
        )
        lines.append(f"MERGE (a)-[r:{relation}]->(b)")
        lines.append(f"SET r.evidence = '{evidence}', r.weight = {weight_float};")

    return "\n".join(lines) + "\n"


def save_result_outputs(result: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_report.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "graph.json").write_text(
        json.dumps(result.get("graph", {}), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "metric.json").write_text(
        json.dumps(result.get("metric", {}), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "mine1.json").write_text(
        json.dumps(result.get("mine1", {}), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "graph.cypher").write_text(
        graph_to_cypher(result.get("graph", {})),
        encoding="utf-8",
    )


def run_neo4j_import(uri: str, user: str, password: str, cypher: str) -> tuple[bool, str]:
    """Выполняет Cypher в Neo4j. Работает только если установлен пакет neo4j."""
    try:
        from neo4j import GraphDatabase
    except Exception as exc:
        return False, f"Пакет neo4j не установлен: {exc}"

    try:
        driver = GraphDatabase.driver(uri, auth=(user, password))
        with driver.session() as session:
            statements = [part.strip() for part in cypher.split(";") if part.strip()]
            for statement in statements:
                session.run(statement)
        driver.close()
        return True, "Граф успешно отправлен в Neo4j."
    except Exception as exc:
        return False, f"Ошибка подключения или импорта в Neo4j: {exc}"



st.title("🍋 LEMON: демонстрация извлечения графа знаний")
st.caption(
    "Загрузка статьи → извлечение графа знаний → визуализация → расчёт LEMON и MINE-1 → экспорт в JSON/Cypher/Neo4j."
)

with st.sidebar:
    st.header("Настройки")
    input_mode = st.radio(
        "Источник текста",
        ["Загрузить PDF/TXT", "Вставить текст вручную"],
        index=0,
    )
    force_ocr = st.checkbox(
        "Принудительный OCR для PDF",
        value=False,
        help="Включайте только для сканов. Для обычных PDF лучше оставить выключенным.",
    )
    show_raw_json = st.checkbox("Показать JSON результата", value=False)
    graph_height = st.slider("Высота визуализации графа", 420, 850, 620, step=20)

    st.divider()
    st.subheader("Neo4j")
    use_neo4j = st.checkbox("Показывать настройки Neo4j", value=False)
    if use_neo4j:
        neo4j_uri = st.text_input("URI", value="bolt://localhost:7687")
        neo4j_user = st.text_input("Пользователь", value="neo4j")
        neo4j_password = st.text_input("Пароль", type="password")
    else:
        neo4j_uri = "bolt://localhost:7687"
        neo4j_user = "neo4j"
        neo4j_password = ""


uploaded_file = None
manual_text = ""

if input_mode == "Загрузить PDF/TXT":
    uploaded_file = st.file_uploader(
        "Загрузите статью в формате PDF или TXT",
        type=["pdf", "txt"],
    )
else:
    manual_text = st.text_area(
        "Вставьте текст статьи",
        height=280,
        placeholder="Вставьте сюда текст научной статьи или её фрагмент...",
    )


col_a, col_b = st.columns([1, 3])
with col_a:
    run_button = st.button("Построить граф и оценить", type="primary", use_container_width=True)
with col_b:
    st.info("Для демонстрации на защите достаточно загрузить PDF или вставить текст и нажать кнопку.")

if run_button:
    if input_mode == "Загрузить PDF/TXT" and uploaded_file is None:
        st.warning("Сначала загрузите PDF или TXT файл.")
        st.stop()

    if input_mode == "Вставить текст вручную" and not manual_text.strip():
        st.warning("Сначала вставьте текст.")
        st.stop()

    system = get_system()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = PROJECT_ROOT / "outputs" / f"streamlit_demo_{timestamp}"

    with st.spinner("Обрабатываю статью, строю граф и считаю метрики..."):
        try:
            if input_mode == "Загрузить PDF/TXT":
                suffix = Path(uploaded_file.name).suffix.lower()
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = Path(tmp.name)

                if suffix == ".pdf":
                    result = system.evaluate_pdf(tmp_path, out_dir=output_dir, force_ocr=force_ocr)
                else:
                    text = tmp_path.read_text(encoding="utf-8", errors="ignore")
                    result = system.evaluate_text(text=text, title=uploaded_file.name)
                    save_result_outputs(result, output_dir)

                try:
                    tmp_path.unlink(missing_ok=True)
                except Exception:
                    pass
            else:
                result = system.evaluate_text(text=manual_text, title="Текст из интерфейса")
                save_result_outputs(result, output_dir)

        except Exception as exc:
            st.error("Во время обработки произошла ошибка.")
            st.exception(exc)
            st.stop()

    graph = result.get("graph", {})
    metric = result.get("metric", {})
    mine1 = result.get("mine1", {})
    nodes_count, edges_count = graph_stats(graph)
    cypher = graph_to_cypher(graph)

    st.success("Готово: граф построен, метрики рассчитаны.")

    # Метрики
    st.subheader("Результаты оценки")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("LEMON", f"{safe_metric_value(metric, 'completeness'):.3f}")
    m2.metric("Precision", f"{safe_metric_value(metric, 'precision'):.3f}")
    m3.metric("F1", f"{safe_metric_value(metric, 'f1'):.3f}")
    m4.metric("MINE-1", f"{safe_metric_value(mine1, 'score'):.3f}")
    m5.metric("Граф", f"{nodes_count} узл. / {edges_count} реб.")

    # Визуализация
    st.subheader("Визуализация графа")
    if nodes_count == 0:
        st.warning("Граф пустой. Проверьте качество текста или загрузите другой документ.")
    else:
        components.html(graph_to_vis_html(graph, height=graph_height), height=graph_height + 20)

    # Таблицы
    left, right = st.columns(2)
    with left:
        st.subheader("Узлы графа")
        node_rows = [
            {
                "id": node.get("id"),
                "label": node.get("label"),
                "type": node.get("node_type"),
                "canonical": node.get("canonical"),
            }
            for node in graph.get("nodes", [])
        ]
        st.dataframe(node_rows, use_container_width=True, height=320)

    with right:
        st.subheader("Рёбра графа")
        edge_rows = [
            {
                "source": edge.get("source"),
                "relation": edge.get("relation"),
                "target": edge.get("target"),
                "weight": edge.get("weight"),
            }
            for edge in graph.get("edges", [])
        ]
        st.dataframe(edge_rows, use_container_width=True, height=320)

    # MINE-1 детали
    with st.expander("Детали MINE-1"):
        fact_results = mine1.get("fact_results", [])
        if fact_results:
            st.dataframe(
                [
                    {
                        "score": item.get("score"),
                        "fact": item.get("fact"),
                        "evidence": compact_label(item.get("evidence"), max_len=120),
                    }
                    for item in fact_results
                ],
                use_container_width=True,
                height=320,
            )
        else:
            st.write(mine1)

    # Экспорт
    st.subheader("Экспорт результатов")
    d1, d2, d3, d4 = st.columns(4)
    with d1:
        st.download_button(
            "Скачать graph.json",
            data=json.dumps(graph, ensure_ascii=False, indent=2),
            file_name="graph.json",
            mime="application/json",
            use_container_width=True,
        )
    with d2:
        st.download_button(
            "Скачать metric.json",
            data=json.dumps(metric, ensure_ascii=False, indent=2),
            file_name="metric.json",
            mime="application/json",
            use_container_width=True,
        )
    with d3:
        st.download_button(
            "Скачать mine1.json",
            data=json.dumps(mine1, ensure_ascii=False, indent=2),
            file_name="mine1.json",
            mime="application/json",
            use_container_width=True,
        )
    with d4:
        st.download_button(
            "Скачать Cypher для Neo4j",
            data=cypher,
            file_name="graph.cypher",
            mime="text/plain",
            use_container_width=True,
        )

    st.caption(f"Файлы также сохранены в папке: {output_dir}")

    # Neo4j
    if use_neo4j:
        st.subheader("Отправка графа в Neo4j")
        st.write("Локальный Neo4j Browser обычно доступен по адресу: http://localhost:7474/browser/")
        if st.button("Импортировать граф в Neo4j", use_container_width=False):
            if not neo4j_password:
                st.warning("Введите пароль Neo4j в боковой панели.")
            else:
                ok, message = run_neo4j_import(neo4j_uri, neo4j_user, neo4j_password, cypher)
                if ok:
                    st.success(message)
                    st.link_button("Открыть Neo4j Browser", "http://localhost:7474/browser/")
                else:
                    st.error(message)

    if show_raw_json:
        with st.expander("Полный JSON результата"):
            st.json(result)

else:
    st.subheader("Как пользоваться")
    st.markdown(
        """
1. Загрузите PDF/TXT статью или вставьте текст вручную.
2. Нажмите **«Построить граф и оценить»**.
3. Система построит граф знаний, рассчитает **LEMON**, **Precision**, **F1** и **MINE-1**.
4. Результаты можно скачать в JSON или экспортировать в Neo4j через Cypher.
        """
    )


