"""
VulnMind Streamlit UI.
Components: query input, source attribution panel, knowledge graph visualizer, risk timeline.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import httpx
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

API_BASE = os.getenv("VULNMIND_API_URL", "http://localhost:8000")


def _show_sample_graph(center_node: str) -> None:
    """Render a sample/demo knowledge graph with Plotly."""
    import math

    nodes = [
        {"id": center_node, "type": "cve", "color": "#e74c3c"},
        {"id": "CWE-502", "type": "cwe", "color": "#f39c12"},
        {"id": "CWE-917", "type": "cwe", "color": "#f39c12"},
        {"id": "T1190", "type": "technique", "color": "#3498db"},
        {"id": "T1059", "type": "technique", "color": "#3498db"},
        {"id": "vendor:apache", "type": "vendor", "color": "#2ecc71"},
    ]
    edges = [
        (center_node, "CWE-502"),
        (center_node, "CWE-917"),
        (center_node, "T1190"),
        (center_node, "T1059"),
        (center_node, "vendor:apache"),
    ]

    n = len(nodes)
    pos: dict[str, tuple[float, float]] = {}
    pos[nodes[0]["id"]] = (0.0, 0.0)
    for i, node in enumerate(nodes[1:], 1):
        angle = 2 * math.pi * i / (n - 1)
        pos[node["id"]] = (math.cos(angle) * 2, math.sin(angle) * 2)

    edge_x, edge_y = [], []
    for src, tgt in edges:
        x0, y0 = pos[src]
        x1, y1 = pos[tgt]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    node_x = [pos[n["id"]][0] for n in nodes]
    node_y = [pos[n["id"]][1] for n in nodes]
    node_colors = [n["color"] for n in nodes]
    node_labels = [n["id"] for n in nodes]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        mode="lines",
        line=dict(width=1, color="#888"),
        hoverinfo="none",
    ))
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        marker=dict(size=20, color=node_colors),
        text=node_labels,
        textposition="bottom center",
        hoverinfo="text",
    ))
    fig.update_layout(
        showlegend=False,
        height=400,
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font=dict(color="white"),
    )
    st.plotly_chart(fig, use_container_width=True)

    legend_cols = st.columns(4)
    legend_cols[0].markdown("🔴 CVE")
    legend_cols[1].markdown("🟡 CWE")
    legend_cols[2].markdown("🔵 ATT&CK Technique")
    legend_cols[3].markdown("🟢 Vendor")


st.set_page_config(
    page_title="VulnMind — Threat Intelligence RAG",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🛡️ VulnMind")
    st.caption("Agentic Threat Intelligence RAG")
    st.divider()

    st.subheader("Query Filters")
    min_cvss = st.slider("Minimum CVSS Score", 0.0, 10.0, 0.0, 0.1)
    vendor_filter = st.text_input("Vendor Filter", placeholder="e.g. microsoft")
    year_start = st.number_input("Year From", 2000, 2026, 2020)
    year_end = st.number_input("Year To", 2000, 2026, 2026)

    st.divider()
    st.subheader("System Status")
    if st.button("Check Health"):
        with st.spinner("Checking..."):
            try:
                r = httpx.get(f"{API_BASE}/health", timeout=5.0)
                h = r.json()
                st.metric("CVE Chunks", h.get("chroma_count", "N/A"))
                st.metric("Graph Nodes", h.get("graph_nodes", "N/A"))
                st.metric("Graph Edges", h.get("graph_edges", "N/A"))
                st.success("Redis: " + ("Connected" if h.get("redis_connected") else "Disconnected"))
            except Exception as e:
                st.error(f"API unreachable: {e}")

    st.divider()
    if st.button("Run Benchmark"):
        st.info("Benchmark requires API to be running. Check logs.")


# ── Main content ──────────────────────────────────────────────────────────────
st.title("VulnMind Threat Intelligence")
st.caption("Powered by hybrid RAG + knowledge graph + ReAct agent")

tab_query, tab_graph, tab_risk, tab_benchmark, tab_actors, tab_surface = st.tabs(
    ["Query", "Knowledge Graph", "Risk Timeline", "Benchmark", "Threat Actors", "Attack Surface"]
)

with tab_query:
    question = st.text_area(
        "Ask a threat intelligence question",
        placeholder="e.g. What ATT&CK techniques does Log4Shell enable and how should I prioritize patching?",
        height=100,
    )

    col1, col2 = st.columns([1, 4])
    with col1:
        submit = st.button("Analyze", type="primary", use_container_width=True)

    if submit and question:
        filters: dict = {}
        if min_cvss > 0:
            filters["min_cvss"] = min_cvss
        if vendor_filter:
            filters["vendor"] = vendor_filter
        if year_start or year_end:
            filters["year_range"] = [int(year_start), int(year_end)]

        payload = {
            "question": question,
            "filters": filters or None,
            "stream": True,
        }

        st.divider()
        answer_container = st.empty()
        sources_container = st.container()
        answer_text = ""

        with st.spinner("Querying VulnMind..."):
            try:
                with httpx.Client(timeout=120.0) as client:
                    with client.stream("POST", f"{API_BASE}/query", json=payload) as resp:
                        resp.raise_for_status()
                        sources_data = []

                        for line in resp.iter_lines():
                            if not line.startswith("data: "):
                                continue
                            event = json.loads(line[6:])
                            etype = event.get("type")

                            if etype == "sources":
                                sources_data = event.get("data", [])
                            elif etype == "token":
                                answer_text += event.get("data", "")
                                answer_container.markdown(answer_text + "▌")
                            elif etype == "done":
                                answer_container.markdown(answer_text)
                                break

                # Source attribution panel
                if sources_data:
                    with sources_container:
                        st.subheader("Source Attribution")
                        for src in sources_data:
                            cve_id = src.get("cve_id", "")
                            cvss = src.get("cvss_score")
                            vm_score = src.get("vulnmind_score")
                            severity = src.get("severity", "Unknown")

                            with st.expander(f"{cve_id} — CVSS: {cvss} | VulnMind: {vm_score}/100"):
                                st.write(src.get("text_preview", ""))
                                col_a, col_b, col_c = st.columns(3)
                                col_a.metric("CVSS", cvss or "N/A")
                                col_b.metric("VulnMind Score", f"{vm_score}/100" if vm_score else "N/A")
                                col_c.metric("Severity", severity)

            except Exception as e:
                st.error(f"Query failed: {e}")


with tab_graph:
    st.subheader("Knowledge Graph Viewer")
    node_query = st.text_input("Enter CVE ID or ATT&CK Technique ID", placeholder="CVE-2021-44228")

    if st.button("Lookup in Graph") and node_query:
        with st.spinner("Traversing graph..."):
            try:
                r = httpx.post(
                    f"{API_BASE}/query",
                    json={"question": f"graph_lookup {node_query}", "stream": False},
                    timeout=30.0,
                )
                # Visualize with plotly
                st.info("Graph visualization requires the API's /graph endpoint. Below is a placeholder network diagram.")
                _show_sample_graph(node_query)
            except Exception as e:
                st.error(f"Graph lookup failed: {e}")
    else:
        _show_sample_graph("CVE-2021-44228")


with tab_risk:
    st.subheader("VulnMind Risk Timeline")
    st.caption("Composite risk scores over time for tracked CVEs")

    # Sample risk timeline data
    sample_data = {
        "CVE": ["CVE-2021-44228"] * 6 + ["CVE-2021-26855"] * 6 + ["CVE-2022-22965"] * 6,
        "Month": ["Jan", "Feb", "Mar", "Apr", "May", "Jun"] * 3,
        "VulnMind Score": [
            95, 94, 92, 90, 88, 85,
            88, 87, 85, 82, 80, 78,
            70, 72, 74, 73, 71, 68,
        ],
        "EPSS": [
            0.97, 0.96, 0.95, 0.93, 0.91, 0.89,
            0.88, 0.86, 0.84, 0.81, 0.79, 0.76,
            0.45, 0.48, 0.51, 0.49, 0.47, 0.44,
        ],
    }
    df = pd.DataFrame(sample_data)

    col_left, col_right = st.columns(2)

    with col_left:
        fig1 = px.line(
            df, x="Month", y="VulnMind Score", color="CVE",
            title="VulnMind Risk Score Over Time",
            template="plotly_dark",
        )
        fig1.update_layout(yaxis_range=[0, 100])
        st.plotly_chart(fig1, use_container_width=True)

    with col_right:
        fig2 = px.line(
            df, x="Month", y="EPSS", color="CVE",
            title="EPSS Exploit Probability Over Time",
            template="plotly_dark",
        )
        fig2.update_layout(yaxis_range=[0, 1])
        st.plotly_chart(fig2, use_container_width=True)

    # Risk score formula explanation
    st.subheader("VulnMind Risk Score Formula")
    st.latex(r"\text{VulnMind Score} = 100 \times (0.4 \times \text{CVSS}_{norm} + 0.4 \times \text{EPSS} + 0.2 \times \text{GraphCentrality}_{pctile})")
    st.markdown("""
    | Component | Weight | Description |
    |-----------|--------|-------------|
    | CVSS Normalized | 40% | CVSS v3 base score / 10 |
    | EPSS Score | 40% | Exploit prediction probability (0-1) |
    | Graph Centrality Percentile | 20% | PageRank percentile among all CVEs |
    | KEV Boost | ×1.15 | Applied if CVE is in CISA KEV catalog |
    """)

    st.divider()
    st.subheader("Trending Threats")
    st.caption("CVEs with the largest recent EPSS score increase")

    if st.button("Load Trending CVEs"):
        with st.spinner("Fetching trending data..."):
            try:
                r = httpx.get(f"{API_BASE}/trending", timeout=30.0)
                r.raise_for_status()
                data = r.json()
                trending_list = data.get("trending", [])

                if trending_list:
                    trend_df = pd.DataFrame(trending_list)

                    def _delta_color(val: float) -> str:
                        if val >= 0.3:
                            return "background-color: #8B0000"
                        if val >= 0.1:
                            return "background-color: #B8860B"
                        return "background-color: #006400"

                    display_cols = [c for c in ["cve_id", "current_epss", "previous_epss", "delta", "vulnmind_score", "severity_label"] if c in trend_df.columns]
                    st.dataframe(
                        trend_df[display_cols].style.applymap(
                            _delta_color, subset=["delta"]
                        ) if "delta" in trend_df.columns else trend_df[display_cols],
                        use_container_width=True,
                    )

                    # EPSS delta bar chart
                    if "delta" in trend_df.columns and "cve_id" in trend_df.columns:
                        fig_trend = px.bar(
                            trend_df.head(10),
                            x="cve_id", y="delta",
                            color="delta",
                            color_continuous_scale="RdYlGn_r",
                            title="EPSS Delta (7-day) — Top Trending CVEs",
                            template="plotly_dark",
                        )
                        fig_trend.update_layout(xaxis_tickangle=-45)
                        st.plotly_chart(fig_trend, use_container_width=True)
                else:
                    st.info("No trending data yet. Run `python -m monitoring.drift_detector` to populate history.")
            except Exception as e:
                st.error(f"Trending fetch failed: {e}")


with tab_benchmark:
    st.subheader("RAGAS Benchmark Scorecard")

    results_path = Path("./eval/benchmark_results.json")
    if results_path.exists():
        results = json.loads(results_path.read_text())
        scores = results.get("scores", {})

        cols = st.columns(5)
        for col, (metric, score) in zip(cols, scores.items()):
            grade = "A" if score >= 0.85 else "B" if score >= 0.75 else "C" if score >= 0.65 else "F"
            col.metric(metric.replace("_", " ").title(), f"{score:.3f}", f"Grade: {grade}")

        # Radar chart
        categories = list(scores.keys())
        values = list(scores.values())
        fig_radar = go.Figure(data=go.Scatterpolar(
            r=values + [values[0]],
            theta=categories + [categories[0]],
            fill="toself",
            line_color="#3498db",
        ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=False,
            template="plotly_dark",
            title="RAGAS Metric Radar",
        )
        st.plotly_chart(fig_radar, use_container_width=True)
    else:
        st.info("No benchmark results yet. Run `python -m eval.benchmark` to generate.")
        st.markdown("""
        | Metric | Expected Score | Grade |
        |--------|---------------|-------|
        | Faithfulness | > 0.85 | A |
        | Answer Relevancy | > 0.80 | A/B |
        | Context Precision | > 0.75 | B |
        | Context Recall | > 0.70 | B |
        | Composite | > 0.78 | B |
        """)


with tab_actors:
    st.subheader("Threat Actor Profiling")
    st.caption("ATT&CK intrusion-set groups and their associated CVEs")

    actor_input = st.text_input(
        "Enter Group ID", placeholder="G0016 (APT28), G0007 (APT28), G0032..."
    )

    if st.button("Lookup Actor", key="actor_lookup") and actor_input:
        with st.spinner(f"Loading profile for {actor_input}..."):
            try:
                r = httpx.get(f"{API_BASE}/actor/{actor_input.strip()}", timeout=30.0)
                r.raise_for_status()
                profile = r.json()

                col_name, col_aliases = st.columns([1, 2])
                col_name.metric("Group", profile.get("name", actor_input))
                aliases = ", ".join(profile.get("aliases", [])) or "None"
                col_aliases.markdown(f"**Aliases:** {aliases}")

                desc = profile.get("description", "")
                if desc:
                    st.markdown(f"**Description:** {desc[:500]}...")

                st.divider()

                techniques = profile.get("techniques_used", [])
                if techniques:
                    st.subheader(f"Techniques Used ({len(techniques)})")
                    tech_df = pd.DataFrame(techniques)
                    if "tactics" in tech_df.columns:
                        tech_df["tactics"] = tech_df["tactics"].apply(
                            lambda x: x.replace(",", ", ") if isinstance(x, str) else x
                        )
                    st.dataframe(tech_df, use_container_width=True)

                cves = profile.get("associated_cves", [])
                if cves:
                    st.subheader(f"Associated CVEs ({len(cves)})")
                    cve_df = pd.DataFrame(cves)

                    def _severity_badge(sev: str) -> str:
                        colors = {
                            "CRITICAL": "🔴", "HIGH": "🟠",
                            "MEDIUM": "🟡", "LOW": "🟢",
                        }
                        return colors.get(str(sev).upper(), "⚪") + f" {sev}"

                    if "severity" in cve_df.columns:
                        cve_df["severity"] = cve_df["severity"].apply(_severity_badge)

                    display_cols = [c for c in ["cve_id", "cvss_score", "epss", "vulnmind_score", "severity", "via_technique"] if c in cve_df.columns]
                    st.dataframe(cve_df[display_cols], use_container_width=True)

                    # Bar chart of top CVEs by VulnMind score
                    if "vulnmind_score" in cve_df.columns and cve_df["vulnmind_score"].notna().any():
                        score_col = "vulnmind_score"
                    elif "cvss_score" in cve_df.columns:
                        score_col = "cvss_score"
                    else:
                        score_col = None

                    if score_col:
                        chart_df = pd.DataFrame(cves).head(10)
                        fig_actor = px.bar(
                            chart_df,
                            x="cve_id", y=score_col,
                            color=score_col,
                            color_continuous_scale="RdYlGn_r",
                            title=f"Top 10 CVEs for {profile.get('name', actor_input)} by {score_col}",
                            template="plotly_dark",
                        )
                        fig_actor.update_layout(xaxis_tickangle=-45, yaxis_range=[0, 100])
                        st.plotly_chart(fig_actor, use_container_width=True)

            except httpx.HTTPStatusError as e:
                if e.response.status_code == 404:
                    st.warning(f"Actor {actor_input} not found in knowledge graph. Run ingestion with threat actor data first.")
                else:
                    st.error(f"API error: {e}")
            except Exception as e:
                st.error(f"Lookup failed: {e}")


with tab_surface:
    st.subheader("Attack Surface Analyzer")
    st.caption("Enter your software stack to find relevant CVEs and prioritize patching")

    software_input = st.text_area(
        "Software Stack (one per line)",
        placeholder="Apache 2.4\nOpenSSL 3.0\nLog4j 2.14\nSpring Boot 2.7",
        height=150,
    )
    include_graph = st.checkbox("Include graph context (ATT&CK techniques)", value=True)

    if st.button("Analyze Attack Surface", type="primary") and software_input:
        software_list = [s.strip() for s in software_input.strip().splitlines() if s.strip()]
        if not software_list:
            st.warning("Please enter at least one software component.")
        else:
            with st.spinner(f"Analyzing {len(software_list)} software components..."):
                try:
                    payload = {"software": software_list, "include_graph_context": include_graph}
                    r = httpx.post(f"{API_BASE}/analyze-surface", json=payload, timeout=120.0)
                    r.raise_for_status()
                    result = r.json()

                    # Summary metrics
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Total CVEs", result.get("total_cves_found", 0))
                    m2.metric("Critical", result.get("critical_count", 0), delta="⚠️ Immediate action")
                    m3.metric("High", result.get("high_count", 0))
                    m4.metric("Software Analyzed", len(software_list))

                    st.divider()

                    cves = result.get("cves", [])
                    if cves:
                        cve_df = pd.DataFrame(cves)
                        display_cols = [c for c in ["cve_id", "cvss_score", "epss", "vulnmind_score", "severity_label", "kev_confirmed", "software_match"] if c in cve_df.columns]
                        cve_df_display = cve_df[display_cols].copy() if display_cols else cve_df

                        def _row_color(row: pd.Series) -> list[str]:
                            score = row.get("vulnmind_score", 0) or 0
                            if score >= 80:
                                return ["background-color: #8B000033"] * len(row)
                            if score >= 60:
                                return ["background-color: #FF8C0033"] * len(row)
                            return [""] * len(row)

                        st.subheader("All CVEs (sorted by VulnMind score)")
                        st.dataframe(
                            cve_df_display.style.apply(_row_color, axis=1),
                            use_container_width=True,
                        )

                        # Severity pie chart
                        if "severity_label" in cve_df.columns:
                            sev_counts = cve_df["severity_label"].value_counts().reset_index()
                            sev_counts.columns = ["Severity", "Count"]
                            color_map = {
                                "Critical": "#e74c3c", "High": "#e67e22",
                                "Medium": "#f1c40f", "Low": "#2ecc71",
                                "Informational": "#95a5a6",
                            }
                            fig_pie = px.pie(
                                sev_counts, names="Severity", values="Count",
                                color="Severity",
                                color_discrete_map=color_map,
                                title="CVE Severity Distribution",
                                template="plotly_dark",
                            )
                            st.plotly_chart(fig_pie, use_container_width=True)

                    # Immediate action expander
                    immediate = result.get("immediate_action", [])
                    if immediate:
                        with st.expander("Immediate Action Required", expanded=True):
                            for i, item in enumerate(immediate, 1):
                                cve_id = item.get("cve_id", "")
                                score = item.get("vulnmind_score", 0)
                                kev = " 🚨 KEV" if item.get("kev_confirmed") else ""
                                st.markdown(f"**{i}. {cve_id}** — Score: {score}/100{kev}")
                                st.info(item.get("patch_recommendation", ""))

                    # Top techniques
                    top_techs = result.get("top_techniques", [])
                    if top_techs:
                        st.subheader("Top ATT&CK Techniques Across Surface")
                        st.markdown(" · ".join(f"`{t}`" for t in top_techs))

                except httpx.HTTPStatusError as e:
                    st.error(f"API error {e.response.status_code}: {e.response.text[:300]}")
                except Exception as e:
                    st.error(f"Analysis failed: {e}")
