import ast
import math
import os

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from matplotlib import cm, colors
from matplotlib.lines import Line2D


PALETTE = [
    "#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2",
    "#B279A2", "#FF9DA6", "#9D755D", "#BAB0AC", "#8CD17D",
]


def load_rules(filepath="output/association_rules.csv"):
    rules = pd.read_csv(filepath)
    print(f"Loaded {len(rules)} rules")
    return rules


def parse_itemset(value):
    """Parse an exported itemset string into product names."""
    if pd.isna(value):
        return tuple()
    text = str(value).strip()
    if not text:
        return tuple()
    if " + " in text:
        return tuple(part.strip() for part in text.split(" + ") if part.strip())
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, (set, frozenset, list, tuple)):
            return tuple(sorted(str(item) for item in parsed))
    except (ValueError, SyntaxError):
        pass
    return (text,)


def iter_rule_items(rules):
    """Yield antecedent/consequent item tuples from current or legacy rule CSVs."""
    for _, row in rules.iterrows():
        if {"antecedents_str", "consequents_str"}.issubset(rules.columns):
            yield parse_itemset(row["antecedents_str"]), parse_itemset(row["consequents_str"]), row
            continue

        rule_text = str(row.get("rule", ""))
        if "->" not in rule_text:
            continue
        left, right = rule_text.split("->", 1)
        yield parse_itemset(left.strip()), parse_itemset(right.strip()), row


def _simple_rule_frame(rules):
    rows = []
    for antecedents, consequents, row in iter_rule_items(rules):
        if len(antecedents) != 1 or len(consequents) != 1:
            continue
        rows.append(
            {
                "antecedent": antecedents[0],
                "consequent": consequents[0],
                "rule": f"{antecedents[0]} -> {consequents[0]}",
                "support": float(row.get("support", 0)),
                "confidence": float(row.get("confidence", 0)),
                "lift": float(row.get("lift", 0)),
            }
        )

    simple = pd.DataFrame(rows)
    if simple.empty:
        return simple

    simple["visual_score"] = (
        simple["lift"] * simple["confidence"] * simple["support"].clip(lower=0.001) ** 0.35
    )
    return simple.sort_values(
        ["visual_score", "lift", "confidence", "support"],
        ascending=False,
    ).reset_index(drop=True)


def _item_support(df_transactions):
    total_transactions = df_transactions["TransactionID"].nunique()
    if total_transactions == 0:
        return pd.Series(dtype=float)
    return (
        df_transactions.groupby("Item")["TransactionID"].nunique()
        / total_transactions
    ).sort_values(ascending=False)


def build_network(rules, max_rules=35, min_lift=1.2, simple_only=True, **_):
    """
    Build a stakeholder-friendly association graph.

    Reciprocal rules are merged into one product-pair edge, keeping the strongest
    direction as the recommendation label.
    """
    simple = _simple_rule_frame(rules)
    if simple.empty:
        print("No one-to-one rules available for the network.")
        return nx.Graph()

    simple = simple[simple["lift"] >= min_lift].head(max_rules)
    G = nx.Graph()

    for _, row in simple.iterrows():
        a = row["antecedent"]
        b = row["consequent"]
        edge_key = tuple(sorted((a, b)))

        if G.has_edge(*edge_key):
            current = G.edges[edge_key]["visual_score"]
            if row["visual_score"] <= current:
                continue

        G.add_edge(
            edge_key[0],
            edge_key[1],
            lift=row["lift"],
            confidence=row["confidence"],
            support=row["support"],
            visual_score=row["visual_score"],
            recommendation=row["rule"],
        )

    print(f"Building market basket network from {G.number_of_edges()} product-pair edges")
    return G


def _community_colors(G):
    if G.number_of_nodes() == 0:
        return {}

    if G.number_of_edges() == 0:
        communities = [{node} for node in G.nodes()]
    else:
        communities = list(
            nx.algorithms.community.greedy_modularity_communities(
                G,
                weight="visual_score",
            )
        )

    color_by_node = {}
    for idx, community in enumerate(communities):
        color = PALETTE[idx % len(PALETTE)]
        for node in community:
            color_by_node[node] = color
    return color_by_node


def _communities(G):
    if G.number_of_nodes() == 0:
        return []
    if G.number_of_edges() == 0:
        return [{node} for node in G.nodes()]
    return list(
        nx.algorithms.community.greedy_modularity_communities(
            G,
            weight="visual_score",
        )
    )


def _community_layout(G):
    communities = sorted(_communities(G), key=len, reverse=True)
    if not communities:
        return {}

    if len(communities) == 1:
        return nx.spring_layout(G, seed=7, weight="visual_score", k=1.8, iterations=250, scale=4)

    pos = {}
    center_radius = 5.0
    for community_index, community in enumerate(communities):
        center_angle = 2 * math.pi * community_index / len(communities)
        center = (
            center_radius * math.cos(center_angle),
            center_radius * math.sin(center_angle),
        )
        nodes = sorted(community)
        local_radius = max(1.45, 0.42 * len(nodes))
        if len(nodes) == 1:
            pos[nodes[0]] = center
            continue
        for node_index, node in enumerate(nodes):
            node_angle = 2 * math.pi * node_index / len(nodes)
            pos[node] = (
                center[0] + local_radius * math.cos(node_angle),
                center[1] + local_radius * math.sin(node_angle),
            )
    return pos


def _edge_widths(G):
    lifts = [data["lift"] for _, _, data in G.edges(data=True)]
    if not lifts:
        return []
    min_lift = min(lifts)
    max_lift = max(lifts)
    spread = max(max_lift - min_lift, 0.001)
    return [1.8 + 5.5 * ((lift - min_lift) / spread) for lift in lifts]


def plot_network(
    G,
    df_transactions,
    filepath="output/network_graph.png",
    title="Market Basket Recommendation Network",
):
    if G.number_of_edges() == 0:
        print(f"No edges to draw for {filepath}")
        return

    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    supports = _item_support(df_transactions)
    node_colors = _community_colors(G)
    weights = [data["visual_score"] for _, _, data in G.edges(data=True)]
    pos = _community_layout(G)

    fig, ax = plt.subplots(figsize=(16, 10), facecolor="white")
    ax.set_title(title, loc="left", fontsize=20, fontweight="bold", pad=18)
    ax.text(
        0.0,
        1.01,
        "Node size = product support | Edge width = lift | Edge color = confidence | Colors = detected product clusters",
        transform=ax.transAxes,
        fontsize=11,
        color="#555555",
    )

    edge_confidences = [data["confidence"] for _, _, data in G.edges(data=True)]
    norm = colors.Normalize(vmin=min(edge_confidences), vmax=max(edge_confidences))
    edge_colors = [cm.viridis(norm(conf)) for conf in edge_confidences]

    nx.draw_networkx_edges(
        G,
        pos,
        ax=ax,
        width=_edge_widths(G),
        edge_color=edge_colors,
        alpha=0.62,
    )

    node_sizes = [
        500 + 6000 * supports.get(node, 0)
        for node in G.nodes()
    ]
    nx.draw_networkx_nodes(
        G,
        pos,
        ax=ax,
        node_color=[node_colors.get(node, "#CCCCCC") for node in G.nodes()],
        node_size=node_sizes,
        edgecolors="white",
        linewidths=2.3,
        alpha=0.98,
    )

    node_labels = {
        node: f"{node}\n{sup:.0%}"
        for node in G.nodes()
        for sup in [supports.get(node, 0)]
    }
    nx.draw_networkx_labels(
        G,
        pos,
        labels=node_labels,
        ax=ax,
        font_size=9,
        font_weight="bold",
        font_color="#222222",
        bbox={"boxstyle": "round,pad=0.28", "fc": "white", "ec": "#DDDDDD", "alpha": 0.88},
    )

    scalar_map = cm.ScalarMappable(norm=norm, cmap=cm.viridis)
    scalar_map.set_array([])
    cbar = fig.colorbar(scalar_map, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Confidence", rotation=90)

    lift_legend = [
        Line2D([0], [0], color="#666666", lw=2.0, label="Lower lift"),
        Line2D([0], [0], color="#666666", lw=6.0, label="Higher lift"),
    ]
    ax.legend(handles=lift_legend, loc="lower left", frameon=True, framealpha=0.92)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(filepath, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {filepath}")


def plot_top_rules(rules, top_n=15, filepath="output/top_rules_chart.png"):
    """Create a compact executive chart for the strongest recommendation rules."""
    simple = _simple_rule_frame(rules)
    if simple.empty:
        print("No simple one-item rules found for the top-rules chart.")
        return

    top = simple.head(top_n).sort_values("lift")
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 9), facecolor="white")
    norm = colors.Normalize(vmin=top["confidence"].min(), vmax=top["confidence"].max())
    cmap = plt.get_cmap("YlGnBu")
    bar_colors = [cmap(0.35 + 0.55 * norm(value)) for value in top["confidence"]]
    ax.barh(top["rule"], top["lift"], color=bar_colors, edgecolor="white", height=0.72)

    for idx, (_, row) in enumerate(top.iterrows()):
        ax.text(
            row["lift"] + 0.03,
            idx,
            f"C {row['confidence']:.0%} | S {row['support']:.1%}",
            va="center",
            fontsize=9,
            color="#333333",
        )

    ax.axvline(1.0, color="#D62728", linestyle="--", linewidth=1.4, alpha=0.8)
    ax.text(1.02, -0.8, "Lift = 1.0", color="#D62728", fontsize=9)
    ax.set_title("Top Product Recommendation Rules", loc="left", fontsize=18, fontweight="bold")
    ax.set_xlabel("Lift")
    ax.set_ylabel("")
    ax.grid(axis="x", color="#EAEAEA", linewidth=0.8)
    ax.spines[["top", "right", "left"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(filepath, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {filepath}")


def plot_rule_quality_scatter(rules, top_n=50, filepath="output/rule_quality_scatter.png"):
    """Plot support, confidence, and lift together for threshold tuning."""
    simple = _simple_rule_frame(rules)
    if simple.empty:
        print("No simple one-item rules found for the rule-quality scatterplot.")
        return

    sample = simple.head(top_n).copy()
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)

    fig, ax = plt.subplots(figsize=(13, 8), facecolor="white")
    norm = colors.Normalize(vmin=sample["lift"].min(), vmax=sample["lift"].max())
    sizes = 260 + 900 * (sample["lift"] - sample["lift"].min()) / max(sample["lift"].max() - sample["lift"].min(), 0.001)

    points = ax.scatter(
        sample["support"],
        sample["confidence"],
        s=sizes,
        c=sample["lift"],
        cmap="plasma",
        norm=norm,
        alpha=0.78,
        edgecolors="white",
        linewidths=1.0,
    )

    offsets = [(10, 10), (12, -18), (-92, 14), (-88, -22), (14, 24), (-96, 30)]
    for offset, (_, row) in zip(offsets, sample.head(6).iterrows()):
        ax.annotate(
            row["rule"],
            (row["support"], row["confidence"]),
            xytext=offset,
            textcoords="offset points",
            fontsize=8,
            color="#222222",
            arrowprops={"arrowstyle": "-", "color": "#999999", "lw": 0.8},
            bbox={"boxstyle": "round,pad=0.18", "fc": "white", "ec": "#DDDDDD", "alpha": 0.86},
        )

    cbar = fig.colorbar(points, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Lift")
    ax.set_title("Rule Quality Map", loc="left", fontsize=18, fontweight="bold")
    ax.set_xlabel("Support: share of all baskets containing both products")
    ax.set_ylabel("Confidence: likelihood of consequent after antecedent")
    ax.set_ylim(max(0, sample["confidence"].min() - 0.04), min(1.06, sample["confidence"].max() + 0.04))
    ax.margins(x=0.05)
    ax.grid(color="#EAEAEA", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(filepath, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {filepath}")


def plot_cross_category_network(rules, df_transactions, min_lift=1.2, filepath="output/cross_category_network.png"):
    """
    Shows only cross-category associations — products from
    different categories that are bought together.
    """
    PRODUCT_CATEGORY = {
        "Bread": "Breakfast", "Butter": "Breakfast", "Eggs": "Breakfast",
        "Milk": "Breakfast", "Jam": "Breakfast", "Coffee": "Breakfast",
        "Chips": "Snacks", "Soda": "Snacks", "Cookies": "Snacks",
        "Chocolate": "Snacks", "Popcorn": "Snacks",
        "Detergent": "Cleaning", "Bleach": "Cleaning", "Sponge": "Cleaning",
        "Mop": "Cleaning", "Disinfectant": "Cleaning",
        "Shampoo": "Personal", "Conditioner": "Personal", "Soap": "Personal",
        "Toothpaste": "Personal", "Razor": "Personal",
    }

    # Filter rules to cross-category only before building graph
    simple = _simple_rule_frame(rules)
    if simple.empty:
        print("No rules available for cross-category network.")
        return

    simple = simple[simple["lift"] >= min_lift].copy()

    # Keep only rules where antecedent and consequent are in different categories
    cross_mask = simple.apply(
        lambda row: (
            PRODUCT_CATEGORY.get(row["antecedent"]) is not None
            and PRODUCT_CATEGORY.get(row["consequent"]) is not None
            and PRODUCT_CATEGORY.get(row["antecedent"]) != PRODUCT_CATEGORY.get(row["consequent"])
        ),
        axis=1
    )
    cross_rules = simple[cross_mask]

    if cross_rules.empty:
        print("No cross-category rules found at this lift threshold.")
        print("Try lowering min_lift or min_support when running the pipeline.")
        return

    print(f"Cross-category rules found: {len(cross_rules)}")
    for _, row in cross_rules.iterrows():
        a, b = row["antecedent"], row["consequent"]
        print(f"  {a} ({PRODUCT_CATEGORY.get(a)}) -> {b} ({PRODUCT_CATEGORY.get(b)}) | Lift: {row['lift']:.3f}")

    # Build a graph from cross-category rules only
    G = nx.Graph()
    for _, row in cross_rules.iterrows():
        a = row["antecedent"]
        b = row["consequent"]
        edge_key = tuple(sorted((a, b)))
        if not G.has_edge(*edge_key):
            G.add_edge(
                edge_key[0], edge_key[1],
                lift=row["lift"],
                confidence=row["confidence"],
                support=row["support"],
                visual_score=row["visual_score"],
                recommendation=row["rule"],
            )

    print(f"Building cross-category network from {G.number_of_edges()} edges")
    plot_network(
        G,
        df_transactions,
        filepath=filepath,
        title="Cross-Category Product Associations",
    )