# Market Basket Analysis for Cross-Selling and Upselling

An end-to-end retail analytics pipeline that uncovers hidden product 
associations from transaction data using Association Rule Mining.
Built as a Data Science Internship Project.

---

## Project Overview

Analyses 10,000 synthetic retail transactions across 21 products in 
4 categories (Breakfast, Snacks, Cleaning, Personal Care) to discover 
which products are frequently bought together — and translates those 
findings into concrete business recommendations.

---

## Pipeline

| Module | File | Purpose |
|---|---|---|
| transactions.csv |
| 'data_preprocessing.py' | Long Format → One-Hot Encoded basket matrix |
| association_rules.py | Apriori + FP-Growth, Support/Confidence/Lift |
| visualize.py | Network graphs + bar chart + scatter plot |
| output/association_rules.csv + 4 visualizations |

---

## Key Results

- **96 association rules** identified (Confidence ≥ 60%, Lift ≥ 1.2)
- **Strongest rule:** Soda → Chips (Lift 4.03, Confidence 100%)
- **Breakfast cluster:** Bread as anchor product (Lift 3.0–4.0)
- **Cross-category finding:** Cleaning ↔ Personal Care (Lift 1.8–1.9)
- FP-Growth and Apriori produced identical results, confirming consistency

---

## Output Files

| File | Description |
|---|---|
| `output/association_rules.csv` | 96 rules sorted by Lift |
| `output/network_graph.png` | Within-category product clusters |
| `output/cross_category_network.png` | Cross-category associations |
| `output/top_rules_chart.png` | Top 15 rules by Lift |
| `output/rule_quality_scatter.png` | Support vs Confidence map |

---

## Installation

```bash
pip install pandas mlxtend networkx matplotlib
```

---

## Usage

```bash
# Step 1: Generate synthetic transaction data
python create_data.py

# Step 2: Run full pipeline (data prep + rules + visualizations)
python main.py

# Optional: custom parameters
python main.py --min-support 0.10 --min-confidence 0.60 --min-lift 1.2
```

---

## Business Strategy Proposal

See `Market Basket Analysis.pdf` for six concrete retail recommendations
derived from the association rules — covering store layout, promotional 
bundles, trigger-based co-display, and online recommendation widgets.

---

## Tech Stack

Python · Pandas · mlxtend · NetworkX · Matplotlib · scikit-learn
