# Market Basket Analysis for Cross-Selling and Upselling

An end-to-end retail analytics pipeline that uncovers hidden product associations from transaction data using Association Rule Mining. Built as a Data Science Internship Project.

---

## Project Overview

Analyses 10,000 synthetic retail transactions across 21 products in 4 categories to discover which products are frequently bought together — and translates those findings into concrete business recommendations.

---

## Pipeline

| Step | File | What it does |
|---|---|---|
| 1 | `create_data.py` | Generates 10,000 synthetic transactions across 4 shopper profiles |
| 2 | `data_preprocessing.py` | Transforms Long Format CSV into One-Hot Encoded basket matrix |
| 3 | `association_rules.py` | Runs Apriori and FP-Growth, calculates Support, Confidence, Lift |
| 4 | `visualize.py` | Produces network graphs and top rules bar chart |
| 5 | `main.py` | Orchestrates the full pipeline end to end |

---

## Key Results

| Metric | Value |
|---|---|
| Total transactions | 10,000 |
| Total products | 21 |
| Association rules found | 96 |
| Minimum confidence | 60% |
| Minimum lift | 1.2 |
| Strongest rule | Soda → Chips (Lift 4.03, Confidence 100%) |
| Best breakfast rule | Butter → Eggs (Lift 3.96, Confidence 85%) |
| Cross-category finding | Cleaning ↔ Personal Care (Lift 1.8–1.9) |

---

## Output Files

| File | Description |
|---|---|
| `output/association_rules.csv` | 96 rules sorted by Lift with Support, Confidence columns |
| `output/network_graph.png` | Within-category product association clusters |
| `output/cross_category_network.png` | Cross-category product associations |
| `output/top_rules_chart.png` | Top 15 rules by Lift with Confidence and Support labels |

---

## Project Structure

| File | Purpose |
|---|---|
| `create_data.py` | Synthetic data generation using shopper mission profiles |
| `data_preprocessing.py` | Long Format to basket matrix transformation |
| `association_rules.py` | Apriori and FP-Growth algorithm implementations |
| `visualize.py` | All visualization functions |
| `main.py` | Pipeline orchestration with CLI argument support |
| `data/transactions.csv` | Generated transaction dataset |
| `Business_Strategy_Proposal.pdf` | Six concrete retail recommendations derived from results |

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

# Step 2: Run the full pipeline
python main.py

# Optional: custom thresholds
python main.py --min-support 0.10 --min-confidence 0.60 --min-lift 1.2
```

---

## Top Findings

| Rule | Confidence | Lift | Category |
|---|---|---|---|
| Soda → Chips | 100% | 4.03 | Snacks |
| Popcorn → Chips | 100% | 4.03 | Snacks |
| Eggs → Butter | 85% | 3.96 | Breakfast |
| Butter → Bread | 100% | 3.07 | Breakfast |
| Soap → Detergent | 67% | 1.89 | Cross-Category |
| Shampoo → Detergent | 67% | 1.79 | Cross-Category |

---

## Business Strategy Proposal

See `Business_Strategy_Proposal.pdf` for six concrete retail recommendations covering:

| Recommendation | Based on |
|---|---|
| Snack Station display | Chips/Soda/Chocolate Lift 4.0 |
| Weekend Snack Bundle | 100% confidence on Chips ↔ Soda |
| Breakfast aisle redesign around Bread | Bread as anchor product |
| Promotional co-display triggers | Soda→Chips 100% confidence |
| Household Essentials end-cap | Cleaning ↔ Personal Care Lift 1.8–1.9 |
| Online recommendation widget | High-confidence rules across all categories |

---

## Tech Stack

| Library | Purpose |
|---|---|
| Pandas | Data transformation and DataFrame operations |
| mlxtend | Apriori and FP-Growth implementations |
| NetworkX | Network graph construction |
| Matplotlib | All visualizations |
