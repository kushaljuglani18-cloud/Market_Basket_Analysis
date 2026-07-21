import time

import pandas as pd
from mlxtend.frequent_patterns import (
    apriori,
    association_rules as mlxtend_association_rules,
    fpgrowth,
)


def _sorted_itemset(itemset):
    return tuple(sorted(str(item) for item in itemset))


def _format_itemset(itemset):
    return " + ".join(_sorted_itemset(itemset))


def _prepare_itemsets(frequent_itemsets):
    itemsets = frequent_itemsets.copy()
    if itemsets.empty:
        return itemsets
    itemsets["itemset_size"] = itemsets["itemsets"].apply(len)
    return itemsets.sort_values(
        ["support", "itemset_size"],
        ascending=[False, True],
    ).reset_index(drop=True)


def run_apriori(basket, min_support=0.15):
    """
    Runs the Apriori algorithm on the basket matrix.

    Apriori works by:
    1. Finding all individual items meeting min_support
    2. Combining them into pairs, filtering by min_support
    3. Combining pairs into triplets, filtering again
    4. Continuing until no new combinations pass the threshold

    This "bottom-up" approach is thorough but can be slow
    on large datasets because it generates many candidate sets.

    Returns frequent itemsets and execution time.
    """
    print("\nRunning Apriori algorithm...")
    start_time = time.time()

    frequent_itemsets = apriori(
        basket,
        min_support=min_support,
        use_colnames=True,
    )
    frequent_itemsets = _prepare_itemsets(frequent_itemsets)

    execution_time = time.time() - start_time
    print(f"Apriori completed in {execution_time:.2f} seconds")
    print(f"Frequent itemsets found: {len(frequent_itemsets)}")

    return frequent_itemsets, execution_time


def run_fpgrowth(basket, min_support=0.15):
    """
    Runs the FP-Growth algorithm on the basket matrix.

    FP-Growth works differently from Apriori:
    1. Builds a compressed tree structure (FP-Tree) of the data
    2. Mines patterns directly from the tree
    3. Never generates explicit candidate sets

    This makes it significantly faster than Apriori on large
    datasets because it only reads the data twice - once to
    build the tree, once to mine it.

    Returns frequent itemsets and execution time.
    """
    print("\nRunning FP-Growth algorithm...")
    start_time = time.time()

    frequent_itemsets = fpgrowth(
        basket,
        min_support=min_support,
        use_colnames=True
    )
    frequent_itemsets = _prepare_itemsets(frequent_itemsets)

    execution_time = time.time() - start_time
    print(f"FP-Growth completed in {execution_time:.2f} seconds")
    print(f"Frequent itemsets found: {len(frequent_itemsets)}")

    return frequent_itemsets, execution_time


def generate_rules(frequent_itemsets, min_confidence=0.5):
    """
    Generates association rules from frequuent itemsets.
    
    Each rule has the form: IF {A} THEN {B}
    with three metrics:
    - Support: how often A and B appear together in the dataset
    - Confidence: p(B|A) - if A is bought, how likely is B to be bought?
    - Lift: is this relationship stronger than random chance?
             Lift > 1 = genuine association
             Lift = 1 = independent (no real connection)
             Lift < 1 = negative association
             
    min_confidence=0.5 means we only keep rules where
    buying A leads to buying B at least 50% of the time.
    """
    print(f"\nGenerating association rules (min_confidence={min_confidence})...")

    if frequent_itemsets.empty:
        print("No frequent itemsets found. Try lowering min_support.")
        return pd.DataFrame()

    rules = mlxtend_association_rules(
        frequent_itemsets,
        metric="confidence",
        min_threshold=min_confidence
    )

    if rules.empty:
        print("No rules met the confidence threshold.")
        return rules

    rules["antecedents_str"] = rules["antecedents"].apply(_format_itemset)
    rules["consequents_str"] = rules["consequents"].apply(_format_itemset)
    rules["antecedent_count"] = rules["antecedents"].apply(len)
    rules["consequent_count"] = rules["consequents"].apply(len)
    rules["rule"] = (
        rules["antecedents_str"] + " -> " + rules["consequents_str"]
    )
    rules = rules.sort_values(
        ["lift", "confidence", "support"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    print(f"Total rules generated: {len(rules)}")
    return rules


def filter_high_lift_rules(rules, min_lift=1.2):
    """
    Filters rules to keep only those with HIGH LIFT associations.
    These are the genuinely interesting ones - products
    that are bought together FAR more than chance would predict.
    
    min_lift=1.2 means the products are bought together
    at least 1.2x more than if they were completely independent.
    """
    if rules.empty:
        return rules.copy()

    high_lift = rules[rules['lift'] >= min_lift].copy()
    high_lift = high_lift.sort_values(
        ["lift", "confidence", "support"],
        ascending=[False, False, False],
    )

    print(f"\nHigh-lift rules (lift >= {min_lift}): {len(high_lift)}")
    return high_lift


def compare_algorithms(basket, min_support=0.15):
    """
    Runs both algorithms and compares their execution times.
    Both should find the same itemsets - the difference is speed.
    """
    print("\n==== ALGORITHM COMPARISON ====")

    apriori_itemsets, apriori_time = run_apriori(basket, min_support)
    fpgrowth_itemsets, fpgrowth_time = run_fpgrowth(basket, min_support)

    print("\nTime comparison:")
    print(f"  Apriori:  {apriori_time:.2f} seconds")
    print(f"  FP-Growth:  {fpgrowth_time:.2f} seconds")

    if 0 < fpgrowth_time < apriori_time:
        speedup = apriori_time / fpgrowth_time
        print(f" FP-Growth was {speedup:.1f}x faster than Apriori")
    else:
        print(f"  Apriori was faster on this dataset size")

    timing = pd.DataFrame(
        [
            {
                "algorithm": "Apriori",
                "seconds": apriori_time,
                "frequent_itemsets": len(apriori_itemsets),
            },
            {
                "algorithm": "FP-Growth",
                "seconds": fpgrowth_time,
                "frequent_itemsets": len(fpgrowth_itemsets),
            },
        ]
    )
    return fpgrowth_itemsets, timing


def save_rules(rules, filepath="output/association_rules.csv"):
    """
    Saves the association rules to a CSV file.
    Rounds numbers to 4 decimal places for readability.
    """
    import os
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)

    preferred_columns = [
        "rule",
        "antecedents_str",
        "consequents_str",
        "antecedent support",
        "consequent support",
        "support",
        "confidence",
        "lift",
        "leverage",
        "conviction",
        "zhangs_metric",
        "antecedent_count",
        "consequent_count",
    ]
    output_columns = [col for col in preferred_columns if col in rules.columns]
    output = rules[output_columns].copy()

    numeric_columns = output.select_dtypes(include="number").columns
    output[numeric_columns] = output[numeric_columns].round(4)
    output.to_csv(filepath, index=False)
    print(f"\nRules saved to {filepath}")


def save_itemsets(frequent_itemsets, filepath="output/frequent_itemsets.csv"):
    """Save frequent itemsets in a CSV-friendly format."""
    import os

    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    output = frequent_itemsets.copy()
    if not output.empty:
        output["itemsets"] = output["itemsets"].apply(_format_itemset)
        numeric_columns = output.select_dtypes(include="number").columns
        output[numeric_columns] = output[numeric_columns].round(4)
    output.to_csv(filepath, index=False)
    print(f"Frequent itemsets saved to {filepath}")
