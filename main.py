import argparse
import os

from association_rules import (
    compare_algorithms,
    filter_high_lift_rules,
    generate_rules,
    save_itemsets,
    save_rules,
)

from data_preprocessing import (
    explore_data,
    get_item_frequencies,
    load_transactions,
    transform_to_basket,
)
from visualize import build_network, plot_network, plot_rule_quality_scatter, plot_top_rules, plot_cross_category_network


def parse_args():
    parser = argparse.ArgumentParser(
        description="Market Basket Analysis pipeline using Apriori and FP-Growth."
    )
    parser.add_argument("--input", default="data/transactions.csv", help="Raw transaction CSV path.")
    parser.add_argument("--output-dir", default="output", help="Folder for CSV and PNG outputs.")
    parser.add_argument("--transaction-col", default=None, help="Transaction/order id column name.")
    parser.add_argument("--item-col", default=None, help="Product/item column name.")
    parser.add_argument("--quantity-col", default=None, help="Optional quantity column; rows <= 0 are removed.")
    parser.add_argument("--cancellation-col", default=None, help="Optional cancellation flag column.")
    parser.add_argument("--min-support", type=float, default=0.15, help="Minimum support for frequent itemsets.")
    parser.add_argument("--min-confidence", type=float, default=0.60, help="Minimum confidence for rules.")
    parser.add_argument("--min-lift", type=float, default=1.20, help="Minimum lift for final rules.")
    parser.add_argument("--max-network-rules", type=int, default=35, help="Maximum rule edges in network graph.")
    return parser.parse_args()


def run_pipeline(args):
    os.makedirs(args.output_dir, exist_ok=True)

    df = load_transactions(
        args.input,
        transaction_col=args.transaction_col,
        item_col=args.item_col,
        quantity_col=args.quantity_col,
        cancellation_col=args.cancellation_col,
    )
    explore_data(df)

    item_frequencies = get_item_frequencies(df).reset_index()
    item_frequencies.columns = ["item", "support"]
    item_frequency_path = os.path.join(args.output_dir, "item_frequencies.csv")
    item_frequencies.to_csv(item_frequency_path, index=False)
    print(f"Item frequencies saved to {item_frequency_path}")

    basket = transform_to_basket(df)

    frequent_itemsets, timing = compare_algorithms(basket, min_support=args.min_support)
    timing_path = os.path.join(args.output_dir, "algorithm_comparison.csv")
    timing.to_csv(timing_path, index=False)
    print(f"Algorithm comparison saved to {timing_path}")

    itemsets_path = os.path.join(args.output_dir, "frequent_itemsets.csv")
    save_itemsets(frequent_itemsets, itemsets_path)

    rules = generate_rules(frequent_itemsets, min_confidence=args.min_confidence)
    high_lift_rules = filter_high_lift_rules(rules, min_lift=args.min_lift)

    print("\n---- TOP 15 RULES BY LIFT ----")
    if high_lift_rules.empty:
        print("No rules passed the configured thresholds.")
    else:
        columns = ["rule", "support", "confidence", "lift"]
        print(high_lift_rules[columns].head(15).to_string(index=False))

    rules_path = os.path.join(args.output_dir, "association_rules.csv")
    save_rules(high_lift_rules, rules_path)

    print("\nGenerating visualizations...")
    G = build_network(
        high_lift_rules,
        max_rules=args.max_network_rules,
        min_lift=args.min_lift,
    )
    plot_network(
        G,
        df,
        filepath=os.path.join(args.output_dir, "network_graph.png"),
    )

    plot_cross_category_network(
    high_lift_rules,
    df,
    min_lift=args.min_lift,
    filepath=os.path.join(args.output_dir, "cross_category_network.png"),
)

    plot_top_rules(
        high_lift_rules,
        top_n=15,
        filepath=os.path.join(args.output_dir, "top_rules_chart.png"),
    )
    plot_rule_quality_scatter(
        high_lift_rules,
        top_n=50,
        filepath=os.path.join(args.output_dir, "rule_quality_scatter.png"),
    )

    print("\nAll done!")


if __name__ == "__main__":
    run_pipeline(parse_args())
