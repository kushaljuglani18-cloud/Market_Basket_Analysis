import pandas as pd
import random
import os

random.seed(42)

# Strategy: shoppers are TYPED - each type has a primary category
# they always buy from, and secondary categories they sometimes visit.
# This keeps item frequencies moderate (30-55%) so lift stays meaningful.

PRODUCTS = {
    "Breakfast": ["Bread", "Butter", "Eggs", "Milk", "Jam", "Coffee"],
    "Snacks":    ["Chips", "Soda", "Cookies", "Chocolate", "Popcorn"],
    "Cleaning":  ["Detergent", "Bleach", "Sponge", "Mop", "Disinfectant"],
    "Personal":  ["Shampoo", "Conditioner", "Soap", "Toothpaste", "Razor"],
}

# Each shopper type picks FROM their primary bundle first,
# then probabilistically adds cross-category items.
# No item should appear in more than ~50% of all transactions.

SHOPPER_TYPES = [
    {
        "name": "Breakfast Shopper",
        "weight": 0.25,
        "primary": {
            "Bread":   1.00,
            "Butter":  0.85,
            "Milk":    0.80,
            "Eggs":    0.70,
            "Jam":     0.65,
            "Coffee":  0.60,
        },
        "cross": {
            # Breakfast shoppers occasionally grab snacks
            "Cookies":   0.35,
            "Chocolate": 0.30,
        },
    },
    {
        "name": "Snack Shopper",
        "weight": 0.25,
        "primary": {
            "Chips":     1.00,
            "Soda":      0.85,
            "Chocolate": 0.80,
            "Cookies":   0.75,
            "Popcorn":   0.65,
        },
        "cross": {
            # Snack shoppers occasionally grab breakfast items
            "Milk":    0.35,
            "Bread":   0.30,
        },
    },
    {
        "name": "Cleaning Shopper",
        "weight": 0.25,
        "primary": {
            "Detergent":    1.00,
            "Bleach":       0.80,
            "Sponge":       0.75,
            "Mop":          0.65,
            "Disinfectant": 0.70,
        },
        "cross": {
            # Cleaning shoppers often restock personal care
            "Soap":      0.55,
            "Shampoo":   0.50,
            "Toothpaste":0.45,
        },
    },
    {
        "name": "Personal Care Shopper",
        "weight": 0.25,
        "primary": {
            "Shampoo":    1.00,
            "Conditioner":0.85,
            "Soap":       0.80,
            "Toothpaste": 0.75,
            "Razor":      0.65,
        },
        "cross": {
            # Personal care shoppers often grab cleaning supplies
            "Detergent": 0.50,
            "Bleach":    0.40,
            "Sponge":    0.45,
        },
    },
]


def generate_basket():
    weights = [s["weight"] for s in SHOPPER_TYPES]
    shopper = random.choices(SHOPPER_TYPES, weights=weights, k=1)[0]
    basket = set()

    # Add primary items based on their probability
    for item, prob in shopper["primary"].items():
        if random.random() < prob:
            basket.add(item)

    # Add cross-category items based on their probability
    for item, prob in shopper["cross"].items():
        if random.random() < prob:
            basket.add(item)

    # Ensure minimum basket size of 2
    if len(basket) < 2:
        all_primary = list(shopper["primary"].keys())
        basket.update(random.sample(all_primary, 2))

    return list(basket)


def generate_transactions(n=10000):
    rows = []
    for tid in range(1, n + 1):
        for item in generate_basket():
            rows.append({"TransactionID": tid, "Item": item})
    return pd.DataFrame(rows)


def main():
    print("Generating transactions...")
    df = generate_transactions(10000)

    os.makedirs("data", exist_ok=True)
    df.to_csv("data/transactions.csv", index=False)

    print(f"Rows: {len(df)}")
    print(f"Transactions: {df['TransactionID'].nunique()}")
    print(f"Avg basket size: {df.groupby('TransactionID')['Item'].count().mean():.2f}")
    print("\nItem frequencies (% of transactions):")
    total = df['TransactionID'].nunique()
    freqs = df.groupby('Item')['TransactionID'].nunique() / total
    print(freqs.sort_values(ascending=False).round(3).to_string())


main()
