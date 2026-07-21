import pandas as pd


TRANSACTION_ALIASES = (
    "TransactionID", "TransactionId", "transaction_id", "InvoiceNo",
    "Invoice", "BillNo", "OrderID", "OrderId", "order_id",
)
ITEM_ALIASES = (
    "Item", "item", "Description", "Product", "ProductName",
    "product_name", "StockCode", "SKU", "sku",
)
QUANTITY_ALIASES = ("Quantity", "Qty", "quantity", "qty")
CANCELLATION_ALIASES = ("Cancelled", "Canceled", "IsCancelled", "is_cancelled")


def _resolve_column(df, explicit_name, candidates, logical_name):
    if explicit_name:
        if explicit_name not in df.columns:
            raise ValueError(
                f"Column '{explicit_name}' was requested for {logical_name}, "
                f"but it is not present. Available columns: {list(df.columns)}"
            )
        return explicit_name

    for candidate in candidates:
        if candidate in df.columns:
            return candidate

    raise ValueError(
        f"Could not infer the {logical_name} column. Pass it explicitly. "
        f"Available columns: {list(df.columns)}"
    )


def standardize_transactions(
    df,
    transaction_col=None,
    item_col=None,
    quantity_col=None,
    cancellation_col=None,
):
    """Return a cleaned long-format frame with TransactionID and Item columns."""
    tx_col = _resolve_column(df, transaction_col, TRANSACTION_ALIASES, "transaction id")
    prod_col = _resolve_column(df, item_col, ITEM_ALIASES, "item/product")

    clean = df.copy()
    clean = clean.rename(columns={tx_col: "TransactionID", prod_col: "Item"})

    clean["TransactionID"] = clean["TransactionID"].astype(str).str.strip()
    clean["Item"] = clean["Item"].astype(str).str.strip()
    clean = clean[
        clean["TransactionID"].ne("")
        & clean["Item"].ne("")
        & clean["Item"].str.lower().ne("nan")
    ].copy()

    if quantity_col is None:
        quantity_col = next((col for col in QUANTITY_ALIASES if col in clean.columns), None)
    if quantity_col is not None:
        if quantity_col not in clean.columns:
            raise ValueError(f"Quantity column '{quantity_col}' is not present.")
        clean[quantity_col] = pd.to_numeric(clean[quantity_col], errors="coerce")
        clean = clean[clean[quantity_col].fillna(0) > 0].copy()

    if cancellation_col is None:
        cancellation_col = next((col for col in CANCELLATION_ALIASES if col in clean.columns), None)
    if cancellation_col is not None:
        if cancellation_col not in clean.columns:
            raise ValueError(f"Cancellation column '{cancellation_col}' is not present.")
        cancelled = clean[cancellation_col].astype(str).str.lower().isin(
            {"1", "true", "yes", "y", "cancelled", "canceled"}
        )
        clean = clean[~cancelled].copy()

    invoice_like = clean["TransactionID"].str.upper().str.startswith("C", na=False)
    if invoice_like.any():
        clean = clean[~invoice_like].copy()

    clean = clean.drop_duplicates(subset=["TransactionID", "Item"])
    return clean


def load_transactions(
    filepath="data/transactions.csv",
    transaction_col=None,
    item_col=None,
    quantity_col=None,
    cancellation_col=None,
):
    """Load and clean raw long-format transaction data."""
    df = pd.read_csv(filepath)
    df = standardize_transactions(
        df,
        transaction_col=transaction_col,
        item_col=item_col,
        quantity_col=quantity_col,
        cancellation_col=cancellation_col,
    )

    print(f"Loaded {len(df)} rows from {filepath}")
    print(f"Unique transactions: {df['TransactionID'].nunique()}")
    print(f"Unique products: {df['Item'].nunique()}")
    return df


def explore_data(df):
    """Print a quick summary of the cleaned dataset."""
    print("\n---- DATA EXPLORATION ----")
    print("\nTop 10 most purchased items:")
    print(df['Item'].value_counts().head(10))

    print("\nAverage basket size (unique items per transaction):")
    basket_sizes = df.groupby('TransactionID')['Item'].nunique()
    print(f" Mean: {basket_sizes.mean():.2f}")
    print(f" Min: {basket_sizes.min()}")
    print(f" Max: {basket_sizes.max()}")


def transform_to_basket(df):
    """Convert long-format transactions to a one-hot encoded basket matrix."""
    print("\nTransforming to Basket Format...")

    basket = (
        df.assign(_purchased=True)
        .pivot_table(
            index="TransactionID",
            columns="Item",
            values="_purchased",
            aggfunc="any",
            fill_value=False,
        )
        .astype(bool)
    )
    basket = basket.reindex(sorted(basket.columns), axis=1)

    print(f"Basket Matrix Shape: {basket.shape}")
    print(f"  {basket.shape[0]} transactions x {basket.shape[1]} products")
    return basket


def get_item_frequencies(df):
    """Return single-item support sorted from most to least common."""
    total_transactions = df['TransactionID'].nunique()
    item_counts = df.groupby('Item')['TransactionID'].nunique()
    item_support = item_counts / total_transactions
    return item_support.sort_values(ascending=False)


def get_transaction_lists(df):
    """Return the list-of-lists basket representation used in MBA explanations."""
    return (
        df.groupby("TransactionID")["Item"]
        .apply(lambda items: sorted(set(items)))
        .tolist()
    )
