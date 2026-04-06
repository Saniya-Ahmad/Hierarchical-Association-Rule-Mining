import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Import mlxtend
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# Load data
print("Loading product data...")
product_df = pd.read_csv('product.csv')
print("Loading transaction data...")
trans_df = pd.read_csv('transaction_data.csv')

# Merge transaction data with product data
print("Merging data...")
merged_df = pd.merge(trans_df, product_df, on='PRODUCT_ID', how='left')

# Sample a subset to handle memory constraints (take 5% of transactions)
print("Sampling data for memory efficiency...")
sample_baskets = merged_df['BASKET_ID'].drop_duplicates().sample(frac=0.05, random_state=42)
merged_df = merged_df[merged_df['BASKET_ID'].isin(sample_baskets)]
print(f"Sampled data has {len(merged_df)} transactions from {len(sample_baskets)} baskets")

# Define hierarchy levels
levels = {
    'Level 1': 'DEPARTMENT',
    'Level 2': 'COMMODITY_DESC',
    'Level 3': 'SUB_COMMODITY_DESC'
}

# Support thresholds for each level (decreasing as we go deeper)
support_thresholds = {
    'Level 1': 0.01,
    'Level 2': 0.02,
    'Level 3': 0.03
}

# Minimum confidence for rules
min_confidence = 0.5

# Function to generate multilevel association rules
def generate_multilevel_rules(level_name, level_column, min_support):
    print(f"\nGenerating rules for {level_name} ({level_column}) with min_support={min_support}")

    # Group by BASKET_ID and get unique items at this level
    baskets = merged_df.groupby('BASKET_ID')[level_column].unique().tolist()

    # Remove NaN values and empty strings
    baskets = [[item for item in basket if pd.notna(item) and item != ''] for basket in baskets]

    # Filter out empty baskets
    baskets = [basket for basket in baskets if len(basket) > 1]

    if len(baskets) == 0:
        print(f"No valid baskets for {level_name}")
        return

    # Encode transactions
    te = TransactionEncoder()
    te_ary = te.fit(baskets).transform(baskets)
    df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

    # Generate frequent itemsets
    frequent_itemsets = apriori(df_encoded, min_support=min_support, use_colnames=True)

    if frequent_itemsets.empty:
        print(f"No frequent itemsets found for {level_name}")
        return

    # Generate association rules
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)

    if rules.empty:
        print(f"No association rules found for {level_name}")
        return

    # Sort by confidence and lift
    rules = rules.sort_values(['confidence', 'lift'], ascending=[False, False])

    print(f"Found {len(rules)} rules for {level_name}")
    print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(10))

    # Save rules to CSV
    rules.to_csv(f'{level_name.replace(" ", "_")}_rules.csv', index=False)

# Generate rules for each level
for level_name, level_column in levels.items():
    generate_multilevel_rules(level_name, level_column, support_thresholds[level_name])

print("\nMultilevel association rule mining completed!")