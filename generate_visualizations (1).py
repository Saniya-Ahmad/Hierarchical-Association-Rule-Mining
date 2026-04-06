import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import numpy as np
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Read all rule files
l1 = pd.read_csv('Level_1_rules.csv')
l2 = pd.read_csv('Level_2_rules.csv')
l3 = pd.read_csv('Level_3_rules.csv')

print("Generating visualizations...")

# ============================================================================
# 1. TOP RULES BY LIFT - BAR CHART (All Levels)
# ============================================================================
fig, axes = plt.subplots(3, 1, figsize=(14, 12))
fig.suptitle('Top 10 Association Rules by LIFT (Strength of Association)', 
             fontsize=16, fontweight='bold', y=0.995)

for ax, rules, level_name in zip(axes, [l1, l2, l3], 
                                  ['Level 1: Department', 'Level 2: Commodity', 'Level 3: Sub-Commodity']):
    top_rules = rules.nlargest(10, 'lift').reset_index(drop=True)
    
    # Create readable labels
    labels = []
    for idx, row in top_rules.iterrows():
        ant = str(row['antecedents'])[:25]
        cons = str(row['consequents'])[:25]
        labels.append(f"{idx+1}. {ant}...→...{cons}...")
    
    bars = ax.barh(range(len(labels)), top_rules['lift'], color=plt.cm.RdYlGn(top_rules['lift']/top_rules['lift'].max()))
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('LIFT (Strength)', fontsize=11, fontweight='bold')
    ax.set_title(level_name, fontsize=12, fontweight='bold', pad=10)
    ax.invert_yaxis()
    
    # Add value labels
    for i, (bar, val, conf) in enumerate(zip(bars, top_rules['lift'], top_rules['confidence'])):
        ax.text(val + 0.05, i, f"{val:.2f} (Conf: {conf:.1%})", 
               va='center', fontsize=8, fontweight='bold')

plt.tight_layout()
plt.savefig('01_Top_Rules_By_Lift.png', dpi=300, bbox_inches='tight')
print("✓ Generated: 01_Top_Rules_By_Lift.png")
plt.close()

# ============================================================================
# 2. CONFIDENCE vs LIFT SCATTER PLOT - All Levels
# ============================================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Confidence vs Lift: Rule Quality Analysis', 
             fontsize=14, fontweight='bold')

for ax, rules, level_name in zip(axes, [l1, l2, l3], 
                                  ['Level 1', 'Level 2', 'Level 3']):
    scatter = ax.scatter(rules['confidence'], rules['lift'], 
                        c=rules['support'], s=100, alpha=0.6, 
                        cmap='viridis', edgecolors='black', linewidth=0.5)
    ax.set_xlabel('Confidence (Rule Strength)', fontsize=10, fontweight='bold')
    ax.set_ylabel('Lift (Association Strength)', fontsize=10, fontweight='bold')
    ax.set_title(level_name, fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=1, color='r', linestyle='--', linewidth=1, alpha=0.5, label='Baseline (Lift=1)')
    ax.legend()
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Support %', fontsize=9)

plt.tight_layout()
plt.savefig('02_Confidence_vs_Lift_Scatter.png', dpi=300, bbox_inches='tight')
print("✓ Generated: 02_Confidence_vs_Lift_Scatter.png")
plt.close()

# ============================================================================
# 3. SUPPORT vs LIFT DISTRIBUTION
# ============================================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Support vs Lift Distribution (Bubble Size = Rule Count)', 
             fontsize=14, fontweight='bold')

for ax, rules, level_name in zip(axes, [l1, l2, l3], 
                                  ['Level 1', 'Level 2', 'Level 3']):
    # Create bins
    support_bins = pd.cut(rules['support'], bins=8)
    lift_bins = pd.cut(rules['lift'], bins=8)
    
    scatter = ax.scatter(rules['support']*100, rules['lift'], 
                        c=rules['confidence'], s=80, alpha=0.7,
                        cmap='plasma', edgecolors='black', linewidth=0.5)
    ax.set_xlabel('Support (%)', fontsize=10, fontweight='bold')
    ax.set_ylabel('Lift', fontsize=10, fontweight='bold')
    ax.set_title(level_name, fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Confidence', fontsize=9)

plt.tight_layout()
plt.savefig('03_Support_vs_Lift.png', dpi=300, bbox_inches='tight')
print("✓ Generated: 03_Support_vs_Lift.png")
plt.close()

# ============================================================================
# 4. METRICS DISTRIBUTION - HISTOGRAMS
# ============================================================================
fig, axes = plt.subplots(3, 3, figsize=(15, 10))
fig.suptitle('Distribution of Key Metrics Across All Levels', 
             fontsize=14, fontweight='bold')

metrics = ['support', 'confidence', 'lift']
all_rules = [l1, l2, l3]
level_names = ['Level 1', 'Level 2', 'Level 3']

for row, (rules, level_name) in enumerate(zip(all_rules, level_names)):
    for col, metric in enumerate(metrics):
        ax = axes[row, col]
        ax.hist(rules[metric], bins=30, color=plt.cm.Set2(row), 
               edgecolor='black', alpha=0.7)
        ax.set_xlabel(metric.capitalize(), fontsize=10, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=10, fontweight='bold')
        ax.set_title(f'{level_name}: {metric.capitalize()}', fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add statistics
        mean_val = rules[metric].mean()
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
        ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig('04_Metrics_Distribution.png', dpi=300, bbox_inches='tight')
print("✓ Generated: 04_Metrics_Distribution.png")
plt.close()

# ============================================================================
# 5. TOP PRODUCTS BY FREQUENCY - Level 2
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle('Top Products by Frequency (Level 2 Analysis)', 
             fontsize=14, fontweight='bold')

# Extract antecedents and consequents
all_items = []
for row in l2['antecedents']:
    items = str(row).replace("frozenset({", "").replace("})", "").replace("'", "").split(', ')
    all_items.extend(items)
for row in l2['consequents']:
    items = str(row).replace("frozenset({", "").replace("})", "").replace("'", "").split(', ')
    all_items.extend(items)

# Count frequency
item_freq = pd.Series(all_items).value_counts().head(15)

# Plot
ax = axes[0]
item_freq.plot(kind='barh', ax=ax, color=plt.cm.Spectral(range(len(item_freq)))/256)
ax.set_xlabel('Frequency in Rules', fontsize=11, fontweight='bold')
ax.set_title('Top 15 Products in Association Rules', fontsize=12, fontweight='bold')
ax.invert_yaxis()

# Add values on bars
for i, v in enumerate(item_freq.values):
    ax.text(v + 0.5, i, str(v), va='center', fontweight='bold')

# Confidence distribution for top items
ax = axes[1]
top_items = item_freq.head(10).index
conf_by_item = []
for item in top_items:
    confs = []
    for idx, row in l2.iterrows():
        if item in str(row['consequents']):
            confs.append(row['confidence'])
    if confs:
        conf_by_item.append(pd.Series(confs).mean())
    else:
        conf_by_item.append(0)

ax.bar(range(len(top_items)), conf_by_item, color=plt.cm.RdYlGn(np.array(conf_by_item)))
ax.set_xticks(range(len(top_items)))
ax.set_xticklabels(top_items, rotation=45, ha='right', fontsize=9)
ax.set_ylabel('Average Confidence', fontsize=11, fontweight='bold')
ax.set_title('Avg Confidence When Item is Consequent', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

for i, v in enumerate(conf_by_item):
    ax.text(i, v + 0.02, f'{v:.2f}', ha='center', fontweight='bold', fontsize=9)

plt.tight_layout()
plt.savefig('05_Top_Products_Frequency.png', dpi=300, bbox_inches='tight')
print("✓ Generated: 05_Top_Products_Frequency.png")
plt.close()

# ============================================================================
# 6. SUMMARY STATISTICS TABLE
# ============================================================================
fig, ax = plt.subplots(figsize=(12, 4))
ax.axis('tight')
ax.axis('off')

summary_data = []
for rules, level_name in zip([l1, l2, l3], ['Level 1', 'Level 2', 'Level 3']):
    summary_data.append([
        level_name,
        len(rules),
        f"{rules['support'].min():.4f}",
        f"{rules['support'].max():.4f}",
        f"{rules['support'].mean():.4f}",
        f"{rules['confidence'].mean():.2%}",
        f"{rules['lift'].mean():.2f}",
        f"{rules['lift'].max():.2f}"
    ])

columns = ['Level', '# Rules', 'Min Support', 'Max Support', 'Avg Support', 
           'Avg Confidence', 'Avg Lift', 'Max Lift']

table = ax.table(cellText=summary_data, colLabels=columns, cellLoc='center', 
                loc='center', colWidths=[0.12, 0.1, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)

# Style header
for i in range(len(columns)):
    table[(0, i)].set_facecolor('#4CAF50')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Alternate row colors
for i in range(1, len(summary_data) + 1):
    for j in range(len(columns)):
        if i % 2 == 0:
            table[(i, j)].set_facecolor('#f0f0f0')
        else:
            table[(i, j)].set_facecolor('#ffffff')

plt.title('Summary Statistics - Multilevel Association Rules', 
         fontsize=14, fontweight='bold', pad=20)
plt.savefig('06_Summary_Statistics_Table.png', dpi=300, bbox_inches='tight')
print("✓ Generated: 06_Summary_Statistics_Table.png")
plt.close()

# ============================================================================
# 7. NETWORK GRAPH - Level 2 Top Rules
# ============================================================================
print("Generating network graph... (this may take a moment)")
fig, ax = plt.subplots(figsize=(16, 14))

G = nx.DiGraph()

# Add top 30 rules by lift
top_rules = l2.nlargest(30, 'lift')

for idx, row in top_rules.iterrows():
    ant = str(row['antecedents']).replace("frozenset({", "").replace("})", "").replace("'", "").strip()
    cons = str(row['consequents']).replace("frozenset({", "").replace("})", "").replace("'", "").strip()
    
    # Abbreviate if too long
    ant = ant[:20] + "..." if len(ant) > 20 else ant
    cons = cons[:20] + "..." if len(cons) > 20 else cons
    
    G.add_edge(ant, cons, weight=row['lift'], confidence=row['confidence'])

# Draw network
pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

# Node colors based on degree
node_colors = [G.degree(node) for node in G.nodes()]
nodes = nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                               node_size=2000, cmap='YlOrRd', 
                               edgecolors='black', linewidths=2, ax=ax)

# Edge widths based on lift
edges = G.edges()
weights = [G[u][v]['weight'] for u, v in edges]
max_weight = max(weights)
edge_widths = [3 * (w / max_weight) for w in weights]

nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color='gray', 
                      arrows=True, arrowsize=20, arrowstyle='->', 
                      connectionstyle='arc3,rad=0.1', alpha=0.6, ax=ax)

# Labels
nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold', ax=ax)

# Colorbar
sm = plt.cm.ScalarMappable(cmap='YlOrRd', 
                          norm=plt.Normalize(vmin=min(node_colors), vmax=max(node_colors)))
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, label='Node Connectivity (In/Out Degree)')

ax.set_title('Association Network Graph (Level 2: Top 30 Rules by Lift)\nNode Size = Frequency | Edge Width = Lift Strength', 
            fontsize=14, fontweight='bold', pad=20)
ax.axis('off')

plt.tight_layout()
plt.savefig('07_Network_Graph_Level2.png', dpi=300, bbox_inches='tight')
print("✓ Generated: 07_Network_Graph_Level2.png")
plt.close()

print("\n" + "="*60)
print("✓ ALL VISUALIZATIONS GENERATED SUCCESSFULLY!")
print("="*60)
print("\nGenerated Files:")
print("  1. 01_Top_Rules_By_Lift.png - Top 10 rules by lift for each level")
print("  2. 02_Confidence_vs_Lift_Scatter.png - Quality analysis scatter plot")
print("  3. 03_Support_vs_Lift.png - Support vs Lift relationship")
print("  4. 04_Metrics_Distribution.png - Histograms of key metrics")
print("  5. 05_Top_Products_Frequency.png - Frequency and confidence analysis")
print("  6. 06_Summary_Statistics_Table.png - Summary statistics table")
print("  7. 07_Network_Graph_Level2.png - Network visualization of rules")
print("\nTotal: 7 visualization files")
