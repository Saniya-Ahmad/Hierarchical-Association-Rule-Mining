# Multi-Level Food Analysis using Hierarchical Association Rule Mining

##  Overview
This project explores customer purchasing behavior using **hierarchical association rule mining**. By applying the Apriori algorithm across multiple levels of product abstraction, the project identifies meaningful patterns in retail transactions.

The goal is to understand how product relationships vary at different levels such as department, commodity, and sub-category.

---

##  Objective
- Discover associations between products at multiple hierarchy levels  
- Compare rule quality across abstraction levels  
- Analyze trade-offs between general and specific patterns  

---

##  Dataset
- **Dunnhumby – The Complete Journey Dataset** (Kaggle)  
- Includes:
  - Transaction data (customer purchases)
  - Product hierarchy (Department → Commodity → Sub-Commodity)

📎 Dataset link: *(https://www.kaggle.com/datasets/frtgnn/dunnhumby-the-complete-journey)*

---

##  Methodology

### 1. Data Preprocessing
- Merged transaction and product datasets using Product ID  
- Removed missing and invalid entries  

### 2. Hierarchical Transformation
- Level 1 → Department  
- Level 2 → Commodity  
- Level 3 → Sub-Commodity  

### 3. Association Rule Mining
- Applied **Apriori algorithm** at each level  
- Used different support thresholds for each hierarchy level  

### 4. Metrics Used
- **Support** – Frequency of itemsets  
- **Confidence** – Likelihood of association  
- **Lift** – Strength of rule beyond random chance  

### 5. Visualization
- Confidence vs Lift plots  
- Metric distribution graphs  
- Network graphs for rule relationships  

---

##  Tech Stack
- Python  
- Pandas, NumPy  
- mlxtend (Apriori)  
- Matplotlib, Seaborn  
- NetworkX (for graph visualization)  

---

##  Results & Insights
- **Level 1 (Department):** Strong but general patterns  
- **Level 2 (Commodity):** Most meaningful and actionable insights  
- **Level 3 (Sub-Commodity):** Highly specific rules with lower support  

Key observation:
> There exists a trade-off between rule specificity and support across hierarchical levels.

---


##  Learnings
- Understanding of hierarchical data mining  
- Practical implementation of Apriori algorithm  
- Importance of tuning support thresholds  
- Data preprocessing and feature engineering  
- Visualization for interpreting association rules  

---

## 🚀 How to Run

1. Clone the repository:
```bash
git clone https://github.com/your-username/Hierarchical-Association-Rule-Mining.git
cd Hierarchical-Association-Rule-Mining
```

Install dependencies:
```
pip install -r requirements.txt

```
Run
```python main.py```

 ## Future Improvements
- Apply FP-Growth for performance optimization
- Extend to real-time recommendation systems
- Experiment with dynamic support thresholds
