# Nykaa Product Performance Analysis  

## Project Overview  
This project analyzes product performance on the Nykaa e-commerce platform using both structured product data and unstructured customer reviews.  

The goal is to understand what drives product success in online beauty retail, where purchasing decisions rely heavily on ratings, reviews, and perceived value.  

The analysis combines:
- Data cleaning and preprocessing  
- Exploratory Data Analysis (EDA)  
- Text analysis (Word Cloud)  
- Machine learning modeling  

---

##  Objectives  
- Identify key factors influencing product success  
- Analyze customer engagement patterns  
- Understand pricing and discount strategies  
- Extract insights from customer reviews  
- Build predictive models for product success  

---

## Datasets  

### **1. Product Dataset**  
- ~9,800+ products scraped from Nykaa  
- Includes:
  - Pricing (**new_price, old_price, discount**)  
  - Product attributes (**brand, item, subitem, shades**)  
  - Engagement (**rating, review_count, rating_count**)  

### **2. Review Dataset**  
- Contains random customer reviews from Nykaa  
- Used for text analysis  
- Focused on identifying patterns in negative feedback  

---

## 🧹 Data Preprocessing  
- Removed irrelevant columns  
- Converted variables to numeric format  
- Handled missing values using pricing relationships  
- Filled categorical nulls with `"unknown"`  
- Removed duplicates  

### **Feature Engineering**
- Created `price_segment` (Low, Mid, High, Premium)  
- Created `success` variable:
  - Rating ≥ 4.0  
  - Review count above median  

---

## Exploratory Data Analysis  

### Key analyses performed:
- Distribution of price, ratings, and discounts  
- Top brands and market share (Pareto analysis)  
- Category-level analysis (item & subitem)  
- Correlation heatmap  
- Price vs rating relationships  
- Review count vs rating (log scale)  

### Product-level insights:
- Top products by review count  
- Value score (rating + discount + engagement)  

---

## Text Analysis (Customer Reviews)  

A second dataset of customer reviews was analyzed to extract qualitative insights.  

### Process:
- Cleaned text using regular expressions  
- Filtered negative reviews  
- Combined text into a corpus  
- Generated a word cloud  

### Key Finding:
Frequent words such as **"shade"**, **"match"**, and **"dark"** indicate that **shade mismatch** is a major customer issue.  

---

##  Machine Learning  

### Model Used:
- **Decision Tree Classifier**

### Approach:
- One-hot encoding  
- Train-test split (80/20)  
- Feature engineering  
- Model evaluation  

### Evaluation Metrics:
- Accuracy  
- Precision  
- Recall  
- Confusion Matrix  
- ROC Curve  

---

## 🔍 Key Insights  
- Customer engagement is the strongest predictor of success  
- Ratings become more reliable with higher review volume  
- Discounts have minimal impact on performance  
- Price affects positioning but not success directly  
- Shade mismatch is a major issue in customer feedback  

---

##  Limitations  
- No sentiment analysis performed on reviews  
- Dataset limited to one platform  
- Success defined using proxy metrics  
- No time-based analysis  

---

##  Future Improvements  
- Apply sentiment analysis (NLP)  
- Use advanced models (XGBoost)  
- Build dashboards (Power BI / Streamlit)  
- Incorporate sales data  

---

## 🛠️ Tools & Technologies  
- **Python** (Pandas, NumPy)  
- **Matplotlib, Seaborn**  
- **Scikit-learn**  
- **WordCloud**  

---
