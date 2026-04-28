
# 1. IMPORTS & DATA LOADING
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# load dataset
df = pd.read_csv("final_dataset.csv")

# preview data
df.head()
df.info()
df.shape

# 2. DATA CLEANING

# drop unnecessary columns
df.drop(columns=["title", "weight"], inplace=True, errors="ignore")

# convert columns to numeric
num_cols = [
    "new_price", "old_price", "discount", "shades",
    "rating_count", "review_count", "weight_ml",
    "weight_g", "rating"
]

for col in num_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# fill missing values using relationships
df["discount"] = df["discount"].fillna(
    (df["old_price"] - df["new_price"]) / df["old_price"]
)

df["new_price"] = df["new_price"].fillna(
    df["old_price"] * (1 - df["discount"])
)

df["old_price"] = df["old_price"].fillna(
    df["new_price"] / (1 - df["discount"])
)

# fill categorical values
df["item"] = df["item"].fillna("unknown")
df["subitem"] = df["subitem"].fillna("unknown")
df["rating"] = df["rating"].fillna(0)

# fill counts
df["rating_count"] = df["rating_count"].fillna(0).astype(int)
df["review_count"] = df["review_count"].fillna(0).astype(int)

# remove duplicates
df.drop_duplicates(inplace=True)


# 3. EXPLORATORY DATA ANALYSIS

# distributions
plt.hist(df["new_price"].dropna(), bins=50)
plt.title("New Price Distribution")
plt.show()

plt.hist(df["discount"].dropna(), bins=40)
plt.title("Discount Distribution")
plt.show()

plt.hist(df["rating"].dropna(), bins=20)
plt.title("Rating Distribution")
plt.show()

# top brands
top_brands = df["brand"].value_counts().head(15)
plt.bar(top_brands.index, top_brands.values)
plt.xticks(rotation=45)
plt.title("Top Brands")
plt.show()

# correlation heatmap
corr = df.select_dtypes(include=["int64", "float64"]).corr()
plt.imshow(corr)
plt.colorbar()
plt.title("Correlation Heatmap")
plt.show()

# 4. FEATURE ENGINEERING

# success variable
df["success"] = np.where(
    (df["rating"] >= 4.0) &
    (df["review_count"] > df["review_count"].median()),
    1, 0
)

# price segmentation
df["price_segment"] = pd.qcut(
    df["new_price"], 4,
    labels=["Low", "Mid", "High", "Premium"]
)


# 5. PREPARING DATA

X = df.drop("success", axis=1)
y = df["success"]

# one-hot encoding
X = pd.get_dummies(X, drop_first=True)

# train-test split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=7
)

# handle missing values
X_train.fillna(X_train.median(numeric_only=True), inplace=True)
X_test.fillna(X_train.median(numeric_only=True), inplace=True)


# 6. DECISION TREE MODEL

from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

dt_model = DecisionTreeClassifier(
    criterion="entropy",
    max_depth=4
)

dt_model.fit(X_train, y_train)
y_pred = dt_model.predict(X_test)

print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
print("Precision:", metrics.precision_score(y_test, y_pred))
print("Recall:", metrics.recall_score(y_test, y_pred))

metrics.ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.show()


# 7. LOGISTIC REGRESSION

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)

y_pred_log = log_model.predict(X_test)
print(classification_report(y_test, y_pred_log))

# 8. RANDOM FOREST

from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

print(classification_report(y_test, y_pred_rf))


# 9. ROC CURVE

from sklearn.metrics import roc_curve, roc_auc_score

auc = roc_auc_score(y_test, y_pred)
fpr, tpr, _ = roc_curve(y_test, y_pred)

plt.plot(fpr, tpr, label=f"AUC={auc:.3f}")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

# 10. TEXT ANALYSIS (WORD CLOUD)


from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd
import re

# load review dataset
reviews_df = pd.read_csv("reviews_dataset.csv")

# assuming you already filtered negative reviews
# (example: reviews_df = reviews_df[reviews_df["rating"] <= 2])

# clean text
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", "", text)  # remove numbers & punctuation
    return text

reviews_df["clean_reviews"] = reviews_df["review_text"].apply(clean_text)

# combine all text
all_text = " ".join(reviews_df["clean_reviews"].dropna())

# generate word cloud
wordcloud = WordCloud(
    width=800,
    height=400,
    background_color="white"
).generate(all_text)

# display
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("Word Cloud - Negative Reviews")
plt.show()

# END
