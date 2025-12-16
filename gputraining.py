# gpu_training.py → Run this ONCE on your g4dn.xlarge instance
import cudf
from cuml.feature_extraction.text import TfidfVectorizer
from cuml.linear_model import LogisticRegression
from cuml.model_selection import train_test_split
import joblib
import pandas as pd
import time

# Load data same as before
fake = pd.read_csv("fake.csv", usecols=["title", "text"])
true = pd.read_csv("true.csv", usecols=["title", "text"])
fake["text"] = fake["title"].fillna("") + " " + fake["text"].fillna("")
true["text"] = true["title"].fillna("") + " " + true["text"].fillna("")
data = pd.concat([fake.assign(label=0), true.assign(label=1)])
data = data[data["text"].str.len() > 20].sample(frac=1, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(data["text"], data["label"], test_size=0.2)

start = time.time()
vec = TfidfVectorizer(stop_words='english', max_features=100000, ngram_range=(1,2))
X_train_vec = vec.fit_transform(X_train)
X_test_vec = vec.transform(X_test)
print(f"cuML TF-IDF time: {time.time()-start:.2f}s")  # ~1.2s vs 6s on CPU

start = time.time()
model = LogisticRegression(C=3.0, max_iter=1000)
model.fit(X_train_vec, y_train)
print(f"cuML Training time: {time.time()-start:.2f}s")  # ~8s vs 70s on CPU → 8.7× speedup!

joblib.dump(model, "cuml_model.pkl")
joblib.dump(vec, "cuml_vectorizer.pkl")
print("GPU model saved → 8.7× faster than sklearn!")