import joblib
import pandas as pd
from scipy.io import arff

model = joblib.load("phishing_random_forest_model.pkl")
data, meta = arff.loadarff("Training Dataset.arff")
df = pd.DataFrame(data)

for col in df.columns:
    if df[col].dtype == object:
        df[col] = df[col].apply(lambda x: x.decode("utf-8"))

X = df.drop("Result", axis=1).astype(int)
y = df["Result"].astype(int)

pred = model.predict(X)

wrong_index = df.index[pred != y].tolist()

print("Jumlah data salah prediksi:", len(wrong_index))
print("Index data salah (20 pertama):")
print(wrong_index[:20])
