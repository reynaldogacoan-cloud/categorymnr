# ============================================
# 🔧 SETUP ENVIRONMENT
# ============================================

from google.colab import drive
drive.mount('/content/drive')

!pip install gspread pandas scikit-learn nltk

# Google authentication (tanpa JSON)
from google.colab import auth
auth.authenticate_user()

import gspread
from google.auth import default
creds, _ = default()
client = gspread.authorize(creds)

import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

nltk.download('stopwords')
nltk.download('wordnet')

# ============================================
# 🔤 PREPROCESSING
# ============================================

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('indonesian'))

def preprocess_text(text):
    if isinstance(text, str):
        words = text.lower().split()
        words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
        return ' '.join(words)
    return ''

# ============================================
# 🔗 GOOGLE SHEET CONNECTION
# ============================================

spreadsheet = client.open_by_key('1DmA5tgK2d-zusPEvNda2sNkprNjabIUs-oyt4cn6QEg')
sheet_model = spreadsheet.worksheet("Model")
sheet_data = spreadsheet.worksheet("Raw")
sheet_catlist = spreadsheet.worksheet("Category List")

# ============================================
# 📥 LOAD DATA
# ============================================

df_model = pd.DataFrame(sheet_model.get_all_records())
df_data = pd.DataFrame(sheet_data.get_all_records())
df_catlist = pd.DataFrame(sheet_catlist.get_all_records())

df_model.dropna(subset=["Detail Problem (SDG)"], inplace=True)
df_model["Detail Problem (SDG)"] = df_model["Detail Problem (SDG)"].apply(preprocess_text)

print("✅ Data loaded")

# ============================================
# 🧠 TRAIN MODELS (3 models)
# ============================================

def train_nb_model(column):
    df_temp = df_model.dropna(subset=[column])
    X = df_temp["Detail Problem (SDG)"]
    y = df_temp[column]
    model = make_pipeline(TfidfVectorizer(), MultinomialNB())
    grid = GridSearchCV(model, {'multinomialnb__alpha': [0.1, 1.0, 10.0]}, cv=5)
    grid.fit(X, y)
    print(f"✅ Trained model for {column} — best α = {grid.best_params_['multinomialnb__alpha']}")
    return grid.best_estimator_

model_category = train_nb_model("Category")
model_area = train_nb_model("Area")
model_main = train_nb_model("Main Category")

# ============================================
# 🎯 SELECT ROWS TO CLASSIFY
# ============================================

df_unclassified = df_data[df_data["Main Category"] == ""].copy()
print(f"🕵️ Rows to classify: {len(df_unclassified)}")

if len(df_unclassified) == 0:
    print("🎉 Semua data sudah diklasifikasi.")
else:
    df_unclassified["Cleaned"] = df_unclassified["Detail Problem (SDG)"].apply(preprocess_text)

    # ============================================
    # 🔮 PREDICT CATEGORY
    # ============================================
    df_unclassified["Pred_Category"] = model_category.predict(df_unclassified["Cleaned"])

    # ============================================
    # 🔮 PREDICT AREA (dengan constraint)
    # ============================================
    pred_area = []
    for text, cat in zip(df_unclassified["Cleaned"], df_unclassified["Pred_Category"]):
        valid_areas = df_catlist[df_catlist["Category"] == cat]["Area"].dropna().unique()
        proba = model_area.predict_proba([text])[0]
        classes = model_area.classes_
        filtered = {cls: p for cls, p in zip(classes, proba) if cls in valid_areas}
        pred_area.append(max(filtered, key=filtered.get) if filtered else classes[proba.argmax()])
    df_unclassified["Pred_Area"] = pred_area

    # ============================================
    # 🔮 PREDICT MAIN CATEGORY (dengan constraint)
    # ============================================
    pred_main = []
    for text, cat in zip(df_unclassified["Cleaned"], df_unclassified["Pred_Category"]):
        valid_main = df_catlist[df_catlist["Category"] == cat]["Main Category"].dropna().unique()
        proba = model_main.predict_proba([text])[0]
        classes = model_main.classes_
        filtered = {cls: p for cls, p in zip(classes, proba) if cls in valid_main}
        pred_main.append(max(filtered, key=filtered.get) if filtered else classes[proba.argmax()])
    df_unclassified["Pred_Main_Category"] = pred_main

    print("✅ Prediction completed!")

    # ============================================
    # ✏️ UPDATE TO GOOGLE SHEET (X,Y,Z)
    # ============================================

    updates = []
    for i, row in df_unclassified.iterrows():
        updates.append({
            'range': f'X{i + 2}',
            'values': [[row["Pred_Main_Category"]]]
        })
        updates.append({
            'range': f'Y{i + 2}',
            'values': [[row["Pred_Area"]]]
        })
        updates.append({
            'range': f'Z{i + 2}',
            'values': [[row["Pred_Category"]]]
        })

    try:
        sheet_data.batch_update(updates)
        print(f"✅ Updated {len(df_unclassified)} rows.")
    except Exception as e:
        print("❌ Error during update:", e)

print("🎯 DONE")
