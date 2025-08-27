"""
Streamlit app: Stock Up/Down Prediction + Backtest
Filename: Streamlit_stock_app.py

Features
- Upload a CSV (cleaned stock CSV expected) or use example file
- Performs feature engineering (Return, MA_5, MA_10, Volatility_5, Volume_MA_5)
- Creates binary target: next-day close > today close
- Trains a Random Forest (or loads cached model)
- Shows evaluation (accuracy, classification report)
- Shows feature importances
- Backtests a simple strategy: go long when model predicts Up
- Provides interactive charts and download of the trained model

Requirements (put this in requirements.txt):
streamlit
pandas
numpy
scikit-learn
matplotlib
joblib

Run:
1) pip install -r requirements.txt
2) streamlit run Streamlit_stock_app.py

Note: This app trains a model on the uploaded data in-browser. For production, pre-train and load a serialized model.
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import io
import joblib
from sklearn.model_selection import TimeSeriesSplit

st.set_page_config(page_title="Stock Up/Down Predictor", layout="wide")

# ---------- Helpers ----------
@st.cache_data
def load_csv(uploaded_file):
    df = pd.read_csv(uploaded_file, parse_dates=["date"]) if uploaded_file is not None else None
    return df

@st.cache_data
def sample_data():
    # small embedded sample data generator if user doesn't upload (synthetic)
    dates = pd.date_range(start="2020-01-01", periods=300, freq='B')
    np.random.seed(42)
    price = np.cumprod(1 + np.random.normal(0, 0.01, size=len(dates))) * 100
    volume = np.random.randint(10000, 50000, size=len(dates))
    df = pd.DataFrame({"date": dates, "open": price * (1 + np.random.normal(0, 0.002, len(dates))),
                       "high": price * (1 + np.random.normal(0.005, 0.01, len(dates))),
                       "low": price * (1 - np.random.normal(0.005, 0.01, len(dates))),
                       "close": price,
                       "adjclose": price,
                       "volume": volume,
                       "ticker": "SAMPLE"})
    return df

@st.cache_data
def feature_engineer(df):
    df = df.copy()
    # Ensure date sorted
    df = df.sort_values("date").reset_index(drop=True)
    # Basic features
    df["Return"] = df["close"].pct_change()
    df["MA_5"] = df["close"].rolling(5).mean()
    df["MA_10"] = df["close"].rolling(10).mean()
    df["Volatility_5"] = df["Return"].rolling(5).std()
    df["Volume_MA_5"] = df["volume"].rolling(5).mean()
    # Target: next day close > today close
    df["Target"] = (df["close"].shift(-1) > df["close"]).astype(int)
    df = df.dropna().reset_index(drop=True)
    return df

@st.cache_data
def train_rf(X, y, params=None):
    if params is None:
        params = {"n_estimators": 200, "random_state": 42}
    model = RandomForestClassifier(**params)
    model.fit(X, y)
    return model

# Backtest calculation (non-leveraged, simple)
def backtest(df_test, preds):
    tmp = df_test.copy().reset_index(drop=True)
    tmp["Pred"] = preds
    # We assume action based on prediction at day t: take return of day t+1, so shift preds by 1
    tmp["Strategy_Return"] = tmp["Pred"].shift(1) * tmp["Return"]
    tmp = tmp.dropna().reset_index(drop=True)
    tmp["Cumulative_Market"] = (1 + tmp["Return"]).cumprod()
    tmp["Cumulative_Strategy"] = (1 + tmp["Strategy_Return"]).cumprod()
    return tmp

# ---------- UI ----------
st.title("📈 Stock Up/Down Predictor — Streamlit Dashboard")
st.markdown("Upload a cleaned stock CSV (date, open, high, low, close, adjclose, volume, ticker) or use sample data.")

col1, col2 = st.columns([2,1])
with col1:
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])   
    if uploaded_file is None:
        use_sample = st.checkbox("Use sample synthetic data", value=True)
    else:
        use_sample = False

with col2:
    st.write("Model settings")
    n_estimators = st.number_input("n_estimators", min_value=50, max_value=1000, value=200, step=50)
    max_depth = st.selectbox("max_depth", options=[None, 5, 10, 15, 20], index=0)
    retrain = st.button("Train / Retrain Model")

# Load data
if uploaded_file is not None:
    df_raw = load_csv(uploaded_file)
elif use_sample:
    df_raw = sample_data()
else:
    st.warning("Please upload a CSV or enable sample data.")
    st.stop()

st.subheader("Preview raw data")
st.dataframe(df_raw.head(10))

# Feature engineering
df = feature_engineer(df_raw)
st.subheader("Feature-engineered data (preview)")
st.dataframe(df[["date","close","Return","MA_5","MA_10","Volatility_5","Volume_MA_5","Target"]].head(10))

features = ["Return","MA_5","MA_10","Volatility_5","Volume_MA_5"]

# Train/test split (time-based 80/20)
split = int(len(df) * 0.8)
train_df = df.iloc[:split]
test_df = df.iloc[split:]
X_train = train_df[features]
y_train = train_df["Target"]
X_test = test_df[features]
y_test = test_df["Target"]

st.write(f"Training rows: {len(train_df)} | Test rows: {len(test_df)}")

# Train model when button clicked or on first load
model = None
model_file = None

if retrain or 'model_obj' not in st.session_state:
    with st.spinner("Training Random Forest..."):
        params = {"n_estimators": int(n_estimators), "max_depth": (None if max_depth is None else int(max_depth)), "random_state": 42}
        model = train_rf(X_train, y_train, params=params)
        st.session_state['model_obj'] = model
        st.success("Model trained and cached in session.")
else:
    model = st.session_state['model_obj']

# Predictions & Evaluation
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

st.subheader("Performance")
col3, col4 = st.columns(2)
with col3:
    st.write("Train Accuracy:", round(accuracy_score(y_train, y_pred_train), 4))
    st.text(classification_report(y_train, y_pred_train))
with col4:
    st.write("Test Accuracy:", round(accuracy_score(y_test, y_pred_test), 4))
    st.text(classification_report(y_test, y_pred_test))

# Feature importances
st.subheader("Feature Importances")
importances = model.feature_importances_
fig1, ax1 = plt.subplots()
ax1.bar(features, importances)
ax1.set_title("Feature Importances (Random Forest)")
ax1.set_ylabel("Importance")
st.pyplot(fig1)

# Backtest
st.subheader("Backtest: Market vs Model Strategy")
back = backtest(test_df, y_pred_test)
fig2, ax2 = plt.subplots(figsize=(10,5))
ax2.plot(back['date'], back['Cumulative_Market'], label='Market (Buy & Hold)')
ax2.plot(back['date'], back['Cumulative_Strategy'], label='Model Strategy')
ax2.legend()
ax2.set_xlabel('Date')
ax2.set_ylabel('Cumulative Return')
st.pyplot(fig2)

# Data and model download
st.subheader("Download")
# Download trained model
model_bytes = io.BytesIO()
joblib.dump(model, model_bytes)
model_bytes.seek(0)

st.download_button(label="Download trained model (.pkl)", data=model_bytes, file_name="random_forest_model.pkl")

# Download processed dataset
csv_bytes = df.to_csv(index=False).encode('utf-8')
st.download_button(label="Download feature-engineered CSV", data=csv_bytes, file_name="INDIAMART_features_streamlit.csv", mime='text/csv')

st.markdown("---")
st.markdown("**Notes / Tips**\n\n- This app trains on the uploaded data in-memory; for reliable production results, train offline on a much larger dataset and version models.\n- The simple strategy shown ignores transaction costs, slippage, and position sizing. Use caution before deploying capital.")

# End of app
