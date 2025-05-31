import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Streamlit config
st.set_page_config(page_title="Scratch Linear Regression", layout="centered")

# Custom CSS styling
st.markdown("""
<style>
body {
    background-color: #002147;
}
h1 {
    color: #FFFFFF;
}
.sidebar .sidebar-content {
    background-color: #002147;
}
</style>
""", unsafe_allow_html=True)

st.title("📈 Linear Regression (Scratch Implementation)")

# --- Scratch Linear Regression Class ---

class LinearRegression:
    def __init__(self, alpha=1e-3, iters=1000):
        self.alpha = alpha
        self.iters = iters
        self.w = None
        self.b = None

    def _init_params(self):
        """Initialize weights and bias to zeros."""
        self.w = np.zeros(self.n)
        self.b = 0

    def update_param(self, dw, db):
        """Update weights and bias using gradients."""
        self.w -= self.alpha * dw
        self.b -= self.alpha * db

    def predict(self, X):
        """Return predicted values for input X."""
        return np.dot(X, self.w) + self.b

    def gradients(self, X, y, y_pred):
        """Compute gradients for weights and bias."""
        e = y_pred - y
        dw = (1 / self.m) * np.dot(X.T, e)
        db = (1 / self.m) * np.sum(e)
        return dw, db

    def fit(self, X, y):
        """Train the model using gradient descent."""
        self.m, self.n = X.shape
        self._init_params()

        for _ in range(self.iters):
            y_pred = self.predict(X)
            dw, db = self.gradients(X, y, y_pred)
            self.update_param(dw, db)

    def final_predict(self, X):
        """Return final predictions after training."""
        return self.predict(X)

# --- Streamlit App Logic ---
uploaded_file = st.file_uploader("📂 Upload CSV", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded!")
    st.dataframe(df.head())

    columns = df.columns.tolist()
    features = st.multiselect("🧮 Select Feature Columns (X)", options=columns)
    target = st.selectbox("🎯 Select Target Column (y)", options=[col for col in columns if col not in features])

    if features and target:
        X = df[features].values
        y = df[target].values

        alpha = st.slider("Learning Rate (alpha)", 1e-4, 1e-1, value=1e-2, step=1e-4, format="%.4f")
        iters = st.slider("Iterations", 100, 5000, step=100, value=1000)

        # Train model
        model = LinearRegression(alpha=alpha, iters=iters)
        model.fit(X, y)
        y_pred = model.final_predict(X)

        # Display results
        st.markdown("### 🔍 Model Parameters")
        st.write(f"**Weights:** {model.w}")
        st.write(f"**Bias:** {model.b:.4f}")

        mse = np.mean((y - y_pred) ** 2)
        r2 = 1 - (np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2))

        st.markdown("### 📊 Performance Metrics")
        st.write(f"**MSE:** {mse:.4f}")
        st.write(f"**R² Score:** {r2:.4f}")

        # Plot (only if single feature)
        if X.shape[1] == 1:
            fig, ax = plt.subplots()
            sns.scatterplot(x=X.flatten(), y=y, color='blue', label='Actual', ax=ax)
            sns.lineplot(x=X.flatten(), y=y_pred, color='red', label='Predicted', ax=ax)
            ax.set_xlabel(features[0])
            ax.set_ylabel(target)
            ax.set_title("📉 Regression Line vs Data")
            st.pyplot(fig)

        # Download predictions
        result_df = df.copy()
        result_df['Predicted'] = y_pred
        csv = result_df.to_csv(index=False)
        st.download_button("📥 Download Predictions CSV", csv, "predictions.csv", "text/csv")
