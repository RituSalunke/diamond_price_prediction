import streamlit as st
import pandas as pd
import joblib

# ==========================
# PAGE CONFIG
# ==========================
st.set_page_config(
    page_title="Diamond Price Prediction",
    page_icon="💎",
    layout="wide"
)

# ==========================
# LOAD DATA
# ==========================
df = pd.read_excel("diamonds.xlsx")

# ==========================
# LOAD SAVED FILES
# ==========================
model = joblib.load("diamond_model.pkl")
scaler = joblib.load("scaler.pkl")

cut_encoder = joblib.load("cut_encoder.pkl")
color_encoder = joblib.load("color_encoder.pkl")
clarity_encoder = joblib.load("clarity_encoder.pkl")

# ==========================
# TITLE
# ==========================
st.title("💎 Diamond Price Prediction")

# ==========================
# DATA PREVIEW
# ==========================
st.subheader("Dataset Preview")
st.dataframe(df.head())

# ==========================
# VALID VALUES
# ==========================
with st.expander("View Valid Values"):

    col1, col2, col3 = st.columns(3)

    with col1:
        st.write("### Cut")
        st.write(list(cut_encoder.classes_))

    with col2:
        st.write("### Color")
        st.write(list(color_encoder.classes_))

    with col3:
        st.write("### Clarity")
        st.write(list(clarity_encoder.classes_))

# ==========================
# FORM
# ==========================
with st.form("prediction_form"):

    st.subheader("Enter Diamond Details")

    carat = st.text_input("Carat")
    cut = st.text_input("Cut")
    color = st.text_input("Color")
    clarity = st.text_input("Clarity")
    depth = st.text_input("Depth")
    table = st.text_input("Table")
    x = st.text_input("X")
    y = st.text_input("Y")
    z = st.text_input("Z")

    submit = st.form_submit_button("Predict Price")

# ==========================
# PREDICTION
# ==========================
if submit:

    try:

        # Encode categorical values
        cut_encoded = cut_encoder.transform([cut])[0]
        color_encoded = color_encoder.transform([color])[0]
        clarity_encoded = clarity_encoder.transform([clarity])[0]

        # Create dataframe
        input_df = pd.DataFrame({
            "carat": [float(carat)],
            "cut": [cut_encoded],
            "color": [color_encoded],
            "clarity": [clarity_encoded],
            "depth": [float(depth)],
            "table": [float(table)],
            "x": [float(x)],
            "y": [float(y)],
            "z": [float(z)]
        })

        # Scale input
        input_scaled = scaler.transform(input_df)

        # Predict
        prediction = model.predict(input_scaled)

        st.success(
            f"💰 Predicted Diamond Price: ${prediction[0]:,.2f}"
        )

    except ValueError:
        st.error(
            "Please enter valid numeric values and valid Cut/Color/Clarity values."
        )

    except Exception as e:
        st.error(f"Error: {e}")
