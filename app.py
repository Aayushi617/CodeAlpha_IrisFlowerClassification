import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Title
st.title("🌸 Iris Flower Classification App")

# Load Data
@st.cache_data
def load_data():
    data = pd.read_csv("Iris.csv")
    return data

df = load_data()

# Show Data
st.subheader("🔍 Dataset Preview")
st.dataframe(df.head())

# Visualizations
st.subheader("📊 Data Visualization")
if st.checkbox("Show Pairplot"):
    fig = sns.pairplot(df.drop("Id", axis=1), hue="Species")
    st.pyplot(fig)
    
st.subheader("📌 Count of Each Iris Species")
fig1, ax1 = plt.subplots()
sns.countplot(data=df, x="Species", palette="Set2", ax=ax1)
ax1.set_title("Count of Iris Species")
st.pyplot(fig1)

st.subheader("📈 Feature Correlation Heatmap")
fig2, ax2 = plt.subplots()
sns.heatmap(df.drop("Id", axis=1).corr(), annot=True, cmap="coolwarm", ax=ax2)
st.pyplot(fig2)

st.subheader("📦 Feature Distribution by Species (Box Plot)")
selected_feature = st.selectbox("Choose a feature", ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm'])
fig3, ax3 = plt.subplots()
sns.boxplot(data=df, x='Species', y=selected_feature, palette="Set3", ax=ax3)
ax3.set_title(f"{selected_feature} Distribution by Species")
st.pyplot(fig3)


# Preprocessing
X = df.drop(["Id", "Species"], axis=1)
y = df["Species"]

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Model Training
model = RandomForestClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluation
st.subheader("📈 Model Evaluation")
acc = accuracy_score(y_test, y_pred)
st.write(f"✅ **Accuracy:** {acc:.2f}")
st.text("Classification Report:")
st.text(classification_report(y_test, y_pred))

# User Input Prediction
st.subheader("🔎 Predict Your Own")
sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.0)
sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.0)
petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 4.0)
petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 1.0)

if st.button("Predict"):
    input_data = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]],
                              columns=X.columns)
    prediction = model.predict(input_data)[0]
    st.success(f"The predicted Iris species is: **{prediction}**")
