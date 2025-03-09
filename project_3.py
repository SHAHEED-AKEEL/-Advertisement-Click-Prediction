import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
from imblearn.over_sampling import SMOTE  

def load_data(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file)
        return df
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

def data_summary(df):
    st.subheader("Data Summary")
    st.write("**Shape of the dataset:**", df.shape)
    st.write("**Column Information:**")
    st.write(df.dtypes)
    st.write("**Missing Values:**")
    st.write(df.isnull().sum())
    st.write("**Statistical Summary:**")
    st.write(df.describe())

def data_visualization(df):
    st.subheader("Data Visualization")
    
    # Correlation Heatmap
    st.write("### Correlation Heatmap")
    fig, ax = plt.subplots()
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)
    
    # Feature Distribution
    st.write("### Feature Distributions")
    for col in df.columns[:-1]:
        fig, ax = plt.subplots()
        sns.histplot(df[col], bins=30, kde=True, ax=ax)
        st.pyplot(fig)

def train_model(df, model_choice):
    X = df.iloc[:, :-1]  # Features
    y = df.iloc[:, -1]   # Target
    
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Model Selection
    if model_choice == "Logistic Regression":
        model = LogisticRegression()
    elif model_choice == "Random Forest":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_choice == "SVM":
        model = SVC()
    
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    
    return model, scaler, accuracy, report, X.columns, y_resampled, cm

def main():
    st.title("Advertisement Click Prediction")
    
    uploaded_file = st.file_uploader("Upload your dataset (CSV format)", type=["csv"])
    
    if uploaded_file:
        df = load_data(uploaded_file)
        
        if df is not None:
            st.write("### Preview of Uploaded Data")
            st.write(df.head())
            
            # Data Summary
            data_summary(df)
            
            # Data Visualization
            data_visualization(df)
            
            # Show class distribution before SMOTE
            st.subheader("Class Distribution (Before SMOTE)")
            fig, ax = plt.subplots()
            sns.countplot(x=df.iloc[:, -1], ax=ax)
            st.pyplot(fig)
            
            # Model Selection
            model_choice = st.selectbox("Choose a Model", ["Logistic Regression", "Random Forest", "SVM"])
            model, scaler, accuracy, report, feature_names, y_resampled, cm = train_model(df, model_choice)
            
            # Show class distribution after SMOTE
            st.subheader("Class Distribution (After SMOTE)")
            fig, ax = plt.subplots()
            sns.countplot(x=y_resampled, ax=ax)
            st.pyplot(fig)
            
            # Display Model Performance
            st.write(f"### **Model Accuracy: {accuracy:.2%}**")
            
            # Display Classification Report as Table
            st.subheader("Classification Report")
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df)
            
            # Display Confusion Matrix
            st.subheader("Confusion Matrix")
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_xlabel("Predicted Labels")
            ax.set_ylabel("True Labels")
            st.pyplot(fig)
            
            # Feature Importance (For models with coefficients)
            if model_choice == "Logistic Regression":
                st.subheader("Feature Importance")
                feature_importance = abs(model.coef_[0])
                imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})
                imp_df = imp_df.sort_values(by='Importance', ascending=False)
                st.write(imp_df)
            
            # Prediction Section
            st.subheader("Make a Prediction")
            user_input = []
            
            for col in feature_names:
                value = st.number_input(f"{col}", value=float(df[col].mean()))
                user_input.append(value)
            
            if st.button("Predict Click"):
                user_data = pd.DataFrame([user_input], columns=feature_names)
                user_data_scaled = scaler.transform(user_data)
                prediction = model.predict(user_data_scaled)
                result = "Clicked" if prediction[0] == 1 else "Not Clicked"
                st.write(f"### Prediction: {result}")
    else:
        st.write("Please upload a dataset to proceed.")

if __name__ == "__main__":
    main()
