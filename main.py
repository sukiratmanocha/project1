import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
# Set page config
st.set_page_config(page_title="Disease Prediction System", page_icon="🩺", layout="wide")

# --- Data Loading and Preprocessing ---
@st.cache_data
def load_and_preprocess_data():
    # Load dataset
    df = pd.read_csv(r"waterborne_diseases_dataset (1).csv")
    
    # Check for data imbalance
    class_counts = df["prognosis"].value_counts()
    
    return df, class_counts

# --- Model Training ---
@st.cache_data
def train_model(_df):
    # Features and target
    X = _df.drop(columns=["prognosis"])
    all_symptoms = list(X.columns)
    symptoms_ascending = sorted(all_symptoms)
    y = _df["prognosis"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=500,
        random_state=42,
        class_weight="balanced",
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2
    )

    model.fit(X_train, y_train)

    train_accuracy = model.score(X_train, y_train)
    test_accuracy = model.score(X_test, y_test)

    return (
        model,
        symptoms_ascending,
        train_accuracy,
        test_accuracy,
        y_test,
        model.predict(X_test),
    )

# --- Main App ---
def main():
    st.title("🩺 Disease Prediction System")
    st.write("Select your symptoms and get disease predictions with confidence scores.")
    
    # Load data and train model
    try:
        df, class_counts = load_and_preprocess_data()
        model, feature_columns, train_acc, test_acc, y_test, y_pred = train_model(df)
        
        # Display total symptoms count
        st.info(f"📋 **Total Symptoms Available:** {len(feature_columns)}")
        
        # Sidebar with model info
        with st.sidebar:
            st.header("📊 Model Information")
            st.write(f"**Total Samples:** {len(df)}")
            
            st.subheader("Samples present for each disease:")
            for disease, count in class_counts.items():
                st.write(f"- {disease}: {count}")
            
            # Prediction threshold slider
            threshold = st.slider("Prediction Threshold", 0.1, 0.9, 0.3, 0.05)
        
        # Main prediction interface
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("🔍 Select Your Symptoms")
            
            # Create symptom checklist in columns for better layout
            num_cols = 3
            cols = st.columns(num_cols)
            symptom_inputs = {}
            
            # Use sorted symptoms for display
            for i, symptom in enumerate(feature_columns):
                with cols[i % num_cols]:
                    symptom_inputs[symptom] = st.checkbox(
                        symptom.replace('_', ' ').title(), 
                        value=False,
                        key=symptom
                    )
        
        with col2:
            st.subheader("🎯 Prediction Results")
            
            # Show selected symptoms
            selected_symptoms = [sym for sym, val in symptom_inputs.items() if val]
            if selected_symptoms:
                st.write("**Selected Symptoms:**")
                for symptom in selected_symptoms:
                    st.write(f"✓ {symptom.replace('_', ' ').title()}")
            else:
                st.info("No symptoms selected yet")
        
        # Predict button
        if st.button("🔬 Predict Disease", type="primary", use_container_width=True):
            if not any(symptom_inputs.values()):
                st.warning("⚠️ Please select at least one symptom before predicting.")
            else:
                original_columns = df.drop(columns=["prognosis"]).columns
                input_df = pd.DataFrame([symptom_inputs]).reindex(
                    columns=original_columns, fill_value=0
                )
                feature_importance = model.feature_importances_
                selected_importance = {
                    sym: feature_importance[list(original_columns).index(sym)]
                    for sym, val in symptom_inputs.items()
                    if val
                }
                # Get probabilities
                probs = model.predict_proba(input_df)[0]
        
                results = [
                    (disease, p)
                    for disease, p in zip(model.classes_, probs)
                    if p >= threshold
                ]
        
                if results:
                    st.success(f"🎯 **Diseases with probability ≥ {threshold:.0%}:**")
                
                    results.sort(key=lambda x: x[1], reverse=True)
                
                    for i, (disease, prob) in enumerate(results, start=1):
                        risk = (
                            "🔴 High" if prob > 0.7
                            else "🟡 Medium" if prob > 0.4
                            else "🟢 Low"
                        )
                
                        with st.expander(
                            f"🦠 Predicted Disease: {disease} — {prob:.2%} ({risk})",
                            expanded=(i == 1)
                        ):
                            st.write(f"**Confidence:** {prob:.2%}")
                            st.progress(prob)
                
                            if prob > 0.7:
                                st.error("⚠️ High probability detected. Please consult a healthcare professional immediately.")
                            elif prob > 0.4:
                                st.warning("⚠️ Moderate probability. Consider consulting a healthcare professional.")
                            else:
                                st.info("ℹ️ Low probability. Monitor symptoms and consult if they persist.")
                
                    # ✅ Symptom relevance belongs here
                    if selected_importance:
                        st.subheader("📊 Symptom Relevance")
                
                        importance_df = pd.DataFrame(
                            list(selected_importance.items()),
                            columns=["Symptom", "Importance"]
                        ).sort_values("Importance", ascending=False)
                
                        for _, row in importance_df.iterrows():
                            st.write(
                                f"- {row['Symptom'].replace('_', ' ').title()}: {row['Importance']:.3f}"
                            )
                
                else:
                    st.info(f"ℹ️ No disease detected with probability ≥ {threshold:.0%}.")

                            
        # Additional insights
        with st.expander("🔍 View Dataset Insights"):
            st.write("**Dataset Overview:**")
            st.write(f"- Total records: {len(df)}")
            st.write(f"- Number of symptoms: {len(feature_columns)}")
            st.write(f"- Number of diseases: {len(df['prognosis'].unique())}")
            
            st.write("**All Symptoms (Alphabetical Order):**")
            symptoms_text = ", ".join([symptom.replace('_', ' ').title() for symptom in feature_columns])
            st.text_area("Complete Symptom List:", symptoms_text, height=100)
            
            st.write("**Top 5 Most Common Diseases:**")
            st.bar_chart(pd.DataFrame({
                "count": class_counts.head(5).values
            }, index=class_counts.head(5).index.astype(str)))

    
    except FileNotFoundError:
        st.error("❌ Dataset file not found. Please check the file path.")
    except Exception as e:
        st.error(f"❌ An error occurred: {str(e)}")

if __name__ == "__main__":

    main()
















