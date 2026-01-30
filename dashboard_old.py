"""
Streamlit Dashboard for Fraud Detection System
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data_loader import DataLoader
from src.feature_engineer import FraudDataFeatureEngineer, CreditCardFeatureEngineer, prepare_features
from src.preprocessor import DataPreprocessor
from src.model_trainer import ModelTrainer, train_all_models
from src.model_evaluator import ModelEvaluator
from src.predictor import load_predictor
from src.config import data_config, model_config

# Page configuration
st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="🛡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = {}
if 'trained_models' not in st.session_state:
    st.session_state.trained_models = {}
if 'predictions' not in st.session_state:
    st.session_state.predictions = None

# Sidebar
st.sidebar.title("Fraud Detection")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    ["Home", "Data Analysis", "Feature Engineering", 
     "Train Models", "Evaluate Models", "Make Predictions",
     "Dashboard"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Dataset Selection")
dataset_type = st.sidebar.selectbox(
    "Choose Dataset",
    ["Fraud (E-commerce)", "Credit Card"]
)

# Helper functions
@st.cache_data
def load_data(dataset_type):
    """Load data with caching"""
    loader = DataLoader()
    try:
        if dataset_type == "Fraud (E-commerce)":
            df = loader.load_fraud_data()
            ds_name = 'fraud'
        else:
            df = loader.load_creditcard_data()
            ds_name = 'creditcard'
        return df, ds_name
    except FileNotFoundError:
        return None, None

def get_class_distribution(df, target_col='class'):
    """Get class distribution"""
    if target_col not in df.columns:
        target_col = 'Class'
    return df[target_col].value_counts()

# ===== HOME PAGE =====
if page == "Home":
    st.title("Fraud Detection System")
    st.markdown("### Advanced Machine Learning for E-Commerce and Banking")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("**Data Analysis**\nExplore and visualize your fraud detection datasets")
    
    with col2:
        st.info("**Train Models**\nTrain ML models with Random Forest, XGBoost, and more")
    
    with col3:
        st.info("**Predictions**\nMake real-time fraud predictions")
    
    st.markdown("---")
    
    # System Status
    st.subheader("System Status")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Available Models:**")
        models_dir = model_config.model_save_dir
        if models_dir.exists():
            models = list(models_dir.glob("*.joblib"))
            if models:
                for model in models:
                    st.success(f"✓ {model.name}")
            else:
                st.warning("No trained models found")
        else:
            st.warning("Models directory not found")
    
    with col2:
        st.markdown("**Data Files:**")
        data_dir = data_config.fraud_data_path.parent
        if data_dir.exists():
            if data_config.fraud_data_path.exists():
                st.success("✓ Fraud_Data.csv")
            else:
                st.warning("✗ Fraud_Data.csv not found")
            
            if data_config.creditcard_data_path.exists():
                st.success("✓ creditcard.csv")
            else:
                st.warning("✗ creditcard.csv not found")
        else:
            st.error("Data directory not found")
    
    st.markdown("---")
    st.info("👈 Select a page from the sidebar to get started!")

# ===== DATA ANALYSIS PAGE =====
elif page == "Data Analysis":
    st.title("Data Analysis")
    
    df, ds_name = load_data(dataset_type)
    
    if df is None:
        st.error(f"Data file not found for {dataset_type}")
    else:
        st.success(f"Loaded {len(df):,} records")
        
        # Basic stats
        col1, col2, col3, col4 = st.columns(4)
        
        target_col = 'class' if 'class' in df.columns else 'Class'
        
        with col1:
            st.metric("Total Records", f"{len(df):,}")
        with col2:
            st.metric("Features", len(df.columns))
        with col3:
            fraud_rate = df[target_col].mean() * 100
            st.metric("Fraud Rate", f"{fraud_rate:.2f}%")
        with col4:
            fraud_count = df[target_col].sum()
            st.metric("Fraud Cases", f"{fraud_count:,}")
        
        # Class distribution
        st.subheader("Class Distribution")
        class_dist = get_class_distribution(df, target_col)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.pie(
                values=class_dist.values,
                names=['Normal', 'Fraud'],
                title="Class Distribution",
                color_discrete_sequence=['#00CC96', '#EF553B']
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = go.Figure(data=[
                go.Bar(
                    x=['Normal', 'Fraud'],
                    y=class_dist.values,
                    marker_color=['#00CC96', '#EF553B']
                )
            ])
            fig.update_layout(title="Class Counts", yaxis_title="Count")
            st.plotly_chart(fig, use_container_width=True)
        
        # Data preview
        st.subheader("Data Preview")
        st.dataframe(df.head(100), use_container_width=True)
        
        # Statistical summary
        with st.expander("Statistical Summary"):
            st.dataframe(df.describe(), use_container_width=True)
        
        # Missing values
        with st.expander("Missing Values Analysis"):
            missing = df.isnull().sum()
            if missing.any():
                st.dataframe(missing[missing > 0], use_container_width=True)
            else:
                st.success("No missing values found!")
        
        # Feature distributions
        if st.checkbox("Show Feature Distributions"):
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if target_col in numeric_cols:
                numeric_cols.remove(target_col)
            
            selected_features = st.multiselect(
                "Select features to visualize",
                numeric_cols,
                default=numeric_cols[:4] if len(numeric_cols) >= 4 else numeric_cols
            )
            
            if selected_features:
                for feature in selected_features:
                    fig = px.histogram(
                        df,
                        x=feature,
                        color=target_col,
                        title=f"Distribution of {feature}",
                        color_discrete_sequence=['#00CC96', '#EF553B'],
                        marginal="box"
                    )
                    st.plotly_chart(fig, use_container_width=True)

# ===== FEATURE ENGINEERING PAGE =====
elif page == "Feature Engineering":
    st.title("Feature Engineering")
    
    df, ds_name = load_data(dataset_type)
    
    if df is None:
        st.error(f"Data file not found for {dataset_type}")
    else:
        st.info(f"Original dataset: {df.shape[0]} rows × {df.shape[1]} columns")
        
        if st.button("Engineer Features", type="primary"):
            with st.spinner("Engineering features..."):
                try:
                    if ds_name == 'fraud':
                        engineer = FraudDataFeatureEngineer()
                    else:
                        engineer = CreditCardFeatureEngineer()
                    
                    df_engineered = engineer.engineer_features(df.copy(), fit=True)
                    
                    st.success(f"Feature engineering complete!")
                    st.info(f"Engineered dataset: {df_engineered.shape[0]} rows × {df_engineered.shape[1]} columns")
                    
                    new_features = [col for col in df_engineered.columns if col not in df.columns]
                    st.success(f"Created {len(new_features)} new features")
                    
                    # Show new features
                    with st.expander("New Features Created"):
                        for i, feature in enumerate(new_features, 1):
                            st.write(f"{i}. {feature}")
                    
                    # Prepare for modeling
                    target_col = 'class' if 'class' in df.columns else 'Class'
                    drop_cols = data_config.fraud_drop_columns if ds_name == 'fraud' else data_config.creditcard_drop_columns
                    
                    df_processed = prepare_features(
                        df.copy(),
                        dataset_type=ds_name,
                        drop_columns=drop_cols,
                        fit=True
                    )
                    
                    df_processed[data_config.target_column] = df[target_col].values
                    
                    # Save to session state
                    st.session_state.processed_data[ds_name] = df_processed
                    
                    # Save to disk
                    loader = DataLoader()
                    loader.save_processed_data(df_processed, ds_name)
                    
                    st.success("Processed data saved!")
                    
                    # Preview
                    st.subheader("Preview of Processed Data")
                    st.dataframe(df_processed.head(), use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")

# ===== TRAIN MODELS PAGE =====
elif page == "Train Models":
    st.title("Train Models")
    
    # Check if processed data exists
    df, ds_name = load_data(dataset_type)
    
    if df is None:
        st.error(f"Data file not found for {dataset_type}")
    else:
        # Model selection
        st.subheader("Model Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            selected_models = st.multiselect(
                "Select Models to Train",
                ["logistic_regression", "random_forest", "xgboost"],
                default=["random_forest"]
            )
        
        with col2:
            use_smote = st.checkbox("Use SMOTE (Recommended for imbalanced data)", value=True)
            perform_cv = st.checkbox("Perform Cross-Validation", value=True)
        
        if st.button("Start Training", type="primary"):
            if not selected_models:
                st.error("Please select at least one model")
            else:
                try:
                    # Load or prepare data
                    if ds_name in st.session_state.processed_data:
                        df_processed = st.session_state.processed_data[ds_name]
                    else:
                        with st.spinner("Loading processed data..."):
                            loader = DataLoader()
                            df_processed = loader.load_processed_data(ds_name)
                    
                    # Split data
                    from sklearn.model_selection import train_test_split
                    
                    y = df_processed[data_config.target_column]
                    X = df_processed.drop(columns=[data_config.target_column])
                    
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y,
                        test_size=data_config.test_size,
                        random_state=data_config.random_state,
                        stratify=y
                    )
                    
                    st.info(f"Training: {len(X_train):,} samples | Testing: {len(X_test):,} samples")
                    
                    # Preprocess
                    with st.spinner("Preprocessing data..."):
                        preprocessor = DataPreprocessor(use_smote=use_smote)
                        X_train_proc, y_train_proc = preprocessor.fit_transform(X_train, y_train)
                        X_test_proc = preprocessor.transform(X_test)
                    
                    if use_smote:
                        st.success(f"SMOTE applied: {len(X_train_proc):,} training samples")
                    
                    # Train models
                    results = {}
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for idx, model_type in enumerate(selected_models):
                        status_text.text(f"Training {model_type}...")
                        
                        trainer = ModelTrainer(model_type=model_type)
                        metrics = trainer.train(X_train_proc, y_train_proc, validate=perform_cv)
                        
                        # Evaluate
                        evaluator = ModelEvaluator()
                        y_pred_proba = trainer.predict_proba(X_test_proc)
                        y_pred = trainer.predict(X_test_proc)
                        
                        eval_metrics = evaluator.evaluate(y_test, y_pred_proba, y_pred)
                        
                        results[model_type] = {
                            'trainer': trainer,
                            'metrics': eval_metrics,
                            'train_metrics': metrics
                        }
                        
                        # Save model
                        trainer.save_model(dataset_name=ds_name)
                        
                        progress_bar.progress((idx + 1) / len(selected_models))
                    
                    status_text.text("Training complete!")
                    
                    # Store in session state
                    st.session_state.trained_models[ds_name] = results
                    st.session_state.test_data = (X_test_proc, y_test)
                    
                    # Display results
                    st.success("All models trained successfully!")
                    
                    st.subheader("Model Performance")
                    
                    # Create comparison table
                    comparison_data = []
                    for model_name, result in results.items():
                        metrics = result['metrics']
                        comparison_data.append({
                            'Model': model_name.replace('_', ' ').title(),
                            'Accuracy': f"{metrics['accuracy']:.4f}",
                            'Precision': f"{metrics['precision']:.4f}",
                            'Recall': f"{metrics['recall']:.4f}",
                            'F1 Score': f"{metrics['f1_score']:.4f}",
                            'ROC AUC': f"{metrics['roc_auc']:.4f}"
                        })
                    
                    comparison_df = pd.DataFrame(comparison_data)
                    st.dataframe(comparison_df, use_container_width=True)
                    
                    # Visualize comparison
                    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
                    fig = go.Figure()
                    
                    for model_name, result in results.items():
                        metrics = result['metrics']
                        values = [metrics[m] for m in metrics_to_plot]
                        
                        fig.add_trace(go.Scatterpolar(
                            r=values,
                            theta=[m.replace('_', ' ').title() for m in metrics_to_plot],
                            fill='toself',
                            name=model_name.replace('_', ' ').title()
                        ))
                    
                    fig.update_layout(
                        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                        showlegend=True,
                        title="Model Performance Comparison"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                except FileNotFoundError:
                    st.error("Processed data not found. Please run Feature Engineering first!")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())

# ===== MODEL EVALUATION PAGE =====
elif page == "Evaluate Models":
    st.title("Model Evaluation")
    
    df, ds_name = load_data(dataset_type)
    
    if ds_name not in st.session_state.trained_models:
        st.warning("No trained models found. Please train models first!")
    else:
        results = st.session_state.trained_models[ds_name]
        X_test_proc, y_test = st.session_state.test_data
        
        # Model selection
        model_names = list(results.keys())
        selected_model = st.selectbox(
            "Select Model to Evaluate",
            model_names,
            format_func=lambda x: x.replace('_', ' ').title()
        )
        
        if selected_model:
            result = results[selected_model]
            trainer = result['trainer']
            metrics = result['metrics']
            
            # Metrics display
            st.subheader("Performance Metrics")
            
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric("Accuracy", f"{metrics['accuracy']:.4f}")
            with col2:
                st.metric("Precision", f"{metrics['precision']:.4f}")
            with col3:
                st.metric("Recall", f"{metrics['recall']:.4f}")
            with col4:
                st.metric("F1 Score", f"{metrics['f1_score']:.4f}")
            with col5:
                st.metric("ROC AUC", f"{metrics['roc_auc']:.4f}")
            
            # Confusion Matrix
            st.subheader("Confusion Matrix")
            
            cm = np.array(metrics['confusion_matrix'])
            
            fig = go.Figure(data=go.Heatmap(
                z=cm,
                x=['Normal', 'Fraud'],
                y=['Normal', 'Fraud'],
                colorscale='Blues',
                text=cm,
                texttemplate='%{text}',
                textfont={"size": 20}
            ))
            
            fig.update_layout(
                title="Confusion Matrix",
                xaxis_title="Predicted",
                yaxis_title="Actual",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # ROC Curve
            st.subheader("ROC Curve")
            
            from sklearn.metrics import roc_curve
            
            y_pred_proba = trainer.predict_proba(X_test_proc)
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1])
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=fpr, y=tpr,
                mode='lines',
                name=f'ROC (AUC = {metrics["roc_auc"]:.4f})',
                line=dict(color='#636EFA', width=3)
            ))
            fig.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1],
                mode='lines',
                name='Random',
                line=dict(color='gray', width=2, dash='dash')
            ))
            
            fig.update_layout(
                title="ROC Curve",
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Feature Importance
            if selected_model in ['random_forest', 'xgboost']:
                st.subheader("Feature Importance")
                
                importance_df = trainer.get_feature_importance()
                
                if importance_df is not None:
                    top_n = st.slider("Number of features to display", 5, 30, 15)
                    
                    top_features = importance_df.head(top_n)
                    
                    fig = go.Figure(go.Bar(
                        x=top_features['importance'],
                        y=top_features['feature'],
                        orientation='h',
                        marker_color='#636EFA'
                    ))
                    
                    fig.update_layout(
                        title=f"Top {top_n} Important Features",
                        xaxis_title="Importance",
                        yaxis_title="Feature",
                        height=600,
                        yaxis={'categoryorder': 'total ascending'}
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)

# ===== MAKE PREDICTIONS PAGE =====
elif page == "Make Predictions":
    st.title("Make Predictions")
    
    df, ds_name = load_data(dataset_type)
    
    # Check for trained models
    models_dir = model_config.model_save_dir
    available_models = list(models_dir.glob(f"*{ds_name}_model.joblib")) if models_dir.exists() else []
    
    if not available_models:
        st.error("No trained models found. Please train a model first!")
    else:
        # Model selection
        model_options = [m.stem.replace(f'_{ds_name}_model', '') for m in available_models]
        selected_model = st.selectbox(
            "Select Model",
            model_options,
            format_func=lambda x: x.replace('_', ' ').title()
        )
        
        threshold = st.slider("Classification Threshold", 0.0, 1.0, 0.5, 0.05)
        
        st.markdown("---")
        
        # Prediction mode
        prediction_mode = st.radio(
            "Prediction Mode",
            ["Single Transaction", "Batch Prediction", "Upload CSV"]
        )
        
        if prediction_mode == "Single Transaction":
            st.subheader("Enter Transaction Details")
            
            # Create input form based on dataset type
            if ds_name == 'fraud':
                col1, col2 = st.columns(2)
                
                with col1:
                    purchase_value = st.number_input("Purchase Value", min_value=0.0, value=150.0)
                    age = st.number_input("Age", min_value=18, max_value=120, value=35)
                    hour_of_day = st.number_input("Hour of Day", min_value=0, max_value=23, value=14)
                
                with col2:
                    day_of_week = st.number_input("Day of Week (0=Mon, 6=Sun)", min_value=0, max_value=6, value=2)
                    time_since_signup = st.number_input("Hours Since Signup", min_value=0.0, value=48.5)
                
                if st.button("Predict", type="primary"):
                    try:
                        from datetime import datetime, timedelta
                        from sklearn.model_selection import train_test_split
                        
                        # Create proper timestamps
                        purchase_time = datetime.now().replace(hour=hour_of_day, minute=0, second=0, microsecond=0)
                        days_to_add = (day_of_week - purchase_time.weekday()) % 7
                        purchase_time = purchase_time + timedelta(days=days_to_add)
                        signup_time = purchase_time - timedelta(hours=time_since_signup)
                        
                        # Get a sample row for structure
                        sample_row = df.iloc[0:1].copy()
                        
                        # Update with user inputs
                        sample_row['purchase_value'] = purchase_value
                        sample_row['age'] = age
                        sample_row['signup_time'] = signup_time.strftime('%Y-%m-%d %H:%M:%S')
                        sample_row['purchase_time'] = purchase_time.strftime('%Y-%m-%d %H:%M:%S')
                        
                        # Load or prepare processed data for fitting preprocessor
                        if ds_name in st.session_state.processed_data:
                            df_processed = st.session_state.processed_data[ds_name]
                        else:
                            # Load processed data
                            loader = DataLoader()
                            df_processed = loader.load_processed_data(ds_name)
                        
                        # Split data to fit preprocessor (same as training)
                        y = df_processed[data_config.target_column]
                        X = df_processed.drop(columns=[data_config.target_column])
                        
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y,
                            test_size=data_config.test_size,
                            random_state=data_config.random_state,
                            stratify=y
                        )
                        
                        # Fit preprocessor on training data
                        preprocessor = DataPreprocessor(use_smote=False)
                        X_train_proc, _ = preprocessor.fit_transform(X_train, y_train)
                        
                        # Now prepare the user's input sample
                        sample_prepared = prepare_features(
                            sample_row.copy(),
                            dataset_type='fraud',
                            drop_columns=data_config.fraud_drop_columns,
                            fit=False
                        )
                        
                        # Remove target column if present
                        target_cols = ['class', 'Class']
                        for col in target_cols:
                            if col in sample_prepared.columns:
                                sample_prepared = sample_prepared.drop(columns=[col])
                        
                        # Align columns with training data (add missing columns with 0s)
                        training_columns = X_train_proc.columns
                        for col in training_columns:
                            if col not in sample_prepared.columns:
                                sample_prepared[col] = 0
                        
                        # Ensure column order matches training data
                        sample_prepared = sample_prepared[training_columns]
                        
                        # Transform user input with fitted preprocessor
                        sample_transformed = preprocessor.transform(sample_prepared)
                        
                        # Load model trainer directly
                        model_path = model_config.model_save_dir / f"{selected_model}_{ds_name}_model.joblib"
                        trainer = ModelTrainer()
                        trainer.load_model(model_path)
                        
                        with st.spinner("Making prediction..."):
                            # Get predictions
                            y_pred_proba = trainer.predict_proba(sample_transformed)
                            y_pred = (y_pred_proba[:, 1] >= threshold).astype(int)
                            
                            fraud_prob = float(y_pred_proba[0, 1])
                            is_fraud = bool(y_pred[0])
                            confidence = max(fraud_prob, 1 - fraud_prob)
                            
                            # Determine risk level
                            if fraud_prob < 0.3:
                                risk_level = 'low'
                            elif fraud_prob < 0.6:
                                risk_level = 'medium'
                            elif fraud_prob < 0.8:
                                risk_level = 'high'
                            else:
                                risk_level = 'critical'
                        
                        # Display result
                        if is_fraud:
                            st.error("FRAUD DETECTED!")
                        else:
                            st.success("Transaction appears legitimate")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Fraud Probability", f"{fraud_prob:.2%}")
                        with col2:
                            st.metric("Confidence", f"{confidence:.2%}")
                        with col3:
                            st.metric("Risk Level", f"{risk_level.title()}")
                        
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())
            
            else:  # Credit card
                st.info("Credit card predictions require V1-V28 features. Please use CSV upload for full predictions.")
        
        elif prediction_mode == "Batch Prediction":
            if df is not None:
                st.info("Using loaded dataset for batch prediction")
                
                num_samples = st.number_input("Number of samples to predict", 1, min(1000, len(df)), 10)
                
                if st.button("Predict Batch", type="primary"):
                    try:
                        predictor = load_predictor(selected_model, ds_name)
                        
                        sample_data = df.sample(n=num_samples, random_state=42)
                        
                        with st.spinner(f"Making predictions on {num_samples} samples..."):
                            predictions = predictor.predict_batch(
                                sample_data,
                                threshold=threshold,
                                include_details=True
                            )
                        
                        # Summary
                        fraud_count = predictions['is_fraud'].sum()
                        fraud_pct = (fraud_count / len(predictions)) * 100
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Total Predictions", len(predictions))
                        with col2:
                            st.metric("Fraud Detected", fraud_count)
                        with col3:
                            st.metric("Fraud Rate", f"{fraud_pct:.2f}%")
                        
                        # Results table
                        st.subheader("Prediction Results")
                        results_df = pd.concat([sample_data.reset_index(drop=True), predictions], axis=1)
                        st.dataframe(results_df, use_container_width=True)
                        
                        # Store in session state
                        st.session_state.predictions = results_df
                        
                        # Download button
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="Download Results",
                            data=csv,
                            file_name=f"predictions_{ds_name}.csv",
                            mime="text/csv"
                        )
                        
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
        
        elif prediction_mode == "Upload CSV":
            uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
            
            if uploaded_file is not None:
                try:
                    upload_df = pd.read_csv(uploaded_file)
                    st.success(f"Loaded {len(upload_df)} records")
                    
                    st.dataframe(upload_df.head(), use_container_width=True)
                    
                    if st.button("Predict", type="primary"):
                        try:
                            predictor = load_predictor(selected_model, ds_name)
                            
                            with st.spinner("Making predictions..."):
                                predictions = predictor.predict_batch(
                                    upload_df,
                                    threshold=threshold,
                                    include_details=True
                                )
                            
                            fraud_count = predictions['is_fraud'].sum()
                            
                            st.success(f"Predictions complete! Found {fraud_count} potential fraud cases")
                            
                            results_df = pd.concat([upload_df.reset_index(drop=True), predictions], axis=1)
                            st.dataframe(results_df, use_container_width=True)
                            
                            csv = results_df.to_csv(index=False)
                            st.download_button(
                                label="Download Results",
                                data=csv,
                                file_name=f"predictions_{ds_name}.csv",
                                mime="text/csv"
                            )
                            
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
                            import traceback
                            st.code(traceback.format_exc())
                            
                except Exception as e:
                    st.error(f"Error reading file: {str(e)}")

# ===== DASHBOARD PAGE =====
elif page == "Dashboard":
    st.title("Analytics Dashboard")
    
    df, ds_name = load_data(dataset_type)
    
    if df is None:
        st.error("No data loaded")
    else:
        target_col = 'class' if 'class' in df.columns else 'Class'
        
        # Overview metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Transactions", f"{len(df):,}")
        with col2:
            fraud_count = df[target_col].sum()
            st.metric("Fraud Cases", f"{fraud_count:,}")
        with col3:
            fraud_rate = (fraud_count / len(df)) * 100
            st.metric("Fraud Rate", f"{fraud_rate:.3f}%")
        with col4:
            normal_count = len(df) - fraud_count
            st.metric("Normal Cases", f"{normal_count:,}")
        
        st.markdown("---")
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # Class distribution pie chart
            class_dist = df[target_col].value_counts()
            fig = px.pie(
                values=class_dist.values,
                names=['Normal', 'Fraud'],
                title="Transaction Distribution",
                color_discrete_sequence=['#00CC96', '#EF553B']
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Class distribution bar chart
            fig = go.Figure(data=[
                go.Bar(
                    x=['Normal', 'Fraud'],
                    y=class_dist.values,
                    marker_color=['#00CC96', '#EF553B'],
                    text=class_dist.values,
                    textposition='auto'
                )
            ])
            fig.update_layout(title="Transaction Counts", yaxis_title="Count")
            st.plotly_chart(fig, use_container_width=True)
        
        # Feature analysis
        if ds_name == 'fraud' and 'purchase_value' in df.columns:
            st.subheader("Purchase Value Analysis")
            
            fig = px.box(
                df,
                x=target_col,
                y='purchase_value',
                color=target_col,
                title="Purchase Value by Class",
                labels={target_col: 'Class', 'purchase_value': 'Purchase Value'},
                color_discrete_sequence=['#00CC96', '#EF553B']
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Predictions dashboard (if available)
        if st.session_state.predictions is not None:
            st.markdown("---")
            st.subheader("Recent Predictions Analysis")
            
            pred_df = st.session_state.predictions
            
            col1, col2 = st.columns(2)
            
            with col1:
                risk_dist = pred_df['risk_level'].value_counts()
                fig = px.pie(
                    values=risk_dist.values,
                    names=risk_dist.index,
                    title="Risk Level Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.histogram(
                    pred_df,
                    x='fraud_probability',
                    nbins=50,
                    title="Fraud Probability Distribution",
                    labels={'fraud_probability': 'Fraud Probability'}
                )
                st.plotly_chart(fig, use_container_width=True)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info(
    "Fraud Detection System v1.0\n\n"
    "Built with Streamlit, scikit-learn, and XGBoost\n\n"
    "© 2026 Adey Innovations Inc."
)
