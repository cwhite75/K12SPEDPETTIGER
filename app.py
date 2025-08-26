import streamlit as st
import anthropic
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import json
from datetime import datetime, timedelta
import io
import base64
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="K-12 Educational Analytics & Strategy AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Claude AI
@st.cache_resource
def initialize_claude():
    try:
        return anthropic.Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])
    except:
        st.error("🔑 Please add your Anthropic API key in Streamlit secrets!")
        st.stop()

claude_client = initialize_claude()

# Header
st.title("📊 K-12 Educational Analytics & Strategy AI")
st.markdown("*Advanced AI-powered data analysis, predictive analytics, and evidence-based teaching strategies*")

# Initialize session state
if "uploaded_data" not in st.session_state:
    st.session_state.uploaded_data = {}
if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = {}
if "predictive_models" not in st.session_state:
    st.session_state.predictive_models = {}
if "presentations" not in st.session_state:
    st.session_state.presentations = {}

# Advanced AI Response System
def get_educational_ai_response(query, context_type="general", data_context=None):
    """Advanced AI response system with educational expertise"""
    try:
        system_prompts = {
            "data_analysis": """You are a Senior Educational Data Analyst and Research Scientist with expertise in:
            - K-12 assessment data interpretation
            - Statistical analysis and predictive modeling
            - Evidence-based educational research
            - Student performance analytics
            - Curriculum effectiveness measurement
            - Learning gap identification
            - Intervention strategy development
            
            Provide scientifically rigorous analysis with specific recommendations backed by educational research.""",
            
            "teaching_strategies": """You are a Master Educational Strategist and Curriculum Expert specializing in:
            - Research-based teaching methodologies
            - Differentiated instruction strategies
            - Multi-tiered intervention systems (MTSS/RTI)
            - Culturally responsive teaching practices
            - Technology integration best practices
            - Assessment-driven instruction
            - Data-informed decision making
            
            Provide practical, research-backed strategies with implementation steps and success metrics.""",
            
            "executive_presentation": """You are a Senior Educational Consultant preparing executive-level presentations with expertise in:
            - District-wide performance analysis
            - Strategic planning and goal setting
            - Budget allocation optimization
            - Policy recommendation development
            - Stakeholder communication
            - ROI analysis for educational programs
            - Compliance and accountability reporting
            
            Create comprehensive, data-driven presentations suitable for superintendents and school boards.""",
            
            "predictive_analytics": """You are a Senior Educational Data Scientist specializing in:
            - Predictive modeling for student outcomes
            - Early warning system development
            - Risk factor identification
            - Intervention effectiveness prediction
            - Resource allocation optimization
            - Long-term trend analysis
            - Machine learning applications in education
            
            Provide actionable insights with confidence intervals and implementation recommendations."""
        }
        
        prompt = system_prompts.get(context_type, system_prompts["data_analysis"])
        
        if data_context:
            query += f"\n\nData Context: {data_context}"
        
        message = claude_client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=4000,
            temperature=0.2,
            messages=[
                {"role": "user", "content": f"{prompt}\n\nQuery: {query}"}
            ]
        )
        return message.content[0].text
    except Exception as e:
        return f"AI Analysis Error: {e}"

# Data Processing Functions
def process_assessment_data(df):
    """Advanced assessment data processing and analysis"""
    try:
        # Basic statistics
        stats_summary = df.describe()
        
        # Performance categories
        if 'score' in df.columns:
            df['performance_level'] = pd.cut(df['score'], 
                                           bins=[0, 60, 70, 80, 90, 100],
                                           labels=['Below Basic', 'Basic', 'Proficient', 'Advanced', 'Exceptional'])
        
        # Grade level analysis
        if 'grade' in df.columns:
            grade_analysis = df.groupby('grade').agg({
                'score': ['mean', 'median', 'std', 'count']
            }).round(2)
        
        # Subject area analysis
        if 'subject' in df.columns:
            subject_analysis = df.groupby('subject').agg({
                'score': ['mean', 'median', 'std', 'count']
            }).round(2)
        
        return {
            'summary_stats': stats_summary,
            'grade_analysis': grade_analysis if 'grade' in df.columns else None,
            'subject_analysis': subject_analysis if 'subject' in df.columns else None,
            'processed_data': df
        }
    except Exception as e:
        st.error(f"Data processing error: {e}")
        return None

def create_predictive_model(df, target_column='score', test_size=0.2):
    """Create predictive models for student performance"""
    try:
        # Prepare features
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_column in numeric_columns:
            numeric_columns.remove(target_column)
        
        X = df[numeric_columns].fillna(df[numeric_columns].mean())
        y = df[target_column].fillna(df[target_column].mean())
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        # Train models
        models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
        }
        
        results = {}
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            results[name] = {
                'model': model,
                'mse': mean_squared_error(y_test, y_pred),
                'r2': r2_score(y_test, y_pred),
                'predictions': y_pred,
                'actual': y_test,
                'feature_importance': model.feature_importances_ if hasattr(model, 'feature_importances_') else None
            }
        
        return results, X.columns.tolist()
    except Exception as e:
        st.error(f"Predictive modeling error: {e}")
        return None, None

def generate_visualizations(df, analysis_type="comprehensive"):
    """Generate comprehensive data visualizations"""
    figures = {}
    
    try:
        if 'score' in df.columns:
            # Score distribution
            fig1 = px.histogram(df, x='score', nbins=20, title='Score Distribution')
            figures['score_distribution'] = fig1
            
            # Box plot by grade
            if 'grade' in df.columns:
                fig2 = px.box(df, x='grade', y='score', title='Score Distribution by Grade')
                figures['grade_boxplot'] = fig2
            
            # Subject comparison
            if 'subject' in df.columns:
                fig3 = px.bar(df.groupby('subject')['score'].mean().reset_index(), 
                             x='subject', y='score', title='Average Score by Subject')
                figures['subject_comparison'] = fig3
            
            # Performance trends over time
            if 'date' in df.columns or 'test_date' in df.columns:
                date_col = 'date' if 'date' in df.columns else 'test_date'
                df[date_col] = pd.to_datetime(df[date_col])
                monthly_avg = df.groupby(df[date_col].dt.to_period('M'))['score'].mean()
                
                fig4 = go.Figure()
                fig4.add_trace(go.Scatter(x=monthly_avg.index.astype(str), y=monthly_avg.values,
                                        mode='lines+markers', name='Average Score'))
                fig4.update_layout(title='Performance Trends Over Time')
                figures['trends'] = fig4
        
        return figures
    except Exception as e:
        st.error(f"Visualization error: {e}")
        return {}

def generate_executive_presentation(analysis_results, school_context="District"):
    """Generate comprehensive executive presentation"""
    
    presentation_prompt = f"""Create a comprehensive executive presentation for {school_context} leadership based on this educational data analysis:

Analysis Results: {json.dumps(analysis_results, default=str, indent=2)}

Create a presentation with these sections:

**SLIDE 1: Executive Summary Dashboard**
- Key performance indicators
- Critical findings summary
- Immediate action items
- Overall district health score

**SLIDE 2: Academic Performance Overview** 
- Grade-level performance analysis
- Subject area comparisons
- Historical trend analysis
- Benchmark comparisons

**SLIDE 3: Achievement Gap Analysis**
- Demographic performance differences
- Equity indicators
- At-risk student identification
- Gap closure progress

**SLIDE 4: Predictive Analytics & Risk Factors**
- Students at risk of not meeting benchmarks
- Early warning indicators
- Intervention opportunity areas
- Success probability modeling

**SLIDE 5: Evidence-Based Intervention Strategies**
- Research-backed recommendations
- Targeted intervention programs
- Resource allocation priorities
- Expected ROI and timelines

**SLIDE 6: Implementation Roadmap**
- 30-60-90 day action plan
- Resource requirements
- Staff development needs
- Success metrics and monitoring

**SLIDE 7: Budget Impact & Resource Allocation**
- Cost-benefit analysis
- Funding recommendations
- Personnel needs
- Technology requirements

**SLIDE 8: Risk Assessment & Mitigation**
- Academic risks identified
- Compliance considerations
- Mitigation strategies
- Contingency planning

**SLIDE 9: Next Steps & Decision Points**
- Board approval items
- Implementation timeline
- Stakeholder communication plan
- Follow-up reporting schedule

Include specific data points, percentages, and actionable recommendations for each slide.
Format for easy copying into PowerPoint with speaker notes."""

    return get_educational_ai_response(presentation_prompt, "executive_presentation")

# Main Interface
st.header("🎯 Analytics Command Center")

# Data Upload Section
with st.expander("📊 Data Upload & Processing", expanded=True):
    st.subheader("Upload Your Educational Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        data_type = st.selectbox("Data Type:", [
            "Beginning of Year (BOY) Assessments",
            "Middle of Year (MOY) Assessments", 
            "End of Year (EOY) Assessments",
            "Benchmark Testing Data",
            "State Assessment Results",
            "Diagnostic Assessment Data",
            "Attendance and Behavior Data",
            "Demographic and Enrollment Data"
        ])
    
    with col2:
        school_context = st.selectbox("School Context:", [
            "Individual School", "District-Wide", "Regional Analysis",
            "State Comparison", "Multi-Year Analysis"
        ])
    
    uploaded_file = st.file_uploader(
        "Upload CSV, Excel, or JSON file",
        type=['csv', 'xlsx', 'json'],
        help="Upload assessment data with columns like: student_id, grade, subject, score, date"
    )
    
    if uploaded_file:
        try:
            # Read file
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith('.xlsx'):
                df = pd.read_excel(uploaded_file)
            elif uploaded_file.name.endswith('.json'):
                df = pd.read_json(uploaded_file)
            
            st.session_state.uploaded_data[data_type] = df
            
            # Show data preview
            st.subheader("📋 Data Preview")
            st.dataframe(df.head())
            
            # Basic info
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Total Records", len(df))
            col_b.metric("Columns", len(df.columns))
            col_c.metric("Data Type", data_type.split(" ")[0])
            
        except Exception as e:
            st.error(f"Error reading file: {e}")

# Data Analysis Section
if st.session_state.uploaded_data:
    st.header("🔬 Advanced Data Analysis")
    
    # Select data for analysis
    selected_data_type = st.selectbox(
        "Select data for analysis:",
        list(st.session_state.uploaded_data.keys())
    )
    
    if selected_data_type and st.button("🚀 Run Comprehensive Analysis"):
        df = st.session_state.uploaded_data[selected_data_type]
        
        with st.spinner("Running advanced educational data analysis..."):
            # Process data
            analysis_results = process_assessment_data(df)
            
            if analysis_results:
                st.session_state.analysis_results[selected_data_type] = analysis_results
                
                # Display results
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📊 Statistical Summary")
                    st.dataframe(analysis_results['summary_stats'])
                    
                    if analysis_results['grade_analysis'] is not None:
                        st.subheader("🎓 Grade Level Analysis")
                        st.dataframe(analysis_results['grade_analysis'])
                
                with col2:
                    if analysis_results['subject_analysis'] is not None:
                        st.subheader("📚 Subject Area Analysis")
                        st.dataframe(analysis_results['subject_analysis'])
                
                # Generate visualizations
                st.subheader("📈 Data Visualizations")
                figures = generate_visualizations(df)
                
                for fig_name, fig in figures.items():
                    st.plotly_chart(fig, use_container_width=True)
                
                # AI Analysis
                st.subheader("🤖 AI-Powered Analysis")
                
                data_summary = f"""
                Data Type: {selected_data_type}
                Total Records: {len(df)}
                Score Statistics: {df['score'].describe() if 'score' in df.columns else 'No score column'}
                Grade Levels: {df['grade'].unique() if 'grade' in df.columns else 'No grade data'}
                Subjects: {df['subject'].unique() if 'subject' in df.columns else 'No subject data'}
                """
                
                analysis_query = f"Provide comprehensive analysis of this {selected_data_type} data including key findings, concerns, and actionable recommendations."
                
                ai_analysis = get_educational_ai_response(analysis_query, "data_analysis", data_summary)
                st.markdown(ai_analysis)

# Predictive Analytics Section
if st.session_state.uploaded_data:
    st.header("🔮 Predictive Analytics")
    
    with st.expander("🎯 Build Predictive Models"):
        prediction_data = st.selectbox(
            "Select data for prediction:",
            list(st.session_state.uploaded_data.keys())
        )
        
        target_variable = st.selectbox(
            "Target variable to predict:",
            ["score", "performance_level", "pass_rate", "growth_rate"]
        )
        
        if st.button("🧠 Build Predictive Model"):
            df = st.session_state.uploaded_data[prediction_data]
            
            if target_variable in df.columns:
                with st.spinner("Building predictive models..."):
                    model_results, features = create_predictive_model(df, target_variable)
                    
                    if model_results:
                        st.session_state.predictive_models[prediction_data] = model_results
                        
                        # Display model performance
                        st.subheader("🎯 Model Performance")
                        
                        for model_name, results in model_results.items():
                            col1, col2 = st.columns(2)
                            col1.metric(f"{model_name} - R² Score", f"{results['r2']:.3f}")
                            col2.metric(f"{model_name} - MSE", f"{results['mse']:.3f}")
                        
                        # Feature importance
                        best_model = max(model_results.keys(), key=lambda x: model_results[x]['r2'])
                        if model_results[best_model]['feature_importance'] is not None:
                            st.subheader("🔍 Feature Importance")
                            importance_df = pd.DataFrame({
                                'Feature': features,
                                'Importance': model_results[best_model]['feature_importance']
                            }).sort_values('Importance', ascending=False)
                            
                            fig_importance = px.bar(importance_df.head(10), 
                                                  x='Importance', y='Feature',
                                                  orientation='h',
                                                  title='Top 10 Most Important Features')
                            st.plotly_chart(fig_importance, use_container_width=True)
                        
                        # AI Interpretation
                        prediction_query = f"Interpret these predictive model results for {prediction_data}. What do the feature importance and model performance tell us about factors affecting {target_variable}?"
                        
                        prediction_analysis = get_educational_ai_response(
                            prediction_query, 
                            "predictive_analytics", 
                            f"Model Performance: {model_results}"
                        )
                        
                        st.markdown("### 🤖 Model Interpretation")
                        st.markdown(prediction_analysis)
            else:
                st.error(f"Target variable '{target_variable}' not found in data")

# Teaching Strategies Section
st.header("🎓 Evidence-Based Teaching Strategies")

with st.expander("📚 Get Research-Based Teaching Recommendations"):
    # Strategy context
    col1, col2 = s
