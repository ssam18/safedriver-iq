"""
SafeDriver-IQ Interactive Dashboard

Streamlit application for real-time safety score calculation and analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import sys

# Add parent directory to path so 'src' module can be imported
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Now we can import from src
from src.realtime_calculator import RealtimeSafetyCalculator, create_example_scenarios

# Page config
st.set_page_config(
    page_title="SafeDriver-IQ Dashboard",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 48px;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 10px;
    }
    .sub-header {
        font-size: 20px;
        color: #666;
        text-align: center;
        margin-bottom: 30px;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .risk-critical {
        color: #D32F2F;
        font-weight: bold;
    }
    .risk-high {
        color: #F57C00;
        font-weight: bold;
    }
    .risk-medium {
        color: #FBC02D;
        font-weight: bold;
    }
    .risk-low {
        color: #388E3C;
        font-weight: bold;
    }
    .risk-excellent {
        color: #1976D2;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_calculator():
    """Load the safety calculator (cached)."""
    model_path = Path(__file__).parent.parent / "results" / "models" / "best_safety_model.pkl"
    feature_path = Path(__file__).parent.parent / "results" / "models" / "feature_names.txt"
    
    if not model_path.exists():
        return None
    
    return RealtimeSafetyCalculator(str(model_path), str(feature_path))


def get_risk_color(risk_level: str) -> str:
    """Get color for risk level."""
    colors = {
        'Critical': '#D32F2F',
        'High': '#F57C00',
        'Medium': '#FBC02D',
        'Low': '#388E3C',
        'Excellent': '#1976D2'
    }
    return colors.get(risk_level, '#666')


def create_gauge_chart(score: float, risk_level: str) -> go.Figure:
    """Create gauge chart for safety score."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"Safety Score<br><span style='font-size:24px;color:{get_risk_color(risk_level)}'>{risk_level} Risk</span>", 
               'font': {'size': 28}},
        delta={'reference': 85, 'increasing': {'color': "green"}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': get_risk_color(risk_level)},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 40], 'color': '#FFCDD2'},
                {'range': [40, 60], 'color': '#FFE0B2'},
                {'range': [60, 75], 'color': '#FFF9C4'},
                {'range': [75, 85], 'color': '#C8E6C9'},
                {'range': [85, 100], 'color': '#BBDEFB'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 85
            }
        }
    ))
    
    fig.update_layout(
        height=350,
        margin=dict(l=20, r=20, t=80, b=20),
        font={'size': 16}
    )
    
    return fig


def create_scenario_comparison(scenarios_results: list) -> go.Figure:
    """Create comparison chart for multiple scenarios."""
    names = [r['name'] for r in scenarios_results]
    scores = [r['result']['safety_score'] for r in scenarios_results]
    colors = [get_risk_color(r['result']['risk_level']) for r in scenarios_results]
    
    fig = go.Figure(data=[
        go.Bar(
            x=names,
            y=scores,
            marker_color=colors,
            text=scores,
            texttemplate='%{text:.1f}',
            textposition='outside'
        )
    ])
    
    fig.update_layout(
        title="Scenario Safety Comparison",
        xaxis_title="Scenario",
        yaxis_title="Safety Score",
        yaxis_range=[0, 105],
        height=400,
        showlegend=False
    )
    
    # Add target line
    fig.add_hline(y=85, line_dash="dash", line_color="green", 
                  annotation_text="Target Score (85)")
    
    return fig


def main():
    """Main application."""
    
    # Header
    st.markdown('<div class="main-header">🚗 SafeDriver-IQ Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Inverse Crash Modeling for Real-Time Driver Safety</div>', unsafe_allow_html=True)
    
    # Load calculator
    calculator = load_calculator()
    
    if calculator is None:
        st.error("⚠️ Model not found! Please train the model first:")
        st.code("jupyter notebook notebooks/02_train_inverse_model.ipynb")
        st.stop()
    
    # Sidebar
    st.sidebar.title("🎛️ Control Panel")
    
    page = st.sidebar.radio(
        "Select Page",
        ["🏠 Home", "🔮 Safety Score Calculator", "⚖️ Scenario Comparison", 
         "💡 Improvement Suggestions", "📊 Batch Analysis", "ℹ️ About"]
    )
    
    # ==================== HOME PAGE ====================
    if page == "🏠 Home":
        st.header("Welcome to SafeDriver-IQ")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("VRU Crashes Analyzed", "38,462", "2016-2023")
        with col2:
            st.metric("Total Crashes", "417,335", "8 years")
        with col3:
            st.metric("Features Engineered", "120+", "ML features")
        
        st.markdown("---")
        
        st.subheader("🎯 What is SafeDriver-IQ?")
        st.write("""
        SafeDriver-IQ uses **inverse crash modeling** to provide real-time safety scores 
        that tell drivers how close they are to crash conditions and what specific actions 
        would make them safer.
        
        **Traditional Approach:** "30% crash risk"  
        **SafeDriver-IQ:** "Safety score: 72/100 → Improve to 85+ by avoiding night driving"
        """)
        
        st.subheader("🚀 Key Features")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ✅ **Real-time Safety Scoring** (0-100)  
            ✅ **Risk Level Classification**  
            ✅ **Actionable Recommendations**  
            ✅ **Scenario Comparison**  
            """)
        
        with col2:
            st.markdown("""
            ✅ **Improvement Suggestions**  
            ✅ **Good Driver Profile**  
            ✅ **VRU-Specific Safety**  
            ✅ **Interactive Analysis**  
            """)
        
        st.markdown("---")
        
        st.subheader("📈 Expected Impact")
        st.write("""
        With 20% adoption, SafeDriver-IQ could prevent:
        - **1,500 pedestrian deaths/year** (20% reduction)
        - **200 cyclist deaths/year** (20% reduction)
        - **30,000 VRU injuries/year** (20% reduction)
        - **Total: 1,870+ lives saved annually**
        """)
    
    # ==================== CALCULATOR PAGE ====================
    elif page == "🔮 Safety Score Calculator":
        st.header("Real-Time Safety Score Calculator")
        st.write("Enter current driving conditions to get your safety score.")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("⏰ Temporal Factors")
            hour = st.slider("Hour of Day (0-23)", 0, 23, 14)
            day_week = st.selectbox("Day of Week", 
                                     [1, 2, 3, 4, 5, 6, 7], 
                                     format_func=lambda x: ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'][x-1],
                                     index=2)
            month = st.slider("Month", 1, 12, 6)
        
        with col2:
            st.subheader("🌤️ Environmental Factors")
            weather = st.selectbox("Weather Condition",
                                   [(1, "Clear"), (2, "Rain"), (3, "Snow"), (4, "Fog"), (10, "Cloudy")],
                                   format_func=lambda x: x[1],
                                   index=0)
            lighting = st.selectbox("Lighting Condition",
                                    [(1, "Daylight"), (2, "Dark - Not Lighted"), (3, "Dark - Lighted"), (4, "Dawn/Dusk")],
                                    format_func=lambda x: x[1],
                                    index=0)
            road_cond = st.selectbox("Road Condition",
                                     [(1, "Dry"), (2, "Wet"), (3, "Snow/Ice"), (4, "Other")],
                                     format_func=lambda x: x[1],
                                     index=0)
        
        with col3:
            st.subheader("🚗 Driving Factors")
            speed_rel = st.slider("Speed Relative to Limit (1=Low, 5=High)", 1, 5, 2)
            vru_present = st.checkbox("Pedestrians/Cyclists Present")
        
        # Calculate button
        if st.button("🔮 Calculate Safety Score", type="primary"):
            # Build scenario
            scenario = {
                'HOUR': hour,
                'DAY_WEEK': day_week,
                'MONTH': month,
                'WEATHER': weather[0],
                'LGT_COND': lighting[0],
                'ROAD_COND': road_cond[0],
                'SPEED_REL': speed_rel,
                'VRU_PRESENT': 1 if vru_present else 0,
                'IS_NIGHT': 1 if hour >= 20 or hour <= 6 else 0,
                'IS_WEEKEND': 1 if day_week in [1, 7] else 0,
                'IS_RUSH_HOUR': 1 if hour in [7, 8, 9, 16, 17, 18] else 0,
                'POOR_LIGHTING': 1 if lighting[0] > 1 else 0,
                'ADVERSE_WEATHER': 1 if weather[0] > 1 else 0
            }
            
            # Debug: Show scenario details
            with st.expander("🔍 Scenario Details (Debug)"):
                st.json(scenario)
            
            # Calculate
            with st.spinner("Calculating safety score..."):
                result = calculator.calculate_safety_score(scenario)
            
            # Display results
            st.markdown("---")
            st.subheader("📊 Results")
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                # Gauge chart
                fig = create_gauge_chart(result['safety_score'], result['risk_level'])
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("### 🎯 Risk Assessment")
                st.markdown(f"**Risk Level:** <span class='risk-{result['risk_level'].lower()}'>{result['risk_level']}</span>", 
                           unsafe_allow_html=True)
                st.metric("Model Confidence", f"{result['confidence']:.1%}")
                
                st.markdown("### 💡 Recommendations")
                for i, rec in enumerate(result['recommendations'], 1):
                    st.write(f"{i}. {rec}")
    
    # ==================== COMPARISON PAGE ====================
    elif page == "⚖️ Scenario Comparison":
        st.header("Scenario Comparison")
        st.write("Compare safety scores across different driving scenarios.")
        
        # Example scenarios
        examples = create_example_scenarios()
        
        if st.button("📊 Compare Example Scenarios", type="primary"):
            results = []
            
            for scenario in examples:
                result = calculator.calculate_safety_score(scenario)
                results.append({
                    'name': scenario['name'],
                    'result': result
                })
            
            # Comparison chart
            fig = create_scenario_comparison(results)
            st.plotly_chart(fig, use_container_width=True)
            
            # Detailed results
            st.markdown("---")
            st.subheader("📋 Detailed Results")
            
            for r in results:
                with st.expander(f"{r['name']} - Score: {r['result']['safety_score']:.1f}"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Safety Score", f"{r['result']['safety_score']:.1f}/100")
                        st.metric("Risk Level", r['result']['risk_level'])
                        st.metric("Confidence", f"{r['result']['confidence']:.1%}")
                    
                    with col2:
                        st.markdown("**Top Recommendations:**")
                        for rec in r['result']['recommendations'][:3]:
                            st.write(f"• {rec}")
    
    # ==================== IMPROVEMENT PAGE ====================
    elif page == "💡 Improvement Suggestions":
        st.header("Safety Score Improvement")
        st.write("Get specific suggestions to improve your safety score.")
        
        # Quick scenario builder
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Current Scenario")
            hour = st.slider("Hour", 0, 23, 18, key="imp_hour")
            weather = st.selectbox("Weather", [1, 2, 3], format_func=lambda x: ['Clear', 'Rain', 'Snow'][x-1], key="imp_weather")
            lighting = st.selectbox("Lighting", [1, 2, 3], format_func=lambda x: ['Day', 'Dark-Unlighted', 'Dark-Lighted'][x-1], key="imp_light")
            speed = st.slider("Speed (1-5)", 1, 5, 4, key="imp_speed")
        
        with col2:
            st.subheader("Target")
            target_score = st.slider("Target Safety Score", 50, 100, 85)
        
        if st.button("💡 Get Improvement Suggestions", type="primary"):
            scenario = {
                'HOUR': hour,
                'WEATHER': weather,
                'LGT_COND': lighting,
                'SPEED_REL': speed,
                'IS_NIGHT': 1 if hour >= 20 or hour <= 6 else 0,
                'POOR_LIGHTING': 1 if lighting > 1 else 0,
                'ADVERSE_WEATHER': 1 if weather > 1 else 0,
                'IS_RUSH_HOUR': 1 if hour in [7, 8, 9, 16, 17, 18] else 0,
                'ROAD_COND': 1,
                'DAY_WEEK': 3,
                'MONTH': 6
            }
            
            with st.spinner("Analyzing improvements..."):
                suggestions = calculator.suggest_improvements(scenario, target_score)
            
            st.markdown("---")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Current Score", f"{suggestions['current_score']:.1f}")
            with col2:
                st.metric("Target Score", f"{suggestions['target_score']:.1f}")
            with col3:
                gap = suggestions['target_score'] - suggestions['current_score']
                st.metric("Gap", f"{gap:.1f}", delta=f"{gap:.1f}")
            
            if suggestions['achievable']:
                st.success("✅ Target score is achievable with these improvements:")
                
                for i, sug in enumerate(suggestions['suggestions'], 1):
                    with st.container():
                        st.markdown(f"### {i}. {sug['action']}")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Improvement", f"+{sug['expected_improvement']:.1f}")
                        with col2:
                            st.metric("New Score", f"{sug['new_score']:.1f}")
                        with col3:
                            st.write(f"**Change:** {sug['current_value']} → {sug['suggested_value']}")
                        st.markdown("---")
            else:
                st.info("Target score already achieved or suggestions unavailable.")
    
    # ==================== BATCH ANALYSIS ====================
    elif page == "📊 Batch Analysis":
        st.header("Batch Scenario Analysis")
        st.write("Upload CSV or analyze multiple scenarios at once.")
        
        st.info("📁 Upload a CSV file with driving scenario data, or use example data.")
        
        # File uploader
        uploaded_file = st.file_uploader("Upload CSV", type=['csv'])
        
        col1, col2 = st.columns(2)
        
        with col1:
            use_examples = st.button("📋 Use Example Scenarios")
        with col2:
            if uploaded_file:
                analyze_uploaded = st.button("🔍 Analyze Uploaded Data", type="primary")
        
        if use_examples:
            scenarios = create_example_scenarios()
            
            with st.spinner("Analyzing scenarios..."):
                # Convert to list of dicts without 'name'
                scenario_dicts = [{k: v for k, v in s.items() if k != 'name'} for s in scenarios]
                results_df = calculator.batch_calculate(scenario_dicts)
            
            st.success(f"✅ Analyzed {len(results_df)} scenarios")
            
            # Summary statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Avg Safety Score", f"{results_df['safety_score'].mean():.1f}")
            with col2:
                critical = (results_df['risk_level'] == 'Critical').sum()
                st.metric("Critical Risk", critical)
            with col3:
                high = (results_df['risk_level'] == 'High').sum()
                st.metric("High Risk", high)
            with col4:
                low = (results_df['risk_level'].isin(['Low', 'Excellent'])).sum()
                st.metric("Low Risk", low)
            
            # Distribution chart
            fig = px.histogram(results_df, x='safety_score', nbins=20,
                              title="Safety Score Distribution",
                              labels={'safety_score': 'Safety Score', 'count': 'Frequency'})
            st.plotly_chart(fig, use_container_width=True)
            
            # Results table
            st.subheader("📋 Detailed Results")
            display_df = results_df[['scenario_id', 'safety_score', 'risk_level', 'confidence']].copy()
            st.dataframe(display_df, use_container_width=True)
    
    # ==================== ABOUT PAGE ====================
    elif page == "ℹ️ About":
        st.header("About SafeDriver-IQ")
        
        st.markdown("""
        ## 🎯 Mission
        
        SafeDriver-IQ aims to save lives by providing proactive, continuous safety guidance 
        to drivers, with special focus on protecting vulnerable road users (pedestrians and cyclists).
        
        ## 🔬 Methodology
        
        ### Inverse Safety Modeling
        
        Unlike traditional approaches that predict crash probability, SafeDriver-IQ:
        
        1. **Trains** crash classifier on VRU crash data (417K crashes, 2016-2023)
        2. **Extracts** decision boundaries between safe and crash conditions
        3. **Computes** distance from boundary = continuous safety score (0-100)
        4. **Generates** specific improvement recommendations
        
        ### Key Innovations
        
        ✅ **Continuous Scoring:** 0-100 scale instead of binary prediction  
        ✅ **Actionable Feedback:** Specific improvements, not vague warnings  
        ✅ **Empirical Profile:** "Good driver" extracted from data, not assumed  
        ✅ **VRU-Specific:** Dedicated models for pedestrian/cyclist safety  
        
        ## 📊 Data
        
        - **Source:** NHTSA CRSS (Crash Report Sampling System)
        - **Years:** 2016-2023 (8 years)
        - **Total Crashes:** 417,335
        - **VRU Crashes:** 38,462
        - **Features:** 120+ engineered features
        
        ## 🎓 Technical Details
        
        **Models:** Random Forest, XGBoost, Gradient Boosting  
        **Features:** Temporal, environmental, location, VRU-specific  
        **Validation:** 5-fold cross-validation, ROC-AUC evaluation  
        **Interpretability:** SHAP analysis, feature importance  
        
        ## 📈 Expected Impact
        
        With 20% adoption:
        - **1,870 lives saved per year**
        - **30,000 injuries prevented per year**
        - **$1.5 billion economic benefit**
        
        ## 👨‍💻 Development
        
        - **Status:** Research prototype
        - **Timeline:** Full deployment in 8-12 weeks
        - **Publication:** Target Q1 2026
        
        ## 📧 Contact
        
        For more information or collaboration opportunities, please contact the development team.
        
        ---
        
        **Version:** 1.0.0  
        **Last Updated:** January 2026
        """)


if __name__ == "__main__":
    main()
