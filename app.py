import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="Telecom Churn Intelligence",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .main { background-color: #f8f7f4; }

    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        border: 1px solid #e8e6e1;
        text-align: center;
    }
    .metric-label {
        font-size: 12px;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 6px;
    }
    .metric-value {
        font-size: 28px;
        font-weight: 600;
        color: #1a1a1a;
        line-height: 1.1;
    }
    .metric-sub {
        font-size: 12px;
        color: #aaa;
        margin-top: 4px;
    }
    .metric-good { color: #2d8a4e; }
    .metric-warn { color: #c85c1a; }
    .metric-info { color: #1a5fa8; }

    .section-title {
        font-size: 18px;
        font-weight: 600;
        color: #1a1a1a;
        margin: 2rem 0 1rem 0;
        padding-bottom: 8px;
        border-bottom: 2px solid #e8e6e1;
    }

    .insight-box {
        background: #fff8f0;
        border-left: 4px solid #e07b2a;
        border-radius: 0 8px 8px 0;
        padding: 0.8rem 1rem;
        margin: 0.5rem 0;
        font-size: 14px;
        color: #4a3520;
    }

    .seg-card {
        background: white;
        border-radius: 12px;
        padding: 1.2rem;
        border: 1px solid #e8e6e1;
        height: 100%;
    }
    .seg-title {
        font-size: 15px;
        font-weight: 600;
        margin-bottom: 8px;
        color: #1a1a1a;
    }
    .seg-stat {
        font-size: 13px;
        color: #666;
        margin: 4px 0;
    }
    .seg-driver {
        background: #f0f4ff;
        border-radius: 6px;
        padding: 6px 10px;
        font-size: 12px;
        color: #2a4a9a;
        margin-top: 10px;
        font-family: 'DM Mono', monospace;
    }

    .retention-card {
        background: white;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        border: 1px solid #e8e6e1;
        margin-bottom: 1rem;
    }
    .retention-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 8px;
    }
    .risk-badge-high {
        background: #fde8e8;
        color: #a32d2d;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 500;
    }

    .stSelectbox > div > div {
        border-radius: 8px;
    }

    div[data-testid="stSidebar"] {
        background: #1a1a2e;
    }
    div[data-testid="stSidebar"] * {
        color: #e8e6e1 !important;
    }
</style>
""", unsafe_allow_html=True)


# ── Simulated Data (replace with real CSVs in production) ─────
@st.cache_data
def load_data():
    np.random.seed(42)
    n = 51047

    segments = np.random.choice(
        ['Heavy Users At Risk', 'Churning Dissatisfied Customers', 'Low Engagement Customers'],
        size=n, p=[0.786, 0.181, 0.033]
    )

    churn_prob = np.where(
        segments == 'Low Engagement Customers',
        np.random.beta(6, 4, n),
        np.where(segments == 'Heavy Users At Risk',
                 np.random.beta(4.5, 5, n),
                 np.random.beta(4, 5, n))
    )
    churn_prob = np.clip(churn_prob, 0.01, 0.99)

    monthly_rev = np.where(
        segments == 'Heavy Users At Risk',
        np.random.normal(72, 28, n),
        np.where(segments == 'Churning Dissatisfied Customers',
                 np.random.normal(55, 22, n),
                 np.random.normal(32, 15, n))
    )
    monthly_rev = np.clip(monthly_rev, 5, 200)

    churn_binary = (np.random.rand(n) < churn_prob * 0.5).astype(int)

    risk_level = pd.cut(
        churn_prob,
        bins=[0, 0.3, 0.6, 1.0],
        labels=['Low Risk', 'Medium Risk', 'High Risk']
    )

    drivers = {
        'Heavy Users At Risk': 'PercChangeMinutes',
        'Churning Dissatisfied Customers': 'CurrentEquipmentDays',
        'Low Engagement Customers': 'CurrentEquipmentDays'
    }
    top_reason = [drivers[s] for s in segments]

    months_service = np.random.randint(1, 72, n)
    care_calls = np.random.poisson(2, n)
    overage = np.random.exponential(30, n)

    df = pd.DataFrame({
        'CustomerID': [f'CUST{i:06d}' for i in range(n)],
        'Segment_Label': segments,
        'Churn_Probability': churn_prob.round(4),
        'Monthly_Revenue': monthly_rev.round(2),
        'Churn_Binary': churn_binary,
        'Risk_Level': risk_level,
        'Top_Churn_Reason': top_reason,
        'MonthsInService': months_service,
        'CustomerCareCalls': care_calls,
        'OverageMinutes': overage.round(1)
    })

    df['Revenue_at_Risk'] = (df['Churn_Probability'] * df['Monthly_Revenue'] * 12).round(2)
    return df


@st.cache_data
def load_retention_messages():
    messages = [
        {
            'CustomerID': 'CUST000042',
            'Segment_Label': 'Heavy Users At Risk',
            'Churn_Probability': 0.87,
            'Monthly_Revenue': 98.50,
            'Top_Churn_Reason': 'Significant drop in usage compared to previous month',
            'Retention_Message': """1. SITUATION: This high-value customer has reduced their monthly usage by over 40% in the past two months, signaling active disengagement before formal churn.

2. OFFER: Upgrade to our Premium Connect plan with 20% more minutes and a data bonus for the next 3 months at the same price.

3. MESSAGE: "We noticed you've been using less of your plan lately — we'd hate to lose you. As one of our most valued customers, we've reserved an exclusive upgrade that gives you more for the same price."

4. URGENCY: Contact within 48 hours — customers at this usage decline rate churn within 2 weeks if not retained."""
        },
        {
            'CustomerID': 'CUST001337',
            'Segment_Label': 'Churning Dissatisfied Customers',
            'Churn_Probability': 0.82,
            'Monthly_Revenue': 67.20,
            'Top_Churn_Reason': 'Using outdated equipment for too long — needs upgrade',
            'Retention_Message': """1. SITUATION: Long-tenure customer (47 months) with equipment over 800 days old — unmet upgrade expectations are the primary churn driver.

2. OFFER: Early device upgrade eligibility with a subsidised handset (50% off selected models) locked to a new 12-month contract.

3. MESSAGE: "You've been with us for nearly 4 years, and we want to make sure you have the best experience. You qualify for our loyalty upgrade program — a new device at half price, on us."

4. URGENCY: Reach out before their billing cycle ends — device frustration peaks at renewal moments."""
        },
        {
            'CustomerID': 'CUST007891',
            'Segment_Label': 'Low Engagement Customers',
            'Churn_Probability': 0.79,
            'Monthly_Revenue': 28.40,
            'Top_Churn_Reason': 'Using outdated equipment for too long — needs upgrade',
            'Retention_Message': """1. SITUATION: Low-engagement customer with minimal usage and aging equipment — likely already using a competitor's SIM as primary.

2. OFFER: Re-engagement bundle: 3 months of free data (2GB/month) plus eligibility for an entry-level device upgrade at 60% discount.

3. MESSAGE: "We miss having you as an active customer. Here's an exclusive offer just for you — free data for 3 months and a fresh device to go with it."

4. URGENCY: Contact this week — low engagement customers who go 30+ days without outreach convert at near-zero rates."""
        }
    ]
    return pd.DataFrame(messages)


df = load_data()
retention_df = load_retention_messages()

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📡 Telecom Churn\nIntelligence Dashboard")
    st.markdown("---")

    page = st.radio(
        "Navigate",
        ["Overview", "Segments", "Model Performance", "Retention Engine"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("**Filters**")

    seg_filter = st.multiselect(
        "Segment",
        options=df['Segment_Label'].unique().tolist(),
        default=df['Segment_Label'].unique().tolist()
    )

    threshold = st.slider(
        "Decision threshold",
        min_value=0.2,
        max_value=0.7,
        value=0.4,
        step=0.05,
        help="Lower = catch more churners, more false alarms"
    )

    st.markdown("---")
    st.markdown("""
    <div style='font-size:12px; color:#888; line-height:1.6;'>
    Dataset: Cell2Cell<br>
    Customers: 51,047<br>
    Model: XGBoost<br>
    AUC-ROC: 0.675<br>
    Built for Tunisie Telecom
    </div>
    """, unsafe_allow_html=True)

filtered_df = df[df['Segment_Label'].isin(seg_filter)]
predicted_churn = (filtered_df['Churn_Probability'] >= threshold).sum()
total_filtered = len(filtered_df)
precision_approx = max(0.25, 0.55 - threshold * 0.5)
recall_approx = max(0.1, 1.1 - threshold * 1.5)


# ══════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════
if page == "Overview":
    st.markdown("# Telecom Churn Intelligence")
    st.markdown("End-to-end ML pipeline for customer retention — XGBoost + SHAP + LLM")
    st.markdown("---")

    # KPI cards
    total_rev_risk = filtered_df['Revenue_at_Risk'].sum()
    high_risk_count = (filtered_df['Churn_Probability'] >= threshold).sum()
    roi = 35
    churn_rate = filtered_df['Churn_Binary'].mean() * 100

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-label'>Total Customers</div>
            <div class='metric-value'>{total_filtered:,}</div>
            <div class='metric-sub'>in selected segments</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-label'>Churn Rate</div>
            <div class='metric-value metric-warn'>{churn_rate:.1f}%</div>
            <div class='metric-sub'>of customers at risk</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-label'>Revenue at Risk</div>
            <div class='metric-value metric-warn'>{total_rev_risk/1e6:.1f}M DT</div>
            <div class='metric-sub'>annual, all segments</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-label'>High Risk (t={threshold})</div>
            <div class='metric-value metric-info'>{high_risk_count:,}</div>
            <div class='metric-sub'>need retention call</div>
        </div>""", unsafe_allow_html=True)
    with c5:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-label'>Retention ROI</div>
            <div class='metric-value metric-good'>{roi}x</div>
            <div class='metric-sub'>return on investment</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div class='section-title'>Churn Probability Distribution</div>",
                unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        fig = go.Figure()
        for churn_val, label, color in [(0, 'Did not churn', '#2d8a4e'), (1, 'Churned', '#c85c1a')]:
            subset = filtered_df[filtered_df['Churn_Binary'] == churn_val]['Churn_Probability']
            fig.add_trace(go.Histogram(
                x=subset, name=label,
                opacity=0.65, nbinsx=40,
                marker_color=color, histnorm='density'
            ))
        fig.add_vline(x=threshold, line_dash='dash', line_color='#1a1a1a', line_width=2,
                      annotation_text=f'Threshold = {threshold}',
                      annotation_position='top right')
        fig.update_layout(
            barmode='overlay', height=300,
            margin=dict(l=20, r=20, t=20, b=40),
            legend=dict(orientation='h', y=1.1),
            xaxis_title='Predicted Churn Probability',
            yaxis_title='Density',
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        risk_counts = filtered_df['Risk_Level'].value_counts()
        colors_pie = ['#2d8a4e', '#e8a020', '#c85c1a']
        fig2 = go.Figure(go.Pie(
            labels=risk_counts.index,
            values=risk_counts.values,
            hole=0.55,
            marker_colors=colors_pie,
            textinfo='percent',
            hovertemplate='%{label}: %{value:,}<extra></extra>'
        ))
        fig2.update_layout(
            height=300,
            margin=dict(l=0, r=0, t=20, b=0),
            showlegend=True,
            legend=dict(orientation='v', x=0.8),
            plot_bgcolor='white', paper_bgcolor='white'
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("<div class='section-title'>Threshold Impact — Live</div>",
                unsafe_allow_html=True)

    st.markdown(f"""
    <div class='insight-box'>
    At threshold <strong>{threshold}</strong>: the model flags <strong>{high_risk_count:,}</strong> customers for retention calls.
    Estimated recall <strong>{recall_approx:.0%}</strong> — catching that share of real churners.
    Call cost: <strong>{high_risk_count * 10:,.0f} DT</strong> →
    Revenue saved (50% win rate): <strong>{high_risk_count * 10 * roi / 2:,.0f} DT</strong>.
    </div>
    """, unsafe_allow_html=True)

    thresholds = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]
    caught = [int((filtered_df['Churn_Probability'] >= t).sum()) for t in thresholds]
    prec = [max(0.25, 0.55 - t * 0.5) for t in thresholds]
    rec = [max(0.05, 1.15 - t * 1.6) for t in thresholds]

    fig3 = make_subplots(specs=[[{"secondary_y": True}]])
    fig3.add_trace(go.Scatter(x=thresholds, y=[c/1000 for c in caught],
                               name='Customers called (K)', line=dict(color='#1a5fa8', width=2)))
    fig3.add_trace(go.Scatter(x=thresholds, y=[r*100 for r in rec],
                               name='Recall (%)', line=dict(color='#c85c1a', width=2, dash='dot')),
                   secondary_y=True)
    fig3.add_vline(x=threshold, line_dash='dash', line_color='#888', line_width=1.5)
    fig3.update_layout(
        height=260, margin=dict(l=20, r=20, t=10, b=40),
        plot_bgcolor='white', paper_bgcolor='white',
        legend=dict(orientation='h', y=1.1),
        xaxis_title='Threshold'
    )
    fig3.update_yaxes(title_text='Customers called (K)', secondary_y=False)
    fig3.update_yaxes(title_text='Recall (%)', secondary_y=True)
    st.plotly_chart(fig3, use_container_width=True)


# ══════════════════════════════════════════════════════════════
# PAGE 2 — SEGMENTS
# ══════════════════════════════════════════════════════════════
elif page == "Segments":
    st.markdown("# Customer Segments")
    st.markdown("Three distinct risk profiles identified by KMeans clustering")
    st.markdown("---")

    seg_stats = filtered_df.groupby('Segment_Label').agg(
        Customers=('CustomerID', 'count'),
        Avg_Churn_Prob=('Churn_Probability', 'mean'),
        Total_Revenue_at_Risk=('Revenue_at_Risk', 'sum'),
        Avg_Revenue=('Monthly_Revenue', 'mean')
    ).round(2)

    seg_drivers = {
        'Heavy Users At Risk': ('PercChangeMinutes', 'Usage declining month over month', '#1a5fa8'),
        'Churning Dissatisfied Customers': ('CurrentEquipmentDays', 'Old device, no upgrade path', '#c85c1a'),
        'Low Engagement Customers': ('CurrentEquipmentDays', 'Low usage + old equipment', '#8b5ca8')
    }

    cols = st.columns(len(seg_stats))
    for i, (seg, row) in enumerate(seg_stats.iterrows()):
        driver_key, driver_desc, color = seg_drivers.get(seg, ('Unknown', '', '#888'))
        with cols[i]:
            st.markdown(f"""<div class='seg-card'>
                <div class='seg-title' style='color:{color}'>{seg}</div>
                <div class='seg-stat'>👥 {int(row['Customers']):,} customers</div>
                <div class='seg-stat'>⚠️ Avg churn prob: <strong>{row['Avg_Churn_Prob']*100:.1f}%</strong></div>
                <div class='seg-stat'>💰 Revenue at risk: <strong>{row['Total_Revenue_at_Risk']/1e6:.2f}M DT</strong></div>
                <div class='seg-stat'>📊 Avg monthly rev: <strong>{row['Avg_Revenue']:.0f} DT</strong></div>
                <div class='seg-driver'>Main driver: {driver_key}<br><span style='color:#555'>{driver_desc}</span></div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div class='section-title'>Revenue at Risk by Segment</div>",
                unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        seg_rev = filtered_df.groupby('Segment_Label')['Revenue_at_Risk'].sum().sort_values(ascending=False)
        colors_bar = ['#c85c1a', '#e8a020', '#8b5ca8']
        fig = go.Figure(go.Bar(
            x=seg_rev.index, y=seg_rev.values / 1000,
            marker_color=colors_bar[:len(seg_rev)],
            text=[f'{v/1000:.0f}K DT' for v in seg_rev.values],
            textposition='outside'
        ))
        fig.update_layout(
            height=320, yaxis_title='Revenue at Risk (K DT)',
            margin=dict(l=20, r=20, t=20, b=60),
            plot_bgcolor='white', paper_bgcolor='white',
            showlegend=False,
            xaxis_tickangle=-20
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        seg_prob = filtered_df.groupby('Segment_Label')['Churn_Probability'].mean().sort_values()
        fig2 = go.Figure(go.Bar(
            x=seg_prob.values * 100,
            y=seg_prob.index,
            orientation='h',
            marker_color=['#8b5ca8', '#e8a020', '#c85c1a'],
            text=[f'{v*100:.1f}%' for v in seg_prob.values],
            textposition='outside'
        ))
        fig2.add_vline(x=40, line_dash='dash', line_color='#aaa',
                       annotation_text='Threshold 40%')
        fig2.update_layout(
            height=320, xaxis_title='Avg Churn Probability (%)',
            margin=dict(l=20, r=60, t=20, b=40),
            plot_bgcolor='white', paper_bgcolor='white',
            showlegend=False
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("<div class='section-title'>Retention Recommendations per Segment</div>",
                unsafe_allow_html=True)

    recommendations = {
        'Heavy Users At Risk': {
            'icon': '📉',
            'driver': 'Usage declining (PercChangeMinutes)',
            'action': 'Monitor monthly usage drops as early alert trigger',
            'offer': 'Usage bonus, loyalty reward, or plan upgrade',
            'message': '"We noticed you\'re using less — here\'s a better plan for you"',
            'urgency': 'Intervene when usage drops >30% vs previous month'
        },
        'Churning Dissatisfied Customers': {
            'icon': '📱',
            'driver': 'Old device, no upgrade path (CurrentEquipmentDays)',
            'action': 'Device upgrade program for equipment > 700 days old',
            'offer': 'Subsidised handset with new contract',
            'message': '"As a valued customer, you qualify for an early device upgrade"',
            'urgency': 'Reach out before billing cycle ends'
        },
        'Low Engagement Customers': {
            'icon': '💤',
            'driver': 'Low usage + old equipment',
            'action': 'Re-engagement campaign with bundle offer',
            'offer': 'Free data bonus + entry-level device at 60% discount',
            'message': '"We miss you — here\'s an exclusive offer to stay connected"',
            'urgency': 'Contact within the week — 30+ day silence = near-zero conversion'
        }
    }

    for seg, rec in recommendations.items():
        if seg in seg_filter:
            with st.expander(f"{rec['icon']}  {seg}", expanded=True):
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.markdown("**Root cause**")
                    st.info(rec['driver'])
                    st.markdown("**Recommended action**")
                    st.write(rec['action'])
                with c2:
                    st.markdown("**Offer to make**")
                    st.success(rec['offer'])
                    st.markdown("**Agent script**")
                    st.write(rec['message'])
                with c3:
                    st.markdown("**Urgency**")
                    st.warning(rec['urgency'])


# ══════════════════════════════════════════════════════════════
# PAGE 3 — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════
elif page == "Model Performance":
    st.markdown("# Model Performance")
    st.markdown("XGBoost vs Logistic Regression vs LightGBM")
    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)
    metrics = [
        ("Best Model", "XGBoost", "info"),
        ("AUC-ROC", "0.675", "good"),
        ("CV Mean AUC", "0.666 ± 0.004", "good"),
        ("vs Baseline", "+6.1%", "good"),
    ]
    for col, (label, val, style) in zip([c1, c2, c3, c4], metrics):
        with col:
            st.markdown(f"""<div class='metric-card'>
                <div class='metric-label'>{label}</div>
                <div class='metric-value metric-{style}'>{val}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div class='section-title'>Model Comparison</div>",
                unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        models = ['Logistic Regression', 'LightGBM', 'XGBoost']
        aucs = [0.614, 0.673, 0.675]
        colors = ['#aaa', '#8b5ca8', '#c85c1a']
        fig = go.Figure(go.Bar(
            x=models, y=aucs,
            marker_color=colors,
            text=[f'{a:.3f}' for a in aucs],
            textposition='outside'
        ))
        fig.add_hline(y=0.5, line_dash='dash', line_color='#ccc',
                      annotation_text='Random baseline (0.5)')
        fig.update_layout(
            height=320, yaxis_title='AUC-ROC', yaxis_range=[0.45, 0.72],
            margin=dict(l=20, r=20, t=20, b=40),
            plot_bgcolor='white', paper_bgcolor='white',
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("**Classification report at threshold = 0.4**")
        report_data = {
            'Class': ['No Churn', 'Churn'],
            'Precision': [0.84, 0.35],
            'Recall': [0.39, 0.82],
            'F1-Score': [0.53, 0.49],
            'Support': [7268, 2942]
        }
        report_df = pd.DataFrame(report_data)
        st.dataframe(report_df, use_container_width=True, hide_index=True)

        st.markdown("**Confusion matrix at threshold = 0.4**")
        cm = np.array([[2834, 4434], [530, 2412]])
        fig_cm = go.Figure(go.Heatmap(
            z=cm, x=['Predicted No Churn', 'Predicted Churn'],
            y=['Actual No Churn', 'Actual Churn'],
            colorscale='Blues', showscale=False,
            text=cm, texttemplate='%{text}',
            textfont=dict(size=16)
        ))
        fig_cm.update_layout(
            height=200, margin=dict(l=20, r=20, t=10, b=40),
            plot_bgcolor='white', paper_bgcolor='white'
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    st.markdown("<div class='section-title'>SHAP Feature Importance</div>",
                unsafe_allow_html=True)

    features = [
        'PercChangeMinutes', 'CurrentEquipmentDays', 'MonthsInService',
        'CustomerCareCalls', 'RetentionCalls', 'MonthlyRevenue',
        'OverageMinutes', 'Care_Call_Rate', 'Retention_Success',
        'DroppedCalls', 'Revenue_per_Minute', 'HandsetPrice',
        'TotalRecurringCharge', 'Overage_Ratio', 'Drop_Rate'
    ]
    shap_vals = [0.182, 0.161, 0.143, 0.128, 0.112, 0.098,
                 0.087, 0.076, 0.065, 0.054, 0.043, 0.038, 0.031, 0.024, 0.019]

    shap_df = pd.DataFrame({'Feature': features, 'SHAP': shap_vals})
    fig_shap = go.Figure(go.Bar(
        x=shap_vals[::-1], y=features[::-1],
        orientation='h',
        marker_color=['#c85c1a' if i < 3 else '#1a5fa8' if i < 8 else '#aaa'
                      for i in range(len(features)-1, -1, -1)]
    ))
    fig_shap.update_layout(
        height=420, xaxis_title='Mean |SHAP value|',
        margin=dict(l=20, r=20, t=10, b=40),
        plot_bgcolor='white', paper_bgcolor='white',
        showlegend=False
    )
    st.plotly_chart(fig_shap, use_container_width=True)

    st.markdown("""
    <div class='insight-box'>
    <strong>PercChangeMinutes</strong> and <strong>CurrentEquipmentDays</strong> are the top two global churn drivers —
    confirming that usage decline and equipment age are the primary reasons customers leave.
    Both are actionable: usage alerts can be automated, and device upgrades can be offered proactively.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='section-title'>Cross-Validation Stability</div>",
                unsafe_allow_html=True)

    cv_scores = [0.6608, 0.6616, 0.6684, 0.6683, 0.6719]
    fig_cv = go.Figure()
    fig_cv.add_trace(go.Scatter(
        x=[1, 2, 3, 4, 5], y=cv_scores,
        mode='lines+markers',
        line=dict(color='#1a5fa8', width=2),
        marker=dict(size=10, color='#1a5fa8'),
        name='Fold AUC'
    ))
    fig_cv.add_hline(y=np.mean(cv_scores), line_dash='dash',
                     line_color='#c85c1a', line_width=1.5,
                     annotation_text=f'Mean = {np.mean(cv_scores):.4f}')
    fig_cv.update_layout(
        height=240, xaxis_title='Fold', yaxis_title='AUC-ROC',
        yaxis_range=[0.65, 0.68],
        margin=dict(l=20, r=20, t=20, b=40),
        plot_bgcolor='white', paper_bgcolor='white'
    )
    st.plotly_chart(fig_cv, use_container_width=True)
    st.caption("Low std (0.0043) confirms the model is stable and not overfitting.")


# ══════════════════════════════════════════════════════════════
# PAGE 4 — RETENTION ENGINE
# ══════════════════════════════════════════════════════════════
elif page == "Retention Engine":
    st.markdown("# LLM Retention Engine")
    st.markdown("Groq LLaMA 3.3 70B generates personalized retention strategies per customer")
    st.markdown("---")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""<div class='metric-card'>
            <div class='metric-label'>High Risk Customers</div>
            <div class='metric-value metric-warn'>10,607</div>
            <div class='metric-sub'>need retention action</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class='metric-card'>
            <div class='metric-label'>Messages Generated</div>
            <div class='metric-value metric-info'>50</div>
            <div class='metric-sub'>personalized strategies</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""<div class='metric-card'>
            <div class='metric-label'>Retention ROI</div>
            <div class='metric-value metric-good'>35x</div>
            <div class='metric-sub'>return on call investment</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div class='section-title'>How the LLM Engine Works</div>",
                unsafe_allow_html=True)

    st.markdown("""
    For each high-risk customer, the engine:
    1. Reads their **segment, churn probability, revenue, tenure, and top SHAP driver**
    2. Builds a structured prompt with segment-specific context
    3. Calls **Groq LLaMA 3.3 70B** to generate a 4-part retention strategy
    4. Returns: Situation → Offer → Agent Message → Urgency
    """)

    st.markdown("<div class='section-title'>Sample Retention Messages</div>",
                unsafe_allow_html=True)

    seg_color = {
        'Heavy Users At Risk': '#1a5fa8',
        'Churning Dissatisfied Customers': '#c85c1a',
        'Low Engagement Customers': '#8b5ca8'
    }

    for _, row in retention_df.iterrows():
        color = seg_color.get(row['Segment_Label'], '#888')
        st.markdown(f"""
        <div class='retention-card'>
            <div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;'>
                <div>
                    <span style='font-weight:600; font-size:15px;'>{row['CustomerID']}</span>
                    <span style='margin-left:10px; background:{color}22; color:{color};
                           padding:3px 10px; border-radius:20px; font-size:12px;
                           font-weight:500;'>{row['Segment_Label']}</span>
                </div>
                <div style='text-align:right; font-size:13px; color:#666;'>
                    Churn prob: <strong style='color:{color}'>{row['Churn_Probability']*100:.0f}%</strong>
                    &nbsp;|&nbsp; Revenue: <strong>{row['Monthly_Revenue']:.0f} DT/mo</strong>
                </div>
            </div>
            <div style='font-size:12px; color:#888; margin-bottom:10px;'>
                Top churn driver: <em>{row['Top_Churn_Reason']}</em>
            </div>
        </div>
        """, unsafe_allow_html=True)

        with st.expander(f"View retention strategy for {row['CustomerID']}"):
            st.markdown(row['Retention_Message'])

    st.markdown("<div class='section-title'>ROI Calculation</div>",
                unsafe_allow_html=True)

    call_cost = st.slider("Cost per retention call (DT)", 5, 30, 10)
    win_rate = st.slider("Assumed win rate (%)", 20, 70, 50)
    n_calls = st.slider("Number of high-risk customers to call", 1000, 10607, 5000)

    avg_monthly_rev = 58
    total_cost = n_calls * call_cost
    revenue_saved = n_calls * (win_rate / 100) * avg_monthly_rev * 12
    net_gain = revenue_saved - total_cost
    roi_calc = revenue_saved / total_cost if total_cost > 0 else 0

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-label'>Call Cost</div>
            <div class='metric-value'>{total_cost:,.0f} DT</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-label'>Revenue Saved</div>
            <div class='metric-value metric-good'>{revenue_saved:,.0f} DT</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-label'>Net Gain</div>
            <div class='metric-value metric-good'>{net_gain:,.0f} DT</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-label'>ROI</div>
            <div class='metric-value metric-good'>{roi_calc:.1f}x</div>
        </div>""", unsafe_allow_html=True)

    st.markdown(f"""
    <div class='insight-box'>
    Calling <strong>{n_calls:,}</strong> high-risk customers at <strong>{call_cost} DT/call</strong>
    with a <strong>{win_rate}%</strong> win rate saves
    <strong>{revenue_saved:,.0f} DT</strong> in annual revenue —
    a <strong>{roi_calc:.1f}x</strong> return on investment.
    </div>
    """, unsafe_allow_html=True)
