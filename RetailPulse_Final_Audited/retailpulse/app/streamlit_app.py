"""RetailPulse AI executive dashboard, styled after the user's reference image."""
import json, sys
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
OUT, PROC = ROOT / "reports", ROOT / "data" / "processed"
st.set_page_config(page_title="RetailPulse AI | Executive Overview", page_icon="📊", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
.stApp{background:#080e18;color:#edf2f8}.block-container{max-width:1800px;padding-top:.8rem;padding-bottom:2rem}
[data-testid="stHeader"]{background:#080e18} section[data-testid="stSidebar"]{background:#0b121e;border-right:1px solid #253142}
.rp-brand{display:flex;align-items:center;gap:12px;padding:4px 0 10px}.rp-logo{display:flex;align-items:center;justify-content:center;width:44px;height:44px;border-radius:13px;background:linear-gradient(135deg,#ff9c45,#f26b21);color:#111923;font-size:25px;font-weight:900}
.rp-name{font-size:1.8rem;font-weight:800;color:#f4f7fb;letter-spacing:-.04em}.rp-name span{color:#ff9138}.rp-subtitle{font-size:.88rem;color:#91a0b4}
h1,h2,h3{color:#f2f5fa;letter-spacing:-.025em}p,label,.stCaption{color:#aab6c6}
[data-testid="stMetric"]{background:linear-gradient(145deg,#151f2c,#101925);border:1px solid #283444;padding:16px;border-radius:12px;min-height:100px}
[data-testid="stMetricLabel"]{color:#bfc9d7}[data-testid="stMetricValue"]{color:#f5f7fb;font-weight:750}
[data-testid="stPlotlyChart"],[data-testid="stDataFrame"]{background:#111b28;border:1px solid #263344;border-radius:12px;overflow:hidden}
.stButton>button{background:linear-gradient(135deg,#ff9438,#ed6c20);border:1px solid #f48b34;color:#111923;border-radius:9px;font-weight:700}
[data-testid="stTabs"] button[role="tab"][aria-selected="true"]{color:#ff9b48;border-bottom-color:#ff8b2b}
.rp-panel{border:1px solid #4b3525;border-radius:13px;padding:17px;background:linear-gradient(120deg,#1c2027,#121b27)}.rp-orange{color:#ff9c48;font-weight:750}
hr{border-color:#273344}
</style>""", unsafe_allow_html=True)
st.markdown('<div class="rp-brand"><div class="rp-logo">ϟ</div><div><div class="rp-name">Retail<span>Pulse AI</span></div><div class="rp-subtitle">Executive Overview · Customer Intelligence · Demand Forecasting</div></div></div>', unsafe_allow_html=True)
with st.sidebar:
    st.markdown("## 🟠 RetailPulse AI")
    st.caption("Predict · Understand · Optimize")
    page = st.radio("NAVIGATION", ["Overview", "Forecasting", "Customer Segments", "Churn Radar", "Inventory Ops", "Model Health"], label_visibility="visible")
    st.divider()
    st.caption("DATA STATUS")
    st.success("Local analytics workspace")
    st.caption("Bundled dataset is synthetic demo data. Replace with authorized business data for real decisions.")

summary_path = OUT / "summary.json"
if not summary_path.exists():
    st.warning("Dashboard reports are not present yet. Prepare the bundled data pipeline first.")
    if st.button("Prepare dashboard data"):
        try:
            from retailpulse.pipeline import run
            with st.spinner("Preparing analytics and model outputs..."): run()
            st.rerun()
        except Exception as e: st.error(f"Pipeline failed: {e}")
    st.stop()

def read_csv(path, dates=None):
    if not path.exists(): return pd.DataFrame()
    return pd.read_csv(path, parse_dates=dates)
try:
    metrics=json.loads(summary_path.read_text(encoding="utf-8"))
    sales=read_csv(PROC/"sales_clean.csv", ["date"])
    daily=read_csv(PROC/"daily_demand.csv", ["date"])
    forecast=read_csv(OUT/"demand_forecast.csv", ["date"])
    segments=read_csv(OUT/"customer_segments.csv")
    churn=read_csv(OUT/"churn_scores.csv")
    inventory=read_csv(OUT/"inventory_recommendations.csv")
except Exception as e:
    st.error(f"Could not load dashboard data: {e}"); st.stop()
if sales.empty:
    st.error("sales_clean.csv is empty or missing. Run the data pipeline again."); st.stop()

if page == "Overview":
    title_col, filter_col = st.columns([3,1.2])
    title_col.markdown("### Executive Overview")
    title_col.caption("Business performance · revenue signals · customer health")
    with filter_col:
        dates=st.date_input("Reporting period", value=(sales.date.min().date(),sales.date.max().date()), key="date_filter")
    view=sales.copy()
    if isinstance(dates,(tuple,list)) and len(dates)==2: view=view[(view.date.dt.date>=dates[0])&(view.date.dt.date<=dates[1])]
    if view.empty: view=sales.copy(); st.info("No records in this period; displaying the complete dataset.")
    orders=view.groupby("order_id").revenue.sum() if "order_id" in view else view.revenue
    highrisk=int((churn.risk_tier.astype(str)=="High").sum()) if not churn.empty and "risk_tier" in churn else 0
    churn_rate=(highrisk/len(churn)) if len(churn) else 0
    conversion=float(metrics.get("conversion_rate",0.0342))
    k=st.columns(5)
    k[0].metric("Total Revenue",f"₹{view.revenue.sum():,.0f}")
    k[1].metric("Users",f"{view.customer_id.nunique():,}" if "customer_id" in view else "N/A")
    k[2].metric("Conversion Rate",f"{conversion:.2%}",help="Demo indicator; session/visit data was not provided")
    k[3].metric("Avg. Order Value",f"₹{orders.mean():,.2f}")
    k[4].metric("Churn Risk",f"{churn_rate:.1%}",help="Share of scored customers labelled high risk")
    main, side=st.columns([2.35,1])
    with main:
        st.subheader("Revenue Over Time")
        st.caption("Daily actual revenue compared with a rolling expected baseline")
        rev=view.set_index("date").resample("D").revenue.sum().rename("Revenue").reset_index()
        rev["Expected"]=rev.Revenue.rolling(7,min_periods=1).mean()
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=rev.date,y=rev.Revenue,mode="lines+markers",name="Revenue",line=dict(color="#ff862d",width=2.5),marker=dict(size=4),fill="tozeroy",fillcolor="rgba(255,134,45,.10)"))
        fig.add_trace(go.Scatter(x=rev.date,y=rev.Expected,mode="lines",name="Revenue (Expected)",line=dict(color="#78869a",dash="dash",width=1.5)))
        fig.update_layout(template="plotly_dark",height=355,margin=dict(l=8,r=8,t=10,b=8),paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font_color="#b7c3d4",legend=dict(orientation="h",y=1.12,x=0),xaxis_title=None,yaxis_title=None,hovermode="x unified")
        fig.update_xaxes(showgrid=False); fig.update_yaxes(gridcolor="#263344",tickprefix="₹")
        st.plotly_chart(fig,use_container_width=True,config={"displaylogo":False,"responsive":True})
    with side:
        st.subheader("✦ AI Insight")
        cat=view.groupby("category",as_index=False).revenue.sum().sort_values("revenue",ascending=False) if "category" in view else pd.DataFrame()
        top=str(cat.iloc[0]["category"]) if not cat.empty else "Top category unavailable"
        share=(cat.iloc[0].revenue/cat.revenue.sum()*100) if not cat.empty and cat.revenue.sum() else 0
        st.markdown(f'<div class="rp-panel"><div class="rp-orange">DEMO INSIGHT · DATA-DERIVED</div><h3 style="margin-top:12px">{top} leads revenue</h3><p>This category contributes <b style="color:#ff9c48">{share:.1f}%</b> of selected-period revenue.</p><hr><p>Review this signal alongside inventory and campaign data. This is correlation/summary, not proof of cause.</p></div>',unsafe_allow_html=True)
        st.write("")
        st.metric("High-risk customers",f"{highrisk:,}")
        risk_items=int(inventory.status.astype(str).str.contains("risk|reorder",case=False).sum()) if not inventory.empty and "status" in inventory else 0
        st.metric("Inventory items to review",f"{risk_items:,}")
    b1,b2,b3=st.columns([1.05,1.05,1.35])
    with b1:
        st.subheader("Revenue by Channel")
        col="channel" if "channel" in view.columns else "category"
        by=view.groupby(col,as_index=False).revenue.sum().sort_values("revenue").tail(7).rename(columns={col:"Channel / Category"})
        f=px.bar(by,x="revenue",y="Channel / Category",orientation="h",template="plotly_dark",color_discrete_sequence=["#ff862d"])
        f.update_layout(height=285,margin=dict(l=4,r=4,t=6,b=6),paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font_color="#b7c3d4",xaxis_title=None,yaxis_title=None); f.update_xaxes(gridcolor="#263344",tickprefix="₹")
        st.plotly_chart(f,use_container_width=True,config={"displaylogo":False})
        if col=="category": st.caption("Channel field unavailable; showing revenue by category instead.")
    with b2:
        st.subheader("Users by Segment")
        if not segments.empty and "segment" in segments:
            sc=segments.segment.value_counts().rename_axis("Segment").reset_index(name="Users")
            f=px.pie(sc,names="Segment",values="Users",hole=.64,template="plotly_dark",color_discrete_sequence=["#377fbd","#ff862d","#7554cf","#17a899","#d8a02d"])
            f.update_traces(textinfo="percent",marker_line_color="#111b28",marker_line_width=3)
            f.update_layout(height=285,margin=dict(l=0,r=0,t=6,b=0),paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font_color="#b7c3d4",legend=dict(orientation="h",y=-.18,x=0,font=dict(size=10)))
            st.plotly_chart(f,use_container_width=True,config={"displaylogo":False})
        else: st.info("Segment data unavailable.")
    with b3:
        st.subheader("Recent Anomalies")
        anomaly=rev.copy(); anomaly["Baseline"]=anomaly.Revenue.rolling(14,min_periods=3).median()
        anomaly["Deviation %"]=(anomaly.Revenue-anomaly.Baseline)/anomaly.Baseline.replace(0,pd.NA)*100
        anomaly=anomaly[anomaly["Deviation %"].abs()>=25].sort_values("date",ascending=False).head(6)
        if anomaly.empty: st.info("No large daily revenue deviations detected in this period.")
        else:
            anomaly=anomaly[["date","Revenue","Deviation %"]].copy(); anomaly["Date"]=anomaly.date.dt.strftime("%d %b %Y"); anomaly["Impact"]=anomaly["Deviation %"].apply(lambda x:"High ↑" if x>=45 else "Medium ↑" if x>=25 else "High ↓" if x<=-45 else "Medium ↓"); anomaly["Deviation"]=anomaly["Deviation %"].map(lambda x:f"{x:+.1f}%")
            st.dataframe(anomaly[["Date","Impact","Deviation","Revenue"]],hide_index=True,use_container_width=True)
        st.caption("Rolling-baseline flags; not a separately trained anomaly model.")
    st.markdown('<div class="rp-panel"><div class="rp-orange">✦ RETAILPULSE AI OPERATIONS BRIEF</div><p>Metrics below are calculated from the included demo data and local model reports. Validate with authorized production data before business decisions.</p></div>',unsafe_allow_html=True)
    a,b,c=st.columns(3); a.metric("Forecast weekly WAPE",f"{metrics.get('forecast',{}).get('weekly_product_wape',0):.1%}"); b.metric("Largest revenue category",top); c.metric("Data source",str(metrics.get("data_source","Unknown")).replace("_"," ").title())
    with st.expander("Data quality / ETL details"): st.json(metrics.get("etl",{}))
elif page == "Forecasting":
    st.title("Demand Forecasting")
    st.caption("Product-level demand projection with a historical holdout backtest.")
    fc=forecast
    if fc.empty or daily.empty:
        st.warning("Forecast files are missing. Run the pipeline to regenerate them.")
    else:
        product=st.selectbox("Select product",sorted(daily.product_id.unique()))
        hist=daily[daily.product_id==product].sort_values("date").tail(90)[["date","units"]].rename(columns={"units":"Historical demand"})
        fut=fc[fc.product_id==product].sort_values("date").copy()
        ffig=go.Figure()
        ffig.add_trace(go.Scatter(x=hist.date,y=hist["Historical demand"],mode="lines",name="Historical demand",line=dict(color="#ff862d",width=2.4)))
        if {"lower","upper"}.issubset(fut.columns):
            ffig.add_trace(go.Scatter(x=fut.date,y=fut.upper,mode="lines",line=dict(width=0),showlegend=False,name="Upper estimate"))
            ffig.add_trace(go.Scatter(x=fut.date,y=fut.lower,mode="lines",line=dict(width=0),fill="tonexty",fillcolor="rgba(255,134,45,.15)",name="Forecast range"))
        if "forecast" in fut:
            ffig.add_trace(go.Scatter(x=fut.date,y=fut.forecast,mode="lines+markers",name="Forecast",line=dict(color="#ffd08a",width=2.4,dash="dash"),marker=dict(size=4)))
        ffig.update_layout(template="plotly_dark",height=390,margin=dict(l=8,r=8,t=18,b=8),paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font_color="#b7c3d4",hovermode="x unified",legend=dict(orientation="h",y=1.12,x=0),xaxis_title=None,yaxis_title="Units")
        ffig.update_xaxes(showgrid=False); ffig.update_yaxes(gridcolor="#263344",rangemode="tozero")
        st.plotly_chart(ffig,use_container_width=True,config={"displaylogo":False,"responsive":True})
        fm=metrics.get("forecast",{})
        m1,m2,m3=st.columns(3)
        m1.metric("Weekly product WAPE",f"{fm.get('weekly_product_wape',0):.1%}")
        m2.metric("Total-level WAPE",f"{fm.get('total_level_wape',0):.1%}")
        m3.metric("Forecast horizon",f"{len(fut)} days")
        st.subheader("Backtest by product")
        pm=read_csv(OUT/"forecast_product_metrics.csv")
        if not pm.empty: st.dataframe(pm,use_container_width=True,hide_index=True)
elif page == "Customer Segments":
    st.title("Customer Segments")
    st.caption("RFM-style segmentation from the bundled demonstration dataset.")
    if not segments.empty and "segment" in segments:
        counts=segments.segment.value_counts().rename_axis("Segment").reset_index(name="Customers")
        a,b=st.columns([1.2,1])
        with a:
            fig=px.bar(counts,x="Segment",y="Customers",color="Segment",template="plotly_dark",color_discrete_sequence=["#ff862d","#377fbd","#7554cf","#17a899"])
            fig.update_layout(height=330,showlegend=False,margin=dict(l=5,r=5,t=10,b=5),paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font_color="#b7c3d4")
            st.plotly_chart(fig,use_container_width=True,config={"displaylogo":False})
        with b:
            st.metric("Customers segmented",f"{segments.customer_id.nunique():,}" if "customer_id" in segments else f"{len(segments):,}")
            st.metric("Segments found",f"{segments.segment.nunique():,}")
            st.dataframe(read_csv(OUT/"segment_summary.csv"),use_container_width=True,hide_index=True)
        with st.expander("Customer-level segment assignments"):
            st.dataframe(segments,use_container_width=True,hide_index=True)
    else: st.info("Segment data unavailable. Run the pipeline to generate customer segments.")
elif page == "Churn Radar":
    st.title("Churn Radar")
    st.caption("Risk tiers are model estimates from synthetic demonstration data; validate before acting on customers.")
    if churn.empty: st.info("Churn scores unavailable. Run the pipeline to generate them.")
    else:
        high=int((churn.risk_tier.astype(str)=="High").sum()) if "risk_tier" in churn else 0
        med=int((churn.risk_tier.astype(str)=="Medium").sum()) if "risk_tier" in churn else 0
        a,b,c=st.columns(3); a.metric("Scored customers",f"{len(churn):,}"); b.metric("High risk",f"{high:,}"); c.metric("Medium risk",f"{med:,}")
        tiers=st.multiselect("Risk tier",sorted(churn.risk_tier.dropna().unique()),default=list(churn.risk_tier.dropna().unique()))
        filtered=churn[churn.risk_tier.isin(tiers)]
        st.dataframe(filtered.head(500),use_container_width=True,hide_index=True)
        if (OUT/"churn_feature_importance.csv").exists():
            st.subheader("Model feature importance")
            imp=read_csv(OUT/"churn_feature_importance.csv")
            if not imp.empty and {"feature","importance"}.issubset(imp.columns):
                fig=px.bar(imp.sort_values("importance").tail(12),x="importance",y="feature",orientation="h",template="plotly_dark",color_discrete_sequence=["#ff862d"])
                fig.update_layout(height=350,margin=dict(l=5,r=5,t=8,b=5),paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font_color="#b7c3d4")
                st.plotly_chart(fig,use_container_width=True,config={"displaylogo":False})
elif page == "Inventory Ops":
    st.title("Inventory Operations")
    st.caption("Suggested actions are calculated from forecast demand and the demo inventory assumptions.")
    if inventory.empty: st.info("Inventory recommendations unavailable. Run the pipeline to regenerate them.")
    else:
        statuses=st.multiselect("Inventory status",sorted(inventory.status.dropna().unique()),default=list(inventory.status.dropna().unique()))
        filtered=inventory[inventory.status.isin(statuses)]
        a,b,c=st.columns(3)
        a.metric("Items reviewed",f"{len(inventory):,}")
        b.metric("Reorder / stockout flags",f"{int(inventory.status.astype(str).str.contains('Reorder|Stockout',case=False).sum()):,}")
        if "revenue_at_risk" in inventory: c.metric("Revenue at risk",f"₹{inventory.revenue_at_risk.sum():,.0f}")
        st.dataframe(filtered,use_container_width=True,hide_index=True)
elif page == "Model Health":
    st.title("Model Health & Data Quality")
    st.caption("Monitoring and validation summary from the latest local pipeline run.")
    fc_metrics=metrics.get("forecast",{}); ch_metrics=metrics.get("churn",{}); drift_metrics=metrics.get("drift",{}); etl=metrics.get("etl",{})
    d=drift_metrics.get("demand",{})
    a,b,c,dcol=st.columns(4)
    a.metric("Weekly forecast WAPE",f"{fc_metrics.get('weekly_product_wape',0):.1%}")
    b.metric("Churn ROC-AUC",f"{ch_metrics.get('models',{}).get(ch_metrics.get('best_model',''),{}).get('roc_auc',0):.3f}")
    c.metric("Demand drift PSI",f"{d.get('psi',0):.4f}")
    dcol.metric("Rows removed by ETL",f"{sum(int(etl.get(k,0)) for k in ['duplicates_removed','null_rows_removed','invalid_value_rows_removed','unknown_product_rows_removed']):,}")
    drift_status=str(d.get("status","unknown")).replace("_"," ").title()
    st.markdown(f'<div class="rp-panel"><div class="rp-orange">PIPELINE STATUS</div><h3 style="margin:8px 0">Demand drift: {drift_status}</h3><p>Data source: {metrics.get("data_source","unknown")} · Last pipeline runtime: {metrics.get("runtime_seconds","N/A")} seconds</p></div>',unsafe_allow_html=True)
    st.subheader("ETL data quality")
    etl_rows=[{"Metric":label,"Rows":int(etl.get(key,0))} for key,label in [("rows_raw","Raw rows"),("rows_clean","Clean rows"),("duplicates_removed","Duplicates removed"),("null_rows_removed","Rows removed for nulls"),("invalid_value_rows_removed","Rows removed for invalid values"),("unknown_product_rows_removed","Rows removed for unknown products")]]
    st.dataframe(pd.DataFrame(etl_rows),use_container_width=True,hide_index=True)
    st.subheader("Feature drift report")
    drift=read_csv(OUT/"drift_report.csv")
    if not drift.empty: st.dataframe(drift,use_container_width=True,hide_index=True)
    else: st.info("Feature drift report is unavailable.")
    st.subheader("Model comparison")
    comparisons=fc_metrics.get("total_level_comparison_wape",{})
    if comparisons: st.dataframe(pd.DataFrame([{"Model":k,"WAPE":v} for k,v in comparisons.items()]),use_container_width=True,hide_index=True)

st.caption(f"Local demo workspace · data source: {metrics.get('data_source','unknown')} · replace synthetic demo data before production use.")
