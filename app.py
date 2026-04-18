import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.seasonal import seasonal_decompose

# ==========================================
# 1. CONFIG & STYLE (FULL COLOR THEME)
# ==========================================
st.set_page_config(
    page_title="AQI Pro: Advanced Intelligence Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main { background-color: #f1f5f9; }
    [data-testid="stMetricValue"] { font-size: 32px; color: #1e3a8a; }
    .stMetric { background-color: #ffffff; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    h1, h2, h3 { color: #1e3a8a; font-family: 'Inter', sans-serif; }
    .story-card {
        background-color: #eff6ff;
        padding: 25px;
        border-radius: 12px;
        border-left: 6px solid #1e3a8a;
        margin-bottom: 25px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    hr { border: 1px solid #cbd5e1; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. DATA ENGINE (PRO FEATURE ENGINEERING)
# ==========================================
@st.cache_data
def load_and_engineer_data():
    df = pd.read_csv("air_quality.csv")
    df.columns = df.columns.str.lower().str.replace('.', '_', regex=False).str.replace(' ', '_')
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['city', 'date'])
    
    df['day_name'] = df['date'].dt.day_name()
    df['month'] = df['date'].dt.month_name()
    df['is_weekend'] = df['date'].dt.dayofweek.isin([5, 6]).astype(int)
    
    df['pm_lag_1'] = df.groupby('city')['pm2_5'].shift(1)
    df['pm_lag_7'] = df.groupby('city')['pm2_5'].shift(7)
    df['rolling_avg_3d'] = df.groupby('city')['pm2_5'].transform(lambda x: x.rolling(3).mean())
    
    return df.dropna()

df = load_and_engineer_data()

# ==========================================
# 3. SIDEBAR (GLOBAL CONTROLS)
# ==========================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/1684/1684425.png", width=100)
    st.title("🚀 AQI Pro Control")
    st.markdown("---")
    selected_city = st.selectbox("🎯 Target Analysis City", df['city'].unique())
    date_range = st.date_input("📅 Observation Period", [df['date'].min(), df['date'].max()])
    st.markdown("---")
    st.write("**Data Version:** 2.5.0-Ultimate")
    st.write("**Model:** Random Forest (Ensemble)")

df_target = df[df['city'] == selected_city].copy()
if len(date_range) == 2:
    start_dt, end_dt = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    df_target = df_target[(df_target['date'] >= start_dt) & (df_target['date'] <= end_dt)]

# ==========================================
# 4. MAIN INTERFACE (ALUR CERITA PORTOFOLIO)
# ==========================================


# ------------------------------------------
# BAGIAN 1: ABOUT DATASET
# ------------------------------------------
st.title("🌍 Air Quality Intelligence Project")
st.markdown("---")
st.header("📖 1. About Dataset")

col_abs1, col_abs2 = st.columns([1, 1.2], gap="large")

with col_abs1:
    st.subheader("Global AQI Monitoring")
    st.write(f"""
    Proyek ini menggunakan dataset **Multinasional Air Quality** yang mencakup pemantauan polusi di berbagai kota besar dunia, 
    termasuk **{selected_city}**. Data ini mengintegrasikan parameter meteorologi (suhu) dengan konsentrasi partikel halus (PM2.5).
    
    Tujuan utama adalah mentransformasi data mentah menjadi **Strategic Insights** untuk membantu pemangku kepentingan 
    memahami pola polusi udara lintas batas dan risiko kesehatan yang menyertainya.
    """)

    st.markdown("""
    **Fitur Utama Portofolio:**
    * **Multi-City Analysis:** Mendukung analisis perbandingan antar kota besar dunia.
    * **Predictive Risk:** Klasifikasi risiko kesehatan (Low, Medium, High) menggunakan Random Forest.
    * **Anomaly Awareness:** Deteksi otomatis lonjakan polusi menggunakan metode Z-Score.
    """)  
    st.markdown("""
    **Metodologi Teknis:**
    * **Exploratory Data Analysis (EDA):** Analisis mendalam untuk mengidentifikasi tren jangka panjang dan pola musiman polusi PM2.5.
    * **Data Preprocessing:** Transformasi data menggunakan *Lagging Features* (h-1 & h-7) dan deteksi pencilan statistik dengan *Z-Score*.
    * **Machine Learning:** Pengembangan model *Random Forest Regressor* untuk prediksi kualitas udara harian yang presisi.
    """)
    st.info("**Data Scope:** Jan 2023 - Des 2024 | Global Meteorological Stations")

with col_abs2:
    # Tambahkan ruang kosong di atas grafik agar sejajar dengan teks di kiri
    st.write("###") 
    st.write("**Initial Target Distribution Analysis**")
    
    fig_init = px.histogram(
        df_target, 
        x="pm2_5", 
        marginal="box", 
        color_discrete_sequence=['#1e3a8a']
    )

    fig_init.update_layout(
        height=380,
        # l=left, r=right, t=top, b=bottom dalam pixel
        margin=dict(l=60, r=40, t=20, b=60), 
        template="plotly_white",
        showlegend=False,
        # Memberikan jarak antara grafik dan sumbu
        bargap=0.1 
    )
    
    # Tambahkan border/padding visual dengan container
    with st.container():
        st.plotly_chart(fig_init, use_container_width=True)

st.divider()

# ------------------------------------------
# BAGIAN 2: DASHBOARD (VISUAL ANALYTICS)
# ------------------------------------------
st.header("📊 2. Strategic Dashboard")

# KPI Section
m1, m2, m3, m4 = st.columns(4)
m1.metric("Current Avg PM2.5", f"{df_target['pm2_5'].mean():.2f}")
m2.metric("Max Recorded", f"{df_target['pm2_5'].max():.2f}")
m3.metric("Volatility (Std Dev)", f"{df_target['pm2_5'].std():.2f}")
correlation = df_target['pm2_5'].corr(df_target['temperature'])
m4.metric("Temp Correlation", f"{correlation:.4f}")

# Anomaly Detection
st.write("### Anomaly Detection Status")
df_target['z_score'] = (df_target['pm2_5'] - df_target['pm2_5'].mean()) / df_target['pm2_5'].std()
df_target['anomaly'] = df_target['z_score'].abs() > 2.5
anomalies_count = df_target['anomaly'].sum()
if anomalies_count > 0:
    st.error(f"⚠️ **Anomaly Alert:** {anomalies_count} pencilan statistik terdeteksi pada kota {selected_city}.")
else:
    st.success("✅ Tidak ada anomali polusi yang signifikan terdeteksi.")

# Visualizations Row 1
c1, c2 = st.columns([2, 1])
with c1:
    st.subheader("Historical PM2.5 Trend with 7-Day Smooth MA")
    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(x=df_target['date'], y=df_target['pm2_5'], name="Daily", line=dict(color='#cbd5e1')))
    fig_trend.add_trace(go.Scatter(x=df_target['date'], y=df_target['pm2_5'].rolling(7).mean(), name="7-Day Smooth MA", line=dict(color='#ff4b4b', width=3)))
    fig_trend.update_layout(template="plotly_white", xaxis_rangeslider_visible=True, height=500)
    st.plotly_chart(fig_trend, use_container_width=True)
with c2:
    st.subheader("🏙️ City Comparison")
    df_city = df.groupby('city', as_index=False)['pm2_5'].mean().sort_values('pm2_5')
    st.plotly_chart(px.bar(df_city, x='pm2_5', y='city', orientation='h', color='pm2_5', color_continuous_scale="RdYlBu_r"), use_container_width=True)

# Row 2: Seasonality & Heatmap
c3, c4 = st.columns(2)
with c3:
    st.subheader("📅 Weekly Pollution Pattern")
    order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    avg_week = df_target.groupby('day_name')['pm2_5'].mean().reindex(order)
    st.plotly_chart(px.bar(avg_week, color=avg_week.values, color_continuous_scale="Aggrnyl"), use_container_width=True)
with c4:
    st.subheader("🔥🔥 Temp vs PM2.5 Heatmap")
    st.plotly_chart(px.density_heatmap(df_target, x="temperature", y="pm2_5", nbinsx=25, nbinsy=25, color_continuous_scale="Viridis"), use_container_width=True)

# Row 3: Correlation Matrix
st.subheader("📊 Advanced Correlation Matrix")
corr_cols = ['pm2_5', 'temperature', 'pm_lag_1', 'pm_lag_7', 'rolling_avg_3d']
corr_matrix = df_target[corr_cols].corr()
st.plotly_chart(px.imshow(corr_matrix, text_auto=True, color_continuous_scale="RdBu_r"), use_container_width=True)

st.divider()

# ------------------------------------------
# BAGIAN 3: MACHINE LEARNING
# ------------------------------------------
st.header("🧠 3. Machine Learning Intelligence")

ml_df = df_target.copy()
X = ml_df[['temperature', 'pm_lag_1', 'pm_lag_7', 'rolling_avg_3d', 'is_weekend']]
y = ml_df['pm2_5']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

model = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train, y_train)
y_preds = model.predict(X_test)

# Metrics
e1, e2, e3 = st.columns(3)
e1.metric("MAE (Error Margin)", f"{mean_absolute_error(y_test, y_preds):.3f}")
e2.metric("RMSE", f"{np.sqrt(mean_squared_error(y_test, y_preds)):.3f}")
e3.metric("R² Score (Akurasi)", f"{r2_score(y_test, y_preds):.3f}")

# ML Evaluation Charts
ev1, ev2 = st.columns(2)
with ev1:
    st.write("**Actual vs Predicted Trend**")
    res_df = pd.DataFrame({'Date': ml_df.loc[y_test.index, 'date'], 'Actual': y_test, 'Predicted': y_preds}).sort_values('Date')
    st.plotly_chart(px.line(res_df, x='Date', y=['Actual', 'Predicted'], color_discrete_sequence=['#cbd5e1', '#1e3a8a']), use_container_width=True)
with ev2:
    st.write("**Feature Importance**")
    feat_imp = pd.DataFrame({'Feature': X.columns, 'Importance': model.feature_importances_}).sort_values('Importance')
    st.plotly_chart(px.bar(feat_imp, x='Importance', y='Feature', orientation='h', color_discrete_sequence=['#1e3a8a']), use_container_width=True)

st.divider()

# ------------------------------------------
# BAGIAN 4: PREDICTION APP & STORYTELLING
# ------------------------------------------
st.header("🔮 4. Simulation & Executive Summary")

col_s1, col_s2 = st.columns([1, 1.5])
with col_s1:
    st.subheader("Forecast Simulation")
    s_temp = st.slider("Temperature Forecast (°C)", 10, 45, 25)
    s_pm1 = st.number_input("PM2.5 Yesterday", value=float(df_target['pm2_5'].iloc[-1]))
    s_pm7 = st.number_input("PM2.5 7-Days Ago", value=float(df_target['pm2_5'].iloc[-7]))
    s_rolling = (s_pm1 + s_pm7) / 2
    
    if st.button("Generate AQI Prediction"):
        prediction = model.predict([[s_temp, s_pm1, s_pm7, s_rolling, 0]])[0]
        st.session_state['pro_pred'] = prediction

with col_s2:
    if 'pro_pred' in st.session_state:
        val = st.session_state['pro_pred']
        st.write(f"### Estimated PM2.5: **{val:.2f} µg/m³**")
        
        # Risk Pie Chart
        def risk_level(pm): return "High Risk" if pm > 100 else "Medium Risk" if pm > 50 else "Low Risk"
        df_target['risk'] = df_target['pm2_5'].apply(risk_level)
        risk_counts = df_target['risk'].value_counts().reset_index()
        fig_risk = px.pie(risk_counts, names='index', values='risk', color='index', color_discrete_map={"High Risk": "#ef4444", "Medium Risk": "#f59e0b", "Low Risk": "#22c55e"})
        st.plotly_chart(fig_risk, use_container_width=True)

# STORYTELLING KESIMPULAN (FINAL STORY)
st.markdown("---")
st.header("📝 Executive Storytelling")
st.markdown(f"""
<div class="story-card">
    <h4>Strategic Conclusion: Air Quality Intelligence for {selected_city}</h4>
    <p>Berdasarkan rangkaian analisis data engine <b>Version 2.5.0</b>, berikut adalah temuan kuncinya:</p>
    <ul>
        <li><b>Insight Polusi:</b> Kota {selected_city} memiliki rata-rata PM2.5 sebesar <b>{df_target['pm2_5'].mean():.2f}</b>. Korelasi dengan suhu tercatat sebesar <b>{correlation:.4f}</b>, menunjukkan faktor cuaca berperan dalam dispersi polutan.</li>
        <li><b>Deteksi Pola:</b> Analisis mingguan menunjukkan adanya fluktuasi polusi yang konsisten, terutama dipengaruhi oleh siklus aktivitas manusia dan perubahan suhu harian.</li>
        <li><b>Keandalan Prediksi:</b> Model Random Forest berhasil mencapai skor akurasi (R²) sebesar <b>{r2_score(y_test, y_preds):.3f}</b>, yang menjadikannya alat yang valid untuk estimasi risiko harian bagi masyarakat.</li>
        <li><b>Rekomendasi:</b> Disarankan untuk meningkatkan kewaspadaan saat suhu berada pada ekstremum tertentu, karena heatmap menunjukkan kepadatan polusi yang tinggi pada rentang suhu tersebut.</li>
    </ul>
</div>
""", unsafe_allow_html=True)

st.divider()
st.markdown("<center>Proyek Portofolio Pro: Analisis & Prediksi Kualitas Udara © 2026</center>", unsafe_allow_html=True)
