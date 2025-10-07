import glob
import os

import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image
from st_aggrid import AgGrid, GridOptionsBuilder
from streamlit_option_menu import option_menu

# Use absolute import so `streamlit run src/dashboard.py` works
try:
    from src.config import load_config  # type: ignore
except Exception:
    load_config = None  # fallback below

if load_config is not None:
    CFG = load_config()
    PROC_DIR = CFG.paths.processed_dir
    VIS_DIR = CFG.paths.visuals_dir
    EXPORTS_DIR = CFG.paths.exports_dir
else:
    PROC_DIR = os.path.join('data', 'processed')
    VIS_DIR = os.path.join(PROC_DIR, 'visualizations')
    EXPORTS_DIR = os.path.join(PROC_DIR, 'exports')

# Cache helpers
@st.cache_data(show_spinner=False)
def _load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def display_dashboard():
    """
    Main function to display the CrystalBall dashboard.
    """
    st.set_page_config(page_title="CrystalBall Dashboard", page_icon="🔮", layout="wide")
    st.markdown("""
        <style>
        .block-container {padding-top: 1rem;}
        </style>
    """, unsafe_allow_html=True)
    st.title("🔮 CrystalBall Analysis Dashboard")

    st.sidebar.title("Navigation")
    # Discover available series from rankings and model summaries
    ranking_csvs = sorted(glob.glob(os.path.join(PROC_DIR, "*_rankings.csv")))
    model_summaries = sorted(glob.glob(os.path.join(PROC_DIR, "*_model_summary.csv")))
    series_keys = sorted({os.path.basename(f).replace('_rankings.csv','') for f in ranking_csvs} |
                         {os.path.basename(f).replace('_model_summary.csv','') for f in model_summaries})

    if not series_keys:
        st.warning("No per-series outputs found. Please run the analysis first.")
        return

    selected_series = st.sidebar.selectbox("Select series", series_keys)
    # Top navigation
    nav = option_menu(
        menu_title="Navigate",
        options=["Overview", "Leaderboard", "Forecasts"],
        icons=["bar-chart", "trophy", "graph-up"],
        orientation="horizontal",
        default_index=0,
    )
    if st.sidebar.button("Refresh data cache"):
        st.cache_data.clear()
    st.sidebar.markdown("---")
    st.sidebar.caption(f"Processed dir: {PROC_DIR}")

    if selected_series and nav == "Overview":
        safe_base = selected_series
        st.header(f"Analysis for: {safe_base}")

        # Display Rankings
        st.subheader("Model Rankings")
        rankings_path = os.path.join(PROC_DIR, f"{safe_base}_rankings.csv")
        if os.path.exists(rankings_path):
            rankings_df = _load_csv(rankings_path)
            # Interactive table with AgGrid
            gob = GridOptionsBuilder.from_dataframe(rankings_df)
            gob.configure_default_column(resizable=True, sortable=True, filter=True)
            gob.configure_selection('single')
            grid_options = gob.build()
            AgGrid(rankings_df, gridOptions=grid_options, height=320, theme="balham")
            # Plotly bar for weighted score
            if 'weighted_score' in rankings_df.columns:
                fig = px.bar(rankings_df.sort_values('weighted_score'),
                             x='weighted_score', y='model', orientation='h',
                             title='Weighted Scores', color='model')
                fig.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            # Download button
            csv_bytes = rankings_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download rankings CSV", csv_bytes, file_name=f"{safe_base}_rankings.csv", mime="text/csv")

            ranking_viz_path = os.path.join(VIS_DIR, f"{safe_base}_rankings.png")
            if os.path.exists(ranking_viz_path):
                st.image(Image.open(ranking_viz_path), caption="Model Performance Rankings")
        else:
            st.info("No ranking data available for this analysis.")

        # Display Adherence Report
        st.subheader("Forecast Adherence Analysis")
        adherence_viz_path = os.path.join(VIS_DIR, f"{safe_base}_adherence_analysis.png")
        if os.path.exists(adherence_viz_path):
            st.image(Image.open(adherence_viz_path), caption="Forecast Pattern Adherence Analysis")
        else:
            st.info("No adherence report available.")

        # Train vs Holdout visual + PDF download
        st.subheader("Train vs Holdout")
        train_test_pngs = sorted(glob.glob(os.path.join(VIS_DIR, f"{safe_base}_*_train_test.png")))
        if train_test_pngs:
            cols = st.columns(2)
            for i, png_path in enumerate(train_test_pngs):
                with cols[i % 2]:
                    png_name = os.path.basename(png_path)
                    # Display the image
                    st.image(Image.open(png_path), caption=png_name)
                    # Matching PDF path lives in exports dir with same basename
                    pdf_basename = png_name.replace('.png', '.pdf')
                    pdf_path = os.path.join(EXPORTS_DIR, pdf_basename)
                    if os.path.exists(pdf_path):
                        with open(pdf_path, 'rb') as f:
                            st.download_button(
                                label=f"Download PDF: {pdf_basename}",
                                data=f.read(),
                                file_name=pdf_basename,
                                mime="application/pdf",
                                use_container_width=True,
                            )
                    else:
                        st.caption("PDF not found for this visual.")
        else:
            st.info("No train/holdout visuals available. Run the visuals generator to create them.")
            
        # Display Forecast Plots
        # Forecasts section moved to separate tab
    if selected_series and nav == "Forecasts":
        safe_base = selected_series
        st.subheader("Forecast Visualizations")
        forecast_plots = glob.glob(os.path.join(VIS_DIR, f"{safe_base}*forecast.png"))
        if forecast_plots:
            cols = st.columns(2)
            for i, plot in enumerate(sorted(forecast_plots)):
                model_name = os.path.basename(plot).replace(f"{safe_base}_", "").replace("_forecast.png", "")
                with cols[i % 2]:
                    st.image(Image.open(plot), caption=f"Forecast: {model_name}")
        else:
            st.info("No forecast plots available.")

    # Global leaderboard
    if nav == "Leaderboard":
        st.markdown("---")
        st.subheader("🏆 Global Leaderboard")
        lb_csv = os.path.join(VIS_DIR, 'leaderboard.csv')
        lb_png = os.path.join(VIS_DIR, 'leaderboard.png')
        if os.path.exists(lb_csv):
            lb = _load_csv(lb_csv)
            gob = GridOptionsBuilder.from_dataframe(lb)
            gob.configure_default_column(resizable=True, sortable=True, filter=True)
            st.markdown("### Leaderboard Table")
            AgGrid(lb, gridOptions=gob.build(), height=320, theme="balham")
            # Plotly for mean_rank
            if 'mean_rank' in lb.columns and 'model' in lb.columns:
                fig = px.bar(lb.sort_values('mean_rank'), x='mean_rank', y='model', orientation='h', title='Mean Rank by Model', color='model')
                fig.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            st.download_button("Download leaderboard CSV", lb.to_csv(index=False).encode('utf-8'), file_name="leaderboard.csv", mime="text/csv")
        if os.path.exists(lb_png):
            st.image(Image.open(lb_png), caption="Leaderboard Plot")

    # Footer
    st.caption("Tip: Use the sidebar to switch series and toggle the global leaderboard view.")

if __name__ == '__main__':
    display_dashboard()
