"""
Strategy Insights Dashboard

Provides automated heuristic extraction and strategy analysis:
- Board type distribution
- Betting frequency by category
- Equity correlation analysis
- Key takeaways
"""

import streamlit as st
import pandas as pd
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.data_loader import list_reports, load_report, get_action_columns, get_equity_columns
from modules.visualizations import (
    create_distribution_chart, create_equity_scatter,
    create_heuristics_table, create_category_comparison_bars
)
from modules.stats import (
    get_board_summary, get_betting_heuristics, get_equity_correlation,
    get_key_takeaways, get_tag_analysis
)

st.set_page_config(
    page_title='Strategy Insights - PioReport',
    page_icon='📊',
    layout='wide'
)

st.title('📊 Strategy Insights')
st.markdown('*Automated heuristic extraction from solver data*')

# Sidebar - Report Selection
st.sidebar.header('Select Report')
available_reports = list_reports()

if not available_reports:
    st.warning('No reports available. Please upload a report on the main page first.')
    st.stop()

selected_report = st.sidebar.selectbox(
    'Choose a report to analyze:',
    available_reports
)

# Load the report
df = load_report(selected_report)

if df is None or df.empty:
    st.error('Failed to load the selected report.')
    st.stop()

# Get column info
action_cols = get_action_columns(df)
equity_cols = get_equity_columns(df)

st.sidebar.markdown('---')
st.sidebar.markdown(f'**Loaded:** {selected_report}')
st.sidebar.markdown(f'**Flops:** {len(df)}')
st.sidebar.markdown(f'**Categories:** {df["Board Category"].nunique()}')

# --- KEY TAKEAWAYS ---
st.header('🎯 Key Takeaways')

takeaways = get_key_takeaways(df)
for takeaway in takeaways:
    st.markdown(f'- {takeaway}')

# --- BOARD DISTRIBUTION ---
st.header('📈 Board Category Distribution')

col1, col2 = st.columns([1, 1])

with col1:
    dist_chart = create_distribution_chart(df)
    st.plotly_chart(dist_chart, use_container_width=True)

with col2:
    # Category summary table
    summary = get_board_summary(df)
    if not summary.empty:
        # Dynamic height based on number of categories
        num_categories = len(summary)
        table_height = min(600, max(300, num_categories * 35 + 50))
        st.dataframe(
            summary.style.format('{:.1f}'),
            use_container_width=True,
            height=table_height
        )

# --- BETTING FREQUENCY BY CATEGORY ---
st.header('📊 Betting Frequency by Board Type')

# Select action to analyze
selected_action = st.selectbox(
    'Select action to analyze:',
    action_cols,
    index=0 if action_cols else None
)

if selected_action:
    col1, col2 = st.columns([2, 1])

    with col1:
        bar_chart = create_category_comparison_bars(
            df, selected_action,
            title=f'{selected_action} by Board Category'
        )
        st.plotly_chart(bar_chart, use_container_width=True)

    with col2:
        # Generate heuristics
        st.subheader('💡 Heuristics')
        heuristics = get_betting_heuristics(df)
        for h in heuristics:
            st.markdown(f'- {h}')

# --- EQUITY CORRELATION ---
st.header('🔗 Equity Correlation Analysis')

col1, col2 = st.columns([2, 1])

with col1:
    # Select equity column
    equity_col = st.selectbox(
        'Equity column:',
        equity_cols,
        index=0 if equity_cols else None
    )

    if equity_col and selected_action:
        scatter = create_equity_scatter(
            df, equity_col, selected_action,
            title=f'{equity_col} vs {selected_action}'
        )
        st.plotly_chart(scatter, use_container_width=True)

with col2:
    if equity_col and selected_action:
        corr, interpretation = get_equity_correlation(df, equity_col, selected_action)
        st.metric('Correlation', f'{corr:.3f}')
        st.markdown(interpretation)

# --- TAG ANALYSIS ---
st.header('🏷️ Secondary Tag Analysis')
st.markdown('*How do tags like Flush Draw, Connected, and Broadway affect strategy?*')

tag_analysis = get_tag_analysis(df, selected_action)

if tag_analysis:
    cols = st.columns(len(tag_analysis))

    for i, (tag, stats) in enumerate(tag_analysis.items()):
        with cols[i]:
            st.subheader(tag)

            diff = stats['difference']
            color = 'red' if diff > 0 else 'green'
            direction = '+' if diff > 0 else ''

            st.metric(
                f'With {tag}',
                f"{stats['with_tag_avg']:.1f}%",
                f"{direction}{diff:.1f}%"
            )

            st.caption(f"Without: {stats['without_tag_avg']:.1f}%")
            st.caption(f"({stats['count_with']} boards with tag)")
else:
    st.info('Tag analysis not available for this dataset.')

# --- DETAILED STATS TABLE ---
st.header('📋 Detailed Statistics')

with st.expander('View Full Statistics Table', expanded=True):
    table_fig = create_heuristics_table(df, 'Board Category', action_cols, equity_cols)
    st.plotly_chart(table_fig, use_container_width=True, height=min(800, len(df['Board Category'].unique()) * 35 + 100))

# --- EXPORT ---
st.header('💾 Export Data')

col1, col2 = st.columns(2)

with col1:
    # Export summary
    summary_csv = get_board_summary(df).to_csv()
    st.download_button(
        '📥 Download Category Summary (CSV)',
        summary_csv,
        file_name=f'{selected_report}_summary.csv',
        mime='text/csv'
    )

with col2:
    # Export full data
    full_csv = df.to_csv(index=False)
    st.download_button(
        '📥 Download Full Data (CSV)',
        full_csv,
        file_name=f'{selected_report}_full.csv',
        mime='text/csv'
    )
