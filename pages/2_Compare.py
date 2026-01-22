"""
Multi-Report Comparison Page

Compare strategies across different spots:
- Side-by-side strategy charts
- Difference highlighting
- Bar charts comparing bet frequencies by board type
"""

import streamlit as st
import pandas as pd
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.data_loader import (
    list_reports, load_report, load_multiple_reports,
    get_action_columns, get_equity_columns, parse_info_file
)
from modules.visualizations import (
    create_comparison_chart, create_strategy_chart,
    create_category_comparison_bars
)
from modules.stats import calculate_comparison_diff, get_board_summary

st.set_page_config(
    page_title='Compare Reports - PioReport',
    page_icon='⚖️',
    layout='wide'
)

st.title('⚖️ Multi-Report Comparison')
st.markdown('*Compare strategies across different positions, stack depths, or spots*')

# Get available reports
available_reports = list_reports()

if len(available_reports) < 2:
    st.warning('Need at least 2 reports to compare. Please upload more reports on the main page.')
    st.stop()

# --- REPORT SELECTION ---
st.sidebar.header('Select Reports to Compare')

selected_reports = st.sidebar.multiselect(
    'Choose 2 or more reports:',
    available_reports,
    default=available_reports[:2] if len(available_reports) >= 2 else available_reports
)

if len(selected_reports) < 2:
    st.info('Please select at least 2 reports to compare.')
    st.stop()

# Load all selected reports
with st.spinner('Loading reports...'):
    reports = load_multiple_reports(selected_reports)

if not reports:
    st.error('Failed to load the selected reports.')
    st.stop()

# --- REPORT INFO ---
st.header('📋 Report Overview')

info_cols = st.columns(len(selected_reports))
for i, (name, df) in enumerate(reports.items()):
    with info_cols[i]:
        st.subheader(name[:30] + '...' if len(name) > 30 else name)

        # Try to get info file data
        folder_path = os.path.join('extracted', name)
        info = parse_info_file(folder_path)

        st.metric('Total Flops', len(df))

        if info.get('spot_type'):
            st.write(f"**Spot Type:** {info['spot_type']}")
        if info.get('pot'):
            st.write(f"**Pot:** {info['pot']}")

        # Category breakdown (show top 5)
        cat_counts = df['Board Category'].value_counts()
        st.write('**Top Board Categories:**')
        for cat, count in cat_counts.head(5).items():
            pct = count / len(df) * 100
            st.write(f'- {cat}: {count} ({pct:.0f}%)')
        if len(cat_counts) > 5:
            st.caption(f'*+{len(cat_counts) - 5} more categories*')

# --- METRIC SELECTION ---
st.header('📊 Strategy Comparison')

# Get common columns
common_actions = None
for df in reports.values():
    actions = set(get_action_columns(df))
    if common_actions is None:
        common_actions = actions
    else:
        common_actions = common_actions.intersection(actions)

common_actions = sorted(list(common_actions))

if not common_actions:
    st.error('No common action columns found across reports.')
    st.stop()

# Select metric to compare
selected_metric = st.selectbox(
    'Select metric to compare:',
    common_actions,
    index=0
)

# --- COMPARISON CHART ---
comparison_chart = create_comparison_chart(
    reports, selected_metric,
    title=f'{selected_metric} Comparison by Board Type'
)
st.plotly_chart(comparison_chart, use_container_width=True)

# --- SIDE BY SIDE BARS ---
st.subheader('Category Breakdown')

bar_cols = st.columns(len(selected_reports))
for i, (name, df) in enumerate(reports.items()):
    with bar_cols[i]:
        st.markdown(f'**{name[:25]}...**' if len(name) > 25 else f'**{name}**')
        bar_chart = create_category_comparison_bars(
            df, selected_metric,
            title=''
        )
        st.plotly_chart(bar_chart, use_container_width=True, key=f'bar_{i}')

# --- DIFFERENCE TABLE ---
st.header('📈 Strategy Differences')

if len(selected_reports) == 2:
    # Two-report comparison with differences
    report1_name, report2_name = list(reports.keys())[:2]
    df1, df2 = reports[report1_name], reports[report2_name]

    diff_df = calculate_comparison_diff(df1, df2, selected_metric)

    if not diff_df.empty:
        diff_df.columns = [report1_name[:20], report2_name[:20], 'Difference', 'Abs Diff']

        # Dynamic height based on categories
        num_cats = len(diff_df)
        table_height = min(600, max(300, num_cats * 35 + 50))

        st.dataframe(
            diff_df.style.background_gradient(
                subset=['Difference'],
                cmap='RdYlGn_r',
                vmin=-30,
                vmax=30
            ).format('{:.1f}'),
            use_container_width=True,
            height=table_height
        )

        # Highlight biggest differences
        st.subheader('💡 Key Differences')
        biggest_diffs = diff_df.nlargest(3, 'Abs Diff')

        for idx, row in biggest_diffs.iterrows():
            diff = row['Difference']
            direction = 'higher' if diff > 0 else 'lower'
            st.markdown(
                f'- **{idx}**: {report2_name[:20]} is **{abs(diff):.1f}%** '
                f'{direction} than {report1_name[:20]}'
            )
else:
    # Multiple reports - show summary table
    st.markdown('*Showing average values for each report*')

    summary_data = []
    for name, df in reports.items():
        row = {'Report': name[:30]}
        summary = df.groupby('Board Category')[selected_metric].mean()
        for cat, val in summary.items():
            row[cat] = val
        summary_data.append(row)

    summary_df = pd.DataFrame(summary_data).set_index('Report').round(1)
    st.dataframe(summary_df, use_container_width=True)

# --- FULL DATA TABLES ---
st.header('📋 Full Category Statistics')

with st.expander('View detailed statistics for all reports', expanded=True):
    for name, df in reports.items():
        st.subheader(name)
        summary = get_board_summary(df)
        # Dynamic height based on categories
        num_cats = len(summary)
        table_height = min(600, max(300, num_cats * 35 + 50))
        st.dataframe(
            summary.style.format('{:.1f}'),
            use_container_width=True,
            height=table_height
        )
        st.markdown('---')

# --- EXPORT ---
st.header('💾 Export Comparison')

# Create comparison export
export_data = []
for name, df in reports.items():
    summary = df.groupby('Board Category')[common_actions].mean()
    summary['Report'] = name
    export_data.append(summary.reset_index())

if export_data:
    export_df = pd.concat(export_data, ignore_index=True)
    csv = export_df.to_csv(index=False)

    st.download_button(
        '📥 Download Comparison Data (CSV)',
        csv,
        file_name='report_comparison.csv',
        mime='text/csv'
    )
