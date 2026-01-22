"""
PioSOLVER Report Analyzer

A Streamlit app for analyzing PioSOLVER aggregated reports.
Simplified board classification, interactive Plotly charts,
and multi-report comparison.
"""

import streamlit as st
import pandas as pd
import os

from modules.data_loader import (
    list_reports, load_report, extract_zip, delete_report,
    get_action_columns, get_equity_columns, parse_info_file, EXTRACTED_DIR
)
from modules.board_classifier import BOARD_CATEGORIES
from modules.visualizations import (
    create_strategy_chart, create_distribution_chart,
    create_category_comparison_bars
)
from modules.stats import get_board_summary, get_key_takeaways

# --- PAGE CONFIG ---
st.set_page_config(
    page_title='PioSOLVER Report Analyzer',
    page_icon='🃏',
    layout='wide',
    initial_sidebar_state='expanded'
)

# --- SIDEBAR ---
st.sidebar.title('🃏 PioReport')
st.sidebar.markdown('*Analyze PioSOLVER reports*')

# File upload
st.sidebar.header('📁 Upload Report')
uploaded_file = st.sidebar.file_uploader(
    'Upload a PioSOLVER .zip file',
    type=['zip'],
    help='Upload an aggregated report ZIP from PioSOLVER'
)

# Handle file upload
if uploaded_file is not None:
    with st.spinner('Extracting report...'):
        extract_dir, file_list = extract_zip(uploaded_file.getvalue())
        if extract_dir:
            st.sidebar.success(f'Extracted: {os.path.basename(extract_dir)}')
            st.rerun()

# Report selection
st.sidebar.header('📂 Select Report')
available_reports = list_reports()

if available_reports:
    selected_report = st.sidebar.selectbox(
        'Choose a report:',
        ['None'] + available_reports
    )
else:
    selected_report = 'None'
    st.sidebar.info('No reports available. Upload a ZIP file above.')

# Delete report option
if selected_report != 'None':
    if st.sidebar.button('🗑️ Delete This Report', type='secondary'):
        delete_report(selected_report)
        st.rerun()

st.sidebar.markdown('---')

# Navigation hint
st.sidebar.markdown('### 📊 Pages')
st.sidebar.markdown('''
- **Home**: View individual report
- **Insights**: Strategy heuristics
- **Compare**: Multi-report comparison
- **Board Explorer**: Full sortable table of all boards
- **Range Builder**: Create custom ranges & study notes
''')

# --- MAIN CONTENT ---
if selected_report == 'None':
    # Welcome screen
    st.title('🃏 PioSOLVER Report Analyzer')
    st.markdown('---')

    st.markdown('''
    ## Welcome!

    This tool helps you analyze PioSOLVER aggregated reports to extract
    actionable poker heuristics.

    ### Features

    - **Detailed Board Classification**: Granular categories with FD/Rainbow distinction
    - **Interactive Charts**: Hover, zoom, and export with Plotly
    - **Strategy Insights**: Automated heuristic extraction
    - **Multi-Report Comparison**: Compare strategies across spots

    ### Getting Started

    1. **Upload** a PioSOLVER aggregated report ZIP file
    2. **Select** the report from the dropdown
    3. **Explore** the data and insights

    ### Board Categories

    | Category | Description | Example |
    |----------|-------------|---------|
    | **Monotone** | All 3 cards same suit | A♠ 7♠ 2♠ |
    | **Paired** | Board with a pair/trips | 8♠ 8♥ 3♦ |
    | **Two Broadway + low** | Two broadway (T+), one low card | K♥ Q♦ 4♣ |
    | **Two Broadway connected** | Two connected broadway cards | K♥ Q♦ J♣ |
    | **High + two low** | One high card (J+), two mid/low | A♥ 6♦ 3♣ |
    | **Mid-connected** | Mid cards (9-T high), connected | T♥ 8♦ 7♣ |
    | **Low-connected** | All low cards (8-), connected | 7♠ 5♥ 4♦ |
    | **A-Low** | Ace with two low cards | A♠ 5♥ 3♦ |
    | **Disconnected** | Unpaired, not fitting above | K♥ 7♦ 2♣ |

    ### Flush Draw Status
    Each unpaired, non-monotone category includes:
    - **(FD)**: Flush draw present (two-tone)
    - **(Rainbow)**: No flush draw (three suits)
    ''')

else:
    # Load and display report
    df = load_report(selected_report)

    if df is None or df.empty:
        st.error('Failed to load the selected report. The CSV may be invalid.')
        st.stop()

    # Get columns
    action_cols = get_action_columns(df)
    equity_cols = get_equity_columns(df)

    # Header with report info
    st.title(f'📊 {selected_report}')

    # Try to get spot info
    folder_path = os.path.join(EXTRACTED_DIR, selected_report)
    info = parse_info_file(folder_path)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric('Total Flops', len(df))
    with col2:
        st.metric('Board Categories', df['Board Category'].nunique())
    with col3:
        if info.get('spot_type'):
            st.metric('Spot Type', info['spot_type'])
        else:
            st.metric('Actions', len(action_cols))
    with col4:
        if 'OOP Equity' in df.columns:
            avg_eq = df['OOP Equity'].mean()
            st.metric('Avg OOP Equity', f'{avg_eq:.1f}%')

    # Key takeaways
    st.markdown('---')
    takeaways = get_key_takeaways(df)
    if takeaways:
        cols = st.columns(len(takeaways))
        for i, takeaway in enumerate(takeaways[:4]):
            with cols[i % len(cols)]:
                st.info(takeaway)

    # --- FILTERS ---
    st.sidebar.markdown('---')
    st.sidebar.header('🔍 Filters')

    # Category filter
    all_categories = ['All'] + sorted(df['Board Category'].unique().tolist())
    selected_category = st.sidebar.selectbox('Board Category:', all_categories)

    # Tag filters
    st.sidebar.markdown('**Secondary Tags:**')
    filter_flush_draw = st.sidebar.checkbox('Flush Draw only')
    filter_connected = st.sidebar.checkbox('Connected only')
    filter_broadway = st.sidebar.checkbox('Broadway only')

    # Apply filters
    filtered_df = df.copy()

    if selected_category != 'All':
        filtered_df = filtered_df[filtered_df['Board Category'] == selected_category]

    if filter_flush_draw and 'Has Flush Draw' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['Has Flush Draw'] == True]

    if filter_connected and 'Is Connected' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['Is Connected'] == True]

    if filter_broadway and 'Has Broadway' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['Has Broadway'] == True]

    # Sorting
    st.sidebar.markdown('---')
    st.sidebar.header('📊 Sorting')

    sort_options = ['Flop'] + action_cols + equity_cols
    sort_by = st.sidebar.selectbox('Sort by:', sort_options)
    sort_order = st.sidebar.radio('Order:', ['Descending', 'Ascending'])

    ascending = sort_order == 'Ascending'
    filtered_df = filtered_df.sort_values(by=sort_by, ascending=ascending)

    # --- VISUALIZATIONS ---
    st.markdown('---')
    st.header('📈 Strategy Visualization')

    # Action highlight selection
    if action_cols:
        col1, col2 = st.columns([3, 1])
        with col1:
            highlight_action = st.selectbox(
                'Highlight Action:',
                action_cols,
                index=0
            )
        with col2:
            group_by_category = st.checkbox('Group by Category', value=True)

    # Strategy chart
    if action_cols and not filtered_df.empty:
        chart = create_strategy_chart(
            filtered_df,
            action_cols,
            highlight_action=highlight_action,
            group_by='Board Category' if group_by_category else None,
            title=f'Strategy Frequency ({len(filtered_df)} flops)'
        )
        st.plotly_chart(chart, use_container_width=True)

        # Category comparison bars
        if group_by_category:
            st.subheader(f'{highlight_action} by Category')
            bar_chart = create_category_comparison_bars(
                filtered_df, highlight_action
            )
            st.plotly_chart(bar_chart, use_container_width=True)
    else:
        st.warning('No data matches the current filters.')

    # --- DATA TABLE ---
    st.markdown('---')
    st.header('📋 Data Table')

    # Column selection
    with st.expander('Select columns to display'):
        default_cols = ['Flop', 'Board Category'] + action_cols[:4] + equity_cols[:2]
        available_cols = filtered_df.columns.tolist()
        selected_cols = st.multiselect(
            'Columns:',
            available_cols,
            default=[c for c in default_cols if c in available_cols]
        )

    if selected_cols:
        display_df = filtered_df[selected_cols]
    else:
        display_df = filtered_df

    # Display with styling
    st.dataframe(
        display_df.style.format(
            {col: '{:.1f}' for col in display_df.columns
             if display_df[col].dtype in ['float64', 'float32']}
        ),
        use_container_width=True,
        height=500
    )

    # Pagination info
    st.caption(f'Showing {len(filtered_df)} of {len(df)} flops')

    # --- CATEGORY SUMMARY ---
    st.markdown('---')
    st.header('📊 Detailed Statistics')

    summary = get_board_summary(filtered_df)
    if not summary.empty:
        # Calculate dynamic height based on number of categories
        num_categories = len(summary)
        table_height = min(800, max(400, num_categories * 35 + 50))

        col1, col2 = st.columns([3, 1])

        with col1:
            st.dataframe(
                summary.style.format('{:.1f}').background_gradient(
                    cmap='Blues',
                    subset=[c for c in summary.columns if c != 'Count']
                ),
                use_container_width=True,
                height=table_height
            )

        with col2:
            dist_chart = create_distribution_chart(filtered_df)
            st.plotly_chart(dist_chart, use_container_width=True)

    # --- EXPORT ---
    st.markdown('---')
    st.header('💾 Export')

    col1, col2 = st.columns(2)

    with col1:
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            '📥 Download Filtered Data (CSV)',
            csv,
            file_name=f'{selected_report}_filtered.csv',
            mime='text/csv'
        )

    with col2:
        summary_csv = summary.to_csv()
        st.download_button(
            '📥 Download Summary (CSV)',
            summary_csv,
            file_name=f'{selected_report}_summary.csv',
            mime='text/csv'
        )
