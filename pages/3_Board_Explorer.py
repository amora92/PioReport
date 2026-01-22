"""
Board Explorer - Full Database View

A comprehensive view of all boards with:
- Sortable, filterable table with color-coded frequencies
- Advanced filtering by texture, high card, action thresholds
- Heatmap visualization
- Board search and quick navigation
- Study mode with flashcards
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.data_loader import list_reports, load_report, get_action_columns, get_equity_columns
from modules.board_classifier import RANK_VALUES

st.set_page_config(
    page_title='Board Explorer - PioReport',
    page_icon='🔍',
    layout='wide'
)

# Custom CSS for better table styling
st.markdown("""
<style>
    .stDataFrame {
        font-size: 14px;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .filter-section {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .highlight-high { background-color: #ff6b6b !important; color: white !important; }
    .highlight-med { background-color: #ffd93d !important; }
    .highlight-low { background-color: #6bcb77 !important; color: white !important; }
</style>
""", unsafe_allow_html=True)

st.title('🔍 Board Explorer')
st.markdown('*Explore all boards with advanced filtering and color-coded action frequencies*')

# --- SIDEBAR ---
st.sidebar.header('📂 Select Report')
available_reports = list_reports()

if not available_reports:
    st.warning('No reports available. Please upload a report on the main page first.')
    st.stop()

selected_report = st.sidebar.selectbox(
    'Choose a report:',
    available_reports
)

# Load report
df = load_report(selected_report)

if df is None or df.empty:
    st.error('Failed to load the selected report.')
    st.stop()

action_cols = get_action_columns(df)
equity_cols = get_equity_columns(df)

# --- QUICK STATS ---
st.markdown('---')
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric('Total Boards', f'{len(df):,}')
with col2:
    st.metric('Categories', df['Board Category'].nunique())
with col3:
    if action_cols:
        # Find check frequency
        check_col = next((c for c in action_cols if 'check' in c.lower()), action_cols[0])
        st.metric('Avg Check %', f"{df[check_col].mean():.1f}%")
with col4:
    if 'OOP Equity' in df.columns:
        st.metric('Avg OOP Equity', f"{df['OOP Equity'].mean():.1f}%")
with col5:
    fd_count = df['Has Flush Draw'].sum() if 'Has Flush Draw' in df.columns else 0
    st.metric('Flush Draw Boards', f'{fd_count:,}')

# --- FILTER TABS ---
st.markdown('---')
tab1, tab2, tab3, tab4 = st.tabs(['📊 Full Table', '🔥 Heatmap', '📚 Study Mode', '📈 Analysis'])

# =============================================================================
# TAB 1: FULL TABLE VIEW
# =============================================================================
with tab1:
    st.header('📊 All Boards - Sortable Table')

    # --- PIO-STYLE FILTERS ---
    # Search bar at top for quick access
    search_col1, search_col2 = st.columns([4, 1])
    with search_col1:
        search_query = st.text_input(
            '🔎 Search boards (e.g., "A♠ K" or "AKs")',
            placeholder='Type to search specific boards...',
            key='board_search'
        )
    with search_col2:
        st.markdown('<br>', unsafe_allow_html=True)
        reset_filters = st.button('🔄 Reset All Filters')

    if reset_filters:
        st.rerun()

    # PioSolver-style filter panel
    with st.expander('🎚️ Flop / Turn Filters (PioSolver Style)', expanded=True):

        # --- CARD RANKS ---
        st.markdown('##### Flop settings - Card ranks')
        rank_col1, rank_col2, rank_col3 = st.columns(3)

        with rank_col1:
            high_card_filter = st.text_input(
                'High Card',
                value='',
                placeholder='e.g., A or AK or AKQ',
                key='high_card_input',
                help='Enter rank(s) to filter. Leave empty for all. Examples: A, AK, AKQJT'
            )
        with rank_col2:
            middle_card_filter = st.text_input(
                'Middle Card',
                value='',
                placeholder='e.g., K or KQJ',
                key='middle_card_input',
                help='Enter rank(s) to filter. Leave empty for all.'
            )
        with rank_col3:
            low_card_filter = st.text_input(
                'Low Card',
                value='',
                placeholder='e.g., 7 or 765432',
                key='low_card_input',
                help='Enter rank(s) to filter. Leave empty for all.'
            )

        st.markdown('---')

        # --- PAIREDNESS, SUITEDNESS, CONNECTIVITY ---
        pair_col, suit_col, conn_col = st.columns(3)

        with pair_col:
            st.markdown('##### Pairedness')
            filter_no_pair = st.checkbox('No pair', value=True, key='filter_no_pair')
            filter_pair = st.checkbox('Pair', value=True, key='filter_pair')
            filter_trips = st.checkbox('Three of a kind', value=True, key='filter_trips')

        with suit_col:
            st.markdown('##### Suitedness')
            filter_rainbow = st.checkbox('Rainbow', value=True, key='filter_rainbow')
            filter_two_tone = st.checkbox('Two-tone (Suited)', value=True, key='filter_two_tone')
            filter_monotone = st.checkbox('Monotone', value=True, key='filter_monotone')

        with conn_col:
            st.markdown('##### Connectivity')
            filter_straight_possible = st.checkbox('Straight possible', value=True, key='filter_straight_possible')
            filter_oesd_possible = st.checkbox('OESD possible', value=True, key='filter_oesd_possible')
            filter_oesd_not_possible = st.checkbox('OESD not possible', value=True, key='filter_oesd_not_possible')

        st.markdown('---')

        # --- HERO HAND EXPRESSION ---
        st.markdown('##### Hero Hand')
        hand_col1, hand_col2 = st.columns(2)
        with hand_col1:
            hero_flop_expr = st.text_input(
                'Hero hand on the Flop expression',
                value='',
                placeholder='e.g., AA, AKs, JJ+, 77-99',
                key='hero_flop_expr',
                help='Filter boards where hero could have specific hands. Leave empty for all.'
            )
        with hand_col2:
            hero_turn_expr = st.text_input(
                'Hero hand on the Turn expression',
                value='',
                placeholder='e.g., top pair, flush draw',
                key='hero_turn_expr',
                help='Filter by made hand type on turn. Leave empty for all.'
            )
        st.caption('[hand expressions: What is the syntax and how does it work?](https://piosolver.com/)')

    # --- OPTIONAL: Category & Action Threshold Filters ---
    with st.expander('📊 Additional Filters (Category & Action Thresholds)', expanded=False):
        add_col1, add_col2 = st.columns(2)

        with add_col1:
            st.markdown('**Board Category**')
            categories = sorted(df['Board Category'].unique().tolist())
            selected_categories = st.multiselect(
                'Filter by category (leave empty for all):',
                categories,
                default=[],
                key='cat_filter',
                help='Select specific categories to include. Empty = all categories.'
            )

        with add_col2:
            st.markdown('**Action Thresholds (Optional)**')
            use_threshold = st.checkbox('Enable action threshold filter', value=False, key='use_threshold')
            threshold_action = None
            threshold_type = None
            threshold_value = None
            if use_threshold and action_cols:
                threshold_action = st.selectbox('Action:', action_cols, key='thresh_action')
                threshold_type = st.radio('Show boards where:', ['Above', 'Below'], horizontal=True, key='thresh_type')
                threshold_value = st.slider('Threshold %:', 0, 100, 50, key='thresh_val')

    # --- APPLY FILTERS ---
    filtered_df = df.copy()

    # Search filter (always applied first)
    if search_query:
        search_upper = search_query.upper()
        # Convert common suit notations
        search_upper = search_upper.replace('S', '♠').replace('H', '♥').replace('D', '♦').replace('C', '♣')
        filtered_df = filtered_df[filtered_df['Flop'].str.upper().str.contains(search_upper, na=False)]

    # Card rank filters (text input - parse allowed ranks)
    def parse_ranks(rank_string):
        """Parse a rank string into a list of valid ranks."""
        if not rank_string:
            return None  # No filter
        valid_ranks = set('AKQJT98765432')
        ranks = [r.upper() for r in rank_string.upper() if r.upper() in valid_ranks]
        return ranks if ranks else None

    high_ranks = parse_ranks(high_card_filter)
    if high_ranks and 'High Card' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['High Card'].isin(high_ranks)]

    middle_ranks = parse_ranks(middle_card_filter)
    if middle_ranks and 'Middle Card' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['Middle Card'].isin(middle_ranks)]

    low_ranks = parse_ranks(low_card_filter)
    if low_ranks and 'Low Card' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['Low Card'].isin(low_ranks)]

    # Pairedness filter (checkbox-based inclusion)
    pairedness_filter = []
    if filter_no_pair:
        pairedness_filter.append('No pair')
    if filter_pair:
        pairedness_filter.append('Pair')
    if filter_trips:
        pairedness_filter.append('Three of a kind')

    if pairedness_filter and 'Pairedness' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['Pairedness'].isin(pairedness_filter)]

    # Suitedness filter (checkbox-based inclusion)
    suitedness_filter = []
    if filter_rainbow:
        suitedness_filter.append('Rainbow')
    if filter_two_tone:
        suitedness_filter.append('Two-tone')
    if filter_monotone:
        suitedness_filter.append('Monotone')

    if suitedness_filter and 'Suitedness' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['Suitedness'].isin(suitedness_filter)]

    # Connectivity filter (checkbox-based - any checked condition matches)
    # This is more complex: show boards where at least one checked condition is true
    if 'Straight Possible' in filtered_df.columns:
        conn_mask = pd.Series([False] * len(filtered_df), index=filtered_df.index)
        if filter_straight_possible:
            conn_mask = conn_mask | filtered_df['Straight Possible']
        if filter_oesd_possible:
            conn_mask = conn_mask | filtered_df['OESD Possible']
        if filter_oesd_not_possible:
            conn_mask = conn_mask | filtered_df['OESD Not Possible']
        # Only apply if at least one is checked
        if filter_straight_possible or filter_oesd_possible or filter_oesd_not_possible:
            filtered_df = filtered_df[conn_mask]

    # Category filter (only if categories are selected)
    if selected_categories:
        filtered_df = filtered_df[filtered_df['Board Category'].isin(selected_categories)]

    # Action threshold filter (only if enabled and configured)
    if use_threshold and threshold_action and threshold_type and threshold_value is not None:
        if threshold_type == 'Above':
            filtered_df = filtered_df[filtered_df[threshold_action] >= threshold_value]
        else:
            filtered_df = filtered_df[filtered_df[threshold_action] <= threshold_value]

    # Hero hand expression filter (placeholder for future implementation)
    # These are complex to implement fully - would need hand-board interaction analysis
    if hero_flop_expr:
        st.info(f'Hero hand filter "{hero_flop_expr}" noted. (Full hand expression parsing coming soon)')

    # --- SORTING ---
    st.markdown('---')
    sort_col1, sort_col2, sort_col3 = st.columns([2, 1, 1])

    with sort_col1:
        sort_options = ['Flop', 'Board Category', 'High Card'] + action_cols + equity_cols
        sort_by = st.selectbox('Sort by:', sort_options, index=0, key='sort_by')
    with sort_col2:
        sort_order = st.radio('Order:', ['Descending', 'Ascending'], horizontal=True, key='sort_order')
    with sort_col3:
        rows_per_page = st.selectbox('Rows per page:', [50, 100, 250, 500, 'All'], index=1, key='rows_page')

    ascending = sort_order == 'Ascending'
    filtered_df = filtered_df.sort_values(by=sort_by, ascending=ascending)

    # --- COLUMN SELECTION ---
    with st.expander('📋 Select Columns to Display'):
        col_select1, col_select2, col_select3 = st.columns(3)
        with col_select1:
            show_basic = st.checkbox('Basic Info (Flop, Category)', value=True)
            show_actions = st.checkbox('Action Frequencies', value=True)
        with col_select2:
            show_equity = st.checkbox('Equity Columns', value=True)
            show_card_ranks = st.checkbox('Card Ranks (High/Mid/Low)', value=False)
        with col_select3:
            show_board_props = st.checkbox('Board Properties (Pairedness, Suitedness, Connectivity)', value=False)
            show_tags = st.checkbox('Legacy Tags', value=False)

    # Build display columns
    display_cols = []
    if show_basic:
        display_cols.extend(['Flop', 'Board Category'])
    if show_card_ranks:
        display_cols.extend(['High Card', 'Middle Card', 'Low Card'])
    if show_actions:
        display_cols.extend(action_cols)
    if show_equity:
        display_cols.extend(equity_cols)
    if show_board_props:
        display_cols.extend(['Pairedness', 'Suitedness', 'Straight Possible', 'OESD Possible'])
    if show_tags:
        display_cols.extend(['Board Tags', 'Has Flush Draw', 'Is Connected', 'Has Broadway'])

    # Remove duplicates while preserving order
    display_cols = list(dict.fromkeys([c for c in display_cols if c in filtered_df.columns]))

    # --- DISPLAY FILTERED COUNT ---
    st.markdown(f'### Showing **{len(filtered_df):,}** of **{len(df):,}** boards')

    # --- COLOR CODING FUNCTION ---
    def color_frequencies(val, col_name):
        """Color code frequency values based on intensity."""
        if not isinstance(val, (int, float)) or pd.isna(val):
            return ''

        # Different color schemes for different action types
        if 'check' in col_name.lower():
            # Check: high = green (passive), low = red (aggressive)
            if val >= 70:
                return 'background-color: #28a745; color: white'
            elif val >= 50:
                return 'background-color: #7dc47d'
            elif val >= 30:
                return 'background-color: #ffc107'
            elif val >= 15:
                return 'background-color: #fd7e14'
            else:
                return 'background-color: #dc3545; color: white'
        elif 'bet' in col_name.lower() or 'raise' in col_name.lower():
            # Bet/Raise: high = red (aggressive), low = green
            if val >= 50:
                return 'background-color: #dc3545; color: white'
            elif val >= 35:
                return 'background-color: #fd7e14'
            elif val >= 20:
                return 'background-color: #ffc107'
            elif val >= 10:
                return 'background-color: #7dc47d'
            else:
                return 'background-color: #28a745; color: white'
        elif 'equity' in col_name.lower():
            # Equity: gradient blue
            if val >= 55:
                return 'background-color: #004085; color: white'
            elif val >= 50:
                return 'background-color: #0056b3; color: white'
            elif val >= 45:
                return 'background-color: #007bff; color: white'
            else:
                return 'background-color: #6c757d; color: white'
        return ''

    def style_dataframe(df_to_style, cols):
        """Apply comprehensive styling to the dataframe."""
        styled = df_to_style.style

        # Apply color coding to each action/equity column
        for col in cols:
            if col in action_cols or col in equity_cols:
                styled = styled.map(lambda x: color_frequencies(x, col), subset=[col])

        # Format numbers
        format_dict = {col: '{:.1f}' for col in cols if df_to_style[col].dtype in ['float64', 'float32']}
        styled = styled.format(format_dict)

        return styled

    # --- DISPLAY TABLE ---
    if display_cols:
        display_df = filtered_df[display_cols].reset_index(drop=True)

        # Handle pagination
        if rows_per_page != 'All':
            rows_per_page = int(rows_per_page)
            total_pages = (len(display_df) - 1) // rows_per_page + 1

            page_col1, page_col2, page_col3 = st.columns([1, 3, 1])
            with page_col2:
                current_page = st.number_input(
                    f'Page (1-{total_pages}):',
                    min_value=1,
                    max_value=max(1, total_pages),
                    value=1,
                    key='page_num'
                )

            start_idx = (current_page - 1) * rows_per_page
            end_idx = min(start_idx + rows_per_page, len(display_df))
            display_df = display_df.iloc[start_idx:end_idx]

            st.caption(f'Showing rows {start_idx + 1} to {end_idx}')

        # Apply styling and display
        styled_df = style_dataframe(display_df, display_cols)
        st.dataframe(
            styled_df,
            use_container_width=True,
            height=min(700, max(400, len(display_df) * 35 + 50))
        )

        # --- LEGEND ---
        st.markdown('---')
        st.markdown('#### Color Legend')
        legend_col1, legend_col2, legend_col3 = st.columns(3)

        with legend_col1:
            st.markdown('**Check Frequency:**')
            st.markdown('🟢 High (70%+) = Very passive')
            st.markdown('🟡 Medium (30-50%) = Mixed')
            st.markdown('🔴 Low (<15%) = Very aggressive')

        with legend_col2:
            st.markdown('**Bet/Raise Frequency:**')
            st.markdown('🔴 High (50%+) = Aggressive')
            st.markdown('🟡 Medium (20-35%) = Balanced')
            st.markdown('🟢 Low (<10%) = Passive')

        with legend_col3:
            st.markdown('**Equity:**')
            st.markdown('🔵 55%+ = Strong advantage')
            st.markdown('🔵 50-55% = Slight edge')
            st.markdown('⚫ <45% = Disadvantage')

    # --- EXPORT ---
    st.markdown('---')
    export_col1, export_col2 = st.columns(2)

    with export_col1:
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            '📥 Download Filtered Data (CSV)',
            csv,
            file_name=f'{selected_report}_filtered_{len(filtered_df)}_boards.csv',
            mime='text/csv'
        )

# =============================================================================
# TAB 2: HEATMAP VIEW
# =============================================================================
with tab2:
    st.header('🔥 Strategy Heatmap')
    st.markdown('*Visualize action frequencies across board textures*')

    import plotly.express as px
    import plotly.graph_objects as go

    heatmap_col1, heatmap_col2 = st.columns([1, 3])

    with heatmap_col1:
        heatmap_action = st.selectbox(
            'Select Action:',
            action_cols,
            key='heatmap_action'
        )

        heatmap_group = st.selectbox(
            'Group By:',
            ['Board Category', 'High Card', 'Pairedness', 'Suitedness', 'Middle Card', 'Low Card'],
            key='heatmap_group'
        )

        color_scale = st.selectbox(
            'Color Scale:',
            ['RdYlGn_r', 'Viridis', 'Blues', 'Reds', 'YlOrRd'],
            key='color_scale'
        )

    with heatmap_col2:
        if heatmap_group == 'Board Category':
            # Category vs Action breakdown
            pivot_data = df.groupby('Board Category')[action_cols].mean().round(1)

            fig = px.imshow(
                pivot_data,
                labels=dict(x='Action', y='Board Category', color='Frequency %'),
                aspect='auto',
                color_continuous_scale=color_scale,
                title=f'Action Frequencies by Board Category'
            )
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)

        elif heatmap_group in ['High Card', 'Middle Card', 'Low Card']:
            # Card rank breakdown
            if heatmap_group in df.columns:
                pivot_data = df.groupby(heatmap_group)[action_cols].mean().round(1)
                # Sort by rank
                rank_order = ['A', 'K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3', '2']
                pivot_data = pivot_data.reindex([r for r in rank_order if r in pivot_data.index])

                fig = px.imshow(
                    pivot_data,
                    labels=dict(x='Action', y=heatmap_group, color='Frequency %'),
                    aspect='auto',
                    color_continuous_scale=color_scale,
                    title=f'Action Frequencies by {heatmap_group}'
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)

        elif heatmap_group in ['Pairedness', 'Suitedness']:
            # Board property breakdown
            if heatmap_group in df.columns:
                pivot_data = df.groupby(heatmap_group)[action_cols].mean().round(1)

                fig = px.imshow(
                    pivot_data,
                    labels=dict(x='Action', y=heatmap_group, color='Frequency %'),
                    aspect='auto',
                    color_continuous_scale=color_scale,
                    title=f'Action Frequencies by {heatmap_group}'
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)

    # Cross-tabulation heatmap
    st.subheader('📊 Detailed Cross-Tabulation')

    cross_col1, cross_col2 = st.columns(2)
    with cross_col1:
        cross_row = st.selectbox('Rows:', ['Board Category', 'High Card', 'Middle Card', 'Low Card', 'Pairedness', 'Suitedness', 'Straight Possible', 'OESD Possible'], key='cross_row')
    with cross_col2:
        cross_metric = st.selectbox('Metric:', action_cols + equity_cols, key='cross_metric')

    if cross_row in df.columns:
        cross_data = df.groupby(cross_row)[cross_metric].agg(['mean', 'std', 'count']).round(1)
        cross_data.columns = ['Average', 'Std Dev', 'Count']

        st.dataframe(
            cross_data.style.background_gradient(subset=['Average'], cmap='RdYlGn_r'),
            use_container_width=True
        )

# =============================================================================
# TAB 3: STUDY MODE
# =============================================================================
with tab3:
    st.header('📚 Study Mode')
    st.markdown('*Test your knowledge of optimal strategies*')

    study_mode = st.radio(
        'Choose study mode:',
        ['🎯 Flashcards', '📝 Quiz', '📊 Strategy Finder'],
        horizontal=True,
        key='study_mode'
    )

    if study_mode == '🎯 Flashcards':
        st.subheader('🎯 Flashcard Practice')
        st.markdown('A random board will be shown. Try to recall the optimal strategy before revealing the answer.')

        # Random board selection
        if st.button('🎲 New Random Board', key='new_card'):
            st.session_state['random_idx'] = np.random.randint(0, len(df))

        if 'random_idx' not in st.session_state:
            st.session_state['random_idx'] = np.random.randint(0, len(df))

        random_board = df.iloc[st.session_state['random_idx']]

        # Display board
        st.markdown('---')
        board_col1, board_col2, board_col3 = st.columns([1, 2, 1])

        with board_col2:
            st.markdown(f"""
            <div style='text-align: center; padding: 30px; background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
                        border-radius: 15px; margin: 20px 0;'>
                <h1 style='color: white; font-size: 48px; margin: 0;'>{random_board['Flop']}</h1>
                <p style='color: #ccc; margin-top: 10px;'>{random_board['Board Category']}</p>
            </div>
            """, unsafe_allow_html=True)

        # Reveal button
        if st.button('👁️ Reveal Strategy', key='reveal'):
            st.session_state['revealed'] = True

        if st.session_state.get('revealed', False):
            st.markdown('### Optimal Strategy:')

            strat_cols = st.columns(len(action_cols))
            for i, col in enumerate(action_cols):
                with strat_cols[i]:
                    val = random_board[col]
                    color = '#28a745' if val >= 50 else '#ffc107' if val >= 25 else '#dc3545'
                    st.markdown(f"""
                    <div style='text-align: center; padding: 15px; background-color: {color};
                                border-radius: 10px; color: white;'>
                        <strong>{col.replace(' freq', '')}</strong><br>
                        <span style='font-size: 24px;'>{val:.1f}%</span>
                    </div>
                    """, unsafe_allow_html=True)

            if equity_cols:
                st.markdown('### Equity:')
                eq_cols = st.columns(len(equity_cols))
                for i, col in enumerate(equity_cols):
                    with eq_cols[i]:
                        st.metric(col, f"{random_board[col]:.1f}%")

            st.session_state['revealed'] = False

    elif study_mode == '📝 Quiz':
        st.subheader('📝 Strategy Quiz')
        st.markdown('Test your understanding of optimal betting frequencies.')

        # Initialize quiz state
        if 'quiz_score' not in st.session_state:
            st.session_state['quiz_score'] = 0
            st.session_state['quiz_total'] = 0

        # Generate quiz question
        if st.button('🎲 New Question', key='new_quiz') or 'quiz_board' not in st.session_state:
            st.session_state['quiz_board'] = df.iloc[np.random.randint(0, len(df))]
            st.session_state['quiz_answered'] = False

        quiz_board = st.session_state['quiz_board']

        st.markdown(f"""
        <div style='text-align: center; padding: 20px; background: #2c3e50; border-radius: 10px;'>
            <h2 style='color: white;'>{quiz_board['Flop']}</h2>
            <p style='color: #95a5a6;'>{quiz_board['Board Category']}</p>
        </div>
        """, unsafe_allow_html=True)

        # Question: Which action has highest frequency?
        st.markdown('### Question: Which action has the HIGHEST frequency on this board?')

        correct_action = action_cols[quiz_board[action_cols].values.argmax()]

        answer = st.radio(
            'Your answer:',
            action_cols,
            key='quiz_answer',
            horizontal=True
        )

        if st.button('Submit Answer', key='submit_quiz'):
            st.session_state['quiz_total'] += 1

            if answer == correct_action:
                st.success(f'✅ Correct! {correct_action} = {quiz_board[correct_action]:.1f}%')
                st.session_state['quiz_score'] += 1
            else:
                st.error(f'❌ Wrong! Correct answer: {correct_action} = {quiz_board[correct_action]:.1f}%')
                st.info(f'Your answer ({answer}) = {quiz_board[answer]:.1f}%')

            # Show all frequencies
            st.markdown('**All frequencies:**')
            for col in action_cols:
                st.write(f'- {col}: {quiz_board[col]:.1f}%')

        # Score display
        st.sidebar.markdown('---')
        st.sidebar.markdown('### Quiz Score')
        st.sidebar.metric(
            'Score',
            f'{st.session_state["quiz_score"]}/{st.session_state["quiz_total"]}',
            f'{(st.session_state["quiz_score"]/max(1,st.session_state["quiz_total"])*100):.0f}%'
        )
        if st.sidebar.button('Reset Score'):
            st.session_state['quiz_score'] = 0
            st.session_state['quiz_total'] = 0

    else:  # Strategy Finder
        st.subheader('📊 Strategy Finder')
        st.markdown('Find boards that match specific strategy criteria.')

        finder_col1, finder_col2 = st.columns(2)

        with finder_col1:
            st.markdown('**Find boards where:**')
            find_action = st.selectbox('Action:', action_cols, key='find_action')
            find_operator = st.selectbox('Is:', ['Greater than', 'Less than', 'Between'], key='find_op')

        with finder_col2:
            st.markdown('**Value:**')
            if find_operator == 'Between':
                find_min = st.number_input('Min %:', 0, 100, 40, key='find_min')
                find_max = st.number_input('Max %:', 0, 100, 60, key='find_max')
            else:
                find_value = st.number_input('Value %:', 0, 100, 50, key='find_val')

        # Apply finder
        if find_operator == 'Greater than':
            found_df = df[df[find_action] > find_value]
        elif find_operator == 'Less than':
            found_df = df[df[find_action] < find_value]
        else:
            found_df = df[(df[find_action] >= find_min) & (df[find_action] <= find_max)]

        st.markdown(f'### Found **{len(found_df)}** boards matching criteria')

        if len(found_df) > 0:
            # Show category distribution of found boards
            st.markdown('**Category Distribution:**')
            cat_dist = found_df['Board Category'].value_counts()
            st.bar_chart(cat_dist)

            # Show sample boards
            st.markdown('**Sample Boards:**')
            sample_size = min(20, len(found_df))
            sample_df = found_df[['Flop', 'Board Category'] + action_cols].head(sample_size)
            st.dataframe(sample_df, use_container_width=True)

# =============================================================================
# TAB 4: ANALYSIS
# =============================================================================
with tab4:
    st.header('📈 Advanced Analysis')

    analysis_type = st.selectbox(
        'Analysis Type:',
        ['Action Distribution', 'Equity vs Action Correlation', 'Category Deep Dive', 'Outlier Detection'],
        key='analysis_type'
    )

    if analysis_type == 'Action Distribution':
        st.subheader('📊 Action Frequency Distribution')

        dist_action = st.selectbox('Select Action:', action_cols, key='dist_action')

        # Histogram
        fig = px.histogram(
            df, x=dist_action,
            nbins=50,
            title=f'Distribution of {dist_action}',
            color='Board Category',
            marginal='box'
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

        # Statistics
        stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
        with stat_col1:
            st.metric('Mean', f"{df[dist_action].mean():.1f}%")
        with stat_col2:
            st.metric('Median', f"{df[dist_action].median():.1f}%")
        with stat_col3:
            st.metric('Std Dev', f"{df[dist_action].std():.1f}%")
        with stat_col4:
            st.metric('Range', f"{df[dist_action].min():.0f}% - {df[dist_action].max():.0f}%")

    elif analysis_type == 'Equity vs Action Correlation':
        st.subheader('🔗 Equity vs Action Correlation')

        if equity_cols:
            corr_equity = st.selectbox('Equity Column:', equity_cols, key='corr_eq')
            corr_action = st.selectbox('Action Column:', action_cols, key='corr_act')

            fig = px.scatter(
                df, x=corr_equity, y=corr_action,
                color='Board Category',
                hover_data=['Flop'],
                trendline='ols',
                title=f'{corr_equity} vs {corr_action}'
            )
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)

            # Correlation coefficient
            corr = df[corr_equity].corr(df[corr_action])
            st.metric('Correlation Coefficient', f'{corr:.3f}')

            if abs(corr) > 0.5:
                st.info(f'Strong {"positive" if corr > 0 else "negative"} correlation detected!')
        else:
            st.warning('No equity columns found in this report.')

    elif analysis_type == 'Category Deep Dive':
        st.subheader('🔬 Category Deep Dive')

        dive_category = st.selectbox(
            'Select Category:',
            sorted(df['Board Category'].unique().tolist()),
            key='dive_cat'
        )

        cat_df = df[df['Board Category'] == dive_category]

        st.markdown(f'### {dive_category}')
        st.markdown(f'**{len(cat_df)} boards** ({len(cat_df)/len(df)*100:.1f}% of total)')

        # Action averages for this category
        st.markdown('#### Action Frequencies')
        action_avgs = cat_df[action_cols].mean()
        overall_avgs = df[action_cols].mean()

        comparison_df = pd.DataFrame({
            'This Category': action_avgs,
            'Overall Average': overall_avgs,
            'Difference': action_avgs - overall_avgs
        }).round(1)

        st.dataframe(
            comparison_df.style.background_gradient(subset=['Difference'], cmap='RdYlGn_r', vmin=-20, vmax=20),
            use_container_width=True
        )

        # Sample boards from category
        st.markdown('#### Sample Boards')
        st.dataframe(
            cat_df[['Flop'] + action_cols].head(10),
            use_container_width=True
        )

    else:  # Outlier Detection
        st.subheader('🎯 Outlier Detection')
        st.markdown('*Find boards with unusual strategy frequencies*')

        outlier_action = st.selectbox('Action to analyze:', action_cols, key='outlier_action')
        outlier_threshold = st.slider('Standard deviations from mean:', 1.0, 3.0, 2.0, 0.5, key='outlier_thresh')

        mean_val = df[outlier_action].mean()
        std_val = df[outlier_action].std()

        upper_bound = mean_val + (outlier_threshold * std_val)
        lower_bound = mean_val - (outlier_threshold * std_val)

        outliers_high = df[df[outlier_action] > upper_bound]
        outliers_low = df[df[outlier_action] < lower_bound]

        out_col1, out_col2 = st.columns(2)

        with out_col1:
            st.markdown(f'### High Outliers (>{upper_bound:.1f}%)')
            st.markdown(f'**{len(outliers_high)} boards**')
            if len(outliers_high) > 0:
                st.dataframe(
                    outliers_high[['Flop', 'Board Category', outlier_action]].sort_values(outlier_action, ascending=False).head(20),
                    use_container_width=True
                )

        with out_col2:
            st.markdown(f'### Low Outliers (<{lower_bound:.1f}%)')
            st.markdown(f'**{len(outliers_low)} boards**')
            if len(outliers_low) > 0:
                st.dataframe(
                    outliers_low[['Flop', 'Board Category', outlier_action]].sort_values(outlier_action).head(20),
                    use_container_width=True
                )
