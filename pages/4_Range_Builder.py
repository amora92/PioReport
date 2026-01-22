"""
Range Builder & Strategy Notes

Build custom ranges and take study notes:
- Create custom board subsets for focused study
- Add personal notes and annotations
- Compare strategies across custom groupings
- Export study materials
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.data_loader import list_reports, load_report, get_action_columns, get_equity_columns

st.set_page_config(
    page_title='Range Builder - PioReport',
    page_icon='🎯',
    layout='wide'
)

st.title('🎯 Range Builder & Study Notes')
st.markdown('*Create custom board subsets and take study notes*')

# Initialize session state for saved ranges
if 'saved_ranges' not in st.session_state:
    st.session_state['saved_ranges'] = {}

if 'study_notes' not in st.session_state:
    st.session_state['study_notes'] = {}

# --- SIDEBAR ---
st.sidebar.header('📂 Select Report')
available_reports = list_reports()

if not available_reports:
    st.warning('No reports available. Please upload a report on the main page first.')
    st.stop()

selected_report = st.sidebar.selectbox(
    'Choose a report:',
    available_reports,
    key='range_report'
)

df = load_report(selected_report)

if df is None or df.empty:
    st.error('Failed to load the selected report.')
    st.stop()

action_cols = get_action_columns(df)
equity_cols = get_equity_columns(df)
all_categories = sorted(df['Board Category'].unique().tolist())

# --- TABS ---
tab1, tab2, tab3, tab4 = st.tabs(['🎯 Range Builder', '📝 Study Notes', '📊 Custom Analysis', '📤 Export'])

# =============================================================================
# TAB 1: RANGE BUILDER
# =============================================================================
with tab1:
    st.header('🎯 Build Custom Board Ranges')
    st.markdown('Create subsets of boards for focused study.')

    builder_col1, builder_col2 = st.columns([2, 1])

    with builder_col1:
        st.subheader('Define Your Range')

        # Multiple filter criteria
        st.markdown('**Board Categories:**')
        all_categories = sorted(df['Board Category'].unique().tolist())
        selected_cats = st.multiselect(
            'Include categories:',
            all_categories,
            default=[],
            key='range_cats'
        )

        st.markdown('**High Cards:**')
        high_cards = ['A', 'K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3', '2']
        selected_high = st.multiselect(
            'Include high cards:',
            high_cards,
            default=[],
            key='range_high'
        )

        st.markdown('**Action Criteria:**')
        crit_col1, crit_col2, crit_col3 = st.columns(3)
        with crit_col1:
            crit_action = st.selectbox('Action:', ['None'] + action_cols, key='crit_action')
        with crit_col2:
            crit_operator = st.selectbox('Operator:', ['>', '<', '>=', '<=', '=='], key='crit_op')
        with crit_col3:
            crit_value = st.number_input('Value %:', 0, 100, 50, key='crit_val')

        st.markdown('**Board Properties:**')
        prop_col1, prop_col2, prop_col3 = st.columns(3)
        with prop_col1:
            include_fd = st.checkbox('Flush Draw', key='inc_fd')
        with prop_col2:
            include_connected = st.checkbox('Connected', key='inc_conn')
        with prop_col3:
            include_broadway = st.checkbox('Broadway', key='inc_bway')

    # Build the range
    range_df = df.copy()

    if selected_cats:
        range_df = range_df[range_df['Board Category'].isin(selected_cats)]

    if selected_high and 'High Card' in range_df.columns:
        range_df = range_df[range_df['High Card'].isin(selected_high)]

    if crit_action != 'None' and crit_action in range_df.columns:
        if crit_operator == '>':
            range_df = range_df[range_df[crit_action] > crit_value]
        elif crit_operator == '<':
            range_df = range_df[range_df[crit_action] < crit_value]
        elif crit_operator == '>=':
            range_df = range_df[range_df[crit_action] >= crit_value]
        elif crit_operator == '<=':
            range_df = range_df[range_df[crit_action] <= crit_value]
        else:
            range_df = range_df[range_df[crit_action] == crit_value]

    if include_fd and 'Has Flush Draw' in range_df.columns:
        range_df = range_df[range_df['Has Flush Draw'] == True]
    if include_connected and 'Is Connected' in range_df.columns:
        range_df = range_df[range_df['Is Connected'] == True]
    if include_broadway and 'Has Broadway' in range_df.columns:
        range_df = range_df[range_df['Has Broadway'] == True]

    with builder_col2:
        st.subheader('Range Preview')
        st.metric('Boards in Range', f'{len(range_df):,}')
        st.metric('% of Total', f'{len(range_df)/len(df)*100:.1f}%')

        if len(range_df) > 0 and action_cols:
            st.markdown('**Average Strategy:**')
            for col in action_cols[:4]:
                avg_val = range_df[col].mean()
                overall_avg = df[col].mean()
                diff = avg_val - overall_avg
                st.metric(
                    col.replace(' freq', ''),
                    f'{avg_val:.1f}%',
                    f'{diff:+.1f}% vs overall'
                )

    # Display range boards
    if len(range_df) > 0:
        st.markdown('---')
        st.subheader(f'📋 Boards in Range ({len(range_df)})')

        display_cols = ['Flop', 'Board Category', 'High Card'] + action_cols
        st.dataframe(
            range_df[display_cols].head(100),
            use_container_width=True,
            height=400
        )

        # Save range
        st.markdown('---')
        save_col1, save_col2 = st.columns([2, 1])
        with save_col1:
            range_name = st.text_input('Range Name:', placeholder='e.g., "High Check Spots"', key='range_name')
        with save_col2:
            if st.button('💾 Save Range', key='save_range'):
                if range_name:
                    st.session_state['saved_ranges'][range_name] = {
                        'flops': range_df['Flop'].tolist(),
                        'count': len(range_df),
                        'criteria': {
                            'categories': selected_cats,
                            'high_cards': selected_high,
                            'action_criteria': f'{crit_action} {crit_operator} {crit_value}' if crit_action != 'None' else None
                        }
                    }
                    st.success(f'Range "{range_name}" saved!')
                else:
                    st.warning('Please enter a range name.')

    # Show saved ranges
    if st.session_state['saved_ranges']:
        st.markdown('---')
        st.subheader('📁 Saved Ranges')

        for name, data in st.session_state['saved_ranges'].items():
            with st.expander(f'📌 {name} ({data["count"]} boards)'):
                st.write(f'**Criteria:** {data["criteria"]}')
                if st.button(f'Load Range: {name}', key=f'load_{name}'):
                    st.info(f'Range "{name}" contains: {", ".join(data["flops"][:10])}...')
                if st.button(f'🗑️ Delete: {name}', key=f'del_{name}'):
                    del st.session_state['saved_ranges'][name]
                    st.rerun()

# =============================================================================
# TAB 2: STUDY NOTES
# =============================================================================
with tab2:
    st.header('📝 Study Notes')
    st.markdown('*Add personal notes and annotations to your study*')

    note_col1, note_col2 = st.columns([1, 2])

    with note_col1:
        st.subheader('Quick Notes')

        # Board-specific note
        st.markdown('**Add Note for Specific Board:**')
        note_board = st.selectbox(
            'Select Board:',
            df['Flop'].tolist()[:100],  # Limit for performance
            key='note_board'
        )

        note_text = st.text_area(
            'Your Note:',
            placeholder='e.g., "Remember to bet more on this texture..."',
            key='note_text'
        )

        if st.button('📝 Save Note', key='save_note'):
            if note_text:
                if selected_report not in st.session_state['study_notes']:
                    st.session_state['study_notes'][selected_report] = {}
                st.session_state['study_notes'][selected_report][note_board] = note_text
                st.success('Note saved!')

        # Category note
        st.markdown('---')
        st.markdown('**Add Note for Category:**')
        note_category = st.selectbox(
            'Select Category:',
            sorted(df['Board Category'].unique().tolist()),
            key='note_cat'
        )

        cat_note = st.text_area(
            'Category Note:',
            placeholder='e.g., "Generally want to bet bigger on paired boards..."',
            key='cat_note'
        )

        if st.button('📝 Save Category Note', key='save_cat_note'):
            if cat_note:
                if selected_report not in st.session_state['study_notes']:
                    st.session_state['study_notes'][selected_report] = {}
                st.session_state['study_notes'][selected_report][f'CAT:{note_category}'] = cat_note
                st.success('Category note saved!')

    with note_col2:
        st.subheader('Your Notes')

        if selected_report in st.session_state['study_notes']:
            notes = st.session_state['study_notes'][selected_report]

            if notes:
                # Board notes
                board_notes = {k: v for k, v in notes.items() if not k.startswith('CAT:')}
                cat_notes = {k.replace('CAT:', ''): v for k, v in notes.items() if k.startswith('CAT:')}

                if board_notes:
                    st.markdown('**Board Notes:**')
                    for board, note in board_notes.items():
                        with st.expander(f'📌 {board}'):
                            st.write(note)
                            # Show board data
                            board_data = df[df['Flop'] == board]
                            if len(board_data) > 0:
                                st.dataframe(board_data[['Flop', 'Board Category'] + action_cols], use_container_width=True)

                if cat_notes:
                    st.markdown('**Category Notes:**')
                    for cat, note in cat_notes.items():
                        with st.expander(f'📁 {cat}'):
                            st.write(note)
            else:
                st.info('No notes saved yet. Add notes using the form on the left.')
        else:
            st.info('No notes saved for this report yet.')

    # Key Insights Summary
    st.markdown('---')
    st.subheader('📊 Key Insights Generator')
    st.markdown('*Generate summary insights based on the data*')

    if st.button('🔄 Generate Insights', key='gen_insights'):
        insights = []

        # Find extreme categories
        cat_summary = df.groupby('Board Category')[action_cols].mean()

        for col in action_cols:
            max_cat = cat_summary[col].idxmax()
            min_cat = cat_summary[col].idxmin()
            max_val = cat_summary[col].max()
            min_val = cat_summary[col].min()

            if max_val - min_val > 15:
                insights.append(f"**{col}**: Highest on {max_cat} ({max_val:.1f}%), lowest on {min_cat} ({min_val:.1f}%)")

        # Flush draw effect
        if 'Has Flush Draw' in df.columns:
            fd_diff = df[df['Has Flush Draw']][action_cols[0]].mean() - df[~df['Has Flush Draw']][action_cols[0]].mean()
            if abs(fd_diff) > 5:
                direction = "more" if fd_diff > 0 else "less"
                insights.append(f"**Flush Draw Effect**: {direction} {action_cols[0].replace(' freq', '')} ({abs(fd_diff):.1f}% difference)")

        if insights:
            for insight in insights:
                st.markdown(f'- {insight}')
        else:
            st.info('No significant insights found.')

# =============================================================================
# TAB 3: CUSTOM ANALYSIS
# =============================================================================
with tab3:
    st.header('📊 Custom Analysis')
    st.markdown('*Run custom analyses on your board ranges*')

    import plotly.express as px

    analysis_mode = st.selectbox(
        'Analysis Mode:',
        ['Compare Two Ranges', 'Strategy Breakdown', 'Correlation Matrix'],
        key='custom_analysis'
    )

    if analysis_mode == 'Compare Two Ranges':
        st.subheader('Compare Two Board Ranges')

        comp_col1, comp_col2 = st.columns(2)

        with comp_col1:
            st.markdown('**Range 1:**')
            range1_cat = st.multiselect('Categories:', all_categories, key='r1_cat')
            if range1_cat:
                range1_df = df[df['Board Category'].isin(range1_cat)]
            else:
                range1_df = df

        with comp_col2:
            st.markdown('**Range 2:**')
            range2_cat = st.multiselect('Categories:', all_categories, key='r2_cat')
            if range2_cat:
                range2_df = df[df['Board Category'].isin(range2_cat)]
            else:
                range2_df = df

        if len(range1_df) > 0 and len(range2_df) > 0:
            # Compare averages
            comparison = pd.DataFrame({
                'Range 1': range1_df[action_cols].mean(),
                'Range 2': range2_df[action_cols].mean(),
                'Difference': range2_df[action_cols].mean() - range1_df[action_cols].mean()
            }).round(1)

            st.dataframe(
                comparison.style.background_gradient(subset=['Difference'], cmap='RdYlGn_r', vmin=-30, vmax=30),
                use_container_width=True
            )

            # Bar chart comparison
            fig = px.bar(
                comparison.reset_index().melt(id_vars='index', value_vars=['Range 1', 'Range 2']),
                x='index', y='value', color='variable',
                barmode='group',
                title='Strategy Comparison',
                labels={'index': 'Action', 'value': 'Frequency %', 'variable': 'Range'}
            )
            st.plotly_chart(fig, use_container_width=True)

    elif analysis_mode == 'Strategy Breakdown':
        st.subheader('Strategy Breakdown by Property')

        breakdown_prop = st.selectbox(
            'Break down by:',
            ['Board Category', 'High Card', 'Has Flush Draw', 'Is Connected'],
            key='breakdown_prop'
        )

        breakdown_action = st.selectbox('Action:', action_cols, key='breakdown_action')

        if breakdown_prop in df.columns:
            breakdown_data = df.groupby(breakdown_prop)[breakdown_action].agg(['mean', 'std', 'count'])
            breakdown_data.columns = ['Average', 'Std Dev', 'Count']
            breakdown_data = breakdown_data.sort_values('Average', ascending=False)

            # Display table
            st.dataframe(
                breakdown_data.style.background_gradient(subset=['Average'], cmap='YlOrRd'),
                use_container_width=True
            )

            # Box plot
            fig = px.box(
                df, x=breakdown_prop, y=breakdown_action,
                title=f'{breakdown_action} by {breakdown_prop}'
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)

    else:  # Correlation Matrix
        st.subheader('Correlation Matrix')

        # Build correlation matrix
        numeric_cols = action_cols + equity_cols
        corr_matrix = df[numeric_cols].corr()

        fig = px.imshow(
            corr_matrix,
            labels=dict(color='Correlation'),
            color_continuous_scale='RdBu_r',
            aspect='auto',
            title='Action & Equity Correlations'
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)

        # Highlight strong correlations
        st.markdown('**Strong Correlations (|r| > 0.5):**')
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr = corr_matrix.iloc[i, j]
                if abs(corr) > 0.5:
                    st.write(f'- {corr_matrix.columns[i]} ↔ {corr_matrix.columns[j]}: {corr:.2f}')

# =============================================================================
# TAB 4: EXPORT
# =============================================================================
with tab4:
    st.header('📤 Export Study Materials')

    st.markdown('Export your study data in various formats.')

    export_col1, export_col2 = st.columns(2)

    with export_col1:
        st.subheader('Export Data')

        # Full data export
        full_csv = df.to_csv(index=False)
        st.download_button(
            '📥 Download Full Dataset (CSV)',
            full_csv,
            file_name=f'{selected_report}_full_data.csv',
            mime='text/csv'
        )

        # Summary export
        summary = df.groupby('Board Category')[action_cols + equity_cols].mean().round(1)
        summary_csv = summary.to_csv()
        st.download_button(
            '📥 Download Category Summary (CSV)',
            summary_csv,
            file_name=f'{selected_report}_summary.csv',
            mime='text/csv'
        )

        # Saved ranges export
        if st.session_state['saved_ranges']:
            ranges_json = json.dumps(st.session_state['saved_ranges'], indent=2)
            st.download_button(
                '📥 Download Saved Ranges (JSON)',
                ranges_json,
                file_name='saved_ranges.json',
                mime='application/json'
            )

    with export_col2:
        st.subheader('Export Study Notes')

        if st.session_state['study_notes']:
            notes_json = json.dumps(st.session_state['study_notes'], indent=2)
            st.download_button(
                '📥 Download Study Notes (JSON)',
                notes_json,
                file_name='study_notes.json',
                mime='application/json'
            )

            # Export as markdown
            md_content = f"# Study Notes - {selected_report}\n\n"
            if selected_report in st.session_state['study_notes']:
                for key, note in st.session_state['study_notes'][selected_report].items():
                    if key.startswith('CAT:'):
                        md_content += f"## Category: {key.replace('CAT:', '')}\n{note}\n\n"
                    else:
                        md_content += f"## Board: {key}\n{note}\n\n"

            st.download_button(
                '📥 Download Study Notes (Markdown)',
                md_content,
                file_name='study_notes.md',
                mime='text/markdown'
            )
        else:
            st.info('No study notes to export.')

    # Import section
    st.markdown('---')
    st.subheader('📥 Import Study Materials')

    uploaded_notes = st.file_uploader(
        'Import Study Notes (JSON):',
        type=['json'],
        key='import_notes'
    )

    if uploaded_notes:
        try:
            imported_data = json.load(uploaded_notes)
            if st.button('Import Notes'):
                st.session_state['study_notes'].update(imported_data)
                st.success('Notes imported successfully!')
        except Exception as e:
            st.error(f'Failed to import notes: {e}')

    uploaded_ranges = st.file_uploader(
        'Import Saved Ranges (JSON):',
        type=['json'],
        key='import_ranges'
    )

    if uploaded_ranges:
        try:
            imported_ranges = json.load(uploaded_ranges)
            if st.button('Import Ranges'):
                st.session_state['saved_ranges'].update(imported_ranges)
                st.success('Ranges imported successfully!')
        except Exception as e:
            st.error(f'Failed to import ranges: {e}')
