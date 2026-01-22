import streamlit as st
import zipfile
import os
from io import BytesIO
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors
import shutil
from matplotlib.patches import Patch

# --- UTILITY FUNCTIONS ---

def extract_zip(zip_bytes):
    try:
        with zipfile.ZipFile(BytesIO(zip_bytes), 'r') as zf:
            root_folder = zf.namelist()[0].split('/')[0]
            extract_dir = os.path.join('extracted', root_folder)
            
            if os.path.exists(extract_dir):
                shutil.rmtree(extract_dir)
            
            zf.extractall(path='extracted/')
            return extract_dir, zf.namelist()
    except Exception as e:
        st.error(f'Failed to extract ZIP file: {e}')
        return None, None


def find_report_csv(directory, report_type):
    """Recursively finds a specified report CSV file in the directory structure."""
    for root, dirs, files in os.walk(directory):
        if report_type in files:
            return os.path.join(root, report_type)
    return None

def list_extracted_folders(base_dir='extracted'):
    """Lists all top-level extracted report folders."""
    if not os.path.exists(base_dir):
        return []
    folders = []
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if os.path.isdir(item_path):
            folders.append(item)
    return folders

def clean_up_folder(folder_path):
    if os.path.exists(folder_path):
        st.info(f"Cleaning up old extraction folder: {folder_path.split(os.path.sep)[-1]}")
        shutil.rmtree(folder_path)


def convert_suits_to_symbols(cards):
    """Converts suit letters (s, d, h, c) to robust Unicode symbols."""
    suit_mapping = {'s': '♠', 'd': '♦', 'h': '♥', 'c': '♣'}
    formatted_cards = []
    for card in cards.split():
        if len(card) == 2:
            rank = card[0]
            suit_letter = card[1]
            suit_symbol = suit_mapping.get(suit_letter, suit_letter)
            formatted_cards.append(rank + suit_symbol)
        else:
            formatted_cards.append(card)
    return ' '.join(formatted_cards)


def equity_color_scale(value, min_val, max_val, midpoint=50):
    """Custom color scale for Streamlit tables (Red < Midpoint < Green)."""
    if pd.isna(value) or min_val == max_val:
        return ''
    normalized_value = (value - min_val) / (max_val - min_val)
    if value < midpoint:
        relative = (value - min_val) / (midpoint - min_val) if (midpoint - min_val) > 0 else 0
        r, g, b = 255, int(255 * relative), 0
    else:
        relative = (value - midpoint) / (max_val - midpoint) if (max_val - midpoint) > 0 else 0
        r, g, b = int(255 * (1 - relative)), 255, 0
    r, g, b = max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b))
    return f'background-color: rgb({r}, {g}, {b})'


# --- CLASSIFICATION FUNCTIONS ---

def get_flop_features(cards):
    """
    Parses the flop string (e.g., 'A♠ K♦ 2♣') to return:
    1. The High Card Rank (string)
    2. Boolean indicating if the board is paired or tripled.
    """
    ranks_map = {'A': 14, 'K': 13, 'Q': 12, 'J': 11, 'T': 10, '9': 9, '8': 8, '7': 7, '6': 6, '5': 5, '4': 4, '3': 3, '2': 2}
    
    try:
        # Extract ranks (assuming format "R+Suit")
        parts = cards.split()
        current_ranks = [c[0] for c in parts]
        current_values = [ranks_map.get(r, 0) for r in current_ranks]
        
        # Determine High Card
        max_val = max(current_values)
        val_to_rank = {v: k for k, v in ranks_map.items()}
        high_card = val_to_rank.get(max_val, '?')
        
        # Determine Pairing (Set length < 3 means duplicates exist)
        is_paired = len(set(current_values)) < 3
        
        return high_card, is_paired
        
    except Exception:
        return '?', False


def classify_board_type(cards):
    """Provides classification of board texture based on suitedness, pairing, and connectedness."""
    try:
        parts = cards.split()
        ranks = [card[0] for card in parts if len(card) >= 2]
        
        # Recover suits for counts (simplified for converted symbols)
        # We just count unique symbols in the string
        suits = [card[1] for card in parts if len(card) >= 2]
        
    except IndexError:
        return 'Invalid Flop'

    # --- 1. Pairing Check ---
    rank_counts = pd.Series(ranks).value_counts()
    if rank_counts.max() == 3: return 'Paired/Tripled Boards'
    if rank_counts.max() == 2: return 'Paired Boards'

    # --- 2. Suitedness Check ---
    suit_counts = pd.Series(suits).value_counts()
    max_suits = suit_counts.max()
    if max_suits == 3: return 'Monotone Boards'
    suitedness_label = 'Two-Tone' if max_suits == 2 else 'Rainbow'

    # --- 3. Connectedness and High Card Check ---
    rank_values = {'A': 14, 'K': 13, 'Q': 12, 'J': 11, 'T': 10, '9': 9, '8': 8, '7': 7, '6': 6, '5': 5, '4': 4, '3': 3, '2': 2}
    values = sorted([rank_values[r] for r in ranks], reverse=True)

    gaps = [values[i] - values[i+1] - 1 for i in range(len(values) - 1)]
    max_gap = max(gaps) if gaps else 0
    is_wheel = (14 in values and 5 in values and 4 in values)
    
    if max_gap <= 0 or is_wheel:
        conn_label = 'Connected'
    elif max_gap <= 1:
        conn_label = 'One-Gapped'
    else:
        conn_label = 'Disconnected'

    broadways = [v for v in values if v >= 10]
    
    if len(broadways) == 3:
        high_card_type = 'Three Broadway'
    elif len(broadways) == 2:
        high_card_type = 'Two Broadway'
    elif values[0] == 14:
        high_card_type = 'Ace High'
    elif values[0] == 13:
        high_card_type = 'King High'
    else:
        high_card_type = 'Low/Mid Cards'
        
    return f'{conn_label} {high_card_type} ({suitedness_label})'


# --- VISUALIZATION FUNCTIONS ---

def create_stacked_bar_plot(df, highlight_column):
    """Creates a stacked horizontal bar plot for strategy frequency."""
    df_plot = df.copy()
    action_columns = [col for col in df_plot.columns if 'freq' in col]
    
    flop_column = df_plot['Flop']
    highlight_data = df_plot[highlight_column]
    
    other_action_cols = [col for col in action_columns if col != highlight_column]
    df_numeric_other = df_plot[other_action_cols]

    df_numeric_other = df_numeric_other.iloc[::-1].reset_index(drop=True)
    flop_column_rev = flop_column.iloc[::-1].reset_index(drop=True)
    highlight_data_rev = highlight_data.iloc[::-1].reset_index(drop=True)
    
    color_palette = plt.cm.tab20(range(len(other_action_cols)))
    HIGHLIGHT_COLOR = '#F59E0B' 

    fig_height = max(8, len(df_plot) * 0.35)
    fig, ax = plt.subplots(figsize=(12, fig_height))

    bottoms = pd.Series([0.0] * len(df_numeric_other))
    
    for i, col in enumerate(other_action_cols):
        ax.barh(flop_column_rev, df_numeric_other[col], left=bottoms, color=color_palette[i], label=col, height=0.7)
        bottoms = bottoms + df_numeric_other[col]

    ax.barh(flop_column_rev, highlight_data_rev, left=bottoms, color=HIGHLIGHT_COLOR, label=highlight_column, height=0.7)

    ax.set_title(f'Strategy Frequency (Highlighting: {highlight_column})', fontsize='x-large')
    ax.set_yticklabels(flop_column_rev, rotation=0, ha='right', fontsize='small')
    ax.set_xlim(0, 100)
    ax.grid(axis='x', linestyle='--')

    full_labels = other_action_cols + [highlight_column]
    color_list = list(color_palette[:len(other_action_cols)])
    color_list.append(HIGHLIGHT_COLOR)

    legend_patches = [Patch(color=color, label=label) for color, label in zip(color_list, full_labels)]
    ax.legend(handles=legend_patches, bbox_to_anchor=(1.02, 1), loc='upper left', fontsize='large')

    plt.tight_layout()
    st.pyplot(fig)


def stacked_equity_plot(dataframe, extract_dir):
    """Creates a side-by-side equity bucket plot for IP vs. OOP."""
    
    IP_path = find_report_csv(extract_dir, 'report_IP_Full.csv')
    OOP_path = find_report_csv(extract_dir, 'report_OOP_Full.csv')
    
    if not IP_path or not OOP_path:
        st.error('report_IP_Full.csv and/or report_OOP_Full.csv not found in the report directory.')
        return

    try:
        IP_df = pd.read_csv(IP_path)
        IP_df = IP_df[IP_df.columns[:4]]
        OOP_df = pd.read_csv(OOP_path)
        OOP_df = OOP_df[OOP_df.columns[:4]]
    except Exception as e:
        st.error(f"Error reading full equity reports: {e}")
        return

    # Filter equity reports based on the Flops currently in the filtered main dataframe
    current_flops = dataframe['Flop'].unique().tolist()
    
    player_dfs = [IP_df, OOP_df]
    for i, df in enumerate(player_dfs):
        df.columns.values[2] = 'Weight'
        df.columns.values[3] = 'Equity'
        df['Flop'] = df['Flop'].apply(convert_suits_to_symbols)
        # Filter strictly to matches
        player_dfs[i] = df[df['Flop'].isin(current_flops)]

    IP_df = player_dfs[0]
    OOP_df = player_dfs[1]
    
    buckets_df = pd.DataFrame()
    buckets_df['Flop'] = current_flops

    def get_bucket_sums(df, flop):
        flop_df = df[df['Flop'] == flop]
        weak_equity_sum = flop_df[flop_df['Equity'] <= 0.25]['Weight'].sum()
        okay_equity_sum = flop_df[(flop_df['Equity'] > 0.25) & (flop_df['Equity'] <= 0.50)]['Weight'].sum()
        good_equity_sum = flop_df[(flop_df['Equity'] > 0.50) & (flop_df['Equity'] <= 0.75)]['Weight'].sum()
        nut_equity_sum = flop_df[flop_df['Equity'] > 0.75]['Weight'].sum()
        return weak_equity_sum, okay_equity_sum, good_equity_sum, nut_equity_sum

    IP_buckets = []
    OOP_buckets = []

    for flop in current_flops:
        IP_buckets.append(get_bucket_sums(IP_df, flop))
        OOP_buckets.append(get_bucket_sums(OOP_df, flop))

    IP_buckets_df = pd.DataFrame(IP_buckets, columns=['IP_Weak', 'IP_Okay', 'IP_Good', 'IP_Nut'])
    OOP_buckets_df = pd.DataFrame(OOP_buckets, columns=['OOP_Weak', 'OOP_Okay', 'OOP_Good', 'OOP_Nut'])

    def convert_to_percentage(df):
        total = df.sum(axis=1)
        percentage_df = df.divide(total, axis=0).fillna(0) * 100 
        return percentage_df

    IP_buckets_df = convert_to_percentage(IP_buckets_df)
    OOP_buckets_df = convert_to_percentage(OOP_buckets_df)

    buckets_df = pd.concat([buckets_df.reset_index(drop=True), IP_buckets_df, OOP_buckets_df], axis=1)
    buckets_df = buckets_df.iloc[::-1].reset_index(drop=True)

    gap = 0.15
    fig, ax = plt.subplots(figsize=(12, max(8, len(current_flops) * 0.5)))
    bar_width = 0.35
    index = range(len(current_flops))

    IP_colors = ['#A5D5F3', '#3AA3E4', '#146090', '#072436'] 
    OOP_colors = ['#F3A5A5', '#E43A3A', '#901414', '#360707']
    
    oop_positions = [p + bar_width + gap for p in index]
    
    ax.barh(oop_positions, buckets_df['OOP_Weak'], bar_width, color=OOP_colors[0])
    ax.barh(oop_positions, buckets_df['OOP_Okay'], bar_width, left=buckets_df['OOP_Weak'], color=OOP_colors[1])
    ax.barh(oop_positions, buckets_df['OOP_Good'], bar_width, left=buckets_df['OOP_Weak'] + buckets_df['OOP_Okay'], color=OOP_colors[2])
    ax.barh(oop_positions, buckets_df['OOP_Nut'], bar_width, left=buckets_df['OOP_Weak'] + buckets_df['OOP_Okay'] + buckets_df['OOP_Good'], color=OOP_colors[3])

    ax.barh(index, buckets_df['IP_Weak'], bar_width, color=IP_colors[0])
    ax.barh(index, buckets_df['IP_Okay'], bar_width, left=buckets_df['IP_Weak'], color=IP_colors[1])
    ax.barh(index, buckets_df['IP_Good'], bar_width, left=buckets_df['IP_Weak'] + buckets_df['IP_Okay'], color=IP_colors[2])
    ax.barh(index, buckets_df['IP_Nut'], bar_width, left=buckets_df['IP_Weak'] + buckets_df['IP_Okay'] + buckets_df['IP_Good'], color=IP_colors[3])

    ax.set_title(f'Equity Buckets (IP vs. OOP)', fontsize='x-large')
    ax.set_yticks([p + (bar_width + gap) / 2 for p in index]) 
    ax.set_yticklabels(buckets_df['Flop'], fontsize='medium')
    ax.set_xlim(0, 100)
    ax.set_xlabel('Range Percentage')

    legend_patches = [
        Patch(color=IP_colors[0], label='IP Weak (0-25%)'),
        Patch(color=IP_colors[1], label='IP Okay (26-50%)'),
        Patch(color=IP_colors[2], label='IP Good (51-75%)'),
        Patch(color=IP_colors[3], label='IP Nut (>75%)'),
        Patch(color=OOP_colors[0], label='OOP Weak (0-25%)'),
        Patch(color=OOP_colors[1], label='OOP Okay (26-50%)'),
        Patch(color=OOP_colors[2], label='OOP Good (51-75%)'),
        Patch(color=OOP_colors[3], label='OOP Nut (>75%)'),
    ]
    ax.legend(handles=legend_patches, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='medium', ncol=2, title="Equity Buckets")
    plt.tight_layout()
    st.pyplot(fig)


# --- STREAMLIT APP LOGIC ---

st.set_page_config(page_title='PioSOLVER Report Analyzer', layout='wide')
st.sidebar.title('PioSOLVER Report')

uploaded_file = st.file_uploader('Upload a PioSOLVER .zip file', type=['zip'])

extract_dir = None
report_path = None
selected_folder = None
report_df = pd.DataFrame()
action_columns = []
is_filter = True
selected_texture = None

if uploaded_file is None:
    existing_folders = list_extracted_folders()
    selected_folder = st.sidebar.selectbox('Or select an already uploaded report folder:', ['None'] + existing_folders)
    if selected_folder != 'None':
        extract_dir = os.path.join('extracted', selected_folder)
        report_path = find_report_csv(extract_dir, 'report.csv')
    
elif uploaded_file is not None and uploaded_file.name.endswith('.zip'):
    extract_dir, file_list = extract_zip(uploaded_file.getvalue())
    if extract_dir is not None:
        report_path = find_report_csv(extract_dir, 'report.csv')
        if report_path is None:
            st.error('report.csv not found in the uploaded ZIP file.')
            clean_up_folder(extract_dir)
            extract_dir = None


if report_path and os.path.exists(report_path):
    try:
        report_df = pd.read_csv(report_path, skiprows=3)
        report_df = report_df.drop(report_df.index[-1])
        
        if 'Flop' in report_df.columns:
            # Apply enhancements
            report_df['Flop'] = report_df['Flop'].apply(convert_suits_to_symbols)
            report_df['Board Texture'] = report_df['Flop'].apply(classify_board_type)
            
            # --- NEW: Extract High Card and Paired status ---
            # We use zip to apply the function and unpack results into two columns
            report_df[['High Card', 'Is Paired']] = report_df['Flop'].apply(
                lambda x: pd.Series(get_flop_features(x))
            )

            equity_columns = ['OOP Equity', 'IP Equity']
            action_columns = [col for col in report_df.columns if 'freq' in col]

            st.sidebar.markdown("---")
            
            # --- 1. Filter by Board Texture ---
            all_textures = report_df['Board Texture'].unique().tolist()
            use_texture_filter = st.sidebar.checkbox('Filter by Board Texture', value=False)
            
            if use_texture_filter:
                try:
                    selected_texture = st.sidebar.selectbox('Select Texture:', all_textures, index=0)
                    report_df = report_df[report_df['Board Texture'] == selected_texture]
                except:
                    st.warning("Could not apply texture filter.")
            
            # --- 2. Filter by High Card (Unpaired) ---
            st.sidebar.markdown("### High Card Filter")
            high_card_options = ['All', 'A', 'K', 'Q', 'J', 'T', '9', '8', '7', '6', '5']
            selected_high_card = st.sidebar.selectbox('Select High Card (Unpaired Flops Only):', high_card_options)
            
            if selected_high_card != 'All':
                # Filter for High Card matching AND ensure board is NOT paired
                report_df = report_df[
                    (report_df['High Card'] == selected_high_card) & 
                    (report_df['Is Paired'] == False)
                ]
                
                # Visual Feedback
                if report_df.empty:
                    st.warning(f"No unpaired flops found with High Card: {selected_high_card}")
                else:
                    st.success(f"Filtered for {selected_high_card}-High Unpaired Flops")

            # --- 3. Sorting ---
            st.sidebar.markdown("---")
            sort_options = [col for col in report_df.columns if col not in ['Flop', 'Board Texture', 'High Card', 'Is Paired']]
            sort_by = st.sidebar.selectbox('Sort table by:', sort_options)
            ascend = st.sidebar.radio('Sort order:', ['Largest to smallest', 'Smallest to largest'])
            ascend_bool = True if ascend == 'Smallest to largest' else False
            report_df = report_df.sort_values(by=[sort_by], ascending=ascend_bool)

            # --- 4. Strategy Highlight ---
            if action_columns:
                highlight_action = st.sidebar.selectbox('Action to Highlight in Strategy Plot:', action_columns)
            else:
                highlight_action = None
            
            st.sidebar.markdown("---")

            # --- Display Table ---
            st.header(f"Aggregated Flop Strategy Report")
            
            styled_df = report_df.style.map(
                lambda x: equity_color_scale(x, report_df['OOP Equity'].min(), report_df['OOP Equity'].max()),
                subset=['OOP Equity']
            ).map(
                lambda x: equity_color_scale(x, report_df['IP Equity'].min(), report_df['IP Equity'].max()),
                subset=['IP Equity']
            )

            action_cmap = plt.cm.get_cmap('YlGnBu')
            for column in action_columns:
                styled_df = styled_df.background_gradient(
                    cmap=action_cmap, 
                    subset=[column], 
                    vmin=0, vmax=100
                )
            
            st.dataframe(styled_df, use_container_width=True, height=700)
            
            # --- Display Plots ---
            st.header("Visualizations")
            
            if not report_df.empty:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader(f"Strategy Frequency: {highlight_action} Highlight")
                    if highlight_action:
                        create_stacked_bar_plot(report_df, highlight_action)
                    else:
                        st.warning("No action frequency columns found.")

                with col2:
                    st.subheader("IP vs. OOP Equity Bucket Distribution")
                    if extract_dir:
                        # Note: We pass the filtered dataframe logic implicitly by filtering inside the function using current flops
                        stacked_equity_plot(report_df, extract_dir)
                    else:
                        st.error("Cannot plot Equity Buckets: Full report files missing.")
            else:
                st.info("The current filters resulted in an empty dataset.")

        else:
            st.error('Flop column data not found in the report.csv.')
            clean_up_folder(extract_dir)
            extract_dir = None
            
    except Exception as e:
        st.error(f'An error occurred: {e}')
        if extract_dir:
             clean_up_folder(extract_dir)

elif not report_path:
    st.info('Upload a PioSOLVER aggregated report ZIP file to begin analysis.')

if uploaded_file is not None and extract_dir:
    if st.sidebar.button('Clear Uploaded Data'):
        clean_up_folder(extract_dir)
        st.rerun()