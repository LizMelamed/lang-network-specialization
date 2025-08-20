import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter, defaultdict
rng = np.random.default_rng(42)
import os
import re

mpl.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,         
    "font.size": 16,            
    "axes.titlesize": 20,
    "axes.labelsize": 18,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "figure.titlesize": 22,
})

def main():

    VEC_PATH = "data/merged_df_with_vectors.parquet"  # Path to the merged DataFrame with GloVe vectors

    RESULTS_DIR = "results/semantic"
    os.makedirs(RESULTS_DIR, exist_ok=True)
    def _ensure_dir(d):
        os.makedirs(d, exist_ok=True)

    def _slug(s: str) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(s)).strip("_")

    def _savefig(fig, filename, subdir=None, dpi=300):
        out_dir = RESULTS_DIR if subdir is None else os.path.join(RESULTS_DIR, subdir)
        _ensure_dir(out_dir)
        path = os.path.join(out_dir, _slug(filename))
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        print(f" Saved: {path}")

    merged_df_with_vectors = pd.read_parquet(VEC_PATH) # import the data with the GloVe vectors

    # perform semantic analysis using GloVe
    def parse_vector_string(vector_str):
        """Parse vector string to numpy array."""
        try:
            if isinstance(vector_str, str):
                # Remove brackets and split
                clean_str = vector_str.strip('[]')
                # Split by spaces and convert to float
                values = [float(x) for x in clean_str.split() if x.strip()]
                return np.array(values)
            else:
                return np.array(vector_str)
        except:
            return None

    _WORD_RE = re.compile(r"[A-Za-z']+")

    def tokenize_chunks(x):
        """Split a chunk/sequence of words into individual tokens.
        - keeps letters and apostrophes (don’t -> dont if you prefer, post-process)
        - lowercases
        """
        if x is None or str(x) == 'nan':
            return []
        return [w.lower() for w in _WORD_RE.findall(str(x))]

    def extract_story_semantic_data(df):
        print(" EXTRACTING SEMANTIC DATA FOR EACH STORY")
        print("=" * 60)

        story_data = {}

        for docid in df['docid'].unique():
            story_df = df[df['docid'] == docid].copy()

            # Extract all words and vectors for this story
            all_words = []
            all_word_vectors = []  # Individual word vectors
            all_chunk_vectors = []  # Average vectors per chunk

            for idx, row in story_df.iterrows():
                # Check if found_words exists and is not null
                found_words_val = row['found_words']
                if found_words_val is not None and str(found_words_val) != 'nan':
                    # Parse words (comma-separated)
                    chunks = str(found_words_val).split(',')        
                    words  = []
                    for ch in chunks:
                        words.extend(tokenize_chunks(ch))

                    # Parse individual word vectors
                    word_vectors_val = row['word_vectors']
                    if word_vectors_val is not None and str(word_vectors_val) != 'nan':
                        try:
                            all_words.extend(words)

                            avg_vector_val = row['avg_word_vector']
                            if avg_vector_val is not None and str(avg_vector_val) != 'nan':
                                chunk_vector = parse_vector_string(avg_vector_val)
                                if chunk_vector is not None and len(chunk_vector) > 0:
                                    # Create one vector per word in this chunk (using the avg vector)
                                    for word in words:
                                        all_word_vectors.append(chunk_vector)
                                    all_chunk_vectors.append(chunk_vector)

                        except Exception as e:
                            print(f"    Warning: Could not parse vectors for {docid}: {e}")
                            # Fall back to just using avg_word_vector
                            avg_vector_val = row['avg_word_vector']
                            if avg_vector_val is not None and str(avg_vector_val) != 'nan':
                                chunk_vector = parse_vector_string(avg_vector_val)
                                if chunk_vector is not None and len(chunk_vector) > 0:
                                    all_chunk_vectors.append(chunk_vector)

                    else:
                        # No word_vectors, just use avg_word_vector for the chunk
                        avg_vector_val = row['avg_word_vector']
                        if avg_vector_val is not None and str(avg_vector_val) != 'nan':
                            chunk_vector = parse_vector_string(avg_vector_val)
                            if chunk_vector is not None and len(chunk_vector) > 0:
                                all_chunk_vectors.append(chunk_vector)

            # Create story profile
            story_data[docid] = {
                'words': all_words,
                'word_vectors': np.array(all_word_vectors) if all_word_vectors else np.array([]),
                'chunk_vectors': np.array(all_chunk_vectors) if all_chunk_vectors else np.array([]),
                'word_counts': Counter(all_words),
                'n_word_vectors': len(all_word_vectors),
                'n_chunks': len(all_chunk_vectors),
                'n_unique_words': len(set(all_words)),
                'total_words': len(all_words)
            }

            print(f"  {docid}: {story_data[docid]['n_chunks']} chunks, "
                f"{story_data[docid]['n_word_vectors']} word vectors, "
                f"{story_data[docid]['n_unique_words']} unique words")

        return story_data

    def create_story_semantic_profiles(story_data):
        print("\n CREATING SEMANTIC PROFILES")
        print("-" * 40)

        profiles = {}

        for docid, data in story_data.items():
            if len(data['word_vectors']) == 0:
                print(f"  No word vectors for {docid}")
                continue

            vectors = data['word_vectors']

            # Basic semantic statistics
            mean_vector = np.mean(vectors, axis=0)

            # Semantic diversity (average pairwise cosine distance)
            if len(vectors) > 1:
                similarities = []
                for i in range(min(100, len(vectors))):  # Sample for efficiency
                    for j in range(i+1, min(100, len(vectors))):
                        sim = cosine_similarity([vectors[i]], [vectors[j]])[0][0]
                        similarities.append(sim)
                semantic_coherence = np.mean(similarities) if similarities else 0
                semantic_diversity = 1 - semantic_coherence
            else:
                semantic_coherence = 1.0
                semantic_diversity = 0.0

            # Most common words
            top_words = data['word_counts'].most_common(10)

            profiles[docid] = {
                'mean_vector': mean_vector,
                'semantic_coherence': semantic_coherence,
                'semantic_diversity': semantic_diversity,
                'vector_dimensionality': len(mean_vector),
                'vocabulary_size': data['n_unique_words'],
                'total_words': data['total_words'],
                'top_words': top_words,
                'all_vectors': vectors  # Keep for further analysis
            }

            print(f"  {docid}: coherence={semantic_coherence:.3f}, "
                f"diversity={semantic_diversity:.3f}, vocab={data['n_unique_words']}")

        return profiles

    def create_story_similarity_matrix(profiles):
        """
        Create similarity matrix between stories based on semantic vectors.
        """
        print("\n CALCULATING STORY SIMILARITIES")
        print("-" * 35)

        story_names = list(profiles.keys())
        n_stories = len(story_names)

        # Create matrix of story mean vectors
        story_vectors = np.array([profiles[story]['mean_vector'] for story in story_names])

        # Calculate pairwise cosine similarities
        similarity_matrix = cosine_similarity(story_vectors)

        return similarity_matrix, story_names

    def plot_story_semantic_analysis(profiles, similarity_matrix, story_names):
        print("\n CREATING SEMANTIC VISUALIZATIONS")
        print("-" * 50)

        # 1) Story similarity heatmap (single figure)
        fig, ax = plt.subplots(figsize=(11, 9))
        sns.heatmap(similarity_matrix,
                    xticklabels=story_names,
                    yticklabels=story_names,
                    annot=True, annot_kws={"size": 10},
                    cmap='viridis', fmt='.2f', ax=ax)
        ax.set_title('Story Semantic Similarity (Cosine)')
        ax.set_xlabel('Story')
        ax.set_ylabel('Story')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        _savefig(fig, "semantic_similarity_heatmap.png")
        plt.close(fig)

        # 2) Semantic coherence vs diversity (scatter, single figure)
        fig, ax = plt.subplots(figsize=(9, 7))
        coherence_vals = [profiles[s]['semantic_coherence'] for s in story_names]
        diversity_vals =  [profiles[s]['semantic_diversity']  for s in story_names]
        ax.scatter(coherence_vals, diversity_vals, s=110, alpha=0.8)
        for i, story in enumerate(story_names):
            ax.annotate(story, (coherence_vals[i], diversity_vals[i]),
                        xytext=(6, 6), textcoords='offset points', fontsize=12)
        ax.set_xlabel('Semantic Coherence')
        ax.set_ylabel('Semantic Diversity')
        ax.set_title('Story Semantic Characteristics')
        ax.grid(True, alpha=0.3)
        _savefig(fig, "semantic_characteristics.png")
        plt.close(fig)

        # 3) Vocabulary size by story (single figure)
        fig, ax = plt.subplots(figsize=(12, 7))
        vocab_sizes = [profiles[s]['vocabulary_size'] for s in story_names]
        bars = ax.bar(range(len(story_names)), vocab_sizes,
                    color=plt.cm.Set3(np.linspace(0, 1, len(story_names))), alpha=0.9)
        ax.set_xticks(range(len(story_names)))
        ax.set_xticklabels(story_names, rotation=45, ha='right')
        ax.set_ylabel('Unique Words')
        ax.set_title('Vocabulary Size by Story')
        for i, bar in enumerate(bars):
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., h + max(2, 0.01*h),
                    f'{int(h)}', ha='center', va='bottom', fontsize=12)
        ax.grid(True, axis='y', alpha=0.25)
        _savefig(fig, "vocabulary_size.png")
        plt.close(fig)

        # 4) PCA of story mean vectors (single figure)
        story_vectors = np.array([profiles[story]['mean_vector'] for story in story_names])
        if len(story_vectors) > 1:
            pca = PCA(n_components=2)
            story_pca = pca.fit_transform(story_vectors)
            fig, ax = plt.subplots(figsize=(9, 7))
            sc = ax.scatter(story_pca[:, 0], story_pca[:, 1], s=110, alpha=0.85,
                            c=range(len(story_names)), cmap='tab10')
            for i, story in enumerate(story_names):
                ax.annotate(story, (story_pca[i, 0], story_pca[i, 1]),
                            xytext=(6, 6), textcoords='offset points', fontsize=12)
            ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var)')
            ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var)')
            ax.set_title('Story Semantic Space (PCA)')
            ax.grid(True, alpha=0.3)
            _savefig(fig, "semantic_pca.png")
            plt.close(fig)

        # 5) Top words per story (one figure per story)
        for story in story_names:
            top_words = profiles[story]['top_words'][:10]
            if not top_words:
                continue
            words  = [w for w, _ in top_words]
            counts = [c for _, c in top_words]

            fig, ax = plt.subplots(figsize=(9, 7))
            bars = ax.barh(range(len(words)), counts, alpha=0.9)
            ax.set_yticks(range(len(words)))
            ax.set_yticklabels(words)
            ax.set_xlabel('Word Frequency')
            ax.set_title(f'Top Words: {story}')
            ax.invert_yaxis()
            max_c = max(counts) if counts else 1
            for j, bar in enumerate(bars):
                bar.set_color(plt.cm.Blues(0.25 + 0.75 * counts[j] / max_c))
            _savefig(fig, f"top_words_{story}.png", subdir="top_words")
            plt.close(fig)


    def create_semantic_story_summary(profiles):

        print("\n SEMANTIC STORY CHARACTERIZATION")
        print("=" * 50)

        # Sort stories by semantic diversity
        sorted_stories = sorted(profiles.items(),
                            key=lambda x: x[1]['semantic_diversity'],
                            reverse=True)

        for story, profile in sorted_stories:
            coherence = profile['semantic_coherence']
            diversity = profile['semantic_diversity']
            vocab_size = profile['vocabulary_size']

            # Characterize based on metrics
            if diversity > 0.3:
                semantic_char = "semantically diverse"
            elif diversity > 0.2:
                semantic_char = "moderately diverse"
            else:
                semantic_char = "semantically focused"

            if vocab_size > 200:
                vocab_char = "rich vocabulary"
            elif vocab_size > 100:
                vocab_char = "moderate vocabulary"
            else:
                vocab_char = "limited vocabulary"

            # Get top words
            top_5_words = [w[0] for w in profile['top_words'][:5]]

            print(f"\n **{story}**")
            print(f"   • {semantic_char} ({diversity:.3f} diversity)")
            print(f"   • {vocab_char} ({vocab_size} unique words)")
            print(f"   • Key words: {', '.join(top_5_words)}")
            print(f"   • Semantic coherence: {coherence:.3f}")

    def suggest_story_categorization(profiles):

        print("\n  SUGGESTED STORY CATEGORIZATION")
        print("=" * 45)

        # Cluster stories based on semantic features
        features = []
        story_names = []

        for story, profile in profiles.items():
            features.append([
                profile['semantic_coherence'],
                profile['semantic_diversity'],
                profile['vocabulary_size'] / 1000,  # Normalize
            ])
            story_names.append(story)

        features = np.array(features)

        if len(features) >= 3:
            # K-means clustering
            kmeans = KMeans(n_clusters=min(3, len(features)), random_state=42)
            clusters = kmeans.fit_predict(features)

            # Group stories by cluster
            cluster_groups = defaultdict(list)
            for story, cluster in zip(story_names, clusters):
                cluster_groups[cluster].append(story)

            cluster_names = ['Narrative/Concrete', 'Abstract/Expository', 'Mixed Content']

            for i, (cluster_id, stories) in enumerate(cluster_groups.items()):
                if i < len(cluster_names):
                    print(f"\n📖 **{cluster_names[i]} Stories:**")
                    for story in stories:
                        profile = profiles[story]
                        print(f"   • {story}: diversity={profile['semantic_diversity']:.3f}, "
                            f"vocab={profile['vocabulary_size']}")

    def run_semantic_story_analysis(df):
        print(" STARTING SEMANTIC STORY ANALYSIS")
        print("=" * 60)

        # Extract semantic data
        story_data = extract_story_semantic_data(df)

        if not story_data:
            print(" No semantic data found!")
            return None

        # Create profiles
        profiles = create_story_semantic_profiles(story_data)

        if not profiles:
            print("No profiles created!")
            return None

        # Calculate similarities
        similarity_matrix, story_names = create_story_similarity_matrix(profiles)

        # Create visualizations
        plot_story_semantic_analysis(profiles, similarity_matrix, story_names)

        # Generate summaries
        create_semantic_story_summary(profiles)

        # Suggest categorization
        suggest_story_categorization(profiles)

        return {
            'story_data': story_data,
            'profiles': profiles,
            'similarity_matrix': similarity_matrix,
            'story_names': story_names
        }

    filtered_df = merged_df_with_vectors[
        (merged_df_with_vectors['subject'] == 's061') &
        (merged_df_with_vectors['fROI'] == 'LangLAngG')
    ]

    semantic_results = run_semantic_story_analysis(filtered_df)

    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")

    def analyze_story_progression(df):
        unique_stories = df['docid'].unique()
        unique_frois = df['fROI'].unique()

        print(f" Dataset Overview:")
        print(f"   Stories: {len(unique_stories)}")
        print(f"   Brain regions (fROIs): {len(unique_frois)}")
        print(f"   Subjects: {df['subject'].nunique()}")
        print(f"   Total observations: {len(df)}")
        print(f"   Stories: {list(unique_stories)}")
        print(f"   Brain regions: {list(unique_frois)}")
        print("="*80)

        return unique_stories, unique_frois

    def create_story_progression_plots(df, story_id, save_plots=True):
        story_data = df[df['docid'] == story_id].copy()
        unique_frois = story_data['fROI'].unique()

        # Pre-aggregate (mean, SE) per fROI × time
        story_avg = (story_data
                    .groupby(['time', 'fROI'])['BOLD']
                    .agg(['mean', 'std', 'count'])
                    .reset_index())
        story_avg['se'] = story_avg['std'] / np.sqrt(story_avg['count'])

        # One figure per fROI
        for froi in unique_frois:
            rd = story_avg[story_avg['fROI'] == froi].sort_values('time')
            if rd.empty:
                continue

            fig, ax = plt.subplots(figsize=(10, 6.5))
            ax.plot(rd['time'], rd['mean'], linewidth=2.6, marker='o', markersize=5, alpha=0.9)
            ax.fill_between(rd['time'], rd['mean'] - rd['se'], rd['mean'] + rd['se'], alpha=0.3)
            ax.set_title(f'BOLD Progression — {story_id} — {froi}')
            ax.set_xlabel('Time (story progression)')
            ax.set_ylabel('BOLD')
            ax.grid(True, alpha=0.3)

            rng_val = (rd['mean'].max() - rd['mean'].min())
            mu_val  = rd['mean'].mean()
            ax.text(0.02, 0.98, f'Range: {rng_val:.3f}\nMean: {mu_val:.3f}',
                    transform=ax.transAxes, fontsize=12, va='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

            if save_plots:
                _savefig(fig, f"story_progression_{story_id}_{froi}.png", subdir="story_progression")
            plt.close(fig)

        return story_avg


    def calculate_bold_statistics(df):

        print(" BOLD Response Statistics by Story and Brain Region")
        print("="*80)

        # Calculate statistics grouped by story and brain region
        stats_df = df.groupby(['docid', 'fROI'])['BOLD'].agg([
            'count', 'mean', 'std', 'min', 'max',
            lambda x: x.max() - x.min(),  # range
            lambda x: np.percentile(x, 25),  # Q1
            lambda x: np.percentile(x, 75),  # Q3
        ]).round(4)

        stats_df.columns = ['N_obs', 'Mean', 'Std', 'Min', 'Max', 'Range', 'Q1', 'Q3']

        stats_df['CV'] = (stats_df['Std'] / abs(stats_df['Mean'])).round(4)

        stats_df = stats_df.reset_index()

        return stats_df

    def analyze_story_patterns(df):

        print(" Cross-Story Pattern Analysis")
        print("="*80)

        # Calculate story-level statistics
        story_stats = df.groupby(['docid', 'fROI']).agg({
            'BOLD': ['mean', 'std', 'min', 'max'],
            'totsurp_hrf': 'mean',
            'fwprob5surp_hrf': 'mean',
            'gpt2_surp_hrf': 'mean'
        }).round(4)

        # Flatten column names
        story_stats.columns = ['_'.join(col).strip() for col in story_stats.columns]
        story_stats = story_stats.reset_index()

        # Calculate BOLD range
        story_stats['BOLD_range'] = story_stats['BOLD_max'] - story_stats['BOLD_min']

        # Find most/least variable regions per story
        print("Most Variable Brain Regions (by BOLD range) per Story:")
        for story in df['docid'].unique():
            story_data = story_stats[story_stats['docid'] == story]
            most_variable = story_data.loc[story_data['BOLD_range'].idxmax()]
            least_variable = story_data.loc[story_data['BOLD_range'].idxmin()]

            print(f"\n  {story}:")
            print(f"    Most variable:  {most_variable['fROI']} (range: {most_variable['BOLD_range']:.4f})")
            print(f"    Least variable: {least_variable['fROI']} (range: {least_variable['BOLD_range']:.4f})")

        return story_stats

    def create_summary_heatmap(stats_df, save_plots=True):
        mean_pivot  = stats_df.pivot(index='fROI', columns='docid', values='Mean')
        range_pivot = stats_df.pivot(index='fROI', columns='docid', values='Range')

        # Mean heatmap
        fig, ax = plt.subplots(figsize=(12.5, 9))
        sns.heatmap(mean_pivot, annot=True, fmt='.3f', cmap='RdBu_r',
                    center=0, ax=ax, cbar_kws={'label': 'Mean BOLD'})
        ax.set_title('Mean BOLD Response by Region and Story')
        ax.set_ylabel('Brain Region (fROI)')
        ax.set_xlabel('Story')
        plt.xticks(rotation=45, ha='right')
        _savefig(fig, "bold_mean_heatmap.png", subdir="summary_heatmaps")
        plt.close(fig)

        # Range heatmap
        fig, ax = plt.subplots(figsize=(12.5, 9))
        sns.heatmap(range_pivot, annot=True, fmt='.3f', cmap='viridis',
                    ax=ax, cbar_kws={'label': 'BOLD Range'})
        ax.set_title('BOLD Response Range by Region and Story')
        ax.set_ylabel('Brain Region (fROI)')
        ax.set_xlabel('Story')
        plt.xticks(rotation=45, ha='right')
        _savefig(fig, "bold_range_heatmap.png", subdir="summary_heatmaps")
        plt.close(fig)

    def correlation_analysis(df):
        print("Correlation Analysis: BOLD vs Surprisal Measures")
        print("="*80)

        # Calculate correlations for each story and brain region
        corr_results = []

        for story in df['docid'].unique():
            for froi in df['fROI'].unique():
                subset = df[(df['docid'] == story) & (df['fROI'] == froi)]

                if len(subset) > 5:  # Need sufficient data points
                    # Calculate correlations
                    corr_pcfg = subset['BOLD'].corr(subset['totsurp_hrf'])
                    corr_5gram = subset['BOLD'].corr(subset['fwprob5surp_hrf'])
                    corr_gpt2 = subset['BOLD'].corr(subset['gpt2_surp_hrf'])

                    corr_results.append({
                        'story': story,
                        'fROI': froi,
                        'corr_pcfg': corr_pcfg,
                        'corr_5gram': corr_5gram,
                        'corr_gpt2': corr_gpt2,
                        'n_obs': len(subset)
                    })

        corr_df = pd.DataFrame(corr_results)

        # Summary statistics
        print("Average correlations across all story-region combinations:")
        print(f"  PCFG Surprisal:  {corr_df['corr_pcfg'].mean():.4f} ± {corr_df['corr_pcfg'].std():.4f}")
        print(f"  5-gram Surprisal: {corr_df['corr_5gram'].mean():.4f} ± {corr_df['corr_5gram'].std():.4f}")
        print(f"  GPT-2 Surprisal:  {corr_df['corr_gpt2'].mean():.4f} ± {corr_df['corr_gpt2'].std():.4f}")

        return corr_df

    def run_complete_analysis(df):

        print(" Starting Natural Stories fMRI Story Progression Analysis")
        print("="*80)

        unique_stories, unique_frois = analyze_story_progression(df)

        stats_df = calculate_bold_statistics(df)
        print("\n BOLD Statistics Summary (first 10 rows):")
        print(stats_df.head(10))

        # 3. Analyze cross-story patterns
        story_patterns = analyze_story_patterns(df)

        # 4. Create progression plots for each story
        print(f"\n Creating progression plots for {len(unique_stories)} stories...")
        story_data_dict = {}

        for story in unique_stories:
            print(f"   Processing {story}...")
            story_avg = create_story_progression_plots(df, story, save_plots=True)
            story_data_dict[story] = story_avg

        # 5. Create summary heatmaps
        print("\n Creating summary heatmaps...")
        create_summary_heatmap(stats_df)

        # 6. Correlation analysis
        corr_df = correlation_analysis(df)

        return {
            'stats_df': stats_df,
            'story_patterns': story_patterns,
            'correlations': corr_df,
            'story_data': story_data_dict
        }

    results = run_complete_analysis(merged_df_with_vectors)

if __name__ == "__main__":
    main()