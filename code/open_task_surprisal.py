import pandas as pd
import os
import re
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings("ignore")
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr

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
    # Paths to your prepared files
    PRED_PATH = "data/pred_use.csv"
    RESP_PATH = "data/resp_use.csv"
    GPT2SURP_PATH = "data/gpt2_surprisal_results.csv"

    # Load directly
    pred_use = pd.read_csv(PRED_PATH)
    resp_use = pd.read_csv(RESP_PATH)

    SAVE_DIR = "results/surprisal"

    def _ensure_dir(d):
        os.makedirs(d, exist_ok=True)

    def _slug(s: str) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(s)).strip("_")

    def _savefig(fig, filename, save_dir=SAVE_DIR, dpi=300):
        _ensure_dir(save_dir)
        path = os.path.join(save_dir, _slug(filename))
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        print(f" saved: {path}")

    def _rp_box(ax, xs=None, ys=None, r=None, p=None,
            loc=(0.02, 0.92), fontsize=16, pad=0.6, alpha=0.9):
        if r is None or p is None:
            if xs is None or ys is None:
                return np.nan, np.nan
            xs = np.asarray(xs, float); ys = np.asarray(ys, float)
            m = np.isfinite(xs) & np.isfinite(ys)
            if m.sum() < 2:
                return np.nan, np.nan
            r, p = pearsonr(xs[m], ys[m])

        ax.text(loc[0], loc[1], f"r={r:.3f}, p={p:.3g}",
                transform=ax.transAxes, fontsize=fontsize,
                bbox=dict(boxstyle=f"round,pad={pad}",
                        facecolor="yellow", edgecolor="black", alpha=alpha),
                zorder=5)
        return r, p

    # Ensure key columns are strings
    for c in ['docid', 'subject', 'fROI']:
        if c in pred_use.columns:
            pred_use[c] = pred_use[c].astype(str)
        if c in resp_use.columns:
            resp_use[c] = resp_use[c].astype(str)

    print("Predictors:", pred_use.shape)
    print("Responses:", resp_use.shape)


    # extract the stories for future use
    def extract_stories_from_df(df, doc_col="docid", time_col="time", word_col="word"):
        required = {doc_col, time_col, word_col}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        df = df.sort_values([doc_col, time_col]).copy()

        stories = {}
        seen = set()

        for docid, g in df.groupby(doc_col, sort=False):
            if docid in seen:
                continue
            seen.add(docid)

            # Deduplicate rows that share the same time (one token per time)
            g_unique = g.drop_duplicates(subset=[time_col], keep="first")

            # Build the story
            words = g_unique[word_col].astype(str).tolist()
            stories[docid] = " ".join(words)

        return stories

    stories = extract_stories_from_df(pred_use)

    for key, value in stories.items():
        print(f"{key}: {value}")

    # get gpt2 surprisal values

    # add thegpt 2 surprisal column to the working file
    df_gpt2_surprisal = pd.read_csv(GPT2SURP_PATH)

    gpt_small = (df_gpt2_surprisal[["docid", "word", "gpt2_surprisal"]]
                .drop_duplicates(["docid", "word"]))
    gpt_small["docid"] = gpt_small["docid"].astype(str).str.strip()
    gpt_small["word"]  = gpt_small["word"].astype(str).str.strip()

    pred_use["docid"] = pred_use["docid"].astype(str).str.strip()
    pred_use["word"]  = pred_use["word"].astype(str).str.strip()

    if "gpt2_surp" not in pred_use.columns:
        pred_use["gpt2_surp"] = pd.NA

    # --- In-place updates per docid (small working sets) ---
    for d in gpt_small["docid"].unique():
        m = (pred_use["docid"] == d)
        if not m.any():
            continue
        dmap = dict(zip(
            gpt_small.loc[gpt_small["docid"] == d, "word"],
            gpt_small.loc[gpt_small["docid"] == d, "gpt2_surprisal"]
        ))
        pred_use.loc[m, "gpt2_surp"] = pred_use.loc[m, "word"].map(dmap)

    print("Done. Nulls (no match):", pred_use["gpt2_surp"].isna().sum())

    # perform HRF alignmnet
    net_map = (
        resp_use[['subject','docid','fROI','network']]
        .drop_duplicates()
        .groupby(['subject','docid','fROI'], as_index=False)['network']
        .agg(lambda s: s.mode().iat[0])
    )
    pred_use = pred_use.merge(net_map, on=['subject','docid','fROI'], how='left')

    if pred_use['network'].isna().any():
        mask_nan = pred_use['network'].isna()
        inferred = np.where(
            pred_use.loc[mask_nan, 'fROI'].str.startswith(('Lang','LANG','lang')), 1,
            np.where(pred_use.loc[mask_nan, 'fROI'].str.startswith(('MD','Md','md')), 0, np.nan)
        )
        pred_use.loc[mask_nan, 'network'] = inferred

    missing_network = pred_use['network'].isna().sum()
    if missing_network:
        print(f"{missing_network} predictor rows still missing 'network'; they will be dropped.")
        pred_use = pred_use[pred_use['network'].notna()].copy()

    # HRF TR-sampled
    def spm_hrf(tr, time_length=32.0):
        t = np.arange(0, time_length + 1e-9, tr)
        hrf = stats.gamma.pdf(t, 6) - stats.gamma.pdf(t, 16) / 6
        s = hrf.sum()
        hrf = hrf / s if s != 0 else hrf
        return hrf

    TR = 2.0
    hrf = spm_hrf(TR)

    #  convolution per (subject, docid, network, fROI)
    def convolve_and_resample(pred_df, colname, tr=2.0, agg="sum", join_with=" "):
        """
        Convolve a word-timed predictor to the TR grid per (subject, docid, network, fROI).
        - agg='sum' (typical for surprisal) or 'mean' (normalize by word count per TR)
        Emits TR rows with integer 'sampleid' (TR index) so merges are exact.
        """
        required = {'subject','docid','network','fROI','time', colname}
        missing = required - set(pred_df.columns)
        if missing:
            raise ValueError(f"Missing columns for convolution: {missing}")

        rows = []
        for (subj, doc, net, roi), g in pred_df.groupby(['subject','docid','network','fROI']):
            g = g.sort_values('time').copy()
            if g.empty:
                continue

            tr_idx = np.floor(g['time'] / tr).astype(int)
            n_tr = int(tr_idx.max()) + 1 if len(tr_idx) else 0
            if n_tr <= 0:
                continue

            vals = g[colname].astype(float).fillna(0).to_numpy()
            agg_bins = np.zeros(n_tr, dtype=float)
            counts   = np.zeros(n_tr, dtype=int)
            np.add.at(agg_bins, tr_idx, vals)
            np.add.at(counts,   tr_idx, 1)
            if agg == "mean":
                nz = counts > 0
                agg_bins[nz] = agg_bins[nz] / counts[nz]

            conv = np.convolve(agg_bins, hrf, mode='full')[:n_tr]

            if 'word' in g.columns:
                word_bins = [[] for _ in range(n_tr)]
                for i, w in zip(tr_idx, g['word'].astype(str).to_numpy()):
                    word_bins[i].append(w)
                words_per_tr = [join_with.join(ws) if ws else np.nan for ws in word_bins]
            else:
                words_per_tr = [np.nan] * n_tr

            times = np.arange(n_tr, dtype=float) * tr
            for t_idx in range(n_tr):
                rows.append((subj, doc, int(net), roi, t_idx, times[t_idx], conv[t_idx], words_per_tr[t_idx]))

        return pd.DataFrame(
            rows,
            columns=['subject','docid','network','fROI','sampleid','time', f'{colname}_hrf','words']
        )

    # build HRF-convolved predictors
    pcfg_df   = convolve_and_resample(pred_use, 'totsurp',     tr=TR, agg="sum")
    ngram_df  = convolve_and_resample(pred_use, 'fwprob5surp', tr=TR, agg="sum")

    if 'gpt2_surp' in pred_use.columns:
        gpt2_df = convolve_and_resample(pred_use, 'gpt2_surp', tr=TR, agg="sum")
    else:
        gpt2_df = None
        print("'gpt2_surp' not found in pred_use; skipping GPT-2 HRF.")

    # merge them on exact TR keys
    pred_hrf = pcfg_df.merge(
        ngram_df[['subject','docid','network','fROI','sampleid','fwprob5surp_hrf']],
        on=['subject','docid','network','fROI','sampleid'],
        how='inner',
        validate='one_to_one'
    )

    if gpt2_df is not None:
        pred_hrf = pred_hrf.merge(
            gpt2_df[['subject','docid','network','fROI','sampleid','gpt2_surp_hrf']],
            on=['subject','docid','network','fROI','sampleid'],
            how='inner',
            validate='one_to_one'
        )

    print("pred_hrf shape:", pred_hrf.shape)
    print(pred_hrf.head())

    # merge the predictors with the BOLD files for conviniance
    merged = resp_use.merge(pred_hrf, on=['subject','docid','fROI','time','network'])
    print("Merged shape:", merged.shape)

    merged.head(10)

    COL_PCFG  = "blue"    # PCFG = BLUE
    COL_G5    = "orange"  # 5-gram = ORANGE
    COL_GPT2  = "green"   # GPT-2 = GREEN

    REQ_SURP_COLS = {
        'pcfg' : ('totsurp_hrf',     'totsurp_hrf_bin',     'totsurp_hrf_surprisal_level', 'totsurp_hrf_is_peak'),
        'g5'   : ('fwprob5surp_hrf', 'fwprob5surp_hrf_bin', 'fwprob5surp_hrf_surprisal_level', 'fwprob5surp_hrf_is_peak'),
        'gpt2' : ('gpt2_surp_hrf',   'gpt2_surp_hrf_bin',   'gpt2_surp_hrf_surprisal_level',   'gpt2_surp_hrf_is_peak'),
    }

    def _ensure_cols(df, cols):
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise KeyError(f"Missing required columns: {missing}")

    def _rename_complexity_cols_to_surprisal(df):
        """Rename any '*_complexity' columns to '*_surprisal_level' for consistency."""
        rename_map = {c: c.replace("_complexity", "_surprisal_level")
                    for c in df.columns if c.endswith("_complexity")}
        return df.rename(columns=rename_map) if rename_map else df

    def add_transformations(df):
        df = df.copy()
        have = [c for c in ['totsurp_hrf','fwprob5surp_hrf','gpt2_surp_hrf'] if c in df]
        if not have:
            raise KeyError("Need at least one of: totsurp_hrf, fwprob5surp_hrf, gpt2_surp_hrf.")

        for c in have:
            x = df[c].clip(lower=0)
            df[f"{c}_log"]  = np.log(x + 1.0)
            df[f"{c}_sqrt"] = np.sqrt(x)
            df[f"{c}_std"]  = StandardScaler().fit_transform(df[[c]]).ravel()

        print("Added transformations for:", ", ".join(have))
        return df

    def create_surprisal_bins(data, surprisal_col, n_bins=10, method="percentile"):
        data = data.copy()
        col_bin = f"{surprisal_col}_bin"

        if method == "percentile":
            try:
                data[col_bin] = pd.qcut(data[surprisal_col], q=n_bins, labels=False, duplicates="drop")
                _, edges = pd.qcut(data[surprisal_col], q=n_bins, retbins=True, duplicates="drop")
            except ValueError:
                method = "equal"

        if method == "equal":
            percentiles = np.linspace(0, 100, n_bins + 1)
            edges = np.unique(np.percentile(data[surprisal_col], percentiles))
            data[col_bin] = pd.cut(data[surprisal_col], bins=edges, labels=False, include_lowest=True)

        return data, edges

    def create_surprisal_thresholds(data, surprisal_col):
        data = data.copy()
        th = {
            'low_threshold'    : data[surprisal_col].quantile(0.33),
            'medium_threshold' : data[surprisal_col].quantile(0.66),
            'high_threshold'   : data[surprisal_col].quantile(0.90),
            'peak_threshold'   : data[surprisal_col].quantile(0.95),
            'extreme_threshold': data[surprisal_col].quantile(0.99),
        }

        conds = [
            data[surprisal_col] < th['low_threshold'],
            (data[surprisal_col] >= th['low_threshold']) & (data[surprisal_col] < th['medium_threshold']),
            (data[surprisal_col] >= th['medium_threshold']) & (data[surprisal_col] < th['high_threshold']),
            (data[surprisal_col] >= th['high_threshold']) & (data[surprisal_col] < th['peak_threshold']),
            data[surprisal_col] >= th['peak_threshold'],
        ]
        labels = ['low','medium','high','very_high','peak']

        lvl_col = f"{surprisal_col}_surprisal_level"
        data[lvl_col] = np.select(conds, labels, default='unknown')
        data[f"{surprisal_col}_is_high"]    = data[surprisal_col] >= th['high_threshold']
        data[f"{surprisal_col}_is_peak"]    = data[surprisal_col] >= th['peak_threshold']
        data[f"{surprisal_col}_is_extreme"] = data[surprisal_col] >= th['extreme_threshold']
        return data, th

    def analyze_surprisal_distributions(merged_df, n_bins=10):
        print("=== SURPRISAL DISTRIBUTION ANALYSIS ===")
        _ensure_cols(merged_df, ['totsurp_hrf','fwprob5surp_hrf','gpt2_surp_hrf'])

        df = add_transformations(merged_df)
        surp_cols = ['totsurp_hrf','fwprob5surp_hrf','gpt2_surp_hrf']

        print("\nBASIC STATISTICS:")
        for c in surp_cols:
            print(f"\n{c}:")
            print(f"  Range: {df[c].min():.3f} .. {df[c].max():.3f}")
            print(f"  Mean±SD: {df[c].mean():.3f} ± {df[c].std():.3f}")
            print(f"  Median: {df[c].median():.3f}")
            print(f"  95th: {df[c].quantile(0.95):.3f} | 99th: {df[c].quantile(0.99):.3f}")

        print("\nCORRELATIONS (Pearson r):")
        for a,b in [('totsurp_hrf','fwprob5surp_hrf'),
                    ('totsurp_hrf','gpt2_surp_hrf'),
                    ('fwprob5surp_hrf','gpt2_surp_hrf')]:
            r = df[a].corr(df[b])
            print(f"  {a} vs {b}: r = {r:.3f}")

        print("\nCREATING SURPRISAL BINS & THRESHOLDS:")
        bin_edges, thresholds = {}, {}
        for c in surp_cols:
            df, edges = create_surprisal_bins(df, c, n_bins=n_bins)
            bin_edges[c] = edges
            df, th = create_surprisal_thresholds(df, c)
            thresholds[c] = th
            print(f"- {c}: {len(edges)-1} bins")

        return df, {'bin_edges': bin_edges, 'thresholds': thresholds}


    def plot_surprisal_overview(df, save_dir=SAVE_DIR):
        _ensure_cols(df, ['totsurp_hrf','fwprob5surp_hrf','gpt2_surp_hrf'])

        # --- Single-figure helpers ---
        def _hist_one(col, title, color, fname):
            fig, ax = plt.subplots(figsize=(8,6))
            ax.hist(df[col].dropna(), bins=60, alpha=0.95, color=color, edgecolor="none")
            ax.set_title(title)
            ax.set_xlabel("Surprisal")
            ax.set_ylabel("Frequency")
            ax.grid(True, axis="y", alpha=0.25)
            _savefig(fig, fname, save_dir)
            plt.close(fig)

        def _scatter_one(xcol, ycol, xt, yt, fname):
            fig, ax = plt.subplots(figsize=(8,6))
            x = df[xcol].values
            y = df[ycol].values
            m = np.isfinite(x) & np.isfinite(y)
            ax.scatter(x[m], y[m], s=6, alpha=0.25, color="black")
            try:
                r, p = pearsonr(x[m], y[m])
                ax.text(0.03, 0.95, f"r={r:.3f}, p={p:.3g}", transform=ax.transAxes,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", edgecolor="black", alpha=0.85))
            except Exception:
                pass
            ax.set_title(f"{xt} vs {yt} Surprisal")
            ax.set_xlabel(f"{xt} Surprisal")
            ax.set_ylabel(f"{yt} Surprisal")
            ax.grid(True, alpha=0.3)
            _savefig(fig, fname, save_dir)
            plt.close(fig)

        def _bin_count_one(bin_col, title, color, fname):
            if bin_col not in df.columns:
                return
            vc = df[bin_col].value_counts().sort_index()
            fig, ax = plt.subplots(figsize=(8,6))
            ax.bar(vc.index.astype(float), vc.values, alpha=0.95, color=color)
            ax.set_title(title)
            ax.set_xlabel("Surprisal Bin")
            ax.set_ylabel("Count")
            ax.grid(True, axis="y", alpha=0.3)
            _savefig(fig, fname, save_dir)
            plt.close(fig)

        # --- Distributions (3 separate figures) ---
        _hist_one('totsurp_hrf',    'PCFG Surprisal — Distribution',   COL_PCFG,  "dist_pcfg.png")
        _hist_one('fwprob5surp_hrf','5-gram Surprisal — Distribution', COL_G5,    "dist_5gram.png")
        _hist_one('gpt2_surp_hrf',  'GPT-2 Surprisal — Distribution',  COL_GPT2,  "dist_gpt2.png")

        # --- Pairwise scatters (3 separate figures) ---
        _scatter_one('totsurp_hrf','fwprob5surp_hrf','PCFG','5-gram', "scatter_pcfg_vs_5gram.png")
        _scatter_one('totsurp_hrf','gpt2_surp_hrf', 'PCFG','GPT-2',   "scatter_pcfg_vs_gpt2.png")
        _scatter_one('fwprob5surp_hrf','gpt2_surp_hrf','5-gram','GPT-2', "scatter_5gram_vs_gpt2.png")

        # --- Bin counts (3 separate figures) ---
        _bin_count_one('totsurp_hrf_bin',     'PCFG Surprisal — Bin Counts',   COL_PCFG, "bin_counts_pcfg.png")
        _bin_count_one('fwprob5surp_hrf_bin', '5-gram Surprisal — Bin Counts', COL_G5,   "bin_counts_5gram.png")
        _bin_count_one('gpt2_surp_hrf_bin',   'GPT-2 Surprisal — Bin Counts',  COL_GPT2, "bin_counts_gpt2.png")

    def summarize_bold_by_surprisal_bins(merged_df):
        merged_df = _rename_complexity_cols_to_surprisal(merged_df.copy())
        _ensure_cols(merged_df, ['fROI','network','BOLD'])

        for key in ['pcfg','g5','gpt2']:
            val, bin_col, _, _ = REQ_SURP_COLS[key]
            _ensure_cols(merged_df, [val, bin_col])

        pcfg_bin = REQ_SURP_COLS['pcfg'][1]
        g5_bin   = REQ_SURP_COLS['g5'][1]
        gpt2_bin = REQ_SURP_COLS['gpt2'][1]

        def _agg(bin_name):
            g = merged_df.groupby(['fROI','network', bin_name]).agg(
                BOLD_mean=('BOLD','mean'),
                BOLD_std=('BOLD','std'),
                BOLD_count=('BOLD','count'),
            ).reset_index().round(3)
            return g

        return _agg(pcfg_bin), _agg(g5_bin), _agg(gpt2_bin)

    def plot_bold_by_surprisal_bins(merged_df, network_filter=None, max_plots=6, save_dir=SAVE_DIR):
        merged_df = _rename_complexity_cols_to_surprisal(merged_df.copy())

        if network_filter is not None:
            data = merged_df[merged_df['network'] == network_filter].copy()
            title_net = f"Network {network_filter}"
        else:
            data = merged_df.copy()
            title_net = "All Networks"

        frois = data['fROI'].dropna().unique()[:max_plots]
        n_plots = len(frois)
        n_cols = min(3, n_plots) if n_plots else 1
        n_rows = (n_plots + n_cols - 1) // n_cols if n_plots else 1

        def _panel(metric_name, bin_col, color, marker='o'):
            # bigger canvas for readability
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(6.5*n_cols, 5.2*n_rows))
            axes = np.atleast_1d(axes).flatten()

            for i, froi in enumerate(frois):
                ax = axes[i]
                fd = data[data['fROI'] == froi]
                if len(fd)==0 or bin_col not in fd:
                    ax.set_visible(False); continue

                grp = fd.groupby(bin_col)['BOLD'].agg(['mean','std','count'])
                if len(grp)==0:
                    ax.set_visible(False); continue

                se = grp['std'] / np.sqrt(grp['count'])
                xs = grp.index.values.astype(float)
                ys = grp['mean'].values

                ax.errorbar(xs, ys, yerr=se.values,
                            marker=marker, capsize=4, linewidth=2.5, markersize=8,
                            color=color, zorder=3)
                ax.set_title(f'{froi} — {metric_name} Surprisal')
                ax.set_xlabel(f'{metric_name} Surprisal Bin')
                ax.set_ylabel('BOLD')
                ax.grid(True, alpha=0.3, zorder=0)

                # add simple linear trend (optional visual guide)
                if len(xs) > 1 and np.nanstd(ys) > 0:
                    z = np.polyfit(xs, ys, 1); pfit = np.poly1d(z)
                    ax.plot(xs, pfit(xs), "--", alpha=0.9, color='red', zorder=4)
                if len(xs) > 1 and np.nanstd(ys) > 0:
                    z = np.polyfit(xs, ys, 1); pfit = np.poly1d(z)
                    ax.plot(xs, pfit(xs), "--", alpha=0.9, color='red', zorder=4)
                    # add the yellow r/p badge
                    _rp_box(ax, xs=xs, ys=ys, loc=(0.04, 0.92), fontsize=16)
            
            # hide unused axes
            for k in range(n_plots, len(axes)):
                axes[k].set_visible(False)

            plt.suptitle(f'{metric_name} Surprisal Effects by fROI — {title_net}', y=1.02)
            plt.tight_layout()

            fname = f"bold_by_bins_{metric_name.replace('-','_')}_{_slug(title_net)}.png"
            _savefig(fig, fname, save_dir)
            plt.close(fig)

        pcfg_bin = REQ_SURP_COLS['pcfg'][1]
        g5_bin   = REQ_SURP_COLS['g5'][1]
        gpt2_bin = REQ_SURP_COLS['gpt2'][1]

        _panel("PCFG",   pcfg_bin,  color=COL_PCFG,  marker='o')  # blue
        _panel("5-gram", g5_bin,    color=COL_G5,    marker='s')  # orange
        _panel("GPT-2",  gpt2_bin,  color=COL_GPT2,  marker='^')  # green

    def run_full_surprisal_pipeline(merged_df, n_bins=10, per_network_max_frois=6, save_dir=SAVE_DIR):
        print(f"Input rows: {len(merged_df):,}")
        merged_df = _rename_complexity_cols_to_surprisal(merged_df.copy())
        _ensure_cols(merged_df, ['totsurp_hrf','fwprob5surp_hrf','gpt2_surp_hrf'])

        # Build bins/thresholds if missing
        for key in ['pcfg','g5','gpt2']:
            val, bin_col, lvl_col, peak_col = REQ_SURP_COLS[key]
            if bin_col not in merged_df.columns:
                merged_df, _ = create_surprisal_bins(merged_df, val, n_bins=n_bins)
            if lvl_col not in merged_df.columns or peak_col not in merged_df.columns:
                merged_df, _ = create_surprisal_thresholds(merged_df, val)

        analyzed_df, dist_info = analyze_surprisal_distributions(merged_df, n_bins=n_bins)

        print("\nCreating overview visualizations …")
        plot_surprisal_overview(analyzed_df, save_dir=save_dir)

        out = {'distribution_info': dist_info}

        if all(c in analyzed_df.columns for c in ['BOLD','fROI','network']):
            print("\nSummarizing BOLD across surprisal bins …")
            pcfg_df, g5_df, gpt2_df = summarize_bold_by_surprisal_bins(analyzed_df)
            out.update({'pcfg_summary': pcfg_df, 'fivegram_summary': g5_df, 'gpt2_summary': gpt2_df})

            nets = list(analyzed_df['network'].dropna().unique())
            for net in nets:
                print(f"\n Plotting per-fROI grids (network={net}) …")
                plot_bold_by_surprisal_bins(analyzed_df, network_filter=net,
                                            max_plots=per_network_max_frois, save_dir=save_dir)
        else:
            print("\n BOLD/fROI/network not all present — neural plots skipped.")

        print("\n Surprisal pipeline complete!")
        return analyzed_df, out

    analyzed_df, outputs = run_full_surprisal_pipeline(merged, n_bins=10, per_network_max_frois=6)

    def plot_all_frois_one_graph(merged_df, metric='pcfg', max_frois=None, save_dir=SAVE_DIR):
        df = _rename_complexity_cols_to_surprisal(merged_df.copy())
        if metric not in REQ_SURP_COLS:
            raise ValueError(f"metric must be one of {list(REQ_SURP_COLS.keys())}")

        value_col, bin_col, _, _ = REQ_SURP_COLS[metric]
        _ensure_cols(df, ['BOLD', 'fROI', value_col])

        if bin_col not in df.columns:
            df, _ = create_surprisal_bins(df, value_col, n_bins=10)

        frois = df['fROI'].dropna().unique()
        if max_frois is not None:
            frois = frois[:max_frois]

        cmap = plt.cm.get_cmap('tab20', max(len(frois), 1))

        fig = plt.figure(figsize=(12, 8))
        for i, froi in enumerate(frois):
            sub = df[df['fROI'] == froi]
            if sub.empty: continue
            grp = sub.groupby(bin_col)['BOLD'].agg(['mean', 'std', 'count'])
            if grp.empty: continue
            se = grp['std'] / np.sqrt(grp['count'])
            xs = grp.index.values.astype(float)
            ys = grp['mean'].values
            plt.errorbar(xs, ys, yerr=se.values,
                        marker='o', capsize=4, linewidth=2.5, markersize=7,
                        color=cmap(i), alpha=0.95, label=str(froi))

        metric_name = {'pcfg': 'PCFG', 'g5': '5-gram', 'gpt2': 'GPT-2'}[metric]
        plt.title(f'{metric_name} Surprisal — All fROIs in One Plot')
        plt.xlabel(f'{metric_name} Surprisal Bin')
        plt.ylabel('BOLD')
        plt.grid(True, alpha=0.3)
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', ncol=1, title="fROI")
        plt.tight_layout()
        plt.subplots_adjust(right=0.78)

        _savefig(fig, f"all_fROIs_{metric_name.replace('-','_')}.png", save_dir)
        plt.close(fig)

    plot_all_frois_one_graph(analyzed_df, metric='pcfg', max_frois=12)
    plot_all_frois_one_graph(analyzed_df, metric='g5', max_frois=12)
    plot_all_frois_one_graph(analyzed_df, metric='gpt2', max_frois=12)

if __name__ == "__main__":
    main()