 # Functional Specialization and Non-Linear Processing in the Human Language Network

This project investigates how different regions within the human language network respond to various measures of linguistic complexity during naturalistic story comprehension. We extend beyond previous network-level analyses by Shain et al. (2020) to examine regional specialization patterns and non-linear complexity processing mechanisms.

## Overview

This repository contains code and analysis for a computational psycholinguistics study that challenges assumptions about uniform complexity processing in the language network. Our research reveals functional specialization across brain regions and uncovers a surprising semantic processing paradox.  

The provided code is focused on the **analysis** itself. If you would like to see how the **GPT-2 surprisal values** or the **GloVe embeddings** were derived, please refer to the accompanying Jupyter notebooks, where the data generation process is fully documented.

### Key Research Questions

1. **Regional Specialization**: Do different areas within the language network exhibit specialized response patterns to different types of prediction difficulty (syntactic vs. lexical vs. contextual)?

2. **Non-Linear Processing**: Does linguistic complexity processing exhibit capacity constraints and non-linear patterns rather than uniform linear scaling?

3. **Semantic Complexity Paradox**: How does story-specific semantic complexity influence language network activation during naturalistic comprehension?

### Major Findings

- **Functional Specialization**: Frontal regions (LangMFG, LangIFG) showed strongest sensitivity to syntactic complexity, temporal regions (LangAntTemp, LangPostTemp) to lexical predictability, while angular gyrus (LangAngG) responded most to GPT-2 contextual complexity.

- **Non-Linear Complexity Processing**: Systematic deviations from linearity across all regions and measures, contradicting linear assumptions of previous approaches.

- **Semantic Processing Paradox**: Stories with highest semantic diversity produced language network deactivation, while coherent narratives elicited maximal activation, suggesting adaptive processing mechanisms.

## Installation

### Prerequisites
- Python 3.11+
- Conda or Miniconda
- Git

### Setup Instructions

1. **Clone the repository**:
   ```bash
   git clone https://github.com/LizMelamed/lang-network-specialization.git
   cd functional-specialization-language-network
   ```

2. **Create the conda environment**:
   ```bash
   conda env create -f environment.yml
   ```

3. **Activate the environment**:
   ```bash
   conda activate cogsci-proj
   ```

4. **Download the Data Folder**:
   ```bash
   python code/data_download.py
   ```

## Usage

### Quick Start - Run Complete Analysis Pipeline

**Open-ended analysis (primary contributions)**:
```bash
# Regional specialization analysis
python code/open_task_surprisal.py

# Semantic complexity analysis  
python code/open_task_semantic_analysis.py
```

**Structured replication tasks**:
```bash
# Open Jupyter notebook for structured tasks
jupyter notebook notebooks/Closed_Task1.ipynb
jupyter notebook notebooks/Closed_Task2.ipynb
jupyter notebook notebooks/Closed_Task3.ipynb
```

**Interactive exploration**:
```bash
jupyter notebook notebooks/Open_Task_Semantic_Analysis.ipynb
jupyter notebook notebooks/Open_Task_Surprisal.ipynb
```

### Analysis Components

1. **Complexity Binning Analysis**: Creates percentile-based surprisal bins to identify non-linear response patterns
2. **Regional Specialization**: Analyzes differential responses across six language network regions
3. **Multi-Model Comparison**: Compares PCFG, 5-gram, and GPT-2 surprisal measures
4. **Semantic Characterization**: Uses GloVe embeddings to quantify story semantic properties
5. **HRF Convolution**: Properly accounts for hemodynamic response delays

## Repository Structure

```
├── code/                           # Analysis scripts
│   ├── open_task_surprisal.py             # Regional specialization analysis
│   ├── open_task_semantic_analysis.py     # Semantic complexity analysis
│   └── utils/                             # Helper functions
├── data/                           # Input datasets
│   ├── LANG_MD_ts_X.csv                   # Combined predictors (PCFG, 5-gram surprisal)
│   ├── LANG_MD_y_train.csv                # Combined BOLD responses (LANG + MD networks)  
│   ├── pred_use.csv                       # Preprocessed predictor variables
│   ├── resp_use.csv                       # Preprocessed brain response data
│   ├── merged.csv                         # Combined analysis-ready dataset
│   ├── gpt2_surprisal_results.csv         # GPT-2 surprisal computations
│   └── merged_df_with_vectors.parquet     # Generated dataset (appears after running data_download.py)
├── notebooks/                      # Jupyter notebooks  
│   ├── Closed_Task1.ipynb                 # Structured Task 1: FastText vs GloVe decoding
│   ├── Closed_Task2.ipynb                 # Structured Task 2: Static vs contextual embeddings
│   ├── Closed_Task3.ipynb                 # Structured Task 3: Neural encoding models
│   ├── Open_Task_Semantic_Analysis.ipynb  # Semantic complexity analysis
│   └── Open_Task_Surprisal.ipynb          # Regional specialization analysis
├── results/                        # Auto-generated outputs
│   ├── all_fROIs_GPT_2.png               # GPT-2 regional responses
│   ├── all_fROIs_PCFG.png                # PCFG regional responses  
│   ├── all_fROIs_5_gram.png              # 5-gram regional responses
│   ├── bold_by_bins_*.png                # Complexity binning visualizations
│   ├── semantic_*.png                    # Semantic analysis plots
│   ├── story_progression_*.png           # Story-specific response patterns
│   └── surprisal_overview.png            # Summary visualizations
├── environment.yml                 # Conda environment specification
└── README.md                      # This file

```

## Data Description

### Primary Datasets

**Natural Stories fMRI Corpus (Shain et al., 2020)**:
- **Participants**: 78 native English speakers
- **Stories**: 8 naturalistic narratives (990-1099 words each)  
- **Brain Networks**: Language (LANG) and Multiple Demand (MD) networks
- **Regions**: 6 language network regions (LangAngG, LangAntTemp, LangIFG, LangIFGorb, LangMFG, LangPostTemp)
- **Predictors**: PCFG surprisal, 5-gram surprisal, word timing information
- **Responses**: BOLD signals temporally aligned to word-level predictors

**Pereira et al. (2018) Structured Tasks**:
- **Participants**: 16 fluent English speakers  
- **Stimuli**: Individual words (180), sentence passages (384), naturalistic sentences (243)
- **Purpose**: Baseline validation for neural decoding approaches

### Key Variables

**Predictor Variables**:
- `totsurp`: PCFG surprisal values (syntactic complexity)
- `fwprob5surp`: 5-gram surprisal values (lexical predictability)  
- `gpt2_surprisal`: GPT-2 surprisal values (contextual complexity)
- `word`: Word tokens with precise timing
- `docid`, `subject`, `time`, `fROI`: Experimental identifiers

**Response Variables**:
- `BOLD`: Brain activation values (HRF-convolved)
- `network`: Brain network labels (LANG, MD)
- `fROI`: Functional region of interest within networks

**Semantic Variables**:
- `semantic_coherence`: Story-level semantic consistency (GloVe-based)
- `semantic_diversity`: Story-level semantic complexity (1 - coherence)
- `vocabulary_size`: Unique words per story

## Methodology

### Analysis Pipeline

1. **Data Preprocessing**:
   - Temporal alignment of word-level predictors with BOLD responses
   - Hemodynamic Response Function (HRF) convolution
   - Z-score normalization across all surprisal measures

2. **Complexity Binning**:
   - Divide each surprisal measure into 10 equal-count percentile bins
   - Ensure balanced sample sizes (~9,500 observations per bin)
   - Capture full range from predictable to highly complex constructions

3. **Regional Specialization Analysis**:
   - Linear correlations between surprisal measures and BOLD responses
   - Region-specific response profiles across complexity bins
   - Cross-measure comparisons (PCFG vs 5-gram vs GPT-2)

4. **Non-linearity Testing**:
   - Systematic deviations from linear fits
   - Identification of peaked responses at intermediate complexity
   - Statistical validation of capacity-limited processing

5. **Semantic Analysis**:
   - GloVe embedding-based story characterization
   - Semantic coherence and diversity metrics
   - Story-specific brain response patterns

### Novel Methodological Contributions

- **Multi-scale surprisal analysis**: First comparison of classical (PCFG, n-gram) and modern (GPT-2) measures in naturalistic fMRI
- **Regional complexity binning**: Beyond network-level analysis to understand within-network specialization
- **Semantic-neural mapping**: Story-level semantic properties linked to regional brain responses
- **Non-linear complexity modeling**: Systematic testing of capacity constraints in language processing

## Key Results

### Regional Functional Specialization

- **LangMFG**: Primary executive control hub with strongest correlations across all surprisal measures (PCFG: r = 0.903)
- **LangIFG**: Core syntactic processor with preferential sensitivity to structural complexity  
- **LangPostTemp**: Integrates semantic complexity across all measures
- **LangAntTemp**: Specializes in lexical processing (5-gram: r = 0.703)
- **LangAngG**: Strongest response to GPT-2 contextual complexity (r = 0.736)
- **LangIFGorb**: Narrative complexity gate with extreme story-specific variability

### The Semantic Processing Paradox

Stories with **highest semantic diversity** ("Tulips": 0.390 diversity) produced language network **deactivation**, while **coherent narratives** ("Elvis": 0.768 coherence) elicited **maximal activation**. This suggests:

- Inverted-U relationship between semantic complexity and network engagement
- Adaptive processing mechanisms preventing resource depletion  
- Potential shifting to MD network under excessive semantic load

### Non-Linear Complexity Processing

- **Systematic deviations** from linearity across all regions and measures
- **Peaked responses** at intermediate complexity levels rather than monotonic increases
- **Most consistent patterns** observed for GPT-2 surprisal, suggesting modern neural language models better capture human comprehension mechanisms

## Notes and Considerations

### Technical Requirements
- **Memory**: GPT-2 analysis requires substantial RAM (8GB+ recommended)
- **Reproducibility**: All random operations use fixed seeds
- **Platform**: Tested on Python 3.11 with provided conda environment
- **Data Size**: Large fMRI files not included in repository due to size constraints

### Important Methodological Notes
- **No synthetic data generation**: All analyses use real fMRI and linguistic data
- **HRF convolution**: Proper temporal alignment accounts for 4-6 second hemodynamic delays  
- **Balanced binning**: Percentile-based approach ensures equal sample sizes across complexity levels
- **Multiple comparison correction**: Statistical results account for multiple regions and measures

### Performance Considerations
- Code optimized for clarity and reproducibility rather than maximum speed
- For large-scale analysis, consider parallelization options in utils modules
- GPT-2 processing can be memory-intensive; reduce batch size if needed

## Citation and References

### Primary Paper
Melamed, E., & Bamberger, N. (2025). Functional Specialization and Non-Linear Processing in the Human Language Network. *Course Project Report, Language, Computation and Cognition (00960222)*, Technion.

### Primary Data Sources
Shain, C., Blank, I. A., van Schijndel, M., Schuler, W., & Fedorenko, E. (2020). fMRI reveals language-specific predictive coding during naturalistic sentence comprehension. *Neuropsychologia*, 138, 107307. [OSF Repository](https://osf.io/eyp8q/)

Pereira, F., Lou, B., Pritchett, B., Ritter, S., Gershman, S. J., Kanwisher, N., & Fedorenko, E. (2018). Toward a universal decoder of linguistic meaning from brain activation. *Nature Communications*, 9(1), 963. 

### Key Software Dependencies
- **Neural Language Models**: Transformers library (Hugging Face) for GPT-2 analysis
- **Brain Analysis**: Nilearn, SciPy, PyGAM for fMRI data processing  
- **Embeddings**: GloVe for semantic analysis
- **Standard Stack**: NumPy, Pandas, Matplotlib, Seaborn, Statsmodels

### Theoretical Framework
- **Predictive Processing**: Bastos et al. (2012), Kuperberg & Jaeger (2016)
- **Language Network Organization**: Fedorenko et al. (2013)
- **Neural Language Models**: Caucheteux & King (2022), Goldstein et al. (2022)

---

**Contact**: For questions about this analysis, to report issues, or to request access to processed data outputs, please open a GitHub issue or contact the project maintainers at elizavetam@campus.technion.ac.il or noabamberger@campus.technion.ac.il.

**GitHub Repository**: [\[(https://github.com/LizMelamed/lang-network-specialization.git)\]]

**Academic Affiliation**: Data and Decision Sciences, Technion - Israel Institute of Technology