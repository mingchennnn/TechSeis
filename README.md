# TechSeis — BERTopic-based Patent Transition Sensing

This repository contains the BERTopic-based analysis pipeline used to (1) build topic models on patent text, (2) compute dynamic topic-change indicators over time windows, and (3) run downstream company-level analyses around “critical” (shift-driving) topics/patents.

The workflow is notebook-first (Google Colab–friendly) and expects a preprocessed patent dataset (IDs + years + document text), with optional precomputed embeddings for speed and reproducibility.

## Repository contents

- `BERTopic_patent_filtering.ipynb`  
  Data access + filtering, embedding precomputation, and BERTopic training setup (incl. topic control + representations).

- `BERTopic_patent_full_analysis.ipynb`  
  End-to-end BERTopic training on the filtered dataset, topic visualization, and export of intermediate objects for indicator computation.

- `BERTopic_patent_dynamic_topics.ipynb`  
  Dynamic topic analysis across rolling year windows:
  - compute change intensity metrics (e.g., rate of meta-topic change, angle-difference–style indicators)
  - identify “change-driving” topics per window
  - recover patents affiliated with identified shift-driving topics

- `BERTopic_patent_predictivity_dynamic_topics.ipynb`  
  Predictivity / truncation experiments (repeat dynamic-topic pipeline with a truncated end year) and compare stability of detected signals.

- `BERTopic_patent_company_analysis.ipynb`  
  Company-level downstream analytics, including:
  - building company technology portfolios with moving windows
  - proximity measures to critical patents (multiple distance/novelty options)
  - incumbent vs entrant trend analysis
  - network centralization and inequality measures over time

- `basic_func.py`  
  Utility functions used across notebooks (reordering topics/clusters by weighted mean year, handling missing years/half-years, angular-difference calculations, divergence metrics, Gini, smoothing, etc.).

## Data expectations

Notebooks are written assuming a folder layout like:

`/content/drive/MyDrive/TechShiftProject/{sector}/{dataset}_data/`

Typical required inputs:
- `df_id_year_document.csv`  
  Patent-level table with at least: `id`, `year`, `document` (and/or merged title+abstract text).
- (optional) `df_company_id_year.csv`  
  Company–patent mapping used in company analyses.
- (optional but recommended) `embeddings.pkl`  
  Precomputed sentence-transformer embeddings for the patent documents.

## Outputs (typical)

Depending on the notebook, the pipeline writes:
- saved BERTopic models (`bertopic_model_saved_.../`)
- window-level topic artifacts (`topic_info.csv`, `avg_vectors.pkl`, `sum_vectors.pkl`)
- cross-window topic clustering results (e.g., `clustered_topics_*.csv`)
- company proximity / centrality time series (various `*_by_year_company_*.csv`, `centralities_by_year.csv`, `gini_coefficients_by_year.csv`)

## Quick start (recommended order)

1. **Filter + prepare data**  
   Run `BERTopic_patent_filtering.ipynb` to load `df_id_year_document.csv`, apply filtering, and (optionally) precompute embeddings.

2. **Train the topic model & export artifacts**  
   Run `BERTopic_patent_full_analysis.ipynb` to fit BERTopic and export topic vectors / summaries.

3. **Dynamic topic indicators & shift-driving topics**  
   Run `BERTopic_patent_dynamic_topics.ipynb` to compute time-window indicators and identify change-driving topics/patents.

4. **(Optional) predictivity / truncation tests**  
   Run `BERTopic_patent_predictivity_dynamic_topics.ipynb` to test robustness when limiting the dataset to earlier end years.

5. **Company-level analysis**  
   Run `BERTopic_patent_company_analysis.ipynb` for firm proximity, entrant/incumbent dynamics, and network measures.

## Environment

These notebooks typically rely on:
- `bertopic`, `hdbscan`, `umap-learn`
- `sentence-transformers`
- `pandas`, `numpy`, `scikit-learn`, `matplotlib`
- (company notebook) `networkx`, `hnswlib`, and some anomaly/distance models

They are configured for Colab usage (GPU enabling cells and Drive paths). If running locally, replace `/content/drive/...` paths with your local directories.

## Notes on reproducibility

- Several notebooks include steps to reduce stochasticity (e.g., fixed seeds / deterministic settings).
- Precomputing embeddings (`embeddings.pkl`) is strongly recommended if you iterate on topic settings.
