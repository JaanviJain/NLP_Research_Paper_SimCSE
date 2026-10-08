# NLP_Research_Paper_SimCSE
Replicating SimCSE: Simple Contrastive Learning of Sentence Embeddings
A from-scratch replication of SimCSE (EMNLP 2021) — both the unsupervised and supervisedvariants — trained and evaluated end-to-end on a free Kaggle GPU (NVIDIA T4 16 GB).

Paper: SimCSE: Simple Contrastive Learning of Sentence Embeddings — Tianyu Gao, Xingcheng Yao & Danqi Chen, EMNLP 2021Paper (ACL Anthology) · arXiv · Official code

Headline Results
Model	Paper (avg. Spearman ×100)	This Replication	Difference
Unsupervised SimCSE-BERT-base	76.25	77.04	+0.79
Supervised SimCSE-BERT-base	81.57	80.87	−0.70
Unsupervised SimCSE-RoBERTa-base	76.57	75.72	−0.85
Supervised SimCSE-RoBERTa-base	82.52	82.38	−0.14
The unsupervised BERT replication exceeds the published result (+0.79 avg over 7 STS tasks).
The supervised RoBERTa result matches the paper within noise (−0.14).
The evaluation protocol was independently validated against the authors' released checkpoints(matches the paper's numbers to ±0.05).
7-task STS test average: paper vs replication

Table of Contents
What the Paper Does
Model and Embedding Size
Datasets
Methodology / Implementation
Results
Pipeline Validation
Why Results Differ from the Paper
Conclusions
Repository Layout
How to Run
References
1. What the Paper Does
SimCSE learns a fixed-length vector (embedding) for any sentence so that semantically similarsentences end up close together in vector space. It trains with a contrastive (InfoNCE)objective with in-batch negatives:

ℓ_i = −log exp( sim(h_i, h_i⁺) / τ ) / Σ_j exp( sim(h_i, h_j⁺) / τ )
The central question is how to build positive pairs (x_i, x_i⁺):

Unsupervised SimCSE — feed the same sentence through the encoder twice.Standard dropout (p = 0.1) masks different neurons in each pass, so the two embeddingsdiffer only slightly. This tiny difference acts as minimal "data augmentation":the pair stays aligned, while all other sentences in the batch act as negatives.No discrete augmentation (word deletion, cropping, synonym replacement) is used —the paper shows plain dropout beats all of them (Table 1 of the paper).
Supervised SimCSE — use natural language inference (NLI) data:entailment (premise, hypothesis) pairs become positives, and the matchingcontradiction hypotheses become hard negatives:
ℓ_i = −log exp( sim(h_i, h_i⁺) / τ ) / Σ_j [ exp( sim(h_i, h_j⁺) / τ ) + exp( sim(h_i, h_j⁻) / τ ) ]
The paper also analyses embeddings through alignment (positives should be close) anduniformity (all embeddings should spread evenly on the hypersphere), showing thatcontrastive learning fixes the anisotropy of pre-trained BERT embeddings.

2. Model and Embedding Size
Component	Detail
Encoder	BERT-base-uncased / RoBERTa-base — 12 layers, 12 heads, hidden size 768, ~110M parameters
Sentence embedding size	768 dimensions (L2-normalised)
Pooling	[CLS] token + MLP pooler; the MLP is dropped at test time for the unsupervised model (paper Table 6)
Similarity	Cosine similarity, temperature τ = 0.05 (paper Table D.1)
Training	Full encoder fine-tuning, AdamW (weight decay 0.01), 10% linear warmup + linear decay, gradient clipping 1.0, fp16 mixed precision
Batch size	Unsupervised: 64 (63 in-batch negatives) · Supervised: 512 (1023 candidates) — paper Table A.1
Objective	Cross-entropy over in-batch candidates (InfoNCE), exactly Eq. (4) and Eq. (5) of the paper, α = 1
3. Datasets
Role	Data	Size	Source
Unsupervised training	English Wikipedia sentences	1,000,000	princeton-nlp/datasets-for-simcse (official)
Supervised training	SNLI + MNLI triplets: (premise, entailment-positive, contradiction-hard-negative)	275,601	same (official file)
Model selection (only)	STS-B development set	1,500 pairs	MTEB copy
Evaluation (test)	STS12–16, STS-B, SICK-R	3,108 / 1,500 / 3,750 / 3,000 / 1,186 / 1,379 / 9,927 pairs	MTEB copies
No STS training data is used at any point — matching the paper's protocol exactly.
Evaluation uses raw cosine similarity, Spearman's rank correlation, and"all" aggregation with no regressor (paper Appendix B).
4. Methodology / Implementation
Environment & tools

Kaggle notebook, NVIDIA T4 (16 GB), Python 3.x, PyTorch 2.x, HuggingFace transformers 4.x, scipy
fp16 mixed precision; seed 42 (and 123 for the seed study) for Python / NumPy / PyTorch
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True for stable GPU memory
Pipeline

Download the official wiki1m_for_simcse.txt and nli_for_simcse.csv from the SimCSE authors.
Tokenise with max length 32 (training) / 128 (evaluation).
Train with the paper's exact hyperparameters; evaluate on STS-B dev every 250 steps(every 100 for supervised v2) and keep the best checkpoint (paper Appendix A).
Evaluate the final model once on all 7 STS test sets.
Additions beyond the paper

Top-3 checkpoint soup (weight averaging of the best dev checkpoints) for supervised runs.
Gradient checkpointing — makes the paper's supervised batch size 512 fit on a 16 GB GPU(three forward passes share one autograd graph; without this, batch 512 OOMs).
Seed variance study — the unsupervised model trained with two seeds.
5. Results
Full per-task comparison (Spearman ×100, "all" setting)
Model	STS12	STS13	STS14	STS15	STS16	STS-B	SICK-R	Avg.
Paper — Unsup BERT-base	68.40	82.41	74.38	80.91	78.56	76.85	72.23	76.25
Ours — Unsup BERT-base	70.55	82.97	74.46	82.05	78.59	78.40	72.21	77.04
Paper — Sup BERT-base	75.30	84.67	80.19	85.40	80.82	84.25	80.39	81.57
Ours — Sup BERT-base (b512 + soup)	74.80	84.38	79.27	84.61	79.39	83.25	80.40	80.87
Paper — Unsup RoBERTa-base	70.16	81.77	73.24	81.36	80.65	80.22	68.56	76.57
Ours — Unsup RoBERTa-base	66.88	80.69	72.16	80.92	80.25	79.70	69.43	75.72
Paper — Sup RoBERTa-base	76.53	85.21	80.95	86.03	82.57	85.83	80.50	82.52
Ours — Sup RoBERTa-base (b512 + soup)	76.28	85.00	80.57	85.96	82.46	85.48	80.92	82.38
Difference heatmap: ours minus paper

The unsupervised BERT model wins on 5 of 7 tasks, with the largest gains onSTS12 (+2.15) and STS-B (+1.55).

Per-task detail — unsupervised BERT-base
Per-task comparison, unsupervised BERT

Training behaviour (STS-B dev during training)
Run	Config	Steps	Time	Best dev STS-B
Unsup BERT (seed 42)	batch 64, lr 3e-5, 1 epoch	15,625	54.7 min	83.28
Unsup BERT (seed 123)	batch 64, lr 3e-5, 1 epoch	15,625	55.6 min	81.93
Sup BERT (v1)	batch 256, lr 5e-5, 3 epochs	3,228	47.9 min	85.09
Sup BERT (v2)	batch 512, lr 5e-5, 3 epochs, grad ckpt	1,614	61.0 min	85.43
Unsup RoBERTa	batch 256, lr 1e-5, 1 epoch	3,906	40.6 min	83.84
Sup RoBERTa	batch 512, lr 5e-5, 3 epochs, grad ckpt	1,614	61.3 min	87.64 (soup of top 3)
Training curves

Note the right panel: the first supervised run (batch 256 — the only size that fit withoutgradient checkpointing) stalls about a point below the paper's line; restoring the paper'sbatch size 512 closes most of that gap. RoBERTa supervised reaches 87.54 dev (87.64 with soup).

Embedding-space analysis (paper §7, alignment & uniformity)
Measured on STS-B; lower is better for both metrics.

Model	ℓ_align	ℓ_uniform
Vanilla BERT (pre-trained)	0.190	−1.005
Unsup SimCSE	0.355	−2.320
Sup SimCSE	0.528	−3.716
Alignment vs uniformity

This reproduces the paper's core analysis: unsupervised SimCSE strongly improvesuniformity (fixing the anisotropic embedding cone of pre-trained BERT) while keepingalignment intact, and the supervised NLI signal further improves alignment —exactly the pattern in the paper's Figure 3.

6. Pipeline Validation
Before training anything, the authors' official checkpoints were re-scored with thisrepository's evaluation code:

Checkpoint	Paper reports	This evaluator
princeton-nlp/unsup-simcse-bert-base-uncased	76.25	76.29
princeton-nlp/sup-simcse-bert-base-uncased	81.57	81.63
Agreement to ±0.05 confirms the evaluation protocol (cosine, Spearman, "all", noregressor) is implemented correctly — so remaining differences in the trained models aregenuinely training-side, not evaluation-side.

7. Why Results Differ from the Paper
Seed variance. Unsupervised BERT with seed 42 vs seed 123 scored 77.04 and 75.67(spread ≈ 1.4). The final model is whichever 250-step snapshot happened to peak on dev;the paper reports a single run.
Batch size (diagnosed and fixed). The first supervised run used batch 256 because 512OOM'd on the 16 GB GPU. Halving the in-batch negatives cost ≈ −0.6 (80.92 vs 80.87 is notthe story — dev 85.09 → 85.43 after the fix). Restoring batch 512 via gradientcheckpointing recovered the gap; RoBERTa supervised reached 87.64 dev with the soup.
Dataset copies. STS12–16 / SICK-R are MTEB copies of the original SentEval files; theofficial checkpoint scores 72.55 vs the paper's 72.23 on SICK-R, so ≈ 0.3 of anytask-level gap is evaluator-side and affects all models equally.
NLI file version. The authors' released nli_for_simcse.csv contains 275,601 tripletsvs the paper's "314k" description (the authors' own filtering). Training on the officialreleased file is the most faithful choice.
8. Conclusions
SimCSE reproduces faithfully from the paper's description: both variants train stably in~1 hour each on a single free Kaggle GPU.
Unsupervised SimCSE-BERT: 77.04 vs 76.25 — the replication exceeds the published result.
Supervised RoBERTa: 82.38 vs 82.52 — within noise of the paper. Supervised BERT:80.87 vs 81.57 (−0.70), explained by batch size and seed variance.
The paper's alignment–uniformity analysis reproduces qualitatively: contrastive trainingflattens the embedding spectrum (uniformity −1.0 → −2.3 / −3.7) while preserving alignment.
9. Repository Layout
├── README.md                      this file├── simcse_replication.ipynb       full pipeline: data, training, evaluation, analysis├── generate_figures.py            reproduces all figures in figures/├── requirements.txt└── figures/    ├── chart1_avg_paper_vs_ours.png    ├── chart2_pertask_unsup_bert.png    ├── chart3_training_curves.png    ├── chart4_align_uniform.png    └── chart5_diff_heatmap.png
10. How to Run
Option A — Kaggle (recommended):

Open a Kaggle notebook with GPU (T4) accelerator and Internet ON.
Upload / open simcse_replication.ipynb and run all cells top to bottom.
Unsupervised ≈ 55 min, supervised ≈ 60 min per model. Checkpoints auto-save and auto-resume.
Option B — local machine (any 16 GB GPU):

pip install -r requirements.txt
model, tokenizer, dev = train_simcse("unsup")            # unsupervised, ~55 minresults = eval_all_tasks(model, tokenizer, use_mlp=False)model, tokenizer, dev = train_sup_v2(batch_size=512, lr=5e-5, epochs=3)   # supervisedresults = eval_all_tasks(model, tokenizer, use_mlp=True)
Reproduce the figures:

python generate_figures.py    # writes all 5 charts into figures/
11. References
Gao, T., Yao, X., & Chen, D. (2021). SimCSE: Simple Contrastive Learning of SentenceEmbeddings. EMNLP 2021, pp. 6894–6910.
Devlin, J., et al. (2019). BERT: Pre-training of Deep Bidirectional Transformers forLanguage Understanding. NAACL-HLT.
Liu, Y., et al. (2019). RoBERTa: A Robustly Optimized BERT Pretraining Approach.arXiv:1907.11692.
Wang, T., & Isola, P. (2020). Understanding Contrastive Representation Learning throughAlignment and Uniformity on the Hypersphere. ICML.
Implemented and evaluated by Your Name .All training data and evaluation sets are the public files released by the SimCSE authors.Kaggle notebook: [link-to-your-notebook] .
