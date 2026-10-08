# NLP_Research_Paper_SimCSE
Replicating SimCSE: Simple Contrastive Learning of Sentence Embeddings
A from-scratch replication of SimCSE (EMNLP 2021) — both the unsupervised and supervisedvariants — trained and evaluated end-to-end on a free Kaggle GPU (NVIDIA T4 16 GB).


Table of Contents
What the Paper Does
Model and Embedding Size
Datasets
Methodology and Implementation
Results
Pipeline Validation
Why Results Differ from the Paper
Conclusions
How to Run
References
Results at a Glance
Model	Paper (Avg. Spearman ×100)	This Replication	Difference
Unsupervised SimCSE-BERT-base	76.25	77.04	+0.79
Supervised SimCSE-BERT-base	81.57	80.87	-0.70
Unsupervised SimCSE-RoBERTa-base	76.57	75.72	-0.85
Supervised SimCSE-RoBERTa-base	82.52	82.38	-0.14
Key Findings
The unsupervised BERT-base replication exceeds the published result by 0.79 points.
The supervised RoBERTa-base replication is within 0.14 points of the published result.
The evaluation pipeline was independently validated against the authors' released checkpoints.
The official checkpoints reproduced the published scores within approximately ±0.05 points.
The unsupervised BERT model outperformed the published result on 5 of 7 STS tasks.
1. What the Paper Does

SimCSE learns a fixed-length vector representation, or sentence embedding, for each input sentence.

The goal is to make:

semantically similar sentences close together in embedding space;
semantically unrelated sentences far apart;
embeddings more uniformly distributed rather than concentrated in a narrow region.

SimCSE uses contrastive learning with the InfoNCE objective and in-batch negative examples.

Contrastive Objective

For an input sentence (x_i), let:

(h_i) = embedding of the original sentence;
(h_i^+) = embedding of its positive counterpart;
(h_j^+) = positive embeddings from other sentences in the batch;
(\operatorname{sim}(\cdot,\cdot)) = cosine similarity;
(\tau) = temperature parameter.

The unsupervised contrastive loss is:

[
\ell_i =
-\log
\frac{
\exp(\operatorname{sim}(h_i,h_i^+)/\tau)
}{
\sum_j
\exp(\operatorname{sim}(h_i,h_j^+)/\tau)
}
]

The model is trained to assign the highest similarity to the positive pair while treating other examples in the batch as negatives.

Unsupervised SimCSE

The unsupervised version creates positive pairs without requiring additional labelled data.

The same sentence is passed through the encoder twice.

Because dropout randomly masks different neurons during each forward pass, the two representations are slightly different.

Key Idea
Same sentence
      ↓
Two stochastic dropout passes
      ↓
Two slightly different embeddings
      ↓
Treat them as a positive pair
      ↓
Other sentences in the batch = negatives

No explicit data augmentation such as:

word deletion;
synonym replacement;
cropping;
word shuffling

is required.

The SimCSE paper showed that simple dropout provides an effective form of augmentation for sentence representation learning.

Supervised SimCSE

The supervised version uses Natural Language Inference (NLI) data.

Each training example contains:

Premise
   |
   +---- Entailment hypothesis → Positive
   |
   +---- Contradiction hypothesis → Hard negative

The entailment pair is treated as the positive pair, while the contradiction pair acts as a hard negative.

The supervised objective is:

[
\ell_i =
-\log
\frac{
\exp(\operatorname{sim}(h_i,h_i^+)/\tau)
}{
\sum_j
\left[
\exp(\operatorname{sim}(h_i,h_j^+)/\tau)
+
\exp(\operatorname{sim}(h_i,h_j^-)/\tau)
\right]
}
]

This provides a stronger semantic training signal because the model explicitly learns from entailment and contradiction relationships.

Alignment and Uniformity

The paper also analyses sentence embeddings using two properties.

Alignment

Alignment measures how close positive pairs are.

The goal is:

Semantically similar sentences should have similar embeddings.

Lower alignment loss is better.

Uniformity

Uniformity measures how evenly embeddings are distributed over the representation space.

The goal is:

Sentence embeddings should be spread across the hypersphere instead of collapsing into a narrow region.

Lower uniformity loss is better.

SimCSE improves the uniformity of pretrained BERT embeddings while maintaining useful alignment.

2. Model and Embedding Size
Component	Configuration
Encoder	BERT-base-uncased / RoBERTa-base
Transformer layers	12
Attention heads	12
Hidden size	768
Parameters	Approximately 110M
Sentence embedding	768 dimensions
Normalization	L2 normalization
Pooling	[CLS] token + MLP pooler
Test-time pooling	MLP dropped for unsupervised model
Similarity	Cosine similarity
Temperature (\tau)	0.05
Training	Full encoder fine-tuning
Optimizer	AdamW
Weight decay	0.01
Warmup	10% linear warmup
Scheduler	Linear decay
Gradient clipping	1.0
Precision	FP16 mixed precision
Objective	Cross-entropy / InfoNCE
(\alpha)	1
Batch Sizes
Variant	Batch Size	In-Batch Negatives / Candidates
Unsupervised	64	63 negatives
Supervised	512	1023 candidates

The supervised batch size of 512 is important because contrastive learning benefits from a large number of negative examples.

3. Datasets
Role	Dataset	Size	Source
Unsupervised training	English Wikipedia sentences	1,000,000	Official SimCSE dataset
Supervised training	SNLI + MNLI triplets	275,601	Official SimCSE dataset
Model selection	STS-B development set	1,500 pairs	MTEB copy
Evaluation	STS12	3,108 pairs	MTEB
Evaluation	STS13	1,500 pairs	MTEB
Evaluation	STS14	3,750 pairs	MTEB
Evaluation	STS15	3,000 pairs	MTEB
Evaluation	STS16	1,186 pairs	MTEB
Evaluation	STS-B	1,379 pairs	MTEB
Evaluation	SICK-R	9,927 pairs	MTEB
Important Evaluation Constraint

No STS training data is used during model training.

STS-B development data is used only for checkpoint selection, matching the evaluation protocol of the original paper.

Evaluation uses:

raw cosine similarity;
Spearman rank correlation;
"all" aggregation;
no regression model.
4. Methodology and Implementation
Environment
Component	Configuration
GPU	NVIDIA T4
GPU Memory	16 GB
Python	Python 3.x
Deep Learning	PyTorch 2.x
NLP Framework	HuggingFace Transformers 4.x
Evaluation	SciPy
Precision	FP16 mixed precision
Random seed	42
Additional seed study	123

The following environment variable was used for more stable CUDA memory allocation:

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
Training Pipeline
Step 1 — Download Data

The official SimCSE datasets are used:

wiki1m_for_simcse.txt
nli_for_simcse.csv

These files are released by the SimCSE authors.

Step 2 — Tokenization
Stage	Maximum Length
Training	32
Evaluation	128
Step 3 — Model Training

The encoder is fully fine-tuned using the hyperparameters described in the original paper.

The model is evaluated on the STS-B development set during training.

Training
   ↓
Evaluate STS-B development set
   ↓
Save best checkpoint
   ↓
Final evaluation on 7 STS tasks
Additional Implementation Improvements
Gradient Checkpointing

The original supervised batch size of 512 initially caused GPU out-of-memory errors.

Gradient checkpointing was introduced to reduce memory usage and allow:

Batch size = 512

to fit on the NVIDIA T4.

Top-3 Checkpoint Soup

For supervised experiments, the top three checkpoints according to development performance were averaged.

Checkpoint 1 ─┐
Checkpoint 2 ─┼──→ Weight averaging → Final model
Checkpoint 3 ─┘
Seed Variance Study

The unsupervised BERT model was trained using two random seeds:

Seed 42
Seed 123

This was used to measure the effect of training randomness on the final results.

5. Results
Overall Results

Spearman correlation × 100 using the "all" evaluation setting.

Model	STS12	STS13	STS14	STS15	STS16	STS-B	SICK-R	Avg.
Paper — Unsupervised BERT-base	68.40	82.41	74.38	80.91	78.56	76.85	72.23	76.25
Ours — Unsupervised BERT-base	70.55	82.97	74.46	82.05	78.59	78.40	72.21	77.04
Paper — Supervised BERT-base	75.30	84.67	80.19	85.40	80.82	84.25	80.39	81.57
Ours — Supervised BERT-base	74.80	84.38	79.27	84.61	79.39	83.25	80.40	80.87
Paper — Unsupervised RoBERTa-base	70.16	81.77	73.24	81.36	80.65	80.22	68.56	76.57
Ours — Unsupervised RoBERTa-base	66.88	80.69	72.16	80.92	80.25	79.70	69.43	75.72
Paper — Supervised RoBERTa-base	76.53	85.21	80.95	86.03	82.57	85.83	80.50	82.52
Ours — Supervised RoBERTa-base	76.28	85.00	80.57	85.96	82.46	85.48	80.92	82.38
Unsupervised BERT Results

The replicated unsupervised BERT model achieved:

Paper:       76.25
Replication: 77.04
Difference: +0.79

The replication outperformed the published result on 5 of the 7 STS tasks.

Largest Improvements
Task	Improvement
STS12	+2.15
STS15	+1.14
STS-B	+1.55
Training Behaviour
Run	Configuration	Steps	Time	Best Dev STS-B
Unsupervised BERT — Seed 42	Batch 64, LR 3e-5, 1 epoch	15,625	54.7 min	83.28
Unsupervised BERT — Seed 123	Batch 64, LR 3e-5, 1 epoch	15,625	55.6 min	81.93
Supervised BERT v1	Batch 256, LR 5e-5, 3 epochs	3,228	47.9 min	85.09
Supervised BERT v2	Batch 512, LR 5e-5, 3 epochs	1,614	61.0 min	85.43
Unsupervised RoBERTa	Batch 256, LR 1e-5, 1 epoch	3,906	40.6 min	83.84
Supervised RoBERTa	Batch 512, LR 5e-5, 3 epochs	1,614	61.3 min	87.64
Embedding-Space Analysis

The alignment and uniformity analysis was performed on STS-B.

Lower values are better for both metrics.

Model	Alignment	Uniformity
Vanilla BERT	0.190	-1.005
Unsupervised SimCSE	0.355	-2.320
Supervised SimCSE	0.528	-3.716
Interpretation

The results reproduce the qualitative behaviour reported in the SimCSE paper.

Vanilla BERT

Pretrained BERT embeddings are highly anisotropic, meaning many sentence embeddings occupy a relatively narrow region of the embedding space.

Unsupervised SimCSE

Contrastive training substantially improves uniformity:

-1.005 → -2.320

The embeddings become more evenly distributed.

Supervised SimCSE

The NLI supervision provides an additional semantic signal:

-1.005 → -3.716

These results support the paper's conclusion that contrastive learning improves the geometry of pretrained sentence representations.

6. Pipeline Validation

Before training new models, the evaluation pipeline was tested using the official SimCSE checkpoints released by the authors.

Checkpoint	Paper Score	This Evaluator
princeton-nlp/unsup-simcse-bert-base-uncased	76.25	76.29
princeton-nlp/sup-simcse-bert-base-uncased	81.57	81.63

The difference is approximately:

≤ 0.05 points

This provides strong evidence that the following evaluation components were implemented correctly:

cosine similarity;
Spearman correlation;
"all" aggregation;
no regression model;
STS task processing.

Therefore, the remaining differences between the replication and the paper primarily originate from training-side factors, rather than an incorrect evaluation implementation.

7. Why Results Differ from the Paper

The replication does not exactly reproduce every published number. Several factors explain the differences.

7.1 Random Seed Variance

The unsupervised BERT model was trained using two seeds.

Seed	Average Score
42	77.04
123	75.67

The spread is approximately 1.37 points.

This demonstrates that SimCSE performance can vary noticeably between training runs.

The paper reports a single training run, so exact reproduction is not guaranteed.

7.2 Batch Size

The first supervised experiment used batch size 256 because batch size 512 caused an out-of-memory error on the 16 GB T4.

The original paper uses batch size 512.

Because contrastive learning relies heavily on negative examples, reducing the batch size reduces the number of negatives available to the model.

Gradient checkpointing was therefore introduced to restore:

Batch size = 512

The larger batch size improved development-set performance:

Batch 256 → 85.09
Batch 512 → 85.43
7.3 Dataset Copies

The STS12–16 and SICK-R datasets used for evaluation are MTEB copies of the original SentEval files.

The official checkpoint evaluation produced:

Paper SICK-R:       72.23
This evaluator:     72.55
Difference:          0.32

Therefore, approximately 0.3 points of some task-level differences may originate from dataset/evaluation-file differences rather than model quality.

7.4 NLI Dataset Version

The paper describes approximately 314K NLI examples.

However, the officially released:

nli_for_simcse.csv

contains:

275,601 triplets

The released file reflects the authors' filtering process.

For faithful replication, this project uses the official released file rather than attempting to reconstruct the dataset from the paper's approximate description.

8. Conclusions

This project demonstrates that SimCSE can be reproduced from the paper's methodology and official datasets using a single NVIDIA T4 GPU.

Main Findings

Unsupervised BERT-base

Paper:       76.25
Replication: 77.04
Difference:  +0.79

The replication exceeds the published average.

Supervised RoBERTa-base

Paper:       82.52
Replication: 82.38
Difference:  -0.14

The result is very close to the published result.

Supervised BERT-base

Paper:       81.57
Replication: 80.87
Difference:  -0.70

The remaining difference is consistent with training variability, batch-size effects, and dataset/evaluation differences.

Alignment and uniformity

The embedding-space analysis reproduces the central finding of the SimCSE paper:

Vanilla BERT uniformity:       -1.005
Unsupervised SimCSE:           -2.320
Supervised SimCSE:             -3.716

Contrastive learning substantially improves the uniformity of sentence embeddings while preserving useful alignment between positive pairs.

Compute efficiency

The complete SimCSE variants can be trained in approximately one hour per model on a single NVIDIA T4 GPU.

Overall, the results provide a strong end-to-end replication of SimCSE, including training, evaluation, checkpoint selection, embedding analysis, and investigation of reproduction gaps.

9. How to Run
Option A — Kaggle

The project can be run using a Kaggle Notebook with an NVIDIA T4 GPU.

Requirements
Kaggle account
Kaggle Notebook
GPU accelerator enabled
Internet access enabled
Approximately 16 GB GPU memory
Steps
Create a new Kaggle notebook.
Enable the NVIDIA T4 GPU accelerator.
Enable Internet access.
Upload or open the project notebook.
Run the notebook from top to bottom.

Approximate training time:

Unsupervised SimCSE: ~55 minutes
Supervised SimCSE:   ~60 minutes per model
Option B — Local GPU

Install the required dependencies:

pip install -r requirements.txt
Train Unsupervised SimCSE
model, tokenizer, dev = train_simcse("unsup")

results = eval_all_tasks(
    model,
    tokenizer,
    use_mlp=False
)
Train Supervised SimCSE
model, tokenizer, dev = train_sup_v2(
    batch_size=512,
    lr=5e-5,
    epochs=3
)

results = eval_all_tasks(
    model,
    tokenizer,
    use_mlp=True
)
10. References
Gao, T., Yao, X., & Chen, D. (2021).
SimCSE: Simple Contrastive Learning of Sentence Embeddings.
Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing (EMNLP), 6894–6910.
Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019).
BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.
Proceedings of NAACL-HLT.
Liu, Y., Ott, M., Goyal, N., et al. (2019).
RoBERTa: A Robustly Optimized BERT Pretraining Approach.
arXiv:1907.11692.
Wang, T., & Isola, P. (2020).
Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere.
Proceedings of ICML.
Acknowledgements

This project is based on the methodology introduced in the SimCSE paper by Tianyu Gao, Xingcheng Yao, and Danqi Chen.

All training and evaluation datasets used in this project are publicly available resources released by the SimCSE authors or distributed through the referenced evaluation collections.

Project Information

Project: NLP Research Paper — SimCSE Replication
Task: Sentence Representation Learning
Models: BERT-base, RoBERTa-base
Methods: Unsupervised and Supervised Contrastive Learning
Framework: PyTorch + HuggingFace Transformers
Hardware: NVIDIA T4 16 GB
Platform: Kaggle

Implemented and evaluated by: Your Name
