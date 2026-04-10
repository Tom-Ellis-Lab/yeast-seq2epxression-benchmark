Evaluation: 
Directly compare performance at predicting tracks on native genome (RNA-seq only)
Pearson 
KL Divergence (Shape-only)
Log error (Magnitude-only)
Not a clean direct comparison as both models are trained on different experiments
Two eQTL datasets (Shorkie):
Binary classification task: is this variant an eQTL or not?
Caudal et al.: significant train test leakage for Shorkie
Kita et al.: more independent test set
Hard to reconstruct how exactly they did their evaluation (from code and paper) – author has not responded to email yet
MPRAs
Can the models predict MPRA outcomes without finetuning
Rafi et al. (Promoter): 
Fixed, original context (from Yorzoi): direct comparison possible
Marginalised over selected genes and positions (from Shorkie): 
some ”finetuning” going on as they select a specific position to evaluate on 
Still a good eval
TODO: stratify by promoter type (truly random, motif tiling, etc.)
Shalem et al. (Terminator):
Clean comparison possible
Fixed, original context (from Yorzoi)
Brooks, rearrangement effect eval (from Yorzoi): 
Direct comparison not possible because Shorkie was not trained on Nanopore data
Possible to compare a “substitute” (i.e. check rearrangement effect as predicted by other tracks)
ExoShorkie evals
Additional artificial chromosomes
Species LM: 
Keren et al.