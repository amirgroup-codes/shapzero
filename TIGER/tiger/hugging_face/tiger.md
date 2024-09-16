## TIGER Online Tool for Cas13 Efficacy Prediction

Welcome to TIGER!
This online tool accompanies our recent study from the labs of [David Knowles](https://daklab.github.io/) and [Neville Sanjana](http://sanjanalab.org/).
TIGER's ability to make accurate on- and off-target predictions enables users to 1) design highly effective gRNAs and 2) precisely modulate transcript expression by engineered gRNA-target mismatches. 

If you use the TIGER Online Tool in your study, please consider citing:
> **[Prediction of on-target and off-target activity of CRISPR–Cas13d guide RNAs using deep learning](http://sanjanalab.org/reprints/WesselsStirn_NBT_2023.pdf).** Wessels, H.-H.<sup>\*</sup>, Stirn, A.<sup>\*</sup>, Méndez-Mancilla, A., Kim, E. J., Hart, S. K., Knowles, D. A.<sup>#</sup>, & Sanjana, N. E.<sup>#</sup> *Nature Biotechnology* (2023).  [https://doi.org/10.1038/s41587-023-01830-8](https://doi.org/10.1038/s41587-023-01830-8)

Please note that this precompiled, online tool differs from the manuscript slightly.
First, this version of TIGER predicts using just target and guide sequence (see [Figure 3c](http://sanjanalab.org/reprints/WesselsStirn_NBT_2023.pdf)). Second, we map TIGER's predictions to the unit interval (0,1) to make estimates more interpretable: A `Guide Score` close to 1 corresponds to high gRNA activity (i.e. desirable for on-target guides).
A `Guide Score` near 0 denotes no/minimal activity (i.e. desirable for predicted off-targets to minimize the activity of these gRNAs on unintended targets).
This transformation is monotonic and therefore preserves Spearman, AUROC, and AUPRC performance.
These estimates (transformations of log-fold-change predictions from TIGER) appear in the `Guide Score` column of this online tool’s output.

### Using the TIGER Online Tool

The tool supports two methods for transcript entry:
1) Manual entry of a single transcript
2) Uploading a FASTA file that can contain one or more transcripts. Each transcript **must** have a unique ID.

The tool has three run modes:
1) Report all on-target gRNAs for each provided transcript.
2) Report the top 10 most active, on-target gRNAs for each provided transcript. This mode allows for the optional identification of off-target effects. For off-target avoidance, please note that a higher `Guide Score` (closer to 1) corresponds to *more* likely off-target effects.
3) Report the top 10 most active, on-target gRNAs for each provided transcript and their titration candidates (all possible single mismatches). A higher `Guide Score` (closer to 1) corresponds to greater transcript knockdown. 

The tool uses Gencode v19 (protein-coding and non-coding RNAs) to identify potential off-target transcripts.
Due to computational limitations, the online tool only supports off-target predictions for the top 10 most active, on-target gRNAs per transcript.

### Future Development Plans

- Off-target scanning speed improvements
- Off-target scanning for titration (engineered mismatch) mode
- Allow users to select more than the top ten guides per transcript
- Incorporate non-scalar features (target accessibility, hybridization energies, etc...)

To report bugs or to request additional features, please click the "Community" button in the top right corner of this screen and start a new discussion.
Alternatively, please email [Andrew Stirn](mailto:andrew.stirn@cs.columbia.edu).

#### Version
You are using version 2.0 of this tool.
All hugging face versions are marked with a `vX.x` tag.
The code used to train this model can be found [here](https://github.com/daklab/tiger)--specifically, please see `tiger_trainer.py` therein.
This GitHub repository has matching `vX.x` tags.
We will increment the major number when a change causes a difference in predictions (e.g. retraining the model).
We will otherwise increment the minor number (e.g. changes to the user interface, speed improvements, etc...).
