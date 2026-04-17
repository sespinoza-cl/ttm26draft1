# Masticatory nociceptive burden predicts cervical comorbidity in patients with temporomandibular disorders

## Overview
#This repository contains the open-access clinical dataset and the MATLAB reproducibility script for the study: **"Masticatory nociceptive burden predicts cervical comorbidity in patients with temporomandibular disorders"**. 
#The study utilizes a multivariate phenotypic approach (PCA and unsupervised K-means clustering) to demonstrate a clinical decoupling between structural joint status and nociceptive symptoms. The findings support that cervical comorbidity in Temporomandibular Disorders (TMD) is driven by a cumulative nociceptive burden rather than mechanical joint derangements, aligning with the neurobiological frameworks of the Trigeminocervical Complex (TCC).

## Repository Structure

* `TMD_Cervical_Dataset.txt`: The complete, anonymized, and tab-separated dataset (n=145) used for the analysis.
* `reproduce_analysis.m`: The master MATLAB script required to reproduce the statistical analysis, predictive models, and figures presented in the manuscript.

## Data Dictionary
The dataset (`TMD_Cervical_Dataset.txt`) contains the following variables based on the DC/TMD Axis I diagnostics and standardized cervical exploration:

| Variable | Type | Description |
| :--- | :--- | :--- |
| **Age** | Numeric | Patient's age in years. |
| **Sex** | String | Patient's biological sex (`Female`, `Male`). |
| **Muscle_Pain** | Boolean (1/0) | Presence of Painful Muscle Disorders (Myalgia/Myofascial Pain). |
| **Joint_Pain** | Boolean (1/0) | Presence of Painful Joint Disorders (Arthralgia). |
| **Intra_Articular_Disorder** | Boolean (1/0) | Presence of Disc Displacements. |
| **Neck_Pain** | Boolean (1/0) | **Target Variable:** Presence of cervical comorbidity (pain on movement AND palpation). |

## Reproducing the Analysis

# The provided MATLAB script is designed for full reproducibility. It performs the dimensionality reduction (PCA), unsupervised clustering, robust logistic regression, and empirical bootstrap validation (2,000 iterations).

### Prerequisites
# * MATLAB (R2021a or newer recommended).
# * **Statistics and Machine Learning Toolbox** (required for `pca`, `kmeans`, `fitglm`, and `randsample` functions).

### Execution Steps
1. Clone or download this repository to your local machine.
2. Open MATLAB and navigate to the repository folder containing both the `.txt` dataset and the `.m` script.
3. Open `reproduce_analysis.m` and click **Run** (or type `reproduce_analysis` in the Command Window).
4. The script will automatically:
   * Load the dataset robustly.
   * Generate **Figure 1** (Silhouette Validation) and **Figure 2** (PCA Mapping).
   * Print the empirical results for **Table 1**, **Table 2**, and **Table 3** directly in the MATLAB Command Window.

## Citation
If you use this dataset or code in your research, please cite the original publication:

Inda, D., Sandoval, G., Lira, D., & Espinoza, S.  (2026). Masticatory nociceptive burden predicts cervical comorbidity in patients with temporomandibular disorders.

## Contact
For any questions regarding the dataset or the statistical methodology, please contact the corresponding author:

**Sebastián Espinoza, PhD.** 
Institute for Health and Wellbeing Technology and Innovation (ITiSB), Universidad Andrés Bello  
Valparaíso, Chile  
inv.itisb@unab.cl ; sebastian.espinoza@hotmail.com

## License
This project is licensed under the [Creative Commons Attribution 4.0 International License (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/). You are free to share and adapt the material for any purpose, even commercially, as long as appropriate credit is given to the original authors.
