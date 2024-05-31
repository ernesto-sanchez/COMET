# ![Comet Logo (1) (1)](https://github.com/ernesto-sanchez/COMET/blob/main/Comet%20Logo%20(3).png)
# **C**ounterfactual **O**utcome **M**odels for **E**nhanced **T**ransplants


## Project description
This project aims to provide a platform to benchmark different counterfactual treatment outcome estimation algorithms in the setting of Organ Transplants. Moreover, It will serve as a tool that supports clinitians in the choice of an optimal patient/donor pair. The porject is structure in 3 main pillars: Data, Model, Evaluation. 
## Data
We will use synthetic data as a proof of concept and to benchmark the performance of counterfactual treatment outcome algorithms, but will design the tool to seamlessly integrate real-world organ transplant data. A script to generate sythetic organs and patients can be found under the "Synthetic Data Generation" folder. Moreover, the matching of these patients and organs can also be synthetically generated. The matching criterion can bec hosen among two, and the strenght of the treatment assignment bias can be tweaked as well via the **TAB** parameter (TAB == 0: Random organ-patient matching, TAB == 1: Optimal matching (according to above criterion) is performed).
Real world data can also be used in this framework, after pre-processing it to be in the desired format (see data_integration_guidelines.md). 
The data used to get the results in the project preprint cannot be published for privacy reasons.
## Model
Several CATE estimators are implementes (TODO: instructions on how to use them) (TODO: Brief description?)
Due to the high-dimensionality of the treatment, many of this methods require some kind of clustering being done to the organs. We therefore also accomodate the use of different clustering methods in our framework. 
## Evaluation
We measure the effectiveness of the models both in terms of how accurately they estimate the CATE and how accurately they estimate the counterfactual outcome of a patient-organ pair. The CATE evaluation is only possible when using synthetic data, as only the factual outcome is present in observational datasets.
# Folder Structure
The porject is structure in 3 main pillars: Data, Model, Evaluation. 
