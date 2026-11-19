# CPE 586 Final Project: Data Submission

## Project title
**Predicting Concrete Compressive Strength from Mixture Composition and Curing Age Using Machine Learning**

## 1. Dataset description

The selected dataset is **Concrete Compressive Strength** (UCI Machine Learning Repository, dataset ID **165**). It contains measurements of concrete mixture composition, curing age, and experimentally measured compressive strength. Each of the **1,030 observations** has **eight numerical input features** and **one numerical output variable**. The UCI repository reports **no missing values**.

- **Engineering field:** Civil engineering / construction materials engineering
- **Problem type:** Supervised machine learning, **regression**
- **Inputs:** Seven concrete mixture-component quantities and curing age
- **Prediction target:** Concrete compressive strength in megapascals (MPa)
- **Data origin:** Real-world experimental measurements; not collected by the author of this course project

## 2. Features and prediction target

| Variable | Description | Unit | Role |
|---|---|---|---|
| Cement | Cement content of the concrete mixture | kg/m³ | Input feature |
| Blast Furnace Slag | Blast-furnace slag content | kg/m³ | Input feature |
| Fly Ash | Fly ash content | kg/m³ | Input feature |
| Water | Water content | kg/m³ | Input feature |
| Superplasticizer | Superplasticizer admixture content | kg/m³ | Input feature |
| Coarse Aggregate | Coarse aggregate content | kg/m³ | Input feature |
| Fine Aggregate | Fine aggregate content | kg/m³ | Input feature |
| Age | Concrete age at strength testing | Days | Input feature |
| **Concrete Compressive Strength** | **Measured concrete compressive strength** | **MPa** | **Target variable** |

There are **no categorical class labels**. The value to be predicted is a **continuous numerical target**, so this project is a regression task rather than classification.

## 3. Problem to be solved

The objective is to **predict the compressive strength of concrete from its constituent quantities and curing age**. Specifically, the project will investigate whether machine learning can provide accurate strength estimates for concrete mixtures represented by the available data.

The tentative modeling plan is to compare **Linear Regression**, **Random Forest Regression**, and **XGBoost Regression**. Linear Regression will provide a simple baseline, while the tree-based models can represent nonlinear relationships. Model performance will be evaluated using metrics such as **R²**, **mean absolute error (MAE)**, and **root mean squared error (RMSE)**. The detailed experimental design will be addressed in the later model submission.

## 4. Why this is an engineering problem

Compressive strength is a key mechanical property used in **concrete mixture design, construction-material assessment, and structural engineering**. It depends on mixture composition and the age of the concrete. Measuring strength involves preparing concrete specimens, curing them, and performing physical compression tests, which require time, laboratory resources, and materials.

A predictive model could help engineers screen candidate mixtures and plan laboratory tests more efficiently. Model predictions would **support**, not replace, physical testing or required engineering safety checks.

## 5. Original source and how to obtain the dataset

**Original dataset creator:** I-Cheng Yeh. The dataset is hosted by the **UCI Machine Learning Repository** and was donated to that repository in **2007**. It was **not experimentally generated for this class project**.

- **Official dataset page:** https://archive.ics.uci.edu/dataset/165/concrete-compressive-strength
- **Persistent dataset DOI:** https://doi.org/10.24432/C5PK67
- **Direct UCI dataset archive:** https://archive.ics.uci.edu/static/public/165/concrete+compressive+strength.zip
- **Original research article:** Yeh, I.-C. (1998). *Modeling of strength of high-performance concrete using artificial neural networks*. *Cement and Concrete Research, 28*(12), 1797–1808. https://doi.org/10.1016/S0008-8846(98)00165-3
- **Dataset license:** Creative Commons Attribution 4.0 International (**CC BY 4.0**)

### Download instructions

**Manual:** Open the official UCI dataset page, select **Download**, and extract the archive. It contains the original spreadsheet (`Concrete_Data.xls`) and supporting description.

## 6. Related research for the later project stages

The intended comparison paper is Elhishi, S., Elashry, A. M., & El-Metwally, S. (2023). *Unboxing machine learning models for concrete strength prediction using XAI*. *Scientific Reports, 13*, 19892. https://doi.org/10.1038/s41598-023-47169-7. It uses the same UCI dataset and evaluates methods relevant to the proposed model comparison. Its results will serve as a published benchmark, **not as results already obtained in this project**.

## References

1. Yeh, I.-C. (1998). *Concrete Compressive Strength* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5PK67
2. Yeh, I.-C. (1998). Modeling of strength of high-performance concrete using artificial neural networks. *Cement and Concrete Research, 28*(12), 1797–1808. https://doi.org/10.1016/S0008-8846(98)00165-3
3. Elhishi, S., Elashry, A. M., & El-Metwally, S. (2023). Unboxing machine learning models for concrete strength prediction using XAI. *Scientific Reports, 13*, 19892. https://doi.org/10.1038/s41598-023-47169-7
