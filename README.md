#  Inequality and Social Issues Analysis (2020)

This project analyzes the relationship between income inequality and key social indicators across European and global regions using data from **Eurostat** and the **World Bank** for the year 2020. The analysis uses three unsupervised learning methods — **Principal Component Analysis (PCA)**, **Exploratory Factor Analysis (EFA)**, and **Hierarchical Clustering Analysis (HCA)** — to uncover patterns, reduce dimensionality, and discover latent factors.

---

##  Data Sources

All data is publicly available and extracted from the following official datasets:

- **[Income Inequality](https://ec.europa.eu/eurostat/databrowser/view/tespm151/default/table?lang=en)**  
- **[Tertiary Education Enrollment](https://ec.europa.eu/eurostat/databrowser/view/educ_uoe_enrt01/default/table?lang=en)**  
- **[Crime, Violence or Vandalism](https://ec.europa.eu/eurostat/databrowser/view/ilc_mddw06/default/table?lang=en)**  
- **[Unemployment by Education Level](https://ec.europa.eu/eurostat/databrowser/view/tps00066/default/table?lang=en)**  
- **[Urban Population (%)](https://databank.worldbank.org/reports.aspx?source=2&series=SP.URB.TOTL.IN.ZS&country=)**  

---

##  Dataset Overview

| Variable                  | Description                                                                 |
|--------------------------|-----------------------------------------------------------------------------|
| **Income Inequality**     | Ratio of income received by the top 20% vs. the bottom 20% of the population |
| **Crime Rate (%)**        | Perceived crime, violence, or vandalism in the local area                   |
| **Education (%)**         | Percentage enrolled in tertiary education (Bachelor’s, Master’s, PhD)       |
| **Unemployment (%)**      | Labor force unemployment rate (ages 25–64)                                 |
| **Urban Population (%)**  | Proportion of the population living in urban areas                          |

- **Observations**: 30  
- **Year**: 2020

---

##  Methods Used

###  Principal Component Analysis (PCA)
- Reduces data dimensionality while retaining most variance.
- Identifies principal components for visual simplification.

###  Hierarchical Clustering Analysis (HCA)
- Groups similar observations into clusters.
- Reveals natural structure and segmentation in the dataset.

###  Exploratory Factor Analysis (EFA)
- Detects hidden relationships between variables.
- Identifies latent variables influencing the observed data.

###  Tools & Libraries
- Python 3.x
- `pandas`, `numpy`, `matplotlib`

