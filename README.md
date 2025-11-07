# popstabilityindex

The popstabilityindex provides functionality to calculate the Population Stability Index (PSI), a common goodness-of-fit metric used in credit risk modeling.

Designed to be highly performant by relying solely on polars.

Supported functionality:
* Numerical columns.
* Categorical columns.
* Missing values are created as a bin.
* inf values are binned with the last bin and -inf values are binned with the first bin.

# Usage

```python
import polars as pl
from popstabilityindex import psi

df_base = pl.read_csv(...)
df_compare = pl.read_csv(...)

df_psi, df_base_freq, df_compare_freq = psi(
    df_base=df_base,
    df_compare=df_compare,
    bins=10,
    numeric_columns=['num_col1', 'num_col2'],
    categorical_columns=['cat_col1', 'cat_col2']   
)
```


# PSI formula
Let $B$ be the number of bins, $C_i$ the proportion of examples in bin $i$ from the comparison data, and $B_i$ the proportion of
examples in the base data.

$$
    PSI = \sum_{i=1}^{B} (C_i - B_i) (\ln(C_i) - \ln(B_i))
$$

PSI can be interpreted as the total percentage change in the bins between two histograms.

# Dependencies

```
polars
```



# Resources

Yurdakul, Bilal, "Statistical Properties of Population Stability Index" (2018). Dissertations. 3208.
https://scholarworks.wmich.edu/dissertations/3208





