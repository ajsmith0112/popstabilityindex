import polars as pl


def psi(*,
        df_base: pl.DataFrame,
        df_compare: pl.DataFrame,
        bins: int = 10,
        # TODO: parameter to include / exclude missing values
        numeric_columns: list[str] = None,
        categorical_columns: list[str] = None) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Calculate the population stability index statistic. If the PSI is undefined, then the value defaults to null.

    PSI is considered undefined if any of the following conditions are true:
        1. Categorical column in the base data contains levels not found in the comparison data. E.g., col1_base: ['a', 'b', 'c'], col1_compare: ['a', 'b']

    Args:
        df_base (polars.DataFrame): The base dataframe.
        df_compare (polars.DataFrame): The comparison dataframe.
        bins (int, optional): The number of bins to use. Defaults to 10.
        numeric_columns (list[str], optional): The numeric columns to use. Defaults to None.
        categorical_columns (list[str], optional): The categorical columns to use. Defaults to None.

    Returns:
        tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        A tuple containing the PSI values, a frequency table for the base data, and a frequency table for the comparison data.
    """

    # Input validation

    # Type checks
    if not (isinstance(df_base, pl.DataFrame) and isinstance(df_compare, pl.DataFrame)):
        raise TypeError("df_base and df_compare must be of type polars.DataFrame")
    if not (bins > 0 and isinstance(bins, int)):
        raise TypeError("bins must be positive integers")
    if not (isinstance(numeric_columns, list) or isinstance(categorical_columns, list)):
        raise TypeError("numeric_columns and categorical_columns must be a list of strings")

    if numeric_columns is None: numeric_columns = []
    if categorical_columns is None: categorical_columns = []

    # Check that all columns exist in both df_base and df_compare
    if missing_cols := set(numeric_columns + categorical_columns) - set(df_base.columns):
        raise Exception(f'Column(s) not found in df_base: {missing_cols}')
    if missing_cols := set(numeric_columns + categorical_columns) - set(df_compare.columns):
        raise Exception(f'Column(s) not found in df_compare: {missing_cols}')

    # Check column overlap
    if overlap := set(numeric_columns) & set(categorical_columns):
        raise Exception(f'numeric_columns and categorical_columns have overlapping columns: {overlap}')

    # Check that all numeric_columns are numeric
    if not all([df_base[col].dtype.is_numeric() for col in numeric_columns] + [df_compare[col].dtype.is_numeric() for col in numeric_columns]):
        raise Exception('Non numeric columns found in numeric_columns')

    # TODO: type checking for categorical columns



    # Data checks
    # inf values?

    # Initialize frequency tables and Lazy dataframes
    df_base_num_count = pl.DataFrame()
    df_compare_num_count = pl.DataFrame()
    df_base_cat_freq = pl.DataFrame()
    df_compare_cat_freq = pl.DataFrame()
    ldf_base = df_base.lazy()
    ldf_compare = df_compare.lazy()

    # Get PSI for numeric columns
    if numeric_columns:
        # Get bins from base using quantiles
        quantiles = pl.linear_space(0, 1, bins + 1, eager=True).to_list()
        dict_cols_cutoffs = (
            pl.concat(
                [ldf_base.select(pl.col(numeric_columns)).quantile(quantile, interpolation='linear') for quantile in quantiles], how='vertical'
            )
            .collect()
            .to_dict(as_series=False)
        )

        # Preprocess bin edges
        for col, list_edges in dict_cols_cutoffs.items():
            dict_cols_cutoffs[col][0] = float('-inf')
            dict_cols_cutoffs[col][-1] = float('inf')

            # Edge case: if first edge is an integer, .hist casts the result column as u64, resulting in a TypeError if the other edges are floats
            # Explicitly cast all values to float
            # TODO: issue casting int to float? e.g., 1.000000001
            dict_cols_cutoffs[col] = [float(val) for val in list_edges]

        # Edge case: Check that bin edges are not strictly monotonic. E.g., [0, 0, 0, 1.5, 2.1, 4.]
        # TODO: is this a general solution?
        dict_cols_cutoffs = {col: list(dict.fromkeys(edges)) for col, edges in dict_cols_cutoffs.items()}

        # Get bin counts for each df
        # TODO: .hist() can fail if all values are NaN or identical — needs explicit handling?
        list_ldfs_counts = [
            pl.concat(
            [ldf.select(pl.col(col).hist(bins=cutoffs, include_category=True)) for col, cutoffs in dict_cols_cutoffs.items()],
            how='horizontal'
        )
            for ldf in [ldf_base, ldf_compare]
        ]

        df_base_num_count, df_compare_num_count = pl.collect_all(list_ldfs_counts)

        # Get null counts
        list_dfs_null_counts = [
            df.select(
                [pl.struct(category=pl.lit('missing', dtype=pl.Categorical), count=pl.col(col).null_count()).alias(col)
                 for col in numeric_columns]
            )
            for df in [df_base, df_compare]
        ]

        # Append null freq, Null count can be 0, which results in log(0)
        # TODO: refactor to avoid indexing list
        df_base_num_count = pl.concat([df_base_num_count, list_dfs_null_counts[0]], how='vertical_relaxed')
        df_compare_num_count = pl.concat([df_compare_num_count, list_dfs_null_counts[1]], how='vertical_relaxed')

    # Get PSI for categorical columns
    if categorical_columns:

        # Get counts for categorical levels. Nulls are a category in value_counts
        list_ldfs_cat_counts = [
            pl.concat(
                [
                    ldf.select(pl.col(col).value_counts(parallel=True).struct.rename_fields(['category', 'count']))
                    .select(pl.col(col).struct.with_fields(pl.field('category').cast(pl.String).cast(pl.Categorical)))
                    .sort(pl.col(col).struct.field('category'), nulls_last=True)
                    for col in categorical_columns
                ],
                how='horizontal'
            )
            for ldf in [ldf_base, ldf_compare]
        ]

        df_base_cat_freq, df_compare_cat_freq = pl.collect_all(list_ldfs_cat_counts)

    # Combine numerical, categorical frequencies
    df_base_freq_struct = pl.concat([df_base_num_count, df_base_cat_freq], how='horizontal')
    df_compare_freq_struct = pl.concat([df_compare_num_count, df_compare_cat_freq], how='horizontal')

    # TODO: validate data of df_base_feq and df_compare_freq
    # DQ check: all percents sum to 1
    # assert df_base_prop.sum().select(((pl.all() - pl.lit(1)).abs() <= 1e-6)).transpose()['column_0'].all()
    # assert df_compare_prop.sum().select(((pl.all() - pl.lit(1)).abs() <= 1e-6)).transpose()['column_0'].all()

    # Check if categories are equal for each column
    df_base_fields = df_base_freq_struct.select(pl.all().struct.field('category').name.keep())
    df_compare_fields = df_compare_freq_struct.select(pl.all().struct.field('category').name.keep())
    list_invalid_psi = []
    for col in df_base_fields.columns:
        if not df_base_fields[col].equals(df_compare_fields[col]):
            list_invalid_psi.append(col)

    # Unnest structs to get counts
    df_base_freq = df_base_freq_struct.with_columns(pl.col(pl.Struct).struct.field('count').name.keep())
    df_compare_freq = df_compare_freq_struct.with_columns(pl.col(pl.Struct).struct.field('count').name.keep())

    # TODO: Temp fix: set 0 frequencies to null
    df_base_freq = df_base_freq.select(pl.when(pl.all() == 0).then(pl.lit(None)).otherwise(pl.all()).name.keep())
    df_compare_freq = df_compare_freq.select(pl.when(pl.all() == 0).then(pl.lit(None)).otherwise(pl.all()).name.keep())

    # Normalize frequencies
    df_base_prop = df_base_freq / df_base.height
    df_compare_prop = df_compare_freq / df_compare.height

    # Apply PSI formula (possible inf if freq is 0)
    df_psi = (df_base_prop - df_compare_prop) * (df_base_prop.select(pl.all().log()) - df_compare_prop.select(pl.all().log()))
    df_psi = df_psi.sum().transpose(include_header=True, header_name='attribute', column_names=['psi'])

    # Set invalid PSI's to null
    df_psi = df_psi.with_columns(
        pl.when(pl.col('attribute').is_in(list_invalid_psi))
        .then(pl.lit(None))
        .otherwise(pl.col('psi'))
        .name.keep()
    )

    # edge cases: bins with 0 volume

    return df_psi, df_base_freq_struct, df_compare_freq_struct

def debug_fn():

    df_base = pl.read_csv('./tests/base_data.csv')
    df_compare = pl.read_csv('./tests/compare_data.csv')

    psi_value = psi(df_base=df_base, df_compare=df_compare, numeric_columns=['col1_norm'])[0]

#debug_fn()