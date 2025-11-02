import polars as pl


def psi(*,
        df_base: pl.DataFrame,
        df_compare: pl.DataFrame,
        bins: int = 10,
        # TODO: parameter to include / exclude missing values
        numeric_columns: list[str] = None,
        categorical_columns: list[str] = None) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    Return: pl.DataFrame containing PSI values. With shape (n_columns, 1).
    """

    # Argument checks
    if missing_cols := set(numeric_columns + categorical_columns) - set(df_base.columns):
        raise Exception(f'Column(s) not found in df_base: {missing_cols}')
    if missing_cols := set(numeric_columns + categorical_columns) - set(df_compare.columns):
        raise Exception(f'Column(s) not found in df_compare: {missing_cols}')

    # Check column overlap
    if overlap := set(numeric_columns) & set(categorical_columns):
        raise Exception(f'numeric_columns and categorical_columns have overlapping columns: {overlap}')

    # Column is included in numeric_columns, but is categorical
    # Column is included in categorical_columns, but is numeric

    # Data checks
    # inf values?

    # Initialize frequency tables and Lazy dataframes
    df_base_num_freq = pl.DataFrame()
    df_compare_num_freq = pl.DataFrame()
    df_base_cat_freq = pl.DataFrame()
    df_compare_cat_freq = pl.DataFrame()
    ldf_base = df_base.lazy()
    ldf_compare = df_compare.lazy()

    # todo: combine the code for base / compare?

    # Get PSI for numeric columns
    if numeric_columns:
        # Get bins from base using quantiles
        quantiles = pl.linear_space(0, 1, bins + 1, eager=True).to_list()
        dict_cols_cutoffs = (
            pl.concat([df_base.select(pl.col(numeric_columns)).quantile(quantile, interpolation='linear') for quantile in quantiles], how='vertical')
            .to_dict(as_series=False)
        )

        # Preprocess bin edges,
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

        # Get frequencies of base and compare df
        ldf_base_freq = pl.concat(
            [ldf_base.select(pl.col(col).hist(bins=cutoffs, include_category=True)) for col, cutoffs in dict_cols_cutoffs.items()],
            how='horizontal'
        )

        ldf_compare_freq = pl.concat(
            [ldf_compare.select(pl.col(col).hist(bins=cutoffs, include_category=True)) for col, cutoffs in dict_cols_cutoffs.items()],
            how='horizontal'
        )

        df_base_num_freq, df_compare_num_freq = pl.collect_all([ldf_base_freq, ldf_compare_freq])

        # Get null frequencies
        # TODO: make lazy, combine with categorical? would allow more flexibility
        df_base_null_freq = df_base.select(
            [pl.struct(category=pl.lit('missing', dtype=pl.Categorical), count=pl.col(col).null_count()).alias(col) for col in numeric_columns]
        )

        df_compare_null_freq = df_base.select(
            [pl.struct(category=pl.lit('missing', dtype=pl.Categorical), count=pl.col(col).null_count()).alias(col) for col in numeric_columns]
        )

        # Append null freq, Null count can be 0, which results in log(0)
        df_base_num_freq = pl.concat([df_base_num_freq, df_base_null_freq], how='vertical')
        df_compare_num_freq = pl.concat([df_compare_num_freq, df_compare_null_freq], how='vertical')

    # Get PSI for categorical columns
    if categorical_columns:
        # Nulls are included in value_counts
        ldf_base_cat_freq = pl.concat(
            [
                ldf_base.select(pl.col(col).value_counts(parallel=True).struct.rename_fields(['category', 'count']))
                .select(pl.col(col).struct.with_fields(pl.field('category').cast(pl.String).cast(pl.Categorical)))
                .sort(pl.col(col).struct.field('category'), nulls_last=True)
                for col in categorical_columns
            ],
            how='horizontal'
        )

        ldf_compare_cat_freq = pl.concat(
            [
                ldf_compare.select(pl.col(col).value_counts(parallel=True).struct.rename_fields(['category', 'count']))
                .select(pl.col(col).struct.with_fields(pl.field('category').cast(pl.String).cast(pl.Categorical)))
                .sort(pl.col(col).struct.field('category'), nulls_last=True)
                for col in categorical_columns
            ],
            how='horizontal'
        )

        df_base_cat_freq, df_compare_cat_freq = pl.collect_all([ldf_base_cat_freq, ldf_compare_cat_freq])

    # Combine numerical, categorical frequencies
    df_base_freq_struct = pl.concat([df_base_num_freq, df_base_cat_freq], how='horizontal')
    df_compare_freq_struct = pl.concat([df_compare_num_freq, df_compare_cat_freq], how='horizontal')

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