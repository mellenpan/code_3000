import pandas as pd

ID_COLS = {"anon_id", "name"}


def load_data(anonymized_path, auxiliary_path):
    """
    Load anonymized and auxiliary datasets.
    """
    anon = pd.read_csv(anonymized_path)
    aux = pd.read_csv(auxiliary_path)
    return anon, aux


def _quasi_identifiers(anon_df, aux_df):
    """
    Determine which columns can be used for matching.
    """
    return [
        c for c in anon_df.columns
        if c in aux_df.columns and c not in ID_COLS
    ]


def link_records(anon_df, aux_df):
    """
    Link anonymized records to auxiliary records using exact matching
    on quasi-identifiers.

    Return only uniquely matched records.
    """
    qi = _quasi_identifiers(anon_df, aux_df)

    if not qi or "anon_id" not in anon_df.columns or "name" not in aux_df.columns:
        return pd.DataFrame(columns=["anon_id", "matched_name"])

    anon = anon_df.dropna(subset=qi)
    aux = aux_df.dropna(subset=qi)

    anon_unique = anon.loc[
        ~anon.duplicated(subset=qi, keep=False)
    ]
    aux_unique = aux.loc[
        ~aux.duplicated(subset=qi, keep=False)
    ]

    merged = anon_unique[["anon_id"] + qi].merge(
        aux_unique[["name"] + qi],
        on=qi,
        how="inner",
        validate="one_to_one"
    )

    return (
        merged.rename(columns={"name": "matched_name"})
        [["anon_id", "matched_name"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )


def deanonymization_rate(matches_df, anon_df):
    """
    Compute the fraction of anonymized records that were uniquely
    re-identified.
    """
    if anon_df.empty:
        return 0.0

    return matches_df["anon_id"].nunique() / len(anon_df)