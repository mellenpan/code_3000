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
    """Helper func to deside which columns to match"""
    return [c for c in anon_df.columns if c in aux_df.columns and c not in ID_COLS]


def link_records(anon_df, aux_df):
    """
    Attempt to link anonymized records to auxiliary records
    using exact matching on quasi-identifiers.

    Returns a DataFrame with columns:
      anon_id, matched_name
    containing ONLY uniquely matched records.
    """
    qi = _quasi_identifiers(anon_df, aux_df)
    if not qi:
        return pd.DataFrame(columns=["anon_id", "matched_name"])

    # Rows w/o quasi-identifiers can't be matched
    anon = anon_df.dropna(subset=qi)
    aux = aux_df.dropna(subset=qi)

    # Keep  quasi-identifier combinations that occur exactly once per side
    anon_unique = anon  
    aux_unique = aux[~aux.duplicated(subset=qi, keep=False)]

    merged = anon_unique[["anon_id"] + qi].merge(
        aux_unique[["name"] + qi], on=qi, how="inner"
    )

    return (
        merged.rename(columns={"name": "matched_name"})[["anon_id", "matched_name"]]
        .reset_index(drop=True)
    )


def deanonymization_rate(matches_df, anon_df):
    """
    Compute the fraction of anonymized records
    that were uniquely re-identified.
    """
    #empty
    if len(anon_df) == 0:
        return 0.0
    #has contents
    return matches_df["anon_id"].nunique() / len(anon_df)
