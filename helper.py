import pandas as pd
from ordered_set import OrderedSet #4.0.2


def choose_label(df_per_row):
    if df_per_row['Preferred Label'] == df_per_row['label']:
        return df_per_row['Preferred Label']
    elif pd.isna(df_per_row['Preferred Label']) & pd.isna(df_per_row['label']):
        return pd.np.nan
    elif pd.isna(df_per_row['Preferred Label']):
        if 'DOID' in df_per_row['label']:
            return pd.np.nan
        else:
            return df_per_row['label']
    elif pd.isna(df_per_row['label']):
        if 'DOID' in df_per_row['Preferred Label']:
            return pd.np.nan
        else:
            return df_per_row['Preferred Label']
    else:
        if len(df_per_row['Preferred Label']) < len(df_per_row['label']):
            return df_per_row['Preferred Label']
        else:
            return df_per_row['label']

def temporally_ordered_list_join(terms_per_date):
    temporally_ordered_terms = []
    for terms in terms_per_date:
        temporally_ordered_terms.append(terms)
    return temporally_ordered_terms