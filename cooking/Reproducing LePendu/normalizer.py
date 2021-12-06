import re
import numpy as np
import pandas as pd


def get_rxnorm_normalizer():
    """
    Deriving normalizer from RXN dataset
    The __RxNORM terminology__ will be later used to normalize the drug having the trade name (e.g. Vioxx into its primary active ingredient)
    
    data source: https://www.nlm.nih.gov/research/umls/rxnorm/index.html
    columnar legends: https://www.nlm.nih.gov/research/umls/rxnorm/docs/techdoc.html#s12_1
    """

    # read the RXNorm dataset
    df_RXNATOMARCHIVE = pd.read_csv('data_pool/RXNorm/RxNorm_full_11012021/rrf/RXNATOMARCHIVE.RRF', delimiter='|', header=None)
    df_RXNCONSO = pd.read_csv('data_pool/RXNorm/RxNorm_full_11012021/rrf/RXNCONSO.RRF', delimiter='|', header=None)

    # Preprocess
    df_RXNATOMARCHIVE[6] = df_RXNATOMARCHIVE[6].astype(str)
    df_normalizer = df_RXNATOMARCHIVE[[2, 6]].merge(df_RXNCONSO[[11, 13, 14]], left_on=6, right_on=13, how='left')
    df_normalizer = df_normalizer[df_normalizer[11]=='RXNORM'].rename(columns={2:'drug_name',14:'ingredients_name'})[['drug_name','ingredients_name']].reset_index()

    df_normalizer['drug_name_modified'] = df_normalizer.apply(lambda x: preprocess(x['drug_name']), axis=1)
    df_normalizer['ingredients_name_modified'] = df_normalizer.apply(lambda x: preprocess(x['ingredients_name']), axis=1)

    df_normalizer['drug_name_guessed_list'] = df_normalizer.apply(lambda x: set(x['drug_name_modified'].split(' ')) - set(x['ingredients_name_modified'].split(' ')), axis=1)
    df_normalizer['ingredient_name_guessed_list'] = df_normalizer.apply(lambda x: set(x['ingredients_name_modified'].split(' ')) - set(x['drug_name_modified'].split(' ')), axis=1)

    df_normalizer['drug_name_guessed'] = df_normalizer.apply(lambda x: join_the_result_back(x['drug_name_guessed_list'], x['drug_name_modified']), axis=1)
    df_normalizer['ingredient_name_guessed'] = df_normalizer.apply(lambda x: join_the_result_back(x['ingredient_name_guessed_list'], x['drug_name_modified']), axis=1)

    # Clean
    ingredient_name_guessed_to_ignore = 'cartridge|autoinjector|system|injection|per injection|medicated|product|product liquid|\
                                         insert|autoinjector|product injectable|suspension|per apap|film|per syrup|pen injector|\
                                         hbr in|isdn|asa apap|jet injector|patch|cartridge in|cartridge dental in|hcl apap|autoinjector per|\
                                         halls|cartridge hcl|hbr hcl apap|per injection meq|metered dose actuat|release delayed granules for|\
                                         per inhalation|armour|injector per pen|syrup|system hr|pen per injector|unt injection|granules for|\
                                         ointment|pen injector in|twicedaily medicated|patch per|lotion|per hcl gm|patch medicated|preparation|dental|\
                                         injection gm in|allergenic|unt inhalation|mapap|syringe in|injection meq|cartridge dental|insert gm|per injection gm\
                                         |inch x|extract|toothpaste|quadrivalent|chicken|injection meq in|pot injection|injector in|\
                                         reconcile|injector in pen|suspension per|suspension inhalation in|as injection|betaa|tablet|\
                                         oral|adult|human|solution|weekly|day|year|equivalent|mcg|per g|billion|per unt|with|as per|as unt|\
                                         release extended|concentrated|type|powder|pack|at least|elemental per|normal|as per hcl|spray|\
                                         nph per|hour|maximum strength|moisturizing|suspension for|titration|per'

    df_normalizer = df_normalizer[(df_normalizer['drug_name_guessed'].str.len() > 3) & \
                                  (df_normalizer['ingredient_name_guessed'].str.len() > 3) & \
                                  (~df_normalizer['ingredient_name_guessed'].str.contains(ingredient_name_guessed_to_ignore))]

    df_normalizer_grouped = df_normalizer[['drug_name_guessed','ingredient_name_guessed']].groupby(['drug_name_guessed']).max()
    normalizer_dict = df_normalizer_grouped.to_dict()['ingredient_name_guessed']

    print('Produced', len(normalizer_dict), 'keys as the RXNorm normalizer dict.')
    
    return normalizer_dict


def preprocess(input_string):

    input_string = input_string.lower()
    to_be_removed_terms = ['mg/ml', 'mg', 'ml']
    
    square_bracket_idx = input_string.find('[')
    if square_bracket_idx > -1:
        input_string = input_string[:(square_bracket_idx-1)]

    for term in to_be_removed_terms:
        input_string = input_string.replace(term,"")

    input_string = re.sub(r'[^a-z ]', '', input_string)
    input_string = re.sub(' +', ' ', input_string)

    input_string = input_string.strip()

    return input_string


def join_the_result_back(set_of_results, full_string):

    list_of_results = list(set_of_results)
    
    if len(list_of_results) > 0:
    
        result_index = []
        for result in list_of_results:
            result_index.append(full_string.find(result))

        array = np.array(result_index)
        order = array.argsort()
        ranks = list(order.argsort())

        result_list = []
        for i in range(max(ranks)+1):
            result_list.append(list_of_results[ranks.index(i)])

        return ' '.join(result_list)
    
    else:
        return ''