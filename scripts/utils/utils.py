import glob
import os
from typing import Any

import numpy as np
import pandas as pd
from typing import Set
from sklearn.model_selection import KFold


def load_data(mode="bimodal") -> pd.DataFrame:
    if mode == "code":
        path = './processed_data/code_only'
    else:
        path = './processed_data/grouped_by_code_snippets'

    all_files = glob.glob(os.path.join(path, '*.pkl'))

    df = pd.DataFrame()
    for filepath in all_files:
        print(filepath)
        df = pd.concat([df, pd.read_pickle(filepath)])

    df = df.reset_index(drop=True)
    df = df.fillna(0)

    return df


# load data split by code snippet stratified by label
def train_test_split_by_code_snippets(
        code_snippet_ids: set,
        X: pd.DataFrame,
        problem_setting: int,
        fold_4: bool = True,
) -> list:
    if os.path.exists("train_test_splits_by_code_snippet.npy") and not fold_4:
        print('Loading existing split "train_test_splits_by_code_snippet.npy"'.center(79, '~'))
        train_test_splits = np.load('train_test_splits_by_code_snippet.npy', allow_pickle=True).tolist()
        return train_test_splits

    print('Splitting data by code snippet'.center(79, '~'))
    train_test_splits = []
    X['subjective_difficulty'] = (
            X['subjective_difficulty'] > X['subjective_difficulty'].median()
    ).astype(int)
    if fold_4:
        print('4 fold cv')
        cs_id_folds = [
            ['10-117-V1', '10-117-V2', '10-181-V2', '10-258-V1', '10-258-V2', '10-258-V3'],
            ['10-928-V1', '10-928-V2', '142-929-V1', '142-929-V2', '189-1871-V1', '189-1871-V2'],
            ['366-143-V1', '49-510-V1', '49-510-V2', '49-76-V3', 'A1117-2696', 'A1117-384'],
            ['A189-895', 'A34-6', 'A49-7086', 'A49-600', 'A84-600'],
        ]
        all_idxs = set(range(0, len(X)))
        for cs_id_list in cs_id_folds:
            cs_idxs = set(X[np.isin(X.iloc[:, 0], cs_id_list)].index.tolist())
            all_other_cs_idxs = all_idxs - cs_idxs
            train_test_splits.append([list(all_other_cs_idxs), list(cs_idxs)])

        print(train_test_splits)
    else:
        cs_id_folds = [
            ['10-117-V1', '10-117-V2', '10-181-V2'],
            ['10-258-V1', '10-258-V2', '10-258-V3'],
            ['10-928-V1', '10-928-V2'],
            ['142-929-V1', '142-929-V2', '189-1871-V1', '189-1871-V2'],
            ['366-143-V1', '49-510-V1', '49-510-V2'],
            ['49-76-V3', 'A1117-2696', 'A1117-384'],
            ['A189-895', 'A34-6', 'A49-7086'],
            ['A49-600', 'A84-600'],
        ]
        all_idxs = set(range(0, len(X)))
        for cs_id_list in cs_id_folds:
            cs_idxs = set(X[np.isin(X.iloc[:, 0], cs_id_list)].index.tolist())
            all_other_cs_idxs = all_idxs - cs_idxs
            train_test_splits.append([list(all_other_cs_idxs), list(cs_idxs)])

        np.save('train_test_splits_by_code_snippet.npy', train_test_splits)
        print(train_test_splits)

    return train_test_splits


# load data split by participant stratified by label
def train_test_split_by_participants(
        participant_ids: set,
        X: pd.DataFrame,
        problem_setting: str,
        fold_4: bool = True,
) -> list:
    if problem_setting == "subjective_difficulty_score":
        problem_setting = "subjective_difficulty"

    if os.path.exists("train_test_splits_by_participant.npy") and not fold_4:
        print('Loading existing split "train_test_splits_by_participant.npy"'.center(79, '~'))
        train_test_splits = np.load('train_test_splits_by_participant.npy', allow_pickle=True).tolist()
        print(train_test_splits)
        return train_test_splits

    print('Splitting data'.center(79, '~'))
    run = True
    seen: set = set()
    train_test_splits = []
    X['subjective_difficulty'] = (
            X['subjective_difficulty'] > X['subjective_difficulty'].mean()
    ).astype(int)
    participant_ids = list(participant_ids)
    if fold_4:
        kfold = KFold(n_splits=4, shuffle=True, random_state=42)
        for train_ids, test_ids in kfold.split(participant_ids):

            all_other_cs_idxs = set(X[np.isin(X.iloc[:, 1], np.array(participant_ids)[train_ids])].index.tolist())
            cs_idxs = set(X[np.isin(X.iloc[:, 1], np.array(participant_ids)[test_ids])].index.tolist())
            train_test_splits.append([list(all_other_cs_idxs), list(cs_idxs)])

        print(train_test_splits)
    else:
        while run:
            cs_id_list = []
            add_ids = False
            add_ids = False
            for cs_id in participant_ids:
                if len(seen) == len(participant_ids):
                    run = False
                    break

                if cs_id in seen:
                    continue
                all_idxs = set(range(0, len(X)))
                cs_id_list.append(cs_id)

                if len(X[np.isin(X.iloc[:, 1], cs_id_list)][problem_setting].unique()) == 2:
                    cs_idxs = set(X[np.isin(X.iloc[:, 1], cs_id_list)].index.tolist())
                    all_other_cs_idxs = all_idxs - cs_idxs
                    seen.update(cs_id_list)
                    add_ids = True
                    break
                elif len(X[np.isin(X.iloc[:, 1], cs_id_list)][problem_setting].unique()) == 1:
                    continue
                else:
                    raise ValueError('got no labels')
            if add_ids:
                train_test_splits.append([list(all_other_cs_idxs), list(cs_idxs)])

        np.save('train_test_splits_by_participant.npy', train_test_splits)
        print(train_test_splits)

    return train_test_splits
