from __future__ import annotations

import argparse
import os
from typing import Any

import numpy as np
import tensorflow as tf
from utils.load_al_madi import load_data_al_madi_2021
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler

import config as C


COL_TO_ID_MAPPING = {
    'accuracy': 0,
    'subjective_difficulty': 1,
    'code_snippet': 2,
    'participant_id': 3,
}


def train_test_split_by_code_snippets(
        code_snippet_ids: set[str],
        X: np.array,
        problem_setting: int,
        fold_4: bool = False,
) -> list[list[list[Any]]]:
    print('Splitting data by code snippet'.center(79, '~'))
    train_test_splits = []
    X[:, COL_TO_ID_MAPPING['subjective_difficulty']] = (
        X[:, COL_TO_ID_MAPPING['subjective_difficulty']].astype(int) > np.median(X[:, COL_TO_ID_MAPPING['subjective_difficulty']].astype(int))
    ).astype(int)
    if fold_4:
        cs_id_folds = [
            ['10-117-V1', '10-117-V2', '10-181-V2', '10-258-V1', '10-258-V2', '10-258-V3'],
            ['10-928-V1', '10-928-V2', '142-929-V1', '142-929-V2', '189-1871-V1', '189-1871-V2'],
            ['366-143-V1', '49-510-V1', '49-510-V2', '49-76-V3', 'A1117-2696', 'A1117-384'],
            ['A189-895', 'A34-6', 'A49-7086', 'A49-600', 'A84-600'],
        ]
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
        cs_idxs = set(np.where(np.isin(X[:, COL_TO_ID_MAPPING['code_snippet']], cs_id_list))[0])
        all_other_cs_idxs = all_idxs - cs_idxs
        train_test_splits.append(
            [list(all_other_cs_idxs), list(cs_idxs)],
        )

    return train_test_splits


def train_test_split_by_participants(
        participant_ids: set[str],
        X: np.array,
        problem_setting: str,
        fold_4: bool = False
) -> list[list[list[Any]]]:
    print('Splitting data'.center(79, '~'))
    seen: set[str] = set()
    train_test_splits = []
    X[:, COL_TO_ID_MAPPING['subjective_difficulty']] = (
        X[:, COL_TO_ID_MAPPING['subjective_difficulty']].astype(int) >
        np.median(X[:, COL_TO_ID_MAPPING['subjective_difficulty']].astype(int))
    ).astype(int)
    run = True
    if fold_4:
        kfold = KFold(n_splits=4, shuffle=True, random_state=42)
        for train_ids, test_ids in kfold.split(list(participant_ids)):
            all_other_cs_idxs = np.isin(X[:, COL_TO_ID_MAPPING['participant_id']], np.array(list(participant_ids))[train_ids])
            cs_idxs = np.isin(X[:, COL_TO_ID_MAPPING['participant_id']], np.array(list(participant_ids))[test_ids])
            train_test_splits.append([list(all_other_cs_idxs), list(cs_idxs)])

    else:
        while run:
            cs_id_list = []
            add_ids = False
            for cs_id in sorted(participant_ids, reverse=True):
                if len(seen) == len(participant_ids):
                    run = False
                    break
                if cs_id in seen:
                    continue
                all_idxs = set(range(0, len(X)))
                cs_id_list.append(cs_id)
                if len(np.unique(X[np.isin(X[:, COL_TO_ID_MAPPING['participant_id']], cs_id_list)][:, COL_TO_ID_MAPPING[problem_setting]])) == 2:
                    cs_idxs = set(
                        np.where(np.isin(X[:, COL_TO_ID_MAPPING['participant_id']], cs_id_list))[0],
                    )
                    all_other_cs_idxs = all_idxs - cs_idxs
                    seen.update(cs_id_list)
                    add_ids = True
                    break
                elif len(np.unique(X[np.isin(X[:, COL_TO_ID_MAPPING['participant_id']], cs_id_list)][:, COL_TO_ID_MAPPING[problem_setting]])) == 1:
                    continue
                else:
                    raise ValueError('got no labels')
            if add_ids:
                train_test_splits.append([list(all_other_cs_idxs), list(cs_idxs)])

    return train_test_splits


def configure_gpu(gpu: str) -> None:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    config = tf.compat.v1.ConfigProto(log_device_placement=True)
    config.gpu_options.per_process_gpu_memory_fraction = .5
    config.gpu_options.allow_growth = True
    tf.compat.v1.Session(config=config)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', '-g', type=int, default=-1)
    parser.add_argument(
        '--problem-setting', type=str, required=True,
        choices=['accuracy', 'subjective_difficulty'],
    )
    parser.add_argument(
        '--split', type=str, required=True,
        choices=['subject', 'code-snippet'],
    )
    parser.add_argument('--less-fold', action='store_true')
    args = parser.parse_args()
    X, y = load_data_al_madi_2021()
    X = np.pad(X, ((0, 0), (0, 4)))
    X = X.astype(float)

    # get train_test_split (s.t. 1 cs is always left out from training entirely)
    code_snippet_ids = set(y[:, COL_TO_ID_MAPPING['code_snippet']])
    participant_ids = set(y[:, COL_TO_ID_MAPPING['participant_id']])

    if args.split == 'code-snippet':
        train_test_splits = train_test_split_by_code_snippets(
            code_snippet_ids,
            y,
            args.problem_setting,
            args.less_fold,
        )
    elif args.split == 'subject':
        train_test_splits = train_test_split_by_participants(
            participant_ids,
            y,
            args.problem_setting,
            args.less_fold,
        )
    else:
        raise NotImplementedError(f'{args.split=}')

    os.makedirs(
        f'{C.results_dir}/cc-{args.split}-{args.problem_setting}',
        exist_ok=True,
    )
    cvscores = []
    aucscores = []
    print(' start evaluation '.center(79, '~'))
    for fold, (train, test) in enumerate(train_test_splits):

        # select what to predict (i.e. task outcome => accuracy, or subj_difficulty)
        code_snip_freqs_dict = {}
        num_user = len(np.unique(y[train][:, COL_TO_ID_MAPPING['participant_id']]))
        for code_snip in np.unique(y[train][:, COL_TO_ID_MAPPING['code_snippet']]):
            code_snip_freqs_dict[code_snip] = {}
            code_snip_len = 0
            for subj_id in np.unique(y[train][:, COL_TO_ID_MAPPING['participant_id']]):
                logical_and = np.logical_and(
                    np.isin(y[train][:, COL_TO_ID_MAPPING['code_snippet']], code_snip),
                    np.isin(y[train][:, COL_TO_ID_MAPPING['participant_id']], subj_id),
                )
                x_tmp = X[train][logical_and]
                num_fix = len(x_tmp)
                if num_fix == 0:
                    continue
                elif num_fix > code_snip_len:
                    code_snip_len = num_fix
                for tok_idx, tok_feat in enumerate(x_tmp):
                    try:
                        code_snip_freqs_dict[code_snip][f'fix_prob_{tok_idx}'] += int(tok_feat[3] > 0)
                        code_snip_freqs_dict[code_snip][f'single_prob_{tok_idx}'] += int(tok_feat[1] > 0)
                        code_snip_freqs_dict[code_snip][f'two_prob_{tok_idx}'] += int(
                                (tok_feat[3] > 0) and
                                (tok_feat[1] == 0),
                        )
                        code_snip_freqs_dict[code_snip][f'skip_prob_{tok_idx}'] += int(
                            sum(tok_feat == 0) == 8,
                        )
                    except KeyError:
                        code_snip_freqs_dict[code_snip][f'fix_prob_{tok_idx}'] = 0
                        code_snip_freqs_dict[code_snip][f'single_prob_{tok_idx}'] = 0
                        code_snip_freqs_dict[code_snip][f'two_prob_{tok_idx}'] = 0
                        code_snip_freqs_dict[code_snip][f'skip_prob_{tok_idx}'] = 0

            for tok_idx in range(code_snip_len):
                code_snip_freqs_dict[code_snip][f'fix_prob_{tok_idx}'] /= num_user
                code_snip_freqs_dict[code_snip][f'single_prob_{tok_idx}'] /= num_user
                code_snip_freqs_dict[code_snip][f'two_prob_{tok_idx}'] /= num_user
                code_snip_freqs_dict[code_snip][f'skip_prob_{tok_idx}'] /= num_user

        prob_feat_arrs = np.zeros(shape=(0, 4))
        if args.split == 'subject':
            for code_snip in np.unique(y[:, COL_TO_ID_MAPPING['code_snippet']]):
                for subj_id in np.unique(y[:, COL_TO_ID_MAPPING['participant_id']]):
                    logical_and = np.logical_and(
                        np.isin(y[:, COL_TO_ID_MAPPING['code_snippet']], code_snip),
                        np.isin(y[:, COL_TO_ID_MAPPING['participant_id']], subj_id),
                    )
                    x_tmp = X[logical_and]
                    if len(x_tmp) == 0:
                        continue
                    else:
                        prob_feat = [
                            (
                                code_snip_freqs_dict[code_snip][f'fix_prob_{tok_idx}'],
                                code_snip_freqs_dict[code_snip][f'single_prob_{tok_idx}'],
                                code_snip_freqs_dict[code_snip][f'two_prob_{tok_idx}'],
                                code_snip_freqs_dict[code_snip][f'skip_prob_{tok_idx}'],
                            )
                            for tok_idx in range(len(x_tmp))
                        ]
                        prob_feat_arr = np.array(prob_feat)
                        X[logical_and][:, 4:8] = prob_feat_arr
                        prob_feat_arrs = np.vstack([prob_feat_arrs, prob_feat_arr])

            X[:, 4:8] = prob_feat_arrs

        if args.problem_setting == 'accuracy':
            y_label = np.array(y[:, COL_TO_ID_MAPPING[args.problem_setting]], dtype=int)
        elif args.problem_setting == 'subjective_difficulty':
            problem_setting = np.array(y[:, COL_TO_ID_MAPPING[args.problem_setting]], dtype=int)
            problem_setting_mean = np.median(problem_setting)
            y_label = np.array(problem_setting > problem_setting_mean, dtype=int)
        else:
            raise NotImplementedError(f'{args.problem_setting=}')

        # scale fixation duration
        scaler = MinMaxScaler()
        for col in range(X.shape[-1]):
            X[train][:, col] = scaler.fit_transform(X[train][:, col].reshape(-1, 1)).squeeze()
            X[test][:, col] = scaler.transform(X[test][:, col].reshape(-1, 1)).squeeze()

        clf = RandomForestClassifier()
        clf.fit(X[train], y_label[train])
        predictions = clf.predict_proba(X[test])[:, 1]
        accuracy = clf.score(X[test], y_label[test])
        _roc_auc_score = roc_auc_score(y_label[test], predictions)
        cvscores.append(accuracy * 100)
        aucscores.append(_roc_auc_score * 100)

        print(f'cvscores: {np.mean(cvscores):.3f}+/-{np.std(cvscores)/np.sqrt(len(cvscores)):.3f}')
        print(f'aucscores: {np.mean(aucscores):.3f}+/-{np.std(aucscores)/np.sqrt(len(aucscores)):.3f}')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())