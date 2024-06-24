from __future__ import annotations

import json
from os import listdir
from os.path import isfile
from os.path import join

import numpy as np
import pandas as pd

import config as C


def participant_fixations_to_df(path_to_txt_file: str) -> pd.DataFrame:
    df_participant_fixations = pd.read_csv(path_to_txt_file, sep='\t', low_memory=False)

    # convert str to list
    try:
        df_participant_fixations['CURRENT_FIX_INTEREST_AREAS'] = df_participant_fixations[
            'CURRENT_FIX_INTEREST_AREAS'
        ].apply(json.loads)
    except KeyError:
        pass

    # add accuracycolumn, accuracy defines whether the comprehension task was answered correctly
    # 1 == correct, 0 == wrong
    try:
        df_participant_fixations['accuracy'] = df_participant_fixations.apply(
            lambda row: 1 if (row.correct_option == row.KEY_STROKE) else 0, axis=1,
        )
    except KeyError:
        pass

    # keep only fixations on code snippet (remove those on task / answer options)
    try:
        df_participant_fixations = df_participant_fixations[
            df_participant_fixations['CURRENT_FIX_X'].between(430, 1400)
        ]
    except KeyError:
        pass

    return df_participant_fixations


# Fritz 2014
def load_data_fritz_2014() -> tuple[np.array, np.array]:
    fix_report_parent_dir = f'{C.code_comp_dir}/Fix-Sac-Reports/'
    res_parent_dir = 'raw-fixation-reports/'
    fix_report_files = [
        f for idx in range(1, 3) for f in listdir(f'{fix_report_parent_dir}/fix_report_{idx}_all')
        if isfile(join(f'{fix_report_parent_dir}/fix_report_{idx}_all', f))
    ]

    skip_snippets = [
        '10-181-V3',
        '10-928-V3',
        '142-929-V3',
        '189-1871-V3',
        # '369-1404-V1', # regenerate reports using IAs for first round participants
        '369-1404-V2',
        '49-76-V1',  # missing IA file
        '49-76-V2',  # missing IA file
    ]

    X_data = []
    y_labels = []
    for fix_report_file in fix_report_files:
        participant_id = fix_report_file[-8:-4]
        print(f'Loading data for {participant_id=}')

        sac_report_file = '_'.join(['sac'] + fix_report_file.split('_')[1:])
        res_report_file = '_'.join(
            fix_report_file.split('_')[:2] + [fix_report_file.split('_')[3].split('.')[0]+'.txt'],
        )
        # get processed Participant Fixation DataFrame
        df_participant_results = participant_fixations_to_df(
            f'{res_parent_dir}/{res_report_file}',
        )
        try:
            df_participant_fixations = participant_fixations_to_df(
                f'{fix_report_parent_dir}/fix_report_1_all/{fix_report_file}',
            )
            df_participant_saccades = participant_fixations_to_df(
                f'{fix_report_parent_dir}/sac_report_1_all/{sac_report_file}',
            )
        except FileNotFoundError:
            df_participant_fixations = participant_fixations_to_df(
                f'{fix_report_parent_dir}/fix_report_2_all/{fix_report_file}',
            )
            df_participant_saccades = participant_fixations_to_df(
                f'{fix_report_parent_dir}/sac_report_2_all/{sac_report_file}',
            )

        # check if data from 2nd experiment round exists for participant and add it

        # get Participant Code Snippet IDs and remove snippets to be skipped
        participant_code_snippet_ids = set(df_participant_fixations['code_snippet_id'])
        participant_code_snippet_ids = participant_code_snippet_ids - set(skip_snippets)

        for code_snippet_id in participant_code_snippet_ids:
            fix_tmp_df = df_participant_fixations.loc[
                df_participant_fixations.code_snippet_id == code_snippet_id
            ].copy()
            sac_tmp_df = df_participant_saccades.loc[
                df_participant_saccades.code_snippet_id == code_snippet_id
            ].copy()
            res_tmp_df = df_participant_results.loc[
                df_participant_results.code_snippet_id == code_snippet_id
            ].copy()
            labels = [
                res_tmp_df.accuracy.iloc[0],
                res_tmp_df.CS_SUBJ_DIFFICULTY.iloc[0],
                code_snippet_id,
                participant_id,
            ]
            batch = 0
            seconds_to_use = 10000
            while True:
                fix_condition = np.logical_and(
                    5000 * batch < fix_tmp_df.CURRENT_FIX_END.values,
                    fix_tmp_df.CURRENT_FIX_END.values < seconds_to_use + 5000 * batch,
                )
                sac_condition = np.logical_and(
                    5000 * batch < sac_tmp_df.CURRENT_SAC_END_TIME.values,
                    sac_tmp_df.CURRENT_SAC_END_TIME.values < seconds_to_use + 5000 * batch,
                )
                if len(fix_tmp_df[fix_condition]) == 0:
                    break
                features = [
                    len(sac_tmp_df[sac_condition])/60,
                    sac_tmp_df.CURRENT_SAC_DURATION[sac_condition].sum()/60,
                    sac_tmp_df.CURRENT_SAC_DURATION[sac_condition].mean()/60,
                    sac_tmp_df.CURRENT_SAC_DURATION[sac_condition].median()/60,
                    sac_tmp_df.CURRENT_SAC_DURATION[sac_condition].std()/60,
                    len(fix_tmp_df[fix_condition])/60,
                    fix_tmp_df.CURRENT_FIX_DURATION[fix_condition].sum()/60,
                    fix_tmp_df.CURRENT_FIX_DURATION[fix_condition].mean()/60,
                    fix_tmp_df.CURRENT_FIX_DURATION[fix_condition].median()/60,
                    fix_tmp_df.CURRENT_FIX_DURATION[fix_condition].std()/60,
                    fix_tmp_df.CURRENT_FIX_PUPIL[fix_condition].max(),
                    fix_tmp_df.CURRENT_FIX_PUPIL[fix_condition].min(),
                    fix_tmp_df.CURRENT_FIX_PUPIL[fix_condition].mean(),
                    fix_tmp_df.CURRENT_FIX_PUPIL[fix_condition].median(),
                    fix_tmp_df.CURRENT_FIX_PUPIL[fix_condition].std(),
                ]
                X_data.append(features)
                y_labels.append(labels)
                batch += 1

    return np.array(X_data), np.array(y_labels)
