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


# Hanaka 2021
def load_data_hanaka_2021() -> tuple[np.array, np.array]:
    fix_report_parent_dir = join(C.code_comp_path, 'Fix-Sac-Reports/')
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
    aoi_df = pd.read_csv(join(C.code_comp_path, 'AOI_IA_IDXS.csv'), delimiter=';')
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

            def eucl_distance(x_1: float, y_1: float, x_2: float, y_2: float) -> float:
                return np.sqrt((x_1 - x_2)**2 + (y_1 - y_2)**2)
            fix_tmp_df['fix_dist'] = eucl_distance(
                fix_tmp_df.CURRENT_FIX_X.shift(),
                fix_tmp_df.CURRENT_FIX_Y.shift(),
                fix_tmp_df.loc[1:, 'CURRENT_FIX_X'],
                fix_tmp_df.loc[1:, 'CURRENT_FIX_Y'],
            )
            aoi_code_snippet_df = aoi_df.loc[aoi_df.code_snippet_id == code_snippet_id]
            # # for aoi 1, 2, 3
            # len(fix_tmp_df[fix_condition]),
            # fix_tmp_df.CURRENT_FIX_DURATION[fix_condition].mean(),
            # fix_tmp_df.CURRENT_FIX_DURATION[fix_condition].sum(),
            # sac_tmp_df.CURRENT_SAC_DURATION[sac_condition].mean(),
            # sum of scanpath length
            # mean of scanpath length
            aoi_feature = []
            for line in range(20):
                try:
                    aoi_df_row = aoi_code_snippet_df.iloc[line]
                except IndexError:
                    aoi_feature.append([0, 0, 0, 0, 0, 0])
                    continue
                fix_aoi_cond = np.logical_and(
                    (aoi_df_row.IA_idx_start <= fix_tmp_df.CURRENT_FIX_INTEREST_AREA_ID.replace('.', 0).values.astype(int)),
                    (fix_tmp_df.CURRENT_FIX_INTEREST_AREA_ID.replace('.', 0).values.astype(int) < aoi_df_row.IA_idx_end),
                )
                sac_aoi_cond = np.logical_and(
                    np.logical_and(
                        (aoi_df_row.IA_idx_start <= sac_tmp_df.PREVIOUS_SAC_NEAREST_END_INTEREST_AREA.replace('.', 0).values.astype(int)),
                        (sac_tmp_df.PREVIOUS_SAC_NEAREST_END_INTEREST_AREA.replace('.', 0).values.astype(int) < aoi_df_row.IA_idx_end),
                    ),
                    np.logical_and(
                        (aoi_df_row.IA_idx_start <= sac_tmp_df.NEXT_SAC_NEAREST_END_INTEREST_AREA.replace('.', 0).values.astype(int)),
                        (sac_tmp_df.NEXT_SAC_NEAREST_END_INTEREST_AREA.replace('.', 0).values.astype(int) < aoi_df_row.IA_idx_end),
                    ),
                )
                if (len(fix_tmp_df[fix_aoi_cond]) == 0) and (len(sac_tmp_df[sac_aoi_cond]) == 0):
                    aoi_feature.append([0, 0, 0, 0, 0, 0])
                elif len(fix_tmp_df[fix_aoi_cond]) == 0:
                    aoi_feature.append([
                        0,
                        0,
                        0,
                        sac_tmp_df.CURRENT_SAC_DURATION[sac_aoi_cond].mean(),
                        0,
                        0,
                    ])
                elif len(sac_tmp_df[sac_aoi_cond]) == 0:
                    aoi_feature.append(
                        [
                            len(fix_tmp_df[fix_aoi_cond]),  # 1
                            fix_tmp_df.CURRENT_FIX_DURATION[fix_aoi_cond].mean(),  # 2
                            fix_tmp_df.CURRENT_FIX_DURATION[fix_aoi_cond].sum(),  # 3
                            0,
                            np.nansum(fix_tmp_df[fix_aoi_cond].fix_dist),  # 5
                            np.nanmean(fix_tmp_df[fix_aoi_cond].fix_dist),  # 6
                        ],
                    )
                else:
                    aoi_feature.append(
                        [
                            len(fix_tmp_df[fix_aoi_cond]),  # 1
                            fix_tmp_df.CURRENT_FIX_DURATION[fix_aoi_cond].mean(),  # 2
                            fix_tmp_df.CURRENT_FIX_DURATION[fix_aoi_cond].sum(),  # 3
                            sac_tmp_df.CURRENT_SAC_DURATION[sac_aoi_cond].mean(),  # 4
                            np.nansum(fix_tmp_df[fix_aoi_cond].fix_dist),  # 5
                            np.nanmean(fix_tmp_df[fix_aoi_cond].fix_dist),  # 6
                        ],
                    )
            features = [
                len(fix_tmp_df),  # 1
                fix_tmp_df.CURRENT_FIX_DURATION.mean(),  # 2
                fix_tmp_df.CURRENT_FIX_DURATION.sum(),  # 3
                sac_tmp_df.CURRENT_SAC_DURATION.mean(),  # 4
                np.nansum(fix_tmp_df.fix_dist),  # 5
                np.nanmean(fix_tmp_df.fix_dist),  # 6
                # trans prob between aoi1 and aoi2
                # trans prob between aoi1 and aoi3
                # trans prob between aoi2 and aoi3
                # # for .1 - 1.5
                # # for 1.6 - 3.0
                # # for 3.1 - 4.5
                # # for 4.5 - 6.0
                # hor freq pow
                # vert freq pow
                # hor fp early
                # vert fp early
                # hor fp late
                # vert fp late
            ]
            features.extend(
                [feat for aoi_feat in aoi_feature for feat in aoi_feat],
            )
            X_data.append(features)
            y_labels.append(labels)

    return np.array(X_data), np.array(y_labels)
