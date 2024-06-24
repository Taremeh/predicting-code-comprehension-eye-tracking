from __future__ import annotations

import json
from os import listdir
from os.path import isfile
from os.path import join
from pathlib import Path

import numpy as np
import pandas as pd
from data_processing import get_IA_df


def participant_fixations_to_df(path_to_txt_file: str) -> pd.DataFrame:
    df_participant_fixations = pd.read_csv(path_to_txt_file, sep='\t')

    # convert str to list
    df_participant_fixations['CURRENT_FIX_INTEREST_AREAS'] = df_participant_fixations[
        'CURRENT_FIX_INTEREST_AREAS'
    ].apply(json.loads)

    # add accuracycolumn, accuracy defines whether the comprehension task was answered correctly
    # 1 == correct, 0 == wrong
    df_participant_fixations['accuracy'] = df_participant_fixations.apply(
        lambda row: 1 if (row.correct_option == row.KEY_STROKE) else 0, axis=1,
    )

    # keep only fixations on code snippet (remove those on task / answer options)
    df_participant_fixations = df_participant_fixations[
        df_participant_fixations['CURRENT_FIX_X'].between(430, 1400)
    ]

    return df_participant_fixations


# Al Madi 2021
# XXX for new code snippets fxiations on token probabilities not applicable
def load_data_al_madi_2021() -> tuple[np.array, np.array]:
    fix_report_parent_dir = './raw-fixation-reports'
    fix_report_parent_dir_additional = './raw-fixation-reports-add'
    fix_report_files = [
        f for f in listdir(fix_report_parent_dir) if isfile(join(fix_report_parent_dir, f))
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
    code_snip_len_dict = {}
    for fix_report_file in fix_report_files:
        participant_id = fix_report_file[-8:-4]

        # get processed Participant Fixation DataFrame
        df_participant_fixations = participant_fixations_to_df(
            f'{fix_report_parent_dir}/{fix_report_file}',
        )

        # check if data from 2nd experiment round exists for participant and add it
        if Path(f'./{fix_report_parent_dir_additional}/fix_report_{participant_id}.txt').exists():
            df_participant_fixations_2 = participant_fixations_to_df(
                f'./{fix_report_parent_dir_additional}/fix_report_{participant_id}.txt',
            )
            df_participant_fixations = pd.concat(
                [df_participant_fixations, df_participant_fixations_2], join='inner',
            )

        # get Participant Code Snippet IDs and remove snippets to be skipped
        participant_code_snippet_ids = set(df_participant_fixations['code_snippet_id'])
        participant_code_snippet_ids = participant_code_snippet_ids - set(skip_snippets)

        for code_snippet_id in participant_code_snippet_ids:
            try:
                ia_df = get_IA_df(code_snippet_id)
            except FileNotFoundError:
                # IAs not found
                continue
            if code_snippet_id in code_snip_len_dict.keys():
                assert code_snip_len_dict[code_snippet_id] == len(ia_df)
            else:
                code_snip_len_dict[code_snippet_id] = len(ia_df)
            tmp_df = df_participant_fixations.loc[
                df_participant_fixations.code_snippet_id == code_snippet_id
            ].copy()
            tmp_df['cur_fix_max'] = tmp_df.CURRENT_FIX_NEAREST_INTEREST_AREA.cummax()
            code_snippet_feat_dict = {}
            labels = [
                tmp_df.accuracy.iloc[0],
                tmp_df.CS_SUBJ_DIFFICULTY.iloc[0],
                code_snippet_id,
                participant_id,
            ]
            for cur_fix_ia in tmp_df.CURRENT_FIX_NEAREST_INTEREST_AREA.unique():
                tmp_cur_fix_ia_df = tmp_df.loc[tmp_df.CURRENT_FIX_NEAREST_INTEREST_AREA == cur_fix_ia]
                if len(tmp_cur_fix_ia_df) == 1:
                    single_fix_dur = tmp_cur_fix_ia_df.iloc[0].CURRENT_FIX_DURATION
                    first_fix_dur = tmp_cur_fix_ia_df.iloc[0].CURRENT_FIX_DURATION
                else:
                    single_fix_dur = 0
                    first_fix_dur = tmp_cur_fix_ia_df.iloc[0].CURRENT_FIX_DURATION
                gaze_duration = tmp_cur_fix_ia_df[tmp_cur_fix_ia_df.CURRENT_FIX_NEAREST_INTEREST_AREA >= tmp_cur_fix_ia_df.cur_fix_max].CURRENT_FIX_DURATION.sum()
                total_time = tmp_cur_fix_ia_df.CURRENT_FIX_DURATION.sum()
                code_snippet_feat_dict[cur_fix_ia] = [
                    first_fix_dur, single_fix_dur, gaze_duration, total_time,
                ]
            for _, row in ia_df.iterrows():
                try:
                    # normalize by length of character
                    X_data.append(code_snippet_feat_dict[row.IA_ID]/len(row.LABEL))
                    y_labels.append(labels)
                except IndexError:
                    X_data.append([0, 0, 0, 0])
                    y_labels.append(labels)

    return np.array(X_data), np.array(y_labels)
