from __future__ import annotations

import argparse
import os
import random
import time
import ast
import json
import warnings
warnings.filterwarnings("ignore")

import keras_tuner
import numpy as np
import tensorflow
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import GlobalAveragePooling1D
from tensorflow.keras.layers import GlobalMaxPooling1D
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import LSTM
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# import config as C
from utils.gpu_selection import configure_gpu
from utils.gpu_selection import select_gpu
from utils.utils import load_data
from utils.utils import train_test_split_by_code_snippets
from utils.utils import train_test_split_by_participants


EMBED_SIZE = 768
MAX_CS_EMBED_SIZE = 166

def help_roc_auc(y_true, y_pred):
    if len(np.unique(y_true)) == 1:
        return .5
    else:
        return roc_auc_score(y_true, y_pred)


def auroc(y_true, y_pred):
    return tf.py_function(help_roc_auc, (y_true, y_pred), tf.double)

class AttentionLayer(tensorflow.keras.layers.Layer):
    def __init__(self, data, window_size, lm="codebert", **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        self.data = tf.Variable(data, dtype=tf.float64, trainable=False)
        self.window_size = window_size
        self.total_window_size = 2 * window_size + 1
        self.lm_idx = 0 if lm == "codebert" else 1
        self.w = self.add_weight(
            name='attention_weight',
            shape=(self.total_window_size,EMBED_SIZE), # shape=(self.total_window_size,),
            dtype=tf.float64,
            initializer='random_normal',
            trainable=True)

    def call(self, inputs):
        inputs, code_snippet_id = inputs
        inputs = tf.cast(inputs, tf.int32)
        code_snippet_id = tf.cast(code_snippet_id, tf.int32)

        # expand dims for proper broadcasting
        indices = tf.range(-self.window_size, self.window_size + 1) + tf.expand_dims(inputs, -1)
        end_clip = MAX_CS_EMBED_SIZE - 1
        indices = tf.clip_by_value(indices, 0, end_clip)

        # bring code_snippet_id into correct shape for gather_nd indices tensor
        expanded_code_snippet_id = tf.expand_dims(code_snippet_id, axis=-1)  # shape: [?, 1, 1]
        tiled_code_snippet_id = tf.tile(expanded_code_snippet_id, [1, 1126, self.total_window_size])  # shape: [?, 1126, 1/3/5]

        lm_idx_tensor = tf.fill(tf.shape(indices), self.lm_idx)

        # Stack the tensors to get [code_snippet_id, lm_idx, indices_i] for each indices_i in indices
        indices_tensor = tf.stack([tiled_code_snippet_id, indices, lm_idx_tensor], axis=-1)

        # gather LLM embeddings in window_size
        gathered_data = tf.gather_nd(params=self.data, indices=indices_tensor)

        # multipy with attention weights and then reduce sum over the window size axis.
        output = gathered_data * self.w

        # Broadcast self.w to match the shape of gathered_data
        #expanded_w = tf.tile(tf.expand_dims(self.w, axis=1), [1, EMBED_SIZE])

        # Multiply the gathered data with the expanded weights
        #output = gathered_data * expanded_w

        output = tf.reduce_sum(output, axis=2)  # Sum over the window size axis

        return output


# main function to execute nested CV
def main() -> int:
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', '-g', type=int, default=-1)
    parser.add_argument(
        '--problem-setting', type=str, required=True,
        choices=['accuracy', 'subjective_difficulty', 'subjective_difficulty_score'],
    )
    parser.add_argument(
        '--split', type=str, required=True,
        choices=['subject', 'code-snippet'],
    )
    parser.add_argument(
        '--mode', type=str, required=True,
        choices=['bimodal', 'fixations', 'code'],
    )
    parser.add_argument('--less-fold', action='store_true')
    parser.add_argument(
        '--split-file', type=str, required=False,
    )
    parser.add_argument(
        '--fold-offset', type=int, required=False,
    )
    args = parser.parse_args()

    configure_gpu(select_gpu(25000))

    # load data
    X = load_data(args.mode)

    # get train_test_split (s.t. 1 cs is always left out from training entirely)

    code_snippet_ids = set(X['code_snippet_id'])
    participant_ids = set(X['participant_id'])

    # check if manual split file is provided
    if args.split_file is not None:
        print(f"Loading existing split {args.split_file}")
        # Load the file
        with open(f'{args.split_file}', 'r') as file:
            data = file.read().replace('\n', '')

        # Convert string to list
        data_list = ast.literal_eval(data)

        # Convert list to numpy array
        train_test_splits = np.array(data_list)
    # split by code-snippet
    elif args.split == 'code-snippet':
        train_test_splits = train_test_split_by_code_snippets(
            code_snippet_ids,
            X,
            args.problem_setting,
            args.less_fold,
        )
    # split by subject
    elif args.split == 'subject':
        train_test_splits = train_test_split_by_participants(
            participant_ids,
            X,
            args.problem_setting,
            args.less_fold,
        )
    else:
        raise NotImplementedError(f'{args.split=}')

    # select what to predict (i.e. task outcome => accuracy, or subj_difficulty)
    if args.problem_setting == 'accuracy':
        y = X.pop('accuracy')
        y = y.to_numpy()
        X.pop('subjective_difficulty')
    elif args.problem_setting == 'subjective_difficulty':
        problem_setting = X.pop('subjective_difficulty')
        problem_setting_mean = np.mean(problem_setting)
        y = np.array(problem_setting > problem_setting_mean, dtype=int)
        X.pop('accuracy')
    elif args.problem_setting == 'subjective_difficulty_score':
        y = X.pop('subjective_difficulty')
        y = y.to_numpy()
        y = (y - 1) / 4 # normalize scale 1-5 to 0-1
        X.pop('accuracy')
    else:
        raise NotImplementedError(f'{args.problem_setting=}')

    if args.fold_offset is None:
        args.fold_offset = 0

    X.pop('participant_id')
    X_code_snippet_ids = X.pop('code_snippet_id')

    # Open the CS_ID to CS_IDX mapping
    with open("CS_ID_to_IDX_MAP_5.json", 'r') as f:
        CS_ID_to_IDX_MAP = json.load(f)

    # get code_snippet embedding IDXs (instead of string IDs)
    X_code_snippet_IDXs = [CS_ID_to_IDX_MAP.get(CS_ID) for CS_ID in X_code_snippet_ids]

    X = X.to_numpy()
    seq_len = 1126

    if args.mode == "code":
        X = X.reshape(X.shape[0], seq_len, 1)
        X = np.repeat(X, 2, axis=2)
    else:
        X = X.reshape(X.shape[0], seq_len, 770+770)


    # reproducibility
    seed = 42  # fix random seed for reproducibility
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tensorflow.random.set_seed(seed)

    # define k-fold cross validation
    cvscores = []
    aucscores = []

    # train and evaluate neural network
    os.makedirs(
        f'{C.results_dir}/cc-{args.split}-{args.problem_setting}-{args.mode}{args.less_fold}',
        exist_ok=True,
    )

    for fold, (train, test) in enumerate(train_test_splits):
        if fold < args.fold_offset:
            print(f"Skipping Fold {fold}")
            continue
        # clear tensorflow keras global state memory to start with blank state at each iteration
        keras.backend.clear_session()

        # scale fixation duration
        scaler = StandardScaler()
        X[train][:, 1] = scaler.fit_transform(X[train][:, 1])
        X[test][:, 1] = scaler.transform(X[test][:, 1])

        X_code_snippet_IDXs = np.array(X_code_snippet_IDXs)

        # define model
        def build_model(hp):
            print(f'{fold=}')
            # position embedding input
            code_snippet_idxs = Input(shape=(1,))
            # IA_ID sequence
            IA_ID_sequence = pos_emb_input = Input(shape=(seq_len,))
            # fixation duration input
            fix_dur_input = Input(shape=(seq_len, 1))

            # decide whether to use graphbert or bert
            _code_lm_model = hp.Choice(
                'code-lm-embedding',
                values=['codebert', 'graphcodebert'],
            )

            attention_window_size = hp.Choice(
                'attention-window-size',
                values=[0, 1, 2], # corresponds to total window sizes [1, 3, 5]
            )

            code_context_input = AttentionLayer(
                data=np.load('PRE_COMPUTED_EMBEDDINGS_5.npy'),
                window_size=attention_window_size,
                lm=_code_lm_model,
            )((IA_ID_sequence, code_snippet_idxs))

            # decide which layer type to use (BiLSTM vs LSTM)
            _seq_model_type = hp.Choice(
                'layer_type',
                values=['bidirectional', 'unidirectional'] # values=['bidirectional', 'unidirectional', 'self_attention']
            )
            # choose number of lstm layers
            no_lstm_layers = hp.Int('lstm_layers', 1, 5, step=1, default=1)
            # choose number of lstm units
            lstm_units = hp.Choice('lstm_units', values=[16, 32, 64, 128, 256, 512, 1024])

            # choose number of attention heads for multi head attention layers
            num_heads = 0 # num_heads = hp.Choice('num_heads', values=[2, 4, 8, 16])

            # decide whether to use bottleneck layer for embeddings
            _use_embedding_bottleneck = hp.Choice(
                'use_embedding_bottleneck',
                values=[True, False],
            )
            if _use_embedding_bottleneck:
                code_context_input = Dense(2*lstm_units)(code_context_input)

            # decide whether to use positional embedding (from token sequence)
            _positional_embedding = hp.Choice(
                'use_positional_embedding',
                values=[True, False],
            )

            if args.mode == 'bimodal':
                # instanitate positional embedding
                if _positional_embedding:
                    if _use_embedding_bottleneck:
                        pos_embedding = Embedding(512, 2*lstm_units)(pos_emb_input) # project index to 512 dim space
                        inputs = Add()([pos_embedding, code_context_input]) # add pos_embedding code_context
                    else:
                        pos_embedding = Embedding(512, 768)(pos_emb_input)
                        inputs = Add()([pos_embedding, code_context_input])
                else:
                    inputs = code_context_input
            elif args.mode == 'fixations':
                # just use positional embedding (i.e. fixated token(s)) without code
                if _use_embedding_bottleneck:
                    pos_embedding = Embedding(512, 2*lstm_units)(pos_emb_input) # project index to 512 dim space
                    inputs = pos_embedding
                else:
                    pos_embedding = Embedding(512, 768)(pos_emb_input)
                    inputs = pos_embedding
            elif args.mode == 'code':
                # just use code embedding without fixations
                inputs = code_context_input

            # if mode bimodal/fixations, add fixation durations
            if args.mode == 'bimodal' or args.mode == 'fixations':
                inputs = Concatenate()([inputs, fix_dur_input])

            # decide on pooling layer
            if _seq_model_type == "self_attention":
                _pooling_type = hp.Choice(
                    'pooling_type',
                    values=['avg', 'max'],
                )
            else:
                _pooling_type = hp.Choice(
                    'pooling_type',
                    values=['last_hidden_state', 'avg', 'max'],
                )

            # instantiate neural sequence layers
            if no_lstm_layers == 1:
                if _seq_model_type == 'bidirectional':
                    if _pooling_type == 'last_hidden_state':
                        lstm_output = Bidirectional(
                            LSTM(lstm_units),
                        )(inputs)
                    elif _pooling_type == 'max':
                        lstm_output = Bidirectional(
                            LSTM(lstm_units, return_sequences=True),
                        )(inputs)
                        lstm_output = GlobalMaxPooling1D()(lstm_output)
                    elif _pooling_type == 'avg':
                        lstm_output = Bidirectional(
                            LSTM(lstm_units, return_sequences=True),
                        )(inputs)
                        lstm_output = GlobalAveragePooling1D()(lstm_output)
                elif _seq_model_type == 'unidirectional':
                    if _pooling_type == 'last_hidden_state':
                        lstm_output = LSTM(lstm_units)(inputs)
                    elif _pooling_type == 'max':
                        lstm_output = LSTM(lstm_units, return_sequences=True)(inputs)
                        lstm_output = GlobalMaxPooling1D()(lstm_output)
                    elif _pooling_type == 'avg':
                        lstm_output = LSTM(lstm_units, return_sequences=True)(inputs)
                        lstm_output = GlobalAveragePooling1D()(lstm_output)
                else:  # using MultiHeadAttention (not used in study)
                    mha = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=lstm_units)
                    mha_output = mha(inputs, inputs)  # self-attention

                    # as there is no last_hidden_state in MHA, we default to Max Pooling
                    if _pooling_type == 'max' or _pooling_type == 'last_hidden_state':
                        lstm_output = GlobalMaxPooling1D()(mha_output)
                    elif _pooling_type == 'avg':
                        lstm_output = GlobalAveragePooling1D()(mha_output)
            else:
                for no_lstm_layer in range(no_lstm_layers):
                    if no_lstm_layer == 0:
                        if _seq_model_type == 'bidirectional':
                            lstm_output = Bidirectional(
                                LSTM(lstm_units, return_sequences=True),
                            )(inputs)
                        elif _seq_model_type == 'unidirectional':
                            lstm_output = LSTM(
                                lstm_units, return_sequences=True,
                            )(inputs)
                        else:
                            mha = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=lstm_units)
                            lstm_output = mha(inputs, inputs)  # self-attention
                    elif no_lstm_layer == (no_lstm_layers - 1):
                        if _seq_model_type == 'bidirectional':
                            if _pooling_type == 'last_hidden_state':
                                lstm_output = Bidirectional(
                                    LSTM(lstm_units),
                                )(lstm_output)
                            elif _pooling_type == 'max':
                                lstm_output = Bidirectional(
                                    LSTM(lstm_units, return_sequences=True),
                                )(lstm_output)
                                lstm_output = GlobalMaxPooling1D()(lstm_output)
                            elif _pooling_type == 'avg':
                                lstm_output = Bidirectional(
                                    LSTM(lstm_units, return_sequences=True),
                                )(lstm_output)
                                lstm_output = GlobalAveragePooling1D()(lstm_output)
                        elif _seq_model_type == 'unidirectional':
                            if _pooling_type == 'last_hidden_state':
                                lstm_output = LSTM(lstm_units)(lstm_output)
                            elif _pooling_type == 'max':
                                lstm_output = LSTM(lstm_units, return_sequences=True)(lstm_output)
                                lstm_output = GlobalMaxPooling1D()(lstm_output)
                            elif _pooling_type == 'avg':
                                lstm_output = LSTM(lstm_units, return_sequences=True)(lstm_output)
                                lstm_output = GlobalAveragePooling1D()(lstm_output)
                        else:
                            mha = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=lstm_units)
                            mha_output = mha(inputs, inputs)  # self-attention

                            # as there is no last_hidden_state in MHA, we default to Max Pooling
                            if _pooling_type == 'max' or _pooling_type == 'last_hidden_state':
                                lstm_output = GlobalMaxPooling1D()(mha_output)
                            elif _pooling_type == 'avg':
                                lstm_output = GlobalAveragePooling1D()(mha_output)
                    else:
                        if _seq_model_type == 'bidirectional':
                            lstm_output = Bidirectional(
                                LSTM(lstm_units, return_sequences=True),
                            )(lstm_output)
                        elif _seq_model_type == 'unidirectional':
                            lstm_output = LSTM(
                                lstm_units, return_sequences=True,
                            )(lstm_output)
                        else:
                            mha = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=lstm_units)
                            lstm_output = mha(inputs, inputs)  # self-attention

            # decide number of dense layers
            no_dense_layers = hp.Int('dense_layers', 1, 5, step=1, default=1)
            # decide number of dense units
            dense_units = hp.Choice('dense_units', values=[16, 32, 64, 128, 256])
            # instantiate dense layers
            if no_dense_layers == 1:
                dense_output = Dense(1, activation='sigmoid')(lstm_output)
            else:
                for no_dense_layer in range(no_dense_layers):
                    if no_dense_layer == 0:
                        dense_output = Dense(dense_units, activation='relu')(lstm_output)
                    elif no_dense_layer == (no_dense_layers - 1):
                        dense_output = Dense(1, activation='sigmoid')(dense_output)
                    else:
                        dense_output = Dense(dense_units, activation='relu')(dense_output)

            # create model using all three inputs
            model = Model(
                inputs=[
                    code_snippet_idxs,
                    pos_emb_input,
                    fix_dur_input,
                ],
                outputs=dense_output,
            )

            opt = Adam(learning_rate=5e-4)
            if args.problem_setting == 'subjective_difficulty_score':
                # compile model with MSE loss and MSE & RMSE metrics
                 model.compile(
                    loss='mean_squared_error',
                    optimizer=opt,
                    metrics=[tf.keras.metrics.MeanAbsoluteError(), tf.keras.metrics.RootMeanSquaredError(name='rmse')],
                )
            else:
                # compile model with BCE loss and accuracy & AUC metrics
                model.compile(
                    loss='binary_crossentropy',
                    optimizer=opt,
                    metrics=[
                        'accuracy',
                        auroc,
                    ],
                )

            return model

        if args.problem_setting != "subjective_difficulty_score":
            # check whether both  label classes are in train and test
            assert len(np.unique(y[train])) == 2
            assert len(np.unique(y[test])) == 2

        # implement Bayesian optimization
        if args.problem_setting == "subjective_difficulty_score":
            objective = keras_tuner.Objective('val_loss', direction='min')
            epochs = 200
        else:
            objective = keras_tuner.Objective('val_loss', direction='min')
            epochs = 1000
        tuner = keras_tuner.RandomSearch(
            hypermodel=build_model,
            objective=objective,
            directory= f'{C.results_dir}/cc-{args.split}-{args.problem_setting}-{args.mode}{args.less_fold}',
            max_trials=100,
            project_name=f'{fold}',
            overwrite=True,
        )

        # implement callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
            ),
        ]

        # tune hyperparameters
        tuner.search(
            x=[
                # CodeSnippet IDX
                X_code_snippet_IDXs[train],
                # IA ID
                X[train][:, :, 0],
                # FIX DUR
                X[train][:, :, 1],
            ],
            y=y[train],
            batch_size=16,
            validation_split=0.1,
            epochs=epochs, # 1000 or 200
            callbacks=callbacks,
            verbose=1,
        )
        # get best parameters
        best_hp = tuner.get_best_hyperparameters(1)[0]
        # build best model
        model = tuner.hypermodel.build(best_hp)

        # fit best model
        model.fit(
            [
                # CodeSnippet IDX
                X_code_snippet_IDXs[train],
                # IA ID
                X[train][:, :, 0],
                # FIX DUR
                X[train][:, :, 1],
            ],
            y[train],
            epochs=epochs, # 1000 or 200
            batch_size=16,
            verbose=1,
            validation_split=0.1,
            callbacks=callbacks,
        )

        # calculate scores
        scores = model.evaluate(
            [
                # CodeSnippet ID
                X_code_snippet_IDXs[test],
                # IA ID
                X[test][:, :, 0],
                # FIX DUR
                X[test][:, :, 1],
            ],
            y[test],
            verbose=0,
        )
        if args.problem_setting != "subjective_difficulty_score":
            print(
                f'{model.metrics_names[1]} {scores[1]*100:.2f}\t'
                f'{model.metrics_names[2]} {scores[2]*100}\t',
            )
            # save scores
            cvscores.append(scores[1] * 100)
            aucscores.append(scores[2] * 100)

            print(
                f'accuracy: {np.mean(cvscores):.2f}% '
                f'(+/- {np.std(cvscores) / np.sqrt(len(cvscores)):.2f}%)',
            )
            print(
                f'AUC: {np.mean(aucscores):.2f}% '
                f'(+/- {np.std(aucscores) / np.sqrt(len(aucscores)):.2f}%)',
            )
        else:
            print(
                f'{model.metrics_names[1]} {scores[1] * 100:.2f}\t'
                f'{model.metrics_names[2]} {scores[2] * 100}\t'
            )
            # save scores
            cvscores.append(scores[1] * 100)
            aucscores.append(scores[2] * 100)

            print(
                f'MAE: {np.mean(cvscores):.2f}% '
                f'(+/- {np.std(cvscores) / np.sqrt(len(cvscores)):.2f}%)',
            )
            print(
                f'RMSE: {np.mean(aucscores):.2f}% '
                f'(+/- {np.std(aucscores) / np.sqrt(len(aucscores)):.2f}%)',
            )

        # Fold Summary
        model.summary()
        total_trainable_params = model.count_params()
        print("Total trainable parameters:", total_trainable_params)

    # Evaluation Summary
    model.summary()
    total_trainable_params = model.count_params()
    print("Total trainable parameters:", total_trainable_params)
    end_time = time.time()
    print('time trained:', end_time - start_time)

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
