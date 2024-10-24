import argparse
import os
import gzip
import pickle
import numpy as np
import pandas as pd

# import sys
# path_to_remove = '/nethome/dtsui31/.local/lib/python3.10/site-packages'
# if path_to_remove in sys.path:
#     sys.path.remove(path_to_remove)
import tensorflow as tf
from Bio import SeqIO

# column names
ID_COL = 'Transcript ID'
SEQ_COL = 'Transcript Sequence'
TARGET_COL = 'Target Sequence'
GUIDE_COL = 'Guide Sequence'
MM_COL = 'Number of Mismatches'
SCORE_COL = 'Guide Score'

# nucleotide tokens
NUCLEOTIDE_TOKENS = dict(zip(['A', 'C', 'G', 'T', 'N'], [0, 1, 2, 3, 255]))
NUCLEOTIDE_COMPLEMENT = dict(zip(['A', 'C', 'G', 'T'], ['T', 'G', 'C', 'A']))

# model hyper-parameters
GUIDE_LEN = 23
CONTEXT_5P = 3
CONTEXT_3P = 0
TARGET_LEN = CONTEXT_5P + GUIDE_LEN + CONTEXT_3P
UNIT_INTERVAL_MAP = 'sigmoid'

# reference transcript files
REFERENCE_TRANSCRIPTS = ('gencode.v19.pc_transcripts.fa.gz', 'gencode.v19.lncRNA_transcripts.fa.gz')

# application configuration
BATCH_SIZE_COMPUTE = 10000
BATCH_SIZE_SCAN = 20
BATCH_SIZE_TRANSCRIPTS = 10000
NUM_TOP_GUIDES = 10
NUM_MISMATCHES = 3
RUN_MODES = dict(
    all='All on-target guides per transcript',
    top_guides='Top {:d} guides per transcript'.format(NUM_TOP_GUIDES),
    titration='Top {:d} guides per transcript & their titration candidates'.format(NUM_TOP_GUIDES)
)

# configure GPUs
for gpu in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, enable=True)
# print(tf.config.list_physical_devices('GPU'))
if len(tf.config.list_physical_devices('GPU')) > 0:
    tf.config.experimental.set_visible_devices(tf.config.list_physical_devices('GPU')[-1], 'GPU')
# print("Visible devices:", tf.config.get_visible_devices('GPU'))


def load_transcripts(fasta_files: list, enforce_unique_ids: bool = True):

    # load all transcripts from fasta files into a DataFrame
    transcripts = pd.DataFrame()
    for file in fasta_files:
        try:
            if os.path.splitext(file)[1] == '.gz':
                with gzip.open(file, 'rt') as f:
                    df = pd.DataFrame([(t.id, str(t.seq)) for t in SeqIO.parse(f, 'fasta')], columns=[ID_COL, SEQ_COL])
            else:
                df = pd.DataFrame([(t.id, str(t.seq)) for t in SeqIO.parse(file, 'fasta')], columns=[ID_COL, SEQ_COL])
        except Exception as e:
            print(e, 'while loading', file)
            continue
        transcripts = pd.concat([transcripts, df])

    # set index
    transcripts[ID_COL] = transcripts[ID_COL].apply(lambda s: s.split('|')[0])
    transcripts.set_index(ID_COL, inplace=True)
    if enforce_unique_ids:
        assert not transcripts.index.has_duplicates, "duplicate transcript ID's detected in fasta file"

    return transcripts


def sequence_complement(sequence: list):
    return [''.join([NUCLEOTIDE_COMPLEMENT[nt] for nt in list(seq)]) for seq in sequence]


def one_hot_encode_sequence(sequence: list, add_context_padding: bool = False):

    # stack list of sequences into a tensor
    sequence = tf.ragged.stack([tf.constant(list(seq)) for seq in sequence], axis=0)

    # tokenize sequence
    nucleotide_table = tf.lookup.StaticVocabularyTable(
        initializer=tf.lookup.KeyValueTensorInitializer(
            keys=tf.constant(list(NUCLEOTIDE_TOKENS.keys()), dtype=tf.string),
            values=tf.constant(list(NUCLEOTIDE_TOKENS.values()), dtype=tf.int64)),
        num_oov_buckets=1)
    sequence = tf.RaggedTensor.from_row_splits(values=nucleotide_table.lookup(sequence.values),
                                               row_splits=sequence.row_splits).to_tensor(255)

    # add context padding if requested
    if add_context_padding:
        pad_5p = 255 * tf.ones([sequence.shape[0], CONTEXT_5P], dtype=sequence.dtype)
        pad_3p = 255 * tf.ones([sequence.shape[0], CONTEXT_3P], dtype=sequence.dtype)
        sequence = tf.concat([pad_5p, sequence, pad_3p], axis=1)

    # one-hot encode
    sequence = tf.one_hot(sequence, depth=4, dtype=tf.float16)

    return sequence


def process_data(transcript_seq: str):

    # convert to upper case
    transcript_seq = transcript_seq.upper()

    # get all target sites
    target_seq = [transcript_seq[i: i + TARGET_LEN] for i in range(len(transcript_seq) - TARGET_LEN + 1)]

    # prepare guide sequences
    guide_seq = sequence_complement([seq[CONTEXT_5P:len(seq) - CONTEXT_3P] for seq in target_seq])

    # model inputs
    model_inputs = tf.concat([
        tf.reshape(one_hot_encode_sequence(target_seq, add_context_padding=False), [len(target_seq), -1]),
        tf.reshape(one_hot_encode_sequence(guide_seq, add_context_padding=True), [len(guide_seq), -1]),
        ], axis=-1)
    return target_seq, guide_seq, model_inputs


def calibrate_predictions(predictions: np.array, num_mismatches: np.array, params: pd.DataFrame = None):
    if params is None:
        params = pd.read_pickle('calibration_params.pkl')
    correction = np.squeeze(params.set_index('num_mismatches').loc[num_mismatches, 'slope'].to_numpy())
    return correction * predictions


def score_predictions(predictions: np.array, params: pd.DataFrame = None):
    if params is None:
        params = pd.read_pickle('scoring_params.pkl')

    if UNIT_INTERVAL_MAP == 'sigmoid':
        params = params.iloc[0]
        return 1 - 1 / (1 + np.exp(params['a'] * predictions + params['b']))

    elif UNIT_INTERVAL_MAP == 'min-max':
        return 1 - (predictions - params['a']) / (params['b'] - params['a'])

    elif UNIT_INTERVAL_MAP == 'exp-lin-exp':
        # regime indices
        active_saturation = predictions < params['a']
        linear_regime = (params['a'] <= predictions) & (predictions <= params['c'])
        inactive_saturation = params['c'] < predictions

        # linear regime
        slope = (params['d'] - params['b']) / (params['c'] - params['a'])
        intercept = -params['a'] * slope + params['b']
        predictions[linear_regime] = slope * predictions[linear_regime] + intercept

        # active saturation regime
        alpha = slope / params['b']
        beta = alpha * params['a'] - np.log(params['b'])
        predictions[active_saturation] = np.exp(alpha * predictions[active_saturation] - beta)

        # inactive saturation regime
        alpha = slope / (1 - params['d'])
        beta = -alpha * params['c'] - np.log(1 - params['d'])
        predictions[inactive_saturation] = 1 - np.exp(-alpha * predictions[inactive_saturation] - beta)

        return 1 - predictions

    else:
        raise NotImplementedError


def get_on_target_predictions(transcripts: pd.DataFrame, model: tf.keras.Model, status_update_fn=None):

    # loop over transcripts
    predictions = pd.DataFrame()
    for i, (index, row) in enumerate(transcripts.iterrows()):

        # parse transcript sequence
        target_seq, guide_seq, model_inputs = process_data(row[SEQ_COL])

        # get predictions
        # print(mirrored_strategy)
        # print('w')
        # with mirrored_strategy.scope():
        #     print('hi123')
        lfc_estimate = model.predict(model_inputs, batch_size=BATCH_SIZE_COMPUTE, verbose=False)[:, 0]
        lfc_estimate = calibrate_predictions(lfc_estimate, num_mismatches=np.zeros_like(lfc_estimate))
        scores = score_predictions(lfc_estimate)
        # print(scores)
        predictions = pd.concat([predictions, pd.DataFrame({
            ID_COL: [index] * len(scores),
            TARGET_COL: target_seq,
            GUIDE_COL: guide_seq,
            SCORE_COL: scores})])

        # progress update
        percent_complete = 100 * min((i + 1) / len(transcripts), 1)
        update_text = 'Evaluating on-target guides for each transcript: {:.2f}%'.format(percent_complete)
        print('\r' + update_text, end='')
        if status_update_fn is not None:
            status_update_fn(update_text, percent_complete)
    print('')

    return predictions


def top_guides_per_transcript(predictions: pd.DataFrame):

    # select and sort top guides for each transcript
    top_guides = pd.DataFrame()
    for transcript in predictions[ID_COL].unique():
        df = predictions.loc[predictions[ID_COL] == transcript]
        df = df.sort_values(SCORE_COL, ascending=False).reset_index(drop=True).iloc[:NUM_TOP_GUIDES]
        top_guides = pd.concat([top_guides, df])

    return top_guides.reset_index(drop=True)


def get_titration_candidates(top_guide_predictions: pd.DataFrame):

    # generate a table of all titration candidates
    titration_candidates = pd.DataFrame()
    for _, row in top_guide_predictions.iterrows():
        for i in range(len(row[GUIDE_COL])):
            nt = row[GUIDE_COL][i]
            for mutation in set(NUCLEOTIDE_TOKENS.keys()) - {nt, 'N'}:
                sm_guide = list(row[GUIDE_COL])
                sm_guide[i] = mutation
                sm_guide = ''.join(sm_guide)
                assert row[GUIDE_COL] != sm_guide
                titration_candidates = pd.concat([titration_candidates, pd.DataFrame({
                    ID_COL: [row[ID_COL]],
                    TARGET_COL: [row[TARGET_COL]],
                    GUIDE_COL: [sm_guide],
                    MM_COL: [1]
                })])

    return titration_candidates


def find_off_targets(top_guides: pd.DataFrame, status_update_fn=None):

    # load reference transcripts
    reference_transcripts = load_transcripts([os.path.join('transcripts', f) for f in REFERENCE_TRANSCRIPTS])

    # one-hot encode guides to form a filter
    guide_filter = one_hot_encode_sequence(sequence_complement(top_guides[GUIDE_COL]), add_context_padding=False)
    guide_filter = tf.transpose(guide_filter, [1, 2, 0])

    # loop over transcripts in batches
    i = 0
    off_targets = pd.DataFrame()
    while i < len(reference_transcripts):
        # select batch
        df_batch = reference_transcripts.iloc[i:min(i + BATCH_SIZE_SCAN, len(reference_transcripts))]
        i += BATCH_SIZE_SCAN

        # find locations of off-targets
        transcripts = one_hot_encode_sequence(df_batch[SEQ_COL].values.tolist(), add_context_padding=False)
        num_mismatches = GUIDE_LEN - tf.nn.conv1d(transcripts, guide_filter, stride=1, padding='SAME')
        loc_off_targets = tf.where(tf.round(num_mismatches) <= NUM_MISMATCHES).numpy()

        # off-targets discovered
        if len(loc_off_targets) > 0:

            # log off-targets
            dict_off_targets = pd.DataFrame({
                'On-target ' + ID_COL: top_guides.iloc[loc_off_targets[:, 2]][ID_COL],
                GUIDE_COL: top_guides.iloc[loc_off_targets[:, 2]][GUIDE_COL],
                'Off-target ' + ID_COL: df_batch.index.values[loc_off_targets[:, 0]],
                'Guide Midpoint': loc_off_targets[:, 1],
                SEQ_COL: df_batch[SEQ_COL].values[loc_off_targets[:, 0]],
                MM_COL: tf.gather_nd(num_mismatches, loc_off_targets).numpy().astype(int),
            }).to_dict('records')

            # trim transcripts to targets
            for row in dict_off_targets:
                start_location = row['Guide Midpoint'] - (GUIDE_LEN // 2)
                del row['Guide Midpoint']
                target = row[SEQ_COL]
                del row[SEQ_COL]
                if start_location < CONTEXT_5P:
                    target = target[0:GUIDE_LEN + CONTEXT_3P]
                    target = 'N' * (TARGET_LEN - len(target)) + target
                elif start_location + GUIDE_LEN + CONTEXT_3P > len(target):
                    target = target[start_location - CONTEXT_5P:]
                    target = target + 'N' * (TARGET_LEN - len(target))
                else:
                    target = target[start_location - CONTEXT_5P:start_location + GUIDE_LEN + CONTEXT_3P]
                if row[MM_COL] == 0 and 'N' not in target:
                    assert row[GUIDE_COL] == sequence_complement([target[CONTEXT_5P:TARGET_LEN - CONTEXT_3P]])[0]
                row[TARGET_COL] = target

            # append new off-targets
            off_targets = pd.concat([off_targets, pd.DataFrame(dict_off_targets)])

        # progress update
        percent_complete = 100 * min((i + 1) / len(reference_transcripts), 1)
        update_text = 'Scanning for off-targets: {:.2f}%'.format(percent_complete)
        print('\r' + update_text, end='')
        if status_update_fn is not None:
            status_update_fn(update_text, percent_complete)
    print('')

    return off_targets


def predict_off_target(off_targets: pd.DataFrame, model: tf.keras.Model):
    if len(off_targets) == 0:
        return pd.DataFrame()

    # compute off-target predictions
    model_inputs = tf.concat([
        tf.reshape(one_hot_encode_sequence(off_targets[TARGET_COL], add_context_padding=False), [len(off_targets), -1]),
        tf.reshape(one_hot_encode_sequence(off_targets[GUIDE_COL], add_context_padding=True), [len(off_targets), -1]),
        ], axis=-1)
    lfc_estimate = model.predict(model_inputs, batch_size=BATCH_SIZE_COMPUTE, verbose=False)[:, 0]
    lfc_estimate = calibrate_predictions(lfc_estimate, off_targets['Number of Mismatches'].to_numpy())
    off_targets[SCORE_COL] = score_predictions(lfc_estimate)

    return off_targets.reset_index(drop=True)


def tiger_exhibit(transcripts: pd.DataFrame, mode: str, check_off_targets: bool, status_update_fn=None):
    # mirrored_strategy = tf.distribute.MirroredStrategy()
    # print('Number of devices: {}'.format(mirrored_strategy.num_replicas_in_sync))

    # print(mirrored_strategy)
    # load model
    if os.path.exists('model'):
        # with mirrored_strategy.scope():
        tiger = tf.keras.models.load_model('model')
    else:
        print('no saved model!')
        exit()

    # evaluate all on-target guides per transcript
    on_target_predictions = get_on_target_predictions(transcripts, tiger, status_update_fn)

    # initialize other outputs
    titration_predictions = off_target_predictions = None

    if mode == 'all' and not check_off_targets:
        off_target_candidates = None

    elif mode == 'top_guides':
        on_target_predictions = top_guides_per_transcript(on_target_predictions)
        off_target_candidates = on_target_predictions

    elif mode == 'titration':
        on_target_predictions = top_guides_per_transcript(on_target_predictions)
        titration_candidates = get_titration_candidates(on_target_predictions)
        titration_predictions = predict_off_target(titration_candidates, model=tiger)
        off_target_candidates = pd.concat([on_target_predictions, titration_predictions])

    else:
        raise NotImplementedError

    # check off-target effects for top guides
    if check_off_targets and off_target_candidates is not None:
        off_target_candidates = find_off_targets(off_target_candidates, status_update_fn)
        off_target_predictions = predict_off_target(off_target_candidates, model=tiger)
        if len(off_target_predictions) > 0:
            off_target_predictions = off_target_predictions.sort_values(SCORE_COL, ascending=False)
            off_target_predictions = off_target_predictions.reset_index(drop=True)

    # finalize tables
    for df in [on_target_predictions, titration_predictions, off_target_predictions]:
        if df is not None and len(df) > 0:
            for col in df.columns:
                if ID_COL in col and set(df[col].unique()) == {'ManualEntry'}:
                    del df[col]
            df[GUIDE_COL] = df[GUIDE_COL].apply(lambda s: s[::-1])  # reverse guide sequences
            df[TARGET_COL] = df[TARGET_COL].apply(lambda seq: seq[CONTEXT_5P:len(seq) - CONTEXT_3P])  # remove context

    return on_target_predictions, titration_predictions, off_target_predictions


if __name__ == '__main__':

    # common arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='titration')
    parser.add_argument('--check_off_targets', action='store_true', default=False)
    parser.add_argument('--fasta_path', type=str, default=None)
    args = parser.parse_args()

    # check for any existing results
    if os.path.exists('on_target.csv') or os.path.exists('titration.csv') or os.path.exists('off_target.csv'):
        raise FileExistsError('please rename or delete existing results')

    # load transcripts from a directory of fasta files
    if args.fasta_path is not None and os.path.exists(args.fasta_path):
        df_transcripts = load_transcripts([os.path.join(args.fasta_path, f) for f in os.listdir(args.fasta_path)])

    # otherwise consider simple test case with first 50 nucleotides from EIF3B-003's CDS
    else:
        df_transcripts = pd.DataFrame({
            ID_COL: ['ManualEntry'],
            SEQ_COL: ['ATGCAGGACGCGGAGAACGTGGCGGTGCCCGAGGCGGCCGAGGAGCGCGC']})
        df_transcripts.set_index(ID_COL, inplace=True)
    
    # process in batches
    batch = 0
    num_batches = len(df_transcripts) // BATCH_SIZE_TRANSCRIPTS
    num_batches += (len(df_transcripts) % BATCH_SIZE_TRANSCRIPTS > 0)
    for idx in range(0, len(df_transcripts), BATCH_SIZE_TRANSCRIPTS):
        batch += 1
        print('Batch {:d} of {:d}'.format(batch, num_batches))

        # run batch
        idx_stop = min(idx + BATCH_SIZE_TRANSCRIPTS, len(df_transcripts))
        df_on_target, df_titration, df_off_target = tiger_exhibit(
            transcripts=df_transcripts[idx:idx_stop],
            mode=args.mode,
            check_off_targets=args.check_off_targets
        )

        # save batch results
        df_on_target.to_csv('on_target.csv', header=batch == 1, index=False, mode='a')
        if df_titration is not None:
            df_titration.to_csv('titration.csv', header=batch == 1, index=False, mode='a')
        if df_off_target is not None:
            df_off_target.to_csv('off_target.csv', header=batch == 1, index=False, mode='a')

        # clear session to prevent memory blow up
        tf.keras.backend.clear_session()
