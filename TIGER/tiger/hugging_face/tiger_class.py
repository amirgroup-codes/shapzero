"""
Converts tiger.py into a callable class, removing the command line
"""
import argparse
import os
import gzip
import pickle
import numpy as np
import pandas as pd
import shap
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(script_dir))
path_to_remove = '/nethome/dtsui31/.local/lib/python3.10/site-packages'
if path_to_remove in sys.path:
    sys.path.remove(path_to_remove)
# import utils
from utils_tiger import load_arguments
from models import build_model
import hugging_face.tiger as hf
from data import model_inputs
import tensorflow as tf
from Bio import SeqIO
tf.keras.utils.set_random_seed(12345)

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
BATCH_SIZE_COMPUTE = 512
BATCH_SIZE_SCAN = 20
BATCH_SIZE_TRANSCRIPTS = 512
NUM_TOP_GUIDES = 10
NUM_MISMATCHES = 3
RUN_MODES = dict(
    all='All on-target guides per transcript',
    top_guides='Top {:d} guides per transcript'.format(NUM_TOP_GUIDES),
    titration='Top {:d} guides per transcript & their titration candidates'.format(NUM_TOP_GUIDES)
)

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


def calibrate_predictions(predictions: np.array, num_mismatches: np.array, params: pd.DataFrame = None, tiger_dir=None):
    if tiger_dir is not None:
        params_path = os.path.join(tiger_dir, 'calibration_params.pkl')
    else:
        params_path = 'calibration_params.pkl'
    if params is None:
        params = pd.read_pickle(params_path)
    correction = np.squeeze(params.set_index('num_mismatches').loc[num_mismatches, 'slope'].to_numpy())
    return correction * predictions


def score_predictions(predictions: np.array, params: pd.DataFrame = None, tiger_dir=None):
    if tiger_dir is not None:
        params_path = os.path.join(tiger_dir, 'scoring_params.pkl')
    else:
        params_path = 'scoring_params.pkl'
    if params is None:
        params = pd.read_pickle(params_path)

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


def get_on_target_predictions(transcripts: pd.DataFrame, model: tf.keras.Model, tiger_dir=None, context=None, tiger=None, postprocess=False):

    # loop over transcripts
    predictions = pd.DataFrame()
    transcript_seq = [row['transcript_seq'] for (_, row) in transcripts.iterrows()]
    target_seq = [row['transcript_seq'][CONTEXT_5P:len(row['transcript_seq']) - CONTEXT_3P] for (_, row) in transcripts.iterrows()]
    guide_seq = [''.join(sequence_complement([row['transcript_seq'][CONTEXT_5P:len(row['transcript_seq']) - CONTEXT_3P]])) for (_, row) in transcripts.iterrows()]
    df = pd.DataFrame({
        'gene': ['qsft' for (_, _) in transcripts.iterrows()],
        '5p_context': [row['transcript_seq'][:CONTEXT_5P] for (_, row) in transcripts.iterrows()],
        'target_seq': target_seq,
        'guide_seq': guide_seq,
        '3p_context': ['' for (_, _) in transcripts.iterrows()],
        'guide_id': ['qsft' for (_, _) in transcripts.iterrows()],
        'guide_type': ['qsft' for (_, _) in transcripts.iterrows()],
    })
    data = model_inputs(df, context=context, scalar_feats=set())
    x, _, _ = tiger.pack_inputs(data)

    # get predictions
    scores = model.predict(x, batch_size=BATCH_SIZE_COMPUTE, verbose=False)[:, 0]
    if postprocess:
        lfc_estimate = calibrate_predictions(scores, num_mismatches=np.zeros_like(scores), tiger_dir=tiger_dir)
        scores = score_predictions(lfc_estimate, tiger_dir=tiger_dir)
    
    index = [index for (index, _) in enumerate(transcripts.iterrows())]
    predictions = pd.concat([predictions, pd.DataFrame({
        ID_COL: index,
        TARGET_COL: transcript_seq, # Will remove context later
        GUIDE_COL: guide_seq,
        SCORE_COL: scores})])
    
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


def predict_off_target(off_targets: pd.DataFrame, model: tf.keras.Model, tiger_dir=None, postprocess=False):
    if len(off_targets) == 0:
        return pd.DataFrame()

    # compute off-target predictions
    model_inputs = tf.concat([
        tf.reshape(one_hot_encode_sequence(off_targets[TARGET_COL], add_context_padding=False), [len(off_targets), -1]),
        tf.reshape(one_hot_encode_sequence(off_targets[GUIDE_COL], add_context_padding=True), [len(off_targets), -1]),
        ], axis=-1)
        
    lfc_estimate = model.predict(model_inputs, batch_size=BATCH_SIZE_COMPUTE, verbose=False)[:, 0]
    if postprocess:
        lfc_estimate = calibrate_predictions(lfc_estimate, off_targets['Number of Mismatches'].to_numpy(), tiger_dir=tiger_dir)
    off_targets[SCORE_COL] = score_predictions(lfc_estimate, tiger_dir=tiger_dir)

    return off_targets.reset_index(drop=True)


def tiger_exhibit(transcripts: pd.DataFrame, model: tf.keras.Model, mode: str, check_off_targets: bool, status_update_fn=None, tiger_dir=None, tiger=None, context=None, postprocess=False):

    # evaluate all on-target guides per transcript
    on_target_predictions = get_on_target_predictions(transcripts, model, context=context, tiger_dir=tiger_dir, tiger=tiger, postprocess=postprocess)

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
        titration_predictions = predict_off_target(titration_candidates, model=model, tiger_dir=tiger_dir, postprocess=postprocess)
        off_target_candidates = pd.concat([on_target_predictions, titration_predictions])

    else:
        raise NotImplementedError

    # check off-target effects for top guides
    if check_off_targets and off_target_candidates is not None:
        off_target_candidates = find_off_targets(off_target_candidates, status_update_fn)
        off_target_predictions = predict_off_target(off_target_candidates, model=model)
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

class TranscriptProcessor:
    def __init__(self, mode='titration', check_off_targets=False, tiger_dir=None, gpu_num=0, postprocess=False):
        """
        Add main target sequence fasta path if you only want the guide score for a particular target sequence
        load_test_data is fed in when loading in the experimental test set. Since the 5' + target sequence length is less than 26 nt, we will pad the sequences
        """
        self.mode = mode
        self.check_off_targets = check_off_targets
        self.df_on_target = pd.DataFrame()
        self.df_titration = pd.DataFrame()
        self.df_off_target = pd.DataFrame()
        self.tiger_dir = tiger_dir
        self.gpu_num = gpu_num
        self._configure_gpu()
        self.postprocess = postprocess
        self._load_model()

    def _load_model(self):
        # load model
        if self.tiger_dir is not None:
            model_path = os.path.join(self.tiger_dir, 'model')
        else:
            model_path = 'model'
        if os.path.exists(model_path):
            self.model = tf.keras.models.load_model(model_path)
        else:
            print('no saved model!')
            exit()

        # load common arguments'
        self.data, normalizer, args = load_arguments()

        # Initialize tiger class that's used for model
        self.context = (hf.CONTEXT_5P, hf.CONTEXT_3P)
        train_data = model_inputs(self.data[self.data.fold == 'training'], context=self.context, scalar_feats=set())
        self.tiger = build_model(name='Tiger2D',
            target_len=train_data['target_tokens'].shape[1], # For the published model, this is fixed
            context_5p=train_data['5p_tokens'].shape[1],
            context_3p=train_data['3p_tokens'].shape[1],
            use_guide_seq=True,
            loss_fn='log_cosh',
            debug=args.debug,
            output_fn=normalizer.output_fn,
            **args.kwargs)

    def _configure_gpu(self):
        # Configure GPUs
        physical_gpus = tf.config.list_physical_devices('GPU')
        if physical_gpus:
            for gpu in physical_gpus:
                tf.config.experimental.set_memory_growth(gpu, enable=True)
            if len(tf.config.list_physical_devices('GPU')) > 0:
                tf.config.experimental.set_visible_devices(tf.config.list_physical_devices('GPU')[self.gpu_num], 'GPU')

    def _load_transcripts(self):
        # Load transcripts from a directory of fasta files
        if self.fasta_path is not None and os.path.exists(self.fasta_path):
            self.df_transcripts = load_transcripts(
                [os.path.join(self.fasta_path, f) for f in os.listdir(self.fasta_path)]
            )
        # Otherwise consider simple test case with first 50 nucleotides from EIF3B-003's CDS
        else:
            self.df_transcripts = pd.DataFrame({
                ID_COL: ['ManualEntry'],
                SEQ_COL: ['ATGCAGGACGCGGAGAACGTGGCGGTGCCCGAGGCGGCCGAGGAGCGCGC']
            })
            self.df_transcripts.set_index(ID_COL, inplace=True)

    def _load_target_transcripts(self):
        # Load target transcripts from a directory of fasta files
        if self.main_target_sequence_path is not None and os.path.exists(self.main_target_sequence_path):
            self.df_main_transcripts = load_transcripts(
                [os.path.join(self.main_target_sequence_path, f) for f in os.listdir(self.main_target_sequence_path)]
            )
        else:
            return ValueError('Target fasta path not specified')

    def process_batches(self, df_seqs=None, rewrite=True):
        
        # Flag for running class multiple times 
        if rewrite == True:
            self.df_on_target = pd.DataFrame()
            self.df_titration = pd.DataFrame()
            self.df_off_target = pd.DataFrame()

        # Process in batches
        batch = 0
        num_batches = len(df_seqs) // BATCH_SIZE_TRANSCRIPTS
        num_batches += (len(df_seqs) % BATCH_SIZE_TRANSCRIPTS > 0)
        for idx in range(0, len(df_seqs), BATCH_SIZE_TRANSCRIPTS):
            batch += 1
            # print('Batch {:d} of {:d}'.format(batch, num_batches))
            # Run batch
            idx_stop = min(idx + BATCH_SIZE_TRANSCRIPTS, len(df_seqs))
            df_on_target_batch, df_titration_batch, df_off_target_batch = tiger_exhibit(
                    transcripts=df_seqs[idx:idx_stop],
                    model=self.model,
                    mode=self.mode,
                    check_off_targets=self.check_off_targets,
                    tiger_dir=self.tiger_dir,
                    tiger=self.tiger,
                    context=self.context,
                    postprocess=self.postprocess
                )

            # Append batch results to the overall results
            self.df_on_target = pd.concat([self.df_on_target, df_on_target_batch])
            if df_titration_batch is not None:
                self.df_titration = pd.concat([self.df_titration, df_titration_batch])
            if df_off_target_batch is not None:
                self.df_off_target = pd.concat([self.df_off_target, df_off_target_batch])

            # Clear session to prevent memory blow up
            tf.keras.backend.clear_session()

    def get_results(self):
        return self.df_on_target, self.df_titration, self.df_off_target
    
    def run_model(self, df_query_indices=None):
        """
        Modified function from /BioMobius/RNA/tiger/experiments.py
        """
        data = model_inputs(df_query_indices, context=self.context, scalar_feats=set())

        # assemble inputs and predict
        x, _, _ = self.tiger.pack_inputs(data)
        scores = self.model.predict(x, verbose=0).flatten().tolist()

        return scores

    def deepshap_explain_model(self, train_data=None, heldout_data=None, num_background_samples=5000):
        """
        Modified function from ../tiger/experiments.py
        """
        from data import model_inputs 
        # load common arguments
        _, normalizer, args = load_arguments()

        # Initialize tiger class that's used for model
        context = (hf.CONTEXT_5P, hf.CONTEXT_3P)
        train_data = model_inputs(train_data, context=context, scalar_feats=set())
        valid_data = model_inputs(heldout_data, context=context, scalar_feats=set())
        tiger = build_model(name='Tiger2D',
            target_len=train_data['target_tokens'].shape[1], # For the published model, this is fixed
            context_5p=train_data['5p_tokens'].shape[1],
            context_3p=train_data['3p_tokens'].shape[1],
            use_guide_seq=True,
            loss_fn='log_cosh',
            debug=args.debug,
            output_fn=normalizer.output_fn,
            **args.kwargs)

        # load model
        if self.tiger_dir is not None:
            model_path = os.path.join(self.tiger_dir, 'model')
        else:
            model_path = 'model'
        if os.path.exists(model_path):
            # with mirrored_strategy.scope():
            model = tf.keras.models.load_model(model_path)
            # model = CustomModel(model)  # Replace the loaded model with CustomModel
            # model.predict = lambda inputs, batch_size=BATCH_SIZE_COMPUTE, verbose=0: custom_predict(model, inputs, batch_size=batch_size, verbose=verbose)
        else:
            print('no saved model!')
            exit()

        # assemble inputs
        x_train, _, _ = tiger.pack_inputs(train_data)
        x_valid, _, _ = tiger.pack_inputs(valid_data)

        # select a set of background examples to take an expectation over
        num_background_samples = min(num_background_samples, x_train.shape[0])
        background = x_train.numpy()[np.random.choice(x_train.shape[0], num_background_samples, replace=False)]

        # compute Shapley values
        e = shap.DeepExplainer(model, background)
        shap_values = e.shap_values(x_valid.numpy())

        # parse Shapley values into a DataFrame and append other relevant information
        df_shap = tiger.parse_input_scores(shap_values)
        current_cols = df_shap.columns.to_list()
        relevant_cols = ['gene', 'target_seq', 'guide_seq', 'guide_type']
        for col in relevant_cols:
            df_shap[col] = valid_data[col].numpy()
            if df_shap[col].dtype == object:
                df_shap[col] = df_shap[col].apply(lambda x: x.decode('utf-8'))
        df_shap = df_shap[relevant_cols + current_cols]

        return df_shap

if __name__ == '__main__':
    processor = TranscriptProcessor(mode='titration', check_off_targets=True, fasta_path='path/to/fasta', tiger_dir=None, gpu_num=0)
    processor.process_batches()
    df_on_target, df_titration, df_off_target = processor.get_results()