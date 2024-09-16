import os
import tiger
import pandas as pd
import streamlit as st
from pathlib import Path

ENTRY_METHODS = dict(
    manual='Manual entry of single transcript',
    fasta="Fasta file upload (supports multiple transcripts if they have unique ID's)"
)


@st.cache_data
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')


def mode_change_callback():
    if st.session_state.mode in {tiger.RUN_MODES['all'], tiger.RUN_MODES['titration']}:  # TODO: support titration
        st.session_state.check_off_targets = False
        st.session_state.disable_off_target_checkbox = True
    else:
        st.session_state.disable_off_target_checkbox = False


def progress_update(update_text, percent_complete):
    with progress.container():
        st.write(update_text)
        st.progress(percent_complete / 100)


def initiate_run():

    # initialize state variables
    st.session_state.transcripts = None
    st.session_state.input_error = None
    st.session_state.on_target = None
    st.session_state.titration = None
    st.session_state.off_target = None

    # initialize transcript DataFrame
    transcripts = pd.DataFrame(columns=[tiger.ID_COL, tiger.SEQ_COL])

    # manual entry
    if st.session_state.entry_method == ENTRY_METHODS['manual']:
        transcripts = pd.DataFrame({
            tiger.ID_COL: ['ManualEntry'],
            tiger.SEQ_COL: [st.session_state.manual_entry]
        }).set_index(tiger.ID_COL)

    # fasta file upload
    elif st.session_state.entry_method == ENTRY_METHODS['fasta']:
        if st.session_state.fasta_entry is not None:
            fasta_path = st.session_state.fasta_entry.name
            with open(fasta_path, 'w') as f:
                f.write(st.session_state.fasta_entry.getvalue().decode('utf-8'))
            transcripts = tiger.load_transcripts([fasta_path], enforce_unique_ids=False)
            os.remove(fasta_path)

    # convert to upper case as used by tokenizer
    transcripts[tiger.SEQ_COL] = transcripts[tiger.SEQ_COL].apply(lambda s: s.upper().replace('U', 'T'))

    # ensure all transcripts have unique identifiers
    if transcripts.index.has_duplicates:
        st.session_state.input_error = "Duplicate transcript ID's detected in fasta file"

    # ensure all transcripts only contain nucleotides A, C, G, T, and wildcard N
    elif not all(transcripts[tiger.SEQ_COL].apply(lambda s: set(s).issubset(tiger.NUCLEOTIDE_TOKENS.keys()))):
        st.session_state.input_error = 'Transcript(s) must only contain upper or lower case A, C, G, and Ts or Us'

    # ensure all transcripts satisfy length requirements
    elif any(transcripts[tiger.SEQ_COL].apply(lambda s: len(s) < tiger.TARGET_LEN)):
        st.session_state.input_error = 'Transcript(s) must be at least {:d} bases.'.format(tiger.TARGET_LEN)

    # run model if we have any transcripts
    elif len(transcripts) > 0:
        st.session_state.transcripts = transcripts


if __name__ == '__main__':

    # app initialization
    if 'mode' not in st.session_state:
        st.session_state.mode = tiger.RUN_MODES['all']
        st.session_state.disable_off_target_checkbox = True
    if 'entry_method' not in st.session_state:
        st.session_state.entry_method = ENTRY_METHODS['manual']
    if 'transcripts' not in st.session_state:
        st.session_state.transcripts = None
    if 'input_error' not in st.session_state:
        st.session_state.input_error = None
    if 'on_target' not in st.session_state:
        st.session_state.on_target = None
    if 'titration' not in st.session_state:
        st.session_state.titration = None
    if 'off_target' not in st.session_state:
        st.session_state.off_target = None

    # title and documentation
    st.markdown(Path('tiger.md').read_text(), unsafe_allow_html=True)
    st.divider()

    # mode selection
    col1, col2 = st.columns([0.65, 0.35])
    with col1:
        st.radio(
            label='What do you want to predict?',
            options=tuple(tiger.RUN_MODES.values()),
            key='mode',
            on_change=mode_change_callback,
            disabled=st.session_state.transcripts is not None,
        )
    with col2:
        st.checkbox(
            label='Find off-target effects (slow)',
            key='check_off_targets',
            disabled=st.session_state.disable_off_target_checkbox or st.session_state.transcripts is not None
        )

    # transcript entry
    st.selectbox(
        label='How would you like to provide transcript(s) of interest?',
        options=ENTRY_METHODS.values(),
        key='entry_method',
        disabled=st.session_state.transcripts is not None
    )
    if st.session_state.entry_method == ENTRY_METHODS['manual']:
        st.text_input(
            label='Enter a target transcript:',
            key='manual_entry',
            placeholder='Upper or lower case',
            disabled=st.session_state.transcripts is not None
        )
    elif st.session_state.entry_method == ENTRY_METHODS['fasta']:
        st.file_uploader(
            label='Upload a fasta file:',
            key='fasta_entry',
            disabled=st.session_state.transcripts is not None
        )

    # let's go!
    st.button(label='Get predictions!', on_click=initiate_run, disabled=st.session_state.transcripts is not None)
    progress = st.empty()

    # input error
    error = st.empty()
    if st.session_state.input_error is not None:
        error.error(st.session_state.input_error, icon="🚨")
    else:
        error.empty()

    # on-target results
    on_target_results = st.empty()
    if st.session_state.on_target is not None:
        with on_target_results.container():
            st.write('On-target predictions:', st.session_state.on_target)
            st.download_button(
                label='Download on-target predictions',
                data=convert_df(st.session_state.on_target),
                file_name='on_target.csv',
                mime='text/csv'
            )
    else:
        on_target_results.empty()

    # titration results
    titration_results = st.empty()
    if st.session_state.titration is not None:
        with titration_results.container():
            st.write('Titration predictions:', st.session_state.titration)
            st.download_button(
                label='Download titration predictions',
                data=convert_df(st.session_state.titration),
                file_name='titration.csv',
                mime='text/csv'
            )
    else:
        titration_results.empty()

    # off-target results
    off_target_results = st.empty()
    if st.session_state.off_target is not None:
        with off_target_results.container():
            if len(st.session_state.off_target) > 0:
                st.write('Off-target predictions:', st.session_state.off_target)
                st.download_button(
                    label='Download off-target predictions',
                    data=convert_df(st.session_state.off_target),
                    file_name='off_target.csv',
                    mime='text/csv'
                )
            else:
                st.write('We did not find any off-target effects!')
    else:
        off_target_results.empty()

    # keep trying to run model until we clear inputs (streamlit UI changes can induce race-condition reruns)
    if st.session_state.transcripts is not None:
        st.session_state.on_target, st.session_state.titration, st.session_state.off_target = tiger.tiger_exhibit(
            transcripts=st.session_state.transcripts,
            mode={v: k for k, v in tiger.RUN_MODES.items()}[st.session_state.mode],
            check_off_targets=st.session_state.check_off_targets,
            status_update_fn=progress_update
        )
        st.session_state.transcripts = None
        st.experimental_rerun()
