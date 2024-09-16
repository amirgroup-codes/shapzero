from qsft.input_signal_subsampled import SubsampledSignal

class BioSubsampledSignal(SubsampledSignal):
    """
    This is a Subsampled signal object, except it implements the unimplemented 'subsample' function.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def subsample(self, query_indices):
        """
        Computes the signal/function values at the queried indicies on the fly
        """
        raise ValueError('Samples not loaded. Check that M{i}_D{j}.pickle exists.')