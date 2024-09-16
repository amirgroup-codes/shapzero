from qsft.utils import qary_ints, qary_vec_to_dec, gwht, load_data, save_data, dec_to_qary_vec, get_qs, tft
from qsft.input_signal import Signal
from qsft.query import get_Ms_and_Ds
from pathlib import Path
from math import floor
from tqdm import tqdm
import numpy as np
import random
import time
import pickle
from random import randint
import sys 

import zlib
def save_data4(data, filename):
    with open(filename, 'wb') as f:
        f.write(zlib.compress(pickle.dumps(data, protocol=4), 9))

class SubsampledSignal(Signal):
    """
    A shell Class for input signal/functions that are too large and cannot be stored in their entirety. In addition to
    the signal itself, this must also contain information about the M and D matricies that are used for subsampling
    Notable attributes are included below.

    Attributes
    ---------
    query_args : dict
    These are the parameters that determine the structure of the Ms and Ds needed for subsampling.
    It contains the following sub-parameters:
        b : int
        The max dimension of subsampling (i.e., we will subsample functions with b inputs, or equivalently a signal of
        length q^b)
        all_bs : list, (optional)
        List of all the b values that should be subsampled. This is most useful when you want to repeat an experiment
        many times with different values of b to see which is most efficient
        For a description of the "delays_method_channel", "delays_method_source", "num_repeat" and "num_subsample", see
        the docstring of the QSFT class.
        subsampling_method
            If set to "simple" the M matricies are generated according to the construction in Appendix C, i.e., a
            block-wise identity structure.
            If set to "complex" the elements of the M matricies are uniformly populated from integers from 0 to q-1.
            It should be noted that these matricies are not checked to be full rank (w.r.t. the module where arithemtic is
            over the integer quotient ring), and so it is possible that the actual dimension of subsampling may be
            lower. For large enough n and b this isn't a problem, since w.h.p. the matricies are full rank.

    L : np.array
    An array that enumerates all q^b q-ary vectors of length b

    foldername : str
    If set, and the file {foldername}/Ms_and_Ds.pickle exists, the Ms and Ds are read directly from the file.
    Furthermore, if the transforms for all the bs are in {foldername}/transforms/U{i}_b{b}.pickle, the transforms can be
    directly loaded into memory.
    """
    def _set_params(self, **kwargs):
        self.n = kwargs.get("n")
        self.q = kwargs.get("q")
        self.N = self.q ** self.n
        self.signal_w = kwargs.get("signal_w")
        self.query_args = kwargs.get("query_args")
        self.b = self.query_args.get("b")
        self.all_bs = self.query_args.get("all_bs", [self.b])   # all b values to sample/transform at
        self.num_subsample = self.query_args.get("num_subsample")
        if "num_repeat" not in self.query_args:
            self.query_args["num_repeat"] = 1
        self.num_repeat = self.query_args.get("num_repeat")
        self.subsampling_method = self.query_args.get("subsampling_method")
        if "method" in self.query_args:
            self.method = self.query_args.get("method")
        else:
            self.method = "N/A"
        self.delays_method_source = self.query_args.get("delays_method_source")
        self.delays_method_channel = self.query_args.get("delays_method_channel")
        self.L = None  # List of all length b qary vectors
        self.foldername = kwargs.get("folder")
        self.type = kwargs.get("type")


    def _init_signal(self):
        if self.method == "generate_samples":
            self._set_Ms_and_Ds_qsft()
            self._generate_train_subsample()
            self._generate_test_subsample()
            exit()
        elif self.subsampling_method == "uniform":
            self._subsample_uniform()
        elif self.subsampling_method == "qsft":
            self._set_Ms_and_Ds_qsft()
            self._subsample_qsft()
        else:
            self._set_Ms_and_Ds_qsft()
            self._subsample_qsft()

    def _check_transforms_qsft(self):
        """
        Returns
        -------
        True if the transform is already computed and saved for all values of b, else False
        """
        if self.foldername:
            Path(f"{self.foldername}/transforms/").mkdir(exist_ok=True)
            for b in self.all_bs:
                for i in range(len(self.Ms)):
                    Us_path = Path(f"{self.foldername}/transforms/U{i}_b{b}.pickle")
                    if not Us_path.is_file():
                        return False
            return True
        else:
            return False

    def _set_Ms_and_Ds_qsft(self):
        """
        Sets the values of Ms and Ds, either by loading from folder if exists, otherwise it loaded from query_args
        """
        if self.foldername:
            Path(f"{self.foldername}").mkdir(exist_ok=True)
            Ms_and_Ds_path = Path(f"{self.foldername}/Ms_and_Ds.pickle")
            if Ms_and_Ds_path.is_file():
                self.Ms, self.Ds = load_data(Ms_and_Ds_path)
            else:
                self.Ms, self.Ds = get_Ms_and_Ds(self.n, self.q, **self.query_args)
                save_data((self.Ms, self.Ds), f"{self.foldername}/Ms_and_Ds.pickle")
        else:
            self.Ms, self.Ds = get_Ms_and_Ds(self.n, self.q, **self.query_args)

    def _subsample_qsft(self):
        """
        Subsamples and computes the sparse fourier transform for each subsampling group if the samples are not already
        present in the folder
        """
        self.Us = [[{} for j in range(len(self.Ds[i]))] for i in range(len(self.Ms))]
        self.transformTimes = [[{} for j in range(len(self.Ds[i]))] for i in range(len(self.Ms))]
        global newrun 
        newrun = False

        if self.foldername:
            Path(f"{self.foldername}/samples").mkdir(exist_ok=True)
            Path(f"{self.foldername}/transforms/").mkdir(exist_ok=True)

        """
        Subsampling
        """
        pbar = tqdm(total=0, position=0)
        for i in range(len(self.Ms)):
            for j in range(len(self.Ds[i])):
                try:
                    if self.type != 'None' and self.type:
                        path_parts = self.type.split('/')
                        transform_file = Path(self.foldername) / "transforms" / Path(*path_parts) / f"U{i}_D{j}.pickle"
                    else:
                        transform_file = Path(self.foldername) / "transforms" / f"U{i}_D{j}.pickle"
                    
                    transform_file.parent.mkdir(parents=True, exist_ok=True)
                except AttributeError:
                    transform_file = Path(f"{self.foldername}/transforms/U{i}_{j}.pickle")
                if self.foldername and transform_file.is_file():
                    self.Us[i][j], self.transformTimes[i][j] = load_data(transform_file)
                    try:
                        if self.type != 'None' and self.type:
                            path_parts = self.type.split('/')
                            sample_file = Path(self.foldername) / "samples" / Path(*path_parts) / f"M{i}_D{j}.pickle"
                        else:
                            sample_file = Path(self.foldername) / "samples" / f"M{i}_D{j}.pickle"
                        
                        sample_file.parent.mkdir(parents=True, exist_ok=True)
                    except AttributeError:
                        sample_file = Path(f"{self.foldername}/samples/M{i}_D{j}.pickle")
                    samples = load_data(sample_file)
                    pbar.total = len(self.Ms) * len(self.Ds[0]) * len(self.Us[i][j])
                    pbar.update(len(self.Us[i][j]))
                else:
                    try:
                        if self.type != 'None' and self.type:
                            path_parts = self.type.split('/')
                            sample_file = Path(self.foldername) / "samples" / Path(*path_parts) / f"M{i}_D{j}.pickle"
                        else:
                            sample_file = Path(self.foldername) / "samples" / f"M{i}_D{j}.pickle"
                        
                        sample_file.parent.mkdir(parents=True, exist_ok=True)
                    except AttributeError:
                        sample_file = Path(f"{self.foldername}/samples/M{i}_D{j}.pickle")
                        
                    if self.foldername and sample_file.is_file():
                        samples = load_data(sample_file)
                        pbar.total = len(self.Ms) * len(self.Ds[0]) * len(samples)
                        pbar.update(len(samples))
                    else:
                        query_indices = self._get_qsft_query_indices(self.Ms[i], self.Ds[i][j])
                        block_length = len(query_indices[0])
                        samples = np.zeros((len(query_indices), block_length), dtype=complex)
                        pbar.total = len(self.Ms) * len(self.Ds[0]) * len(query_indices)
                        if block_length > 10000:
                            for k in range(len(query_indices)):
                                samples[k] = self.subsample(query_indices[k])
                                pbar.update()
                        else:
                            all_query_indices = np.concatenate(query_indices)
                            all_samples = self.subsample(all_query_indices)
                            for k in range(len(query_indices)):
                                samples[k] = all_samples[k * block_length: (k+1) * block_length]
                                pbar.update()
                        if self.foldername:
                            save_data(samples, sample_file)

        for i in range(len(self.Ms)):
            for j in range(len(self.Ds[i])):
                try:
                    if self.type != 'None':
                        sample_file = Path(f"{self.foldername}/samples/{self.type}/M{i}_D{j}.pickle")
                    else: 
                        sample_file = Path(f"{self.foldername}/samples/M{i}_D{j}.pickle")
                except AttributeError:
                    sample_file = Path(f"{self.foldername}/samples/M{i}_D{j}.pickle")
                try:
                    if self.type != 'None' and self.type:
                        path_parts = self.type.split('/')
                        transform_file = Path(self.foldername) / "transforms" / Path(*path_parts) / f"U{i}_D{j}.pickle"
                    else:
                        transform_file = Path(self.foldername) / "transforms" / f"U{i}_D{j}.pickle"
                    
                    transform_file.parent.mkdir(parents=True, exist_ok=True)
                except AttributeError:
                    transform_file = Path(f"{self.foldername}/transforms/U{i}_{j}.pickle")
                samples = load_data(sample_file)
                for b in self.all_bs:
                    start_time = time.time()
                    self.Us[i][j][b] = self._compute_subtransform(samples, b)
                    self.transformTimes[i][j][b] = time.time() - start_time
                if self.foldername:
                    save_data((self.Us[i][j], self.transformTimes[i][j]), transform_file)

    def _generate_train_subsample(self):
        """
        Get all train sample indices and save them out 
        """
        self.Us = [[{} for j in range(len(self.Ds[i]))] for i in range(len(self.Ms))]
        self.transformTimes = [[{} for j in range(len(self.Ds[i]))] for i in range(len(self.Ms))]
        global newrun 
        newrun = False

        if self.foldername:
            Path(f"{self.foldername}/samples").mkdir(exist_ok=True)
            Path(f"{self.foldername}/transforms/").mkdir(exist_ok=True)

        """
        Extract indices
        """
        for i in range(len(self.Ms)):
            for j in range(len(self.Ds[i])):
                sample_file_indices = Path(f"{self.foldername}/samples/M{i}_D{j}_queryindices.pickle")
                sample_file_qaryindices = Path(f"{self.foldername}/samples/M{i}_D{j}_qaryindices.pickle")
                if self.foldername and not sample_file_indices.is_file():
                    query_indices = self._get_qsft_query_indices(self.Ms[i], self.Ds[i][j])
                    random_samples = np.array(dec_to_qary_vec(query_indices, self.q, self.n)).T
                    save_data4(random_samples, sample_file_indices)
                    save_data4(query_indices, sample_file_qaryindices)

    def _generate_test_subsample(self):
        """
        Get all test sample indices and save them out 
        """
        test_file_indices = Path(f"{self.foldername}/../test/signal_t_queryindices.pickle")
        test_file_qaryindices = Path(f"{self.foldername}/../test/signal_t_query_qaryindices.pickle")
        query_indices = self._get_random_query_indices(10000)
        save_data4(query_indices, test_file_qaryindices)
        random_samples = np.array(dec_to_qary_vec(query_indices, self.q, self.n)).T
        save_data4(random_samples, test_file_indices)

    def _generate_qary_vectors(self, n_samples, q, n, max_nonzero=2):
        """
        Generates q-ary vectors with a maximum number of nonzero indices
        """
        qary_vectors = []
        for _ in range(n_samples):
            # Generate a vector with at most max_nonzero nonzero indices
            non_zero_indices = np.random.choice(n, size=np.random.randint(1, max_nonzero + 1), replace=False)
            qary_vector = np.zeros(n, dtype=int)
            qary_vector[non_zero_indices] = np.random.randint(1, q, size=len(non_zero_indices))
            qary_vectors.append(qary_vector)
        return np.array(qary_vectors)

    def _subsample_uniform(self):
        """
        Uniformly subsamples the signal. Useful when you are solving via LASSO
        """
        if self.foldername:
            Path(f"{self.foldername}").mkdir(exist_ok=True)
        try:
            if self.type != 'None' and self.type:
                path_parts = self.type.split('/')
                sample_file = Path(self.foldername) / Path(*path_parts) / "signal_t.pickle"
            else:
                sample_file = Path(self.foldername) / "signal_t.pickle"

            sample_file.parent.mkdir(parents=True, exist_ok=True)
        except AttributeError:
            sample_file = Path(f"{self.foldername}/signal_t.pickle")
        if self.foldername and sample_file.is_file():
            signal_t = load_data(sample_file)
        else:
            query_indices = self._get_random_query_indices(self.query_args["n_samples"])
            samples = self.subsample(query_indices)
            signal_t = dict(zip(query_indices, samples))
            if self.foldername:
                save_data(signal_t, sample_file)
        self.signal_t = signal_t

    def get_all_qary_vectors(self):
        if self.L is None:
            self.L = np.array(qary_ints(self.b, self.q))  # List of all length b qary vectors
        return self.L

    def subsample(self, query_indices):
        raise NotImplementedError

    def _get_qsft_query_indices(self, M, D_sub):
        """
        Gets the indicies to be queried for a given M and D

        Parameters
        ----------
        M
        D_sub

        Returns
        -------
        base_inds_dec : list
        The i-th element in the list is the affine space {Mx + d_i, forall x}, but in a decimal index, because it is
        more efficient, where d_i is the i-th row of D_sub.
        """
        b = M.shape[1]
        L = self.get_all_qary_vectors()
        ML = (M @ L) % self.q
        base_inds = [(ML + np.outer(d, np.ones(self.q ** b, dtype=int))) % self.q for d in D_sub]
        base_inds = np.array(base_inds)
        base_inds_dec = []
        for i in range(len(base_inds)):
            base_inds_dec.append(qary_vec_to_dec(base_inds[i], self.q))
        return base_inds_dec
    

    def _get_random_query_indices(self, n_samples):
        """
        Returns random indicies to be sampled.

        Parameters
        ----------
        n_samples

        Returns
        -------
        base_ids_dec
        Indicies to be queried in decimal representation
        """
        n_samples = np.minimum(n_samples, self.N)
        base_inds_dec = [randint(0, self.N-1) for _ in range(n_samples)]
        return base_inds_dec

    def get_MDU(self, ret_num_subsample, ret_num_repeat, b, trans_times=False):
        """
        Allows the QSFT Class to get the effective Ms, Ds and Us (subsampled transforms).
        Parameters
        ----------
        ret_num_subsample
        ret_num_repeat
        b
        trans_times

        Returns
        -------
        Ms_ret
        Ds_ret
        Us_ret
        """
        Ms_ret = []
        Ds_ret = []
        Us_ret = []
        Ts_ret = []
        if ret_num_subsample <= self.num_subsample and ret_num_repeat <= self.num_repeat and b <= self.b:
            subsample_idx = np.random.choice(self.num_subsample, ret_num_subsample, replace=False)
            delay_idx = np.random.choice(self.num_repeat, ret_num_repeat, replace=False)
            for i in subsample_idx:
                Ms_ret.append(self.Ms[i][:, :b])
                Ds_ret.append([])
                Us_ret.append([])
                Ts_ret.append([])
                for j in delay_idx:
                    Ds_ret[-1].append(self.Ds[i][j])
                    Us_ret[-1].append(self.Us[i][j][b])
                    Ts_ret[-1].append(self.transformTimes[i][j][b])
            if trans_times:
                return Ms_ret, Ds_ret, Us_ret, Ts_ret
            else:
                return Ms_ret, Ds_ret, Us_ret
        else:
            raise ValueError("There are not enough Ms or Ds.")

    def _compute_subtransform(self, samples, b):
        transform = [gwht(row[::(self.q ** (self.b - b))], self.q, b) for row in samples]
        return transform

    def get_source_parity(self):
        return self.Ds[0][0].shape[0]
