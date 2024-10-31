'''
Class for computing the q-ary fourier transform of a function/signal
'''
import time
import numpy as np
from qsft.reconstruct import singleton_detection
from qsft.input_signal_subsampled import SubsampledSignal
from qsft.input_signal import Signal
from qsft.utils import bin_to_dec, qary_vec_to_dec, sort_qary_vecs, calc_hamming_weight, dec_to_qary_vec, load_data
from pathlib import Path

import cmath
from sklearn.linear_model import Ridge
import math

class QSFT(Signal):
    '''
    Class to encapsulate the configuration of our fourier algorithm.

    Attributes
    ---------
    reconstruct_method_source : str
    method of reconstruction for source coding: "identity" - default setting, should be used unless you know that all
                                                indicies have low hamming weight
                                                "coded" - Currently only supports prime q, if you know the max hamming
                                                weight of less than t this option should be used and will greatly reduce
                                                complexity. Note a source_decoder object must also be passed
    reconstruct_method_channel : str
    Method of reconstruction for channel coding: "mle" - exact MLE computation. Fine for small problems but not
                                                         recommended it is exponential in n
                                                 "nso" - symbol-wise recovery suitable when a repetition type code is used
                                                 "identity" - no channel coding, only use when there is no noise
    num_subsamples : int
    The number of different subsampling groups M used

    num_repeat : int
    When a repetition code is used for channel coding, (NSO) this is the number of repetitions

    b : int
    Size of the sub-sampling signal. In general, we need q^b = O(K) where K is the number of nonzero terms in the
    transform. In practice, any q^b > K typically works well.

    noise_sd : scalar
    A noise parameter. Roughly, the standard deviation of the noise if it was an additive gaussian.

    source_decoder : function
    A function that takes in a source coded index, and returns decoded value of that index. Only needed when
    reconstruct_method_source = "coded"
    '''
    def __init__(self, **kwargs):
        self.reconstruct_method_source = kwargs.get("reconstruct_method_source")
        self.reconstruct_method_channel = kwargs.get("reconstruct_method_channel")
        self.num_subsample = kwargs.get("num_subsample")
        self.num_repeat = kwargs.get("num_repeat")
        self.b = kwargs.get("b")
        self.source_decoder = kwargs.get("source_decoder", None)

    def transform(self, signal, verbosity=0, report=False, timing_verbose=False, **kwargs):
        """
         Computes the q-ary fourier transform of a signal object

         Arguments
         ---------

         signal : Signal
         Signal object to be transformed.

         verbosity : int
         Larger numbers lead to increased number of printouts

         timing_verbose : Boolean
         If set to True, outputs detailed information about the amount of time each transform step takes.

         report : Boolean
         If set to True this function returns optional outputs "runtime": transform_time + peeling_time,
         "n_samples": total number of samples,"locations": locations of nonzero indicies,"avg_hamming_weight" average
          hamming weight of non-zero indicies and "max_hamming_weight": the maximum hamming weight of a nonzero index

          Returns
          -------
          gwht : dict
          Fourier transform (WHT) of the input signal

          runtime : scalar
          transform time + peeling time.

          n_samples : int
          number of samples used in computing the transform.

          locations : list
          List of nonzero indicies in the transform.

          avg_hamming_weight : scalar
          Average hamming wieght of non-zero indicies.


          max_hamming_weight : int
          Max hamming weight among the non-zero indicies.
         """
        q = signal.q
        n = signal.n
        b = self.b

        omega = np.exp(2j * np.pi / q)
        result = []

        gwht = {}
        gwht_counts = {}

        peeling_max = q ** n
        peeled = set([])
        if isinstance(signal, SubsampledSignal):
            Ms, Ds, Us, Ts = signal.get_MDU(self.num_subsample, self.num_repeat, self.b, trans_times=True)
        else:
            raise NotImplementedError("QSFT currently only supports signals that inherit from SubsampledSignal")
        for i in range(len(Ds)):
            Us[i] = np.vstack(Us[i])
            Ds[i] = np.vstack(Ds[i])
        transform_time = np.sum(Ts)
        if timing_verbose:
            print(f"Transform Time:{transform_time}", flush=True)
        Us = np.array(Us)


        gamma = 0.5
        cutoff = 1e-9 + (1 + gamma) * (signal.noise_sd ** 2) / (q ** b)  # noise threshold
        cutoff = kwargs.get("cutoff", cutoff)

        if verbosity >= 2:
            print("cutoff = ", cutoff, flush=True)

        # begin peeling
        # index convention for peeling: 'i' goes over all M/U/S values
        # i.e. it refers to the index of the subsampling group (zero-indexed - off by one from the paper).
        # 'j' goes over all columns of the WHT subsample matrix, going from 0 to 2 ** b - 1.
        # e.g. (i, j) = (0, 2) refers to subsampling group 0, and aliased bin 2 (10 in binary)
        # which in the example of section 3.2 is the multiton X[0110] + X[1010] + W1[10]

        # a multiton will just store the (i, j)s in a list
        # a singleton will map from the (i, j)s to the true (binary) values k.
        # e.g. the singleton (0, 0), which in the example of section 3.2 is X[0100] + W1[00]
        # would be stored as the dictionary entry (0, 0): array([0, 1, 0, 0]).
        max_iter = 1000
        iter_step = 0
        cont_peeling = True
        num_peeling = 0
        conf_fraction = 0.1

        nonzero_bins = np.ones((len(Us), len(Us[0].T)))
        bins_to_check = np.ones((len(Us), len(Us[0].T)))

        peeling_start = time.time()
        singletons = {}  # dictionary from (i, j) values to the true index of the singleton, k.
        if timing_verbose:
            start_time = time.time()
        while cont_peeling and num_peeling < peeling_max and iter_step < max_iter:
            iter_step += 1
            if verbosity >= 2:
                print('-----')
                print("iter ", iter_step, flush=True)
                # print('the measurement matrix')
                # for U in Us:
                #     print(U)
            # first step: find all the singletons and multitons.
            # multitons = []  # list of (i, j) values indicating where multitons are.

            for i, (U, M, D) in enumerate(zip(Us, Ms, Ds)):
                for j, col in enumerate(U.T):
                    if bins_to_check[i, j] == 1 and nonzero_bins[i, j] == 1:
                        if np.linalg.norm(col) ** 2 > cutoff * len(col):
                            k = singleton_detection(
                                col,
                                method_channel=self.reconstruct_method_channel,
                                method_source=self.reconstruct_method_source,
                                q=q,
                                source_parity=signal.get_source_parity(),
                                nso_subtype="nso1",
                                source_decoder=self.source_decoder
                            )
                            signature = omega ** (D @ k)
                            rho = np.dot(np.conjugate(signature), col) / D.shape[0]
                            residual = col - rho * signature
                            res_norm_sq = np.linalg.norm(residual) ** 2
                            j_qary = dec_to_qary_vec([j], q, b).T[0]
                            bin_matching = np.all((M.T @ k) % q == j_qary)

                            if verbosity >= 5:
                                print((i, j), np.linalg.norm(residual) ** 2, cutoff * len(col))
                            if (not bin_matching) or np.linalg.norm(residual) ** 2 > cutoff * len(col):                                
                                if verbosity >= 6:
                                    print("We have a Multiton")
                            else:  # declare as singleton
                                singletons[(i, j)] = (k, rho, res_norm_sq)

                                if verbosity >= 3:
                                    print("We have a Singleton at " + str(k))
                        else:
                            nonzero_bins[i, j] = 0

                            # Find residual 
                            k = singleton_detection(
                                col,
                                method_channel=self.reconstruct_method_channel,
                                method_source=self.reconstruct_method_source,
                                q=q,
                                source_parity=signal.get_source_parity(),
                                nso_subtype="nso1",
                                source_decoder=self.source_decoder
                            )
                            signature = omega ** (D @ k)
                            rho = np.dot(np.conjugate(signature), col) / D.shape[0]
                            residual = col - rho * signature
                            res_norm_sq = np.linalg.norm(residual) ** 2
                            j_qary = dec_to_qary_vec([j], q, b).T[0]
                            bin_matching = np.all((M.T @ k) % q == j_qary)

                            if verbosity >= 6:
                                print("We have a Zeroton")
                    bins_to_check[i, j] = 0

            # all singletons and multitons are discovered
            # sort and display all discovered singletons
            singletons = dict(sorted(singletons.items(), key=lambda item: item[1][2]))
            if verbosity >= 5:
                print('singletons:')
                for ston in singletons.items():
                    print("\t{0} {1}\n".format(ston, bin_to_dec(ston[1][0])))


            # if there were no multi-tons or single-tons, decrease cutoff
            if len(singletons) == 0:
                cont_peeling = False


            # balls to peel (only peel the best conf_fraction of singletons)
            num_to_peel = max(int(conf_fraction * len(singletons)), 1)
            singletons_to_peel = dict(list(singletons.items())[:num_to_peel])
            balls_to_peel = set()
            ball_values = {}
            for (i, j) in singletons_to_peel:
                k, rho, res = singletons_to_peel[(i, j)]
                ball = tuple(k)  # Must be a hashable type
                balls_to_peel.add(ball)
                ball_values[ball] = rho
                result.append((k, ball_values[ball]))

            if verbosity >= 5:
                print('these balls will be peeled')
                print(balls_to_peel)
            # peel
            for ball in balls_to_peel:
                num_peeling += 1

                k = np.array(ball)[..., np.newaxis]
                potential_peels = [(l, qary_vec_to_dec(M.T.dot(k) % q, q)[0]) for l, M in enumerate(Ms)]
                if verbosity >= 6:
                    k_dec = qary_vec_to_dec(k, q)
                    peeled.add(int(k_dec))
                    print("Processing Singleton {0}".format(k_dec))
                    print(k)
                    for (l, j) in potential_peels:
                        print("The singleton appears in M({0}), U({1})".format(l, j))
                for peel in potential_peels:
                    signature_in_stage = omega ** (Ds[peel[0]] @ k)
                    to_subtract = ball_values[ball] * signature_in_stage.reshape(-1, 1)
                    # only peel if the bin is not a zeroton
                    if nonzero_bins[peel[0], peel[1]] == 1:
                        if verbosity >= 6:
                            print("Peeled ball {0} off bin {1}".format(qary_vec_to_dec(k, q), peel))
                        Us[peel[0]][:, peel[1]] -= np.array(to_subtract)[:, 0]
                        # reactivate the bin calculations
                        bins_to_check[peel[0], peel[1]] = 1
                        if peel in singletons:
                            singletons.pop(peel)

                if verbosity >= 5:
                    print("Iteration Complete: The peeled indicies are:")
                    print(np.sort(list(peeled)))

        loc = set()
        for k, value in result: # iterating over (i, j)s
            loc.add(tuple(k))
            if tuple(k) in gwht_counts:
                gwht[tuple(k)] = (gwht[tuple(k)] * gwht_counts[tuple(k)] + value) / (gwht_counts[tuple(k)] + 1)
                gwht_counts[tuple(k)] = gwht_counts[tuple(k)] + 1
            else:
                gwht[tuple(k)] = value
                gwht_counts[tuple(k)] = 1
        if timing_verbose:
            print(f"Peeling Time:{time.time() - start_time}", flush=True)

        peeling_time = time.time() - peeling_start

        if not report:
            return gwht
        else:
            n_samples = np.prod(np.shape(np.array(Us)))
            if len(loc) > 0:
                loc = list(loc)
                if kwargs.get("sort", False):
                    loc = sort_qary_vecs(loc)
                avg_hamming_weight = np.mean(calc_hamming_weight(loc))
                max_hamming_weight = np.max(calc_hamming_weight(loc))
            else:
                loc, avg_hamming_weight, max_hamming_weight = [], 0, 0
            result = {
                "gwht": gwht,
                "runtime": transform_time + peeling_time,
                "n_samples": n_samples,
                "locations": loc,
                "avg_hamming_weight": avg_hamming_weight,
                "max_hamming_weight": max_hamming_weight
            }

            return result
        
    def get_alpha(n):
        n = n / n**2
        sgn = -1 if n < 0 else 1
        scale = int(-math.floor(math.log10(abs(n))))
        if scale <= 0:
            scale = 1
        factor = 10**scale
        number = sgn*math.floor(abs(n)*factor)/factor
        scaling_factor = int(str(sgn*math.floor(abs(n)*factor))[:1])
        if scaling_factor == 0:
            scaling_factor = 1
        return number/scaling_factor
    
    def residual_sq(A, gwht, y):
        residual = np.dot(A, list(gwht)) - y
        residual_squared = np.linalg.norm(residual)**2 
        return residual_squared
