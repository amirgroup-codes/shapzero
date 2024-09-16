import numpy as np
from qsft.lasso import lasso_decode
from qsft.qsft import QSFT
from qsft.utils import gwht, dec_to_qary_vec, NpEncoder
import json
from qsft.query import get_reed_solomon_dec
from pathlib import Path
from qsft.utils import load_data
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import sys 

class TestHelper:

    def __init__(self, signal_args, methods, subsampling_args, test_args, exp_dir, subsampling=True):

        self.n = signal_args["n"]
        self.q = signal_args["q"]
        if "type" in signal_args:
            self.type = signal_args["type"]
        else:
            self.type = None

        self.exp_dir = exp_dir
        self.subsampling = subsampling

        self.signal_args = signal_args
        self.subsampling_args = subsampling_args
        self.test_args = test_args

        if self.subsampling:
            if len(set(methods).intersection(["qsft"])) > 0:
                self.train_signal = self.load_train_data()
            # print("Quaternary Training data loaded.", flush=True)
            if len(set(methods).intersection(["qsft_binary"])) > 0:
                self.train_signal_binary = self.load_train_data_binary()
                # print("Binary Training data loaded.", flush=True)
            if len(set(methods).intersection(["lasso"])) > 0:
                self.train_signal_uniform = self.load_train_data_uniform()
                # print("Uniform Training data loaded.", flush=True)
            if len(set(methods).intersection(["qsft_coded"])) > 0:
                self.train_signal_coded = self.load_train_data_coded()
                # print("Uniform Training data loaded.", flush=True)
            self.test_signal = self.load_test_data()
            # print("Test data loaded.", flush=True)
        else:
            self.train_signal = self.load_full_data()
            self.test_signal = self.train_signal
            if any([m.startswith("binary") for m in methods]):
                raise NotImplementedError  # TODO: implement the conversion
            # print("Full data loaded.", flush=True)

    def generate_signal(self, signal_args):
        raise NotImplementedError

    def load_train_data(self):
        signal_args = self.signal_args.copy()
        query_args = self.subsampling_args.copy()
        query_args.update({
            "subsampling_method": "qsft",
            "query_method": "complex",
            "delays_method_source": "identity",
            "delays_method_channel": "nso"
        })
        signal_args["folder"] = self.exp_dir / "train"
        signal_args["query_args"] = query_args
        return self.generate_signal(signal_args)

    def load_train_data_coded(self):
        signal_args = self.signal_args.copy()
        query_args = self.subsampling_args.copy()
        query_args.update({
            "subsampling_method": "qsft",
            "query_method": "complex",
            "delays_method_source": "coded",
            "delays_method_channel": "nso",
            "t": signal_args["t"]
        })
        signal_args["folder"] = self.exp_dir / "train_coded"
        signal_args["query_args"] = query_args
        return self.generate_signal(signal_args)

    def load_train_data_binary(self):
        return None

    def load_train_data_uniform(self):
        signal_args = self.signal_args.copy()
        query_args = self.subsampling_args.copy()
        n_samples = query_args["num_subsample"] * (signal_args["q"] ** query_args["b"]) *\
                    query_args["num_repeat"] * (signal_args["n"] + 1)
        query_args = {"subsampling_method": "uniform", "n_samples": n_samples}
        signal_args["folder"] = self.exp_dir / "train_uniform"
        signal_args["query_args"] = query_args
        return self.generate_signal(signal_args)

    def load_test_data(self):
        signal_args = self.signal_args.copy()
        (self.exp_dir / "test").mkdir(exist_ok=True)
        signal_args["query_args"] = {"subsampling_method": "uniform", "n_samples": self.test_args.get("n_samples")}
        signal_args["folder"] = self.exp_dir / "test"
        signal_args["noise_sd"] = 0
        return self.generate_signal(signal_args)

    def load_full_data(self):
        #   TODO: implement
        return None

    def compute_model(self, method, model_kwargs, report=False, verbosity=0):
        if method == "gwht":
            return self._calculate_gwht(model_kwargs, report, verbosity)
        elif method == "qsft":
            return self._calculate_qsft(model_kwargs, report, verbosity)
        elif method == "qsft_binary":
            return self._calculate_qsft_binary(model_kwargs, report, verbosity)
        elif method == "qsft_coded":
            return self._calculate_qsft_coded(model_kwargs, report, verbosity)
        elif method == "lasso":
            return self._calculate_lasso(model_kwargs, report, verbosity)
        else:
            raise NotImplementedError()

    def test_model(self, method, **kwargs):
        if method == "qsft" or method == "qsft_coded" or method == "lasso":
            return self._test_qary(**kwargs)
        elif method == "qsft_binary":
            return self._test_binary(**kwargs)
        else:
            raise NotImplementedError()

    def _calculate_gwht(self, model_kwargs, report=False, verbosity=0):
        """
        Calculates GWHT coefficients of the RNA fitness function. This will try to load them
        from the results folder, but will otherwise calculate from scratch. If save=True,
        then coefficients will be saved to the results folder.
        """
        if verbosity >= 1:
            print("Finding all GWHT coefficients")

        beta = gwht(self.train_signal, q=4, n=self.n)
        print("Found GWHT coefficients")
        return beta

    def _calculate_qsft(self, model_kwargs, report=False, verbosity=0):
        """
        Calculates GWHT coefficients of the RNA fitness function using QSFT.
        """
        if verbosity >= 1:
            print("Estimating GWHT coefficients with QSFT")
        qsft = QSFT(
            reconstruct_method_source="identity",
            reconstruct_method_channel="nso",
            num_subsample=model_kwargs["num_subsample"],
            num_repeat=model_kwargs["num_repeat"],
            b=model_kwargs["b"]
        )
        self.train_signal.noise_sd = model_kwargs["noise_sd"]
        out = qsft.transform(self.train_signal, verbosity=verbosity, timing_verbose=(verbosity >= 1), report=report)
        if verbosity >= 1:
            print("Found GWHT coefficients")
        return out

    def _calculate_qsft_coded(self, model_kwargs, report=False, verbosity=0):
        """
        Calculates GWHT coefficients of the RNA fitness function using QSFT.
        """
        if verbosity >= 1:
            print("Estimating GWHT coefficients with QSFT")

        decoder = get_reed_solomon_dec(self.signal_args["n"], self.signal_args["t"], self.signal_args["q"])
        qsft = QSFT(
            reconstruct_method_source="coded",
            reconstruct_method_channel="nso",
            num_subsample=model_kwargs["num_subsample"],
            num_repeat=model_kwargs["num_repeat"],
            b=model_kwargs["b"],
            source_decoder=decoder
        )
        self.train_signal_coded.noise_sd = model_kwargs["noise_sd"]
        out = qsft.transform(self.train_signal_coded, verbosity=verbosity, timing_verbose=(verbosity >= 1), report=report)
        if verbosity >= 1:
            print("Found GWHT coefficients")
        return out

    def _calculate_qsft_binary(self, model_kwargs, report=False, verbosity=0):
        """
        Calculates GWHT coefficients of the RNA fitness function using QSFT.
        """
        factor = round(np.log(self.q) / np.log(2))

        if verbosity >= 1:
            print("Estimating GWHT coefficients with QSFT")
        qsft = QSFT(
            reconstruct_method_source="identity",
            reconstruct_method_channel="nso",
            num_subsample=model_kwargs["num_subsample"],
            num_repeat=max(1, model_kwargs["num_repeat"] // factor),
            b=factor * model_kwargs["b"],
        )
        self.train_signal_binary.noise_sd = model_kwargs["noise_sd"] / factor
        out = qsft.transform(self.train_signal_binary, verbosity=verbosity, timing_verbose=(verbosity >= 1), report=report)
        if verbosity >= 1:
            print("Found GWHT coefficients")
        return out

    def _calculate_lasso(self, model_kwargs, report=False, verbosity=0):
        """
        Calculates GWHT coefficients of the RNA fitness function using LASSO. This will try to load them
        from the results folder, but will otherwise calculate from scratch. If save=True,
        then coefficients will be saved to the results folder.
        """
        if verbosity > 0:
            print("Finding Fourier coefficients with LASSO")

        self.train_signal_uniform.noise_sd = model_kwargs["noise_sd"]
        out = lasso_decode(self.train_signal_uniform, model_kwargs["n_samples"], noise_sd=model_kwargs["noise_sd"])

        if verbosity > 0:
            print("Found Fourier coefficients")

        return out
    
    def _test_qary(self, beta):
        """
        :param beta:
        :return:
        """
        if len(beta.keys()) > 0:
            
            # Import test signal 
            try:
                if self.type != 'None':
                    sample_file = Path(f"{self.exp_dir}/test/{self.type}/signal_t.pickle")
                else: 
                    sample_file = Path(f"{self.exp_dir}/test/signal_t.pickle")
            except AttributeError:
                sample_file = Path(f"{self.exp_dir}/test/signal_t.pickle")
            signal_t = load_data(sample_file)
            self.test_signal.signal_t = signal_t

            test_signal = self.test_signal.signal_t
            (sample_idx_dec, samples) = list(test_signal.keys()), list(test_signal.values())
            batch_size = 10000

            beta_keys = list(beta.keys())
            beta_values = list(beta.values())
            y_hat = []
            for i in range(0, len(sample_idx_dec), batch_size):
                sample_idx_dec_batch = sample_idx_dec[i:i + batch_size]
                sample_idx_batch = dec_to_qary_vec(sample_idx_dec_batch, self.q, self.n)
                freqs = np.array(sample_idx_batch).T @ np.array(beta_keys).T
                H = np.exp(2j * np.pi * freqs / self.q)
                y_hat.append(H @ np.array(beta_values))

            y_hat = np.concatenate(y_hat)
            # print(len(y_hat))


            """
            Compute R^2 and NMSE
            """
            nmse = np.linalg.norm(y_hat - samples) ** 2 / np.linalg.norm(samples) ** 2
            y = np.real(samples)
            y_hat = np.real(y_hat)
            r2_value = r2_score(y, y_hat)


            """
            Plotting test samples
            """
            marker = 'o'
            color = '#3c5488'
            font_size = 15

            fig, ax = plt.subplots(figsize=(7, 7))
            scatter = plt.scatter(y, y_hat, marker=marker, color=color, label='Empirical samples', s=10)  # Adjust marker size with 's' parameter

            # Fit a linear regression line
            coefficients = np.polyfit(y, y_hat, 1)
            poly_fit = np.poly1d(coefficients)
            line_of_best_fit = poly_fit(y)
            plt.plot(y, line_of_best_fit, color='#9b9ca0', linewidth=2, label='Line of Best Fit')

            # Set labels and fonts
            ax.set_ylabel('q-sft predicted', fontsize=font_size)
            ax.set_xlabel('Model scores', fontsize=font_size)
            ax.tick_params(axis='both', labelsize=font_size)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            max_limit = max(max(y), max(y_hat))+1
            min_limit = min(min(y), min(y_hat))
            ax.set_xlim(min_limit, max_limit)
            ax.set_ylim(min_limit, max_limit)

            # Print the R^2 value in the legend
            legend_label = f'Test samples (R$^2$ = {r2_value:.2f})'
            plt.legend([scatter], [legend_label], fontsize=font_size, loc='lower right')

            # Save the plot
            plot_file = Path(f"{self.exp_dir}/test_samples.png")
            plt.savefig(f"{plot_file}")

            return nmse, r2_value
        else:
            return 1, -1

    def _test_binary(self, beta):
        """
        :param beta:
        :return:
        """
        if len(beta.keys()) > 0:
            test_signal = self.test_signal.signal_t
            (sample_idx_dec, samples) = list(test_signal.keys()), list(test_signal.values())
            batch_size = 10000

            beta_keys = list(beta.keys())
            beta_values = list(beta.values())

            y_hat = []
            for i in range(0, len(sample_idx_dec), batch_size):
                sample_idx_dec_batch = sample_idx_dec[i:i + batch_size]
                sample_idx_batch = dec_to_qary_vec(sample_idx_dec_batch, 2, 2 * self.n)
                freqs = np.array(sample_idx_batch).T @ np.array(beta_keys).T
                H = np.exp(2j * np.pi * freqs / 2)
                y_hat.append(H @ np.array(beta_values))

            # TODO: Write with an if clause
            y_hat = np.abs(np.concatenate(y_hat))



            return np.linalg.norm(y_hat - samples) ** 2 / np.linalg.norm(samples) ** 2
        else:
            return 1
