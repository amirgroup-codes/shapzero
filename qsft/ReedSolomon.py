import galois
from galois._codes._reed_solomon import decode_jit
import numpy as np
import math


class ReedSolomon(galois.ReedSolomon):
    """
    Class that extends galois.ReedSolomon. Mainly it is needed to implement syndrome decoding.

    Attributes
    ---------
    prime_field : GF.field
    The galois feild of base q

    s :
    """
    def __init__(self, n: int, t: int, q: int):
        self.prime_field = galois.GF(q)
        self.s = math.ceil(math.log(n) / math.log(q))
        nt = (q ** self.s) - 1 if n <= (q ** self.s) - 1 else (q ** (self.s + 1)) - 1
        self.s = self.s if n <= (q ** self.s) - 1 else self.s + 1
        self.ns = n
        super().__init__(n=nt, k=nt-2*t)

    def syndrome_decode(self, syndrome):
        """
        Syndrome Decoding for the RS decoder

        Parameters
        ----------
        syndrome : self.prime_field.array()
        The syndrome as represented in the prime field. (i.e., the raw output of D*k)

        Returns
        -------
        err : prime_field.array
        The decoded value of k

        n_errors : the number of decoding errors
        """
        # Invoke the JIT compiled function
        q = self.field.characteristic
        codeword = self.field.Zeros(self.n)
        syndrome_qs = self.field.Vector([syndrome[self.s * i:self.s * (i + 1)] for i in range(2 * self.t)])
        dec_codeword, n_errors = decode_jit(self.field)(codeword[:, np.newaxis].T, syndrome_qs[:, np.newaxis].T, self.c,
                                                        self.t, int(self.field.primitive_element))
        return -dec_codeword[:, -self.ns:], n_errors

    def get_delay_matrix(self):
        """
        Generates teh parity check matrix of the constructed ReedSolomon code in the prime field

        Returns
        -------
        D : GF.array
        Coded delay matrix
        """
        Hvec = self.H[:, -self.ns:].vector()
        p = self.get_parity_length()
        D = self.prime_field.Zeros((p+1, self.ns))
        for i in range(self.ns):
            for j in range(2 * self.t):
                D[(self.s * j + 1):(self.s * (j + 1) + 1), i] = Hvec[j, i, :]
        return D

    def get_parity_length(self):
        """
        Returns the number of parity symbols in the code
        Returns
        -------
        P
        """
        return 2*self.t*self.s
