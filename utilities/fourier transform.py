# credit: https://towardsdatascience.com/fast-fourier-transform-937926e591cb
import numpy as np


def dft(X):
    X = np.asarray(X, dtype=float)
    N = X.shape[0]
    n = np.arange(N)
    k = n.reshape((N, 1))
    M = np.exp(-2j * np.pi * k * n / N)

    return np.dot(M, X)


x = np.random.random(1024)
print(np.allclose(dft(x), np.fft.fft(x)))


def fft(X):
    X = np.asarray(X, dtype=float)
    N = X.shape[0]
    if N % 2 > 0:
        raise ValueError("must be a power of 2")
    elif N <= 2:
        return dft(X)
    else:
        X_even = fft(X[::2])
        X_odd = fft(X[1::2])
        terms = np.exp(-2j * np.pi * np.arange(N) / N)
        return np.concatenate([X_even + terms[:int(N / 2)] * X_odd,
                               X_even + terms[int(N / 2):] * X_odd])


x = np.random.random(1024)
print(np.allclose(fft(x), np.fft.fft(x)))


def fft_vector(X):
    X = np.asarray(X, dtype=float)
    N = X.shape[0]
    if np.log2(N) % 1 > 0:
        raise ValueError("must be a power of 2")

    N_min = min(N, 2)

    n = np.arange(N_min)
    k = n[:, None]
    M = np.exp(-2j * np.pi * n * k / N_min)
    X = np.dot(M, X.reshape((N_min, -1)))

    while X.shape[0] < N:
        X_even = X[:, :int(X.shape[1] / 2)]
        X_odd = X[:, int(X.shape[1] / 2):]
        terms = np.exp(-1j * np.pi * np.arange(X.shape[0])
                       / X.shape[0])[:, None]
        X = np.vstack([X_even + terms * X_odd,
                       X_even - terms * X_odd])

    return X.ravel()


x = np.random.random(1024)
print(np.allclose(fft_vector(x), np.fft.fft(x)))

#  the FFTPACK algorithm behind numpy’s fft is a Fortran implementation
