import random
import math
from fractions import Fraction
import scipy.special as sc

SEED = 1234567890

def LCG(a, c, m, seed):     
    xi = seed
    while True:
        xf = (a * xi + c) % m
        xi = xf
        yield xf

def lcg_random_sample(n, interval, seed = SEED):
    lower, upper = interval[0], interval[1]
    sample = []
    glibc = LCG(1103515245, 12345, 2 ** 32, seed)

    for i in range(n):
        observation = (upper - lower) * (next(glibc) / (2 ** 32 - 1)) + lower
        sample.append(int(observation))

    return sample

def mersenne_random_sample(n, interval):
    lower, upper = interval[0], interval[1]
    sample = []
    
    for i in range(n):
        observation = random.randint(lower,upper)
        sample.append(int(observation))
    
    return sample

def list_to_string(lst):
    lst = [str(x) for x in lst]
    return " ".join(lst)

def count_ones_zeros(bits_arr):
    return (sum(bits_arr), len(bits_arr)-sum(bits_arr)) # number of ones, number of zeroes

def monobit_test(bits_arr):
    '''
    In:
    n - liczba bitów
    Out:
    status (True: ciąg oceniony jako losowy, False: ciąg oceniony jako nielosowy)
    Kroki:
    1. Policz liczbę jedynek i zer w ciągu bitów.
    2. Policz liczbę S odpowiadającą przełożeniu jedynek na (1) i zer na (-1). Weź jej bezwględną wartość.
    3. Policz |S|/(sqrt(liczba cyfr)).
    4. Policz P-value jako erfc(wartość z punktu 3 / sqrt(2)). erfc jest uzupełniającą funkcją funkcją błędu (https://pl.wikipedia.org/wiki/Funkcja_b%C5%82%C4%99du)
    5. Jeśli P-value jest mniejsze niż ustalony poziom istotności, to ciąg NIE JEST losowy. W innym przypadku ciąg JEST losowy.

    Rekomendacja: n >= 100.
    '''
    n = len(bits_arr)
    n_ones, n_zeros = count_ones_zeros(bits_arr)

    S_abs = abs(n_ones - n_zeros)
    p = math.erfc(S_abs/math.sqrt(n) * math.sqrt(2))
    return(p >= 0.01)

def frequency_test(bits_arr):
    '''
    In:
    M - długość bloku
    n - liczba bitów
    Out:
    status (True: ciąg oceniony jako losowy, False: ciąg oceniony jako nielosowy)
    Kroki:
    1. Podziel bity wejściowe na N = floor(n/M) nienakładających się bloków. Odrzuć pozostałe bity.
    2. Dla każdego bloku M-bitów wylicz proporcję jedynek korzystając z podanego wzoru.
    3. Policz chi-kwadrat.
    4. Policz P-value jako igamc(N/2, chi-kwadrat/2).
    5. Jeśli P-value jest mniejsze niż ustalony poziom istotności, to ciąg NIE JEST losowy. W innym przypadku ciąg JEST losowy.

    Rekomendacja: n >= 100, M >= 20, M > .01n i N < 100.
    '''
    n = len(bits_arr)
    M = 20
    N = int(math.floor(n/M))
    if N > 99:
        N = 99
        M = int(n/N)
    
    proportions = []
    for i in list(range(N)):
        block = bits_arr[i*M : (i+1)*M]
        n_ones, _ = count_ones_zeros(block)
        proportions.append(Fraction(n_ones, M))

    chi = 0
    for pr in proportions:
        chi += 4*M*((pr - Fraction(1,2))**2)
    
    p = 1 - sc.gammainc(N/2, float(chi/2))
    return(p >= 0.01)

if __name__ == "__main__":
    lower = 0
    upper = 1
    n = 1000
    random.seed(SEED)

    lcg_numbers = lcg_random_sample(n, [lower,upper])
    mersenne_numbers = mersenne_random_sample(n, [lower,upper])

    print(monobit_test(lcg_numbers))
    print(monobit_test(mersenne_numbers))

    print(frequency_test(lcg_numbers))
    print(frequency_test(mersenne_numbers))

    # test the below with M=3 for NIST example
    # bits = [0,1,1,0,0,1,1,0,1,0]
    # print(frequency_test(bits))