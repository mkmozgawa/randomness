import random
from random import SystemRandom
import math
from fractions import Fraction
import scipy.special as sc
from scipy.stats import norm
import numpy as np

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

def urandom_sample(n, interval):
    lower, upper = interval[0], interval[1]
    cryptogen = SystemRandom()
    return [cryptogen.randrange(lower, upper) for i in range(n)]

def list_to_string(lst):
    lst = [str(x) for x in lst]
    return " ".join(lst)

def count_ones_zeros(bits_arr):
    return (sum(bits_arr), len(bits_arr)-sum(bits_arr)) # number of ones, number of zeroes

def monobit_test(bits_arr):
    '''
    In:
    n bitów wejścia
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
    s_obs = S_abs / math.sqrt(n)
    p = math.erfc(s_obs/math.sqrt(2))
    return(p >= 0.01)

def block_frequency_test(bits_arr, M):
    '''
    In:
    M - długość bloku
    n bitów wejścia
    Out:
    status (True: ciąg oceniony jako losowy, False: ciąg oceniony jako nielosowy)
    Kroki:
    1. Podziel bity wejściowe na N = floor(n/M) nienakładających się bloków. Odrzuć pozostałe bity.
    2. Dla każdego bloku M-bitów wylicz proporcję jedynek korzystając z podanego wzoru.
    3. Policz chi-kwadrat.
    4. Policz P-value jako (1- igamc(N/2, chi-kwadrat/2)).
    5. Jeśli P-value jest mniejsze niż ustalony poziom istotności, to ciąg NIE JEST losowy. W innym przypadku ciąg JEST losowy.

    Rekomendacja: n >= 100, M >= 20, M > .01n i N < 100.
    '''
    n = len(bits_arr)
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

def cumulative_sums(bits_arr, mode):
    '''
    In:
    n bitów wejścia
    mode (True = forward, False = backward)
    Out:
    status (True: ciąg oceniony jako losowy, False: ciąg oceniony jako nielosowy)
    Kroki:
    1. Znormalizuj bity wejścia poprzez przypisanie im wartości: (1) dla jedynki i (-1) dla zera.
    2. Policz sumy kolejnych coraz dłuższych podciągów bitów.
    3. Oblicz z = max_{1<=k<=n}|S_{k}} będące największą z wartości bezwzględnych sum z punktu 2.
    4. Policz P-value z podanego wzoru.
    5. Jeśli P-value jest mniejsze niż ustalony poziom istotności, to ciąg NIE JEST losowy. W innym przypadku ciąg JEST losowy.
    Rekomendacja: co najmniej 100 bitów wejścia.
    W mode=0 wysokie P-value świadczy o zbyt wielu zerach lub jedynkach na początku ciągu, w mode=1 - na końcu ciągu.
    Niskie P-value świadczy o zbyt równym wymieszaniu zer i jedynek.
    '''
    n = len(bits_arr)
    bits_norm = [1 if b==1 else -1 for b in bits_arr]

    sums = []

    if mode:
        for index, value in enumerate(bits_norm):
            sums.append(sum(bits_norm[:index+1]))
    else:
        for i in list(range(n-1, -1, -1)):
            sums.append(sum(bits_norm[i:]))

    sums_abs = [abs(x) for x in sums]
    z = max(sums_abs)

    sub_k_start = math.floor((-n/z+1)/4)
    sub_k_end = math.floor((n/z-1)/4)
    sub = 0
    for k in list(range(sub_k_start, sub_k_end+1)):
        sub += norm.cdf(((4*k+1)*z)/math.sqrt(n))
        sub -= norm.cdf(((4*k-1)*z)/math.sqrt(n))
    
    add_k_start = math.floor((-n/z-3)/4)
    add_k_end = math.floor((n/z-1)/4)
    add = 0
    for k in list(range(add_k_start, add_k_end+1)):
        add += norm.cdf(((4*k+3)*z)/math.sqrt(n))
        add -= norm.cdf(((4*k+1)*z)/math.sqrt(n))
    
    p = 1 - sub + add
    return(p >= 0.01)

def process_blocks_of_m_length(bits_arr, m):
    '''
    Kroki 1-4 metody approximate_entropy
    '''
    n = len(bits_arr)
    bits_to_append = bits_arr[:m-1]
    bits_arr = bits_arr + bits_to_append

    seqs = []
    for i in list(range(0, n)):
        seqs.append(''.join(str(b) for b in bits_arr[i:i+m]))
    
    seqs_count = {}
    for el in set(seqs):
        seqs_count[el] = seqs.count(el)
    
    c_i_n = {}
    for key, value in seqs_count.items():
        c_i_n[key] = value / n

    phi_m = 0
    for key, value in c_i_n.items():
        phi_m += value * math.log(value)

    return phi_m

def approximate_entropy(bits_arr, m):
    '''
    Input:
    n bitów wejścia
    m - długość bloku
    Output:
    status (True: ciąg oceniony jako losowy, False: ciąg oceniony jako nielosowy)
    Kroki:
    1. Dla m-długich nakładających się podciągów ciągu bitów wejścia policz ile jest różnych typów (np. dla 3-bitowych ciągów jest 2**3=8 podtypów) i policz ich entropię.
    2. Dla m+1-długich ciągów zrób to samo.
    3. Policz parametr ApEn będący różnicą (1) i (2). (Małe wartości ApEn sugerują regularność ciągu.)
    4. Policz P-value jako (1- igamc(2**(m-1), chi-kwadrat/2)).
    5. Jeśli P-value jest mniejsze niż ustalony poziom istotności, to ciąg NIE JEST losowy. W innym przypadku ciąg JEST losowy.
    Rekomendacje: m, n takie że m < floor(log_2(n)) - 5
    '''
    phi_m = process_blocks_of_m_length(bits_arr, m)
    phi_m_1 = process_blocks_of_m_length(bits_arr, m+1)
    
    ap_en = phi_m - phi_m_1
    chi_sq = 2*n*(math.log(2) - ap_en)
    p = 1 - sc.gammainc(2**(m-1), chi_sq/2)
    return(p >= 0.01)


if __name__ == "__main__":
    lower = 0
    upper = 1
    n = 500
    random.seed(SEED)

    lcg_numbers = lcg_random_sample(n, [lower,upper+1])
    mersenne_numbers = mersenne_random_sample(n, [lower,upper])
    urandom_numbers = urandom_sample(n, [lower, upper+1])

    print('LCG:')
    print('Monobit test zdany: %s' % monobit_test(lcg_numbers))
    print('Frequency test within a block zdany: %s' % block_frequency_test(lcg_numbers, 20))
    print('Cumulative sums test zdany: %s' % cumulative_sums(lcg_numbers, False))
    print('Approximate entropy test zdany: %s' % approximate_entropy(lcg_numbers, 2))

    print('\nMersenne-Twister:')
    print('Monobit test zdany: %s' % monobit_test(mersenne_numbers))
    print('Frequency test within a block zdany: %s' % block_frequency_test(mersenne_numbers, 20))
    print('Cumulative sums test zdany: %s' % cumulative_sums(mersenne_numbers, False))
    print('Approximate entropy test zdany: %s' % approximate_entropy(mersenne_numbers, 2))

    print('\nurandom:')
    print('Monobit test zdany: %s' % monobit_test(urandom_numbers))
    print('Frequency test within a block zdany: %s' % block_frequency_test(urandom_numbers, 20))
    print('Cumulative sums test zdany: %s' % cumulative_sums(urandom_numbers, False))
    print('Approximate entropy test zdany: %s' % approximate_entropy(urandom_numbers, 2))

    ## test z NISTa
    # bits = [1,0,1,1,0,1,0,1,0,1]
    # print(monobit_test(bits)) # p =~ 0.527

    ## test z NISTa
    # bits = [0,1,1,0,0,1,1,0,1,0]
    # print(block_frequency_test(bits, 3))

    ## test z NISTa
    # bits_s = list('1100100100001111110110101010001000100001011010001100001000110100110001001100011001100010100010111000')
    # bits = [int(b) for b in bits_s]
    # print(cumulative_sums(bits, True)) # p =~ 0.219194
    # print(cumulative_sums(bits, False)) # p =~ 0.114866

    ## test z NISTa
    # bits_s = list('1100100100001111110110101010001000100001011010001100001000110100110001001100011001100010100010111000')
    # bits = [int(b) for b in bits_s]
    # print(approximate_entropy(bits, 2)) # p =~ 0.2353