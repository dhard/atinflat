#! /usr/bin/python
from __future__ import division 
from __future__ import print_function
from optparse import OptionParser, OptionValueError
#from types import FloatType
from scipy import stats
from bitstring import Bits
from math import log, exp
from decimal import *
import itertools
from itertools import repeat, chain, combinations, product, combinations_with_replacement
from sympy import multinomial_coefficients
from functools import reduce
from collections import Counter
import multiprocessing
import time, sys, re, os, operator
import numpy as np
import pdb
import joblib

def printline(m):
    return re.sub('  ',' ',re.sub('\n',',',np.array2string(m)))

def powerset(iterable):
    """
    powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)
    https://docs.python.org/3/library/itertools.html
    """
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

def masked_genotypes_gen(tRNAs,aaRSs):
    for sm in powerset(range(tRNAs)):
        setm = set(sm)
        if bool(setm):
            for sn in powerset(range(aaRSs)):
                setn = set(sn)
                if bool(setn):
                    for st in powerset(range(tRNAs)):
                        sett     = set(st)
                        if bool(sett & setm):
                            for sa in powerset(range(aaRSs)):
                                seta     = set(sa)
                                if bool(seta & setn):
                                    yield setm,setn,sett,seta

def genotypes_gen(tRNAs,aaRSs):
    for st in powerset(range(tRNAs)):
        sett     = set(st)
        for sa in powerset(range(aaRSs)):
            seta     = set(sa)
            yield sett,seta

from tempfile import mkdtemp
cachedir = mkdtemp()
from joblib import Memory
memory = Memory(cachedir=cachedir, mmap_mode='r', verbose=0)

sbmmd     = dict() # site-block match matrix dictionary keyed by joblib hashes 
@memory.cache
def sbm_matrix(k,m=None):
    if k in sbmmd:
        return sbmmd[k]
    elif m is not None:
        sbmmd[k] = m
        return m
    else:
        return None

mmd     = dict() # match matrix dictionary keyed by fitness 
@memory.cache
def mmatrix(k,m=None):
    if k in mmd:
        return mmd[k]
    elif m is not None:
        mmd[k] = m
        return m
    else:
        return None

degeneracy = dict()
off        = dict()
# this could be rewritten to take advantage of pool.imap().
def compute_degeneracy(tRNAs,aaRSs,mask,cache):
    """
    This function computes all possible site-block-match-matrices
    and their encodable genetic degeneracies
    """
    uni_t       = set(range(tRNAs))
    uni_a       = set(range(aaRSs))
    if mask:
        zeros   = 2**(2*(tRNAs+aaRSs))
    else:
        zeros   = 2**(tRNAs+aaRSs)

    if mask:
        genotypes = masked_genotypes_gen(tRNAs,aaRSs)
        for setm,setn,sett,seta in genotypes:
            offm = uni_t - setm
            offn = uni_a - setn
            settc    = uni_t - sett
            sett     &= setm
            settc    &= setm
            setac     = uni_a - seta
            seta     &= setn
            setac    &= setn

            m = np.zeros((tRNAs,aaRSs),dtype=np.int16)
            for match in chain(product(sett,seta),product(settc,setac)):
                m[match] += 1
            if (m==0).all():
                print ('# huh! in compute_degeneracy') # why do we get here?
                continue
            key = joblib.hash(m)
            
            if key in degeneracy:
                degeneracy[key] += 1
                zeros -= 1
                off[key] += len(offm | offn)
            else:
                degeneracy[key] = 1
                if cache:
                    sbm_matrix(key,m)
                else:
                    sbmmd[key] = m
            off[key] = len(offm | offn)
            zeros -= 1
            
    else:
        genotypes = genotypes_gen(tRNAs,aaRSs)
        for sett,seta in genotypes:
            settc    = uni_t - sett
            setac     = uni_a - seta        
            m = np.zeros((tRNAs,aaRSs),dtype=np.int16)
            for match in chain(product(sett,seta),product(settc,setac)):
                m[match] += 1
            key = joblib.hash(m)
            
            if key in degeneracy:
                degeneracy[key] += 1
                zeros -= 1
            else:
                degeneracy[key] = 1
                if cache:
                    sbm_matrix(key,m)
                else:
                    sbmmd[key] = m
                zeros -= 1
    if zeros:
        m0  = np.zeros((tRNAs,aaRSs),dtype=np.int16)
        key = joblib.hash(m0)
        degeneracy[key] = zeros
        off[key]        = 0
        if cache:
            sbm_matrix(key,m0) # matrix[key] = mmatrix[key] = m0
        else:
            sbmmd[key] = m0

def compute_match_matrix(g,width,pairs,mask):
    length  = 2 * width * (pairs)
    tbegin  = 0
    tend    = width
    abegin  = width
    aend    = 2*width
    if mask:
        length  *= 2
        mtbegin = int(length / 2)
        mtend   = mtbegin + width
        mabegin = mtbegin + width
        maend   = mtbegin + 2 * width

    t = []
    a = []
    if mask:
        mt = []
        ma = []


    for i in range(pairs): 
        t.append(g[tbegin:tend])
        a.append(g[abegin:aend])
        tbegin += 2 * width
        tend   += 2 * width
        abegin += 2 * width
        aend   += 2 * width
        if mask:
            mt.append(g[mtbegin:mtend])
            ma.append(g[mabegin:maend])
            mtbegin += 2 * width
            mtend   += 2 * width
            mabegin += 2 * width
            maend   += 2 * width
            
    ## compute matching function
    m = np.zeros((pairs,pairs),dtype=np.int16)
    for i in range(pairs):
        for j in range(pairs): 
            taxnor       = ~(t[i] ^ a[j])
            if mask:
                maskand  =  mt[i] & ma[j]  
                m[i,j]   = (maskand & taxnor).count(1)
            else:
                m[i,j]   = (taxnor).count(1)
    return m

def compute_coding_matrix(m,kdmax,epsilon,square):
    ## compute energies and kinetic off-rates between tRNAs and aaRSs from the match matrix
    #kd = 1/((1 / kdmax) * np.exp(m * epsilon)) # # K_ij = (kdmax)^-1 * exp [m_ij * epsilon], and k_d = (1/K_ij)

    kd = kdmax / np.exp(m * epsilon)
    if square:
        kd = kd**2

    ## compute the "code matrix" of conditional decoding probabilities 
    c  = np.zeros((tRNAs,aaRSs),dtype=float)
    for i in range(tRNAs):
        for j in range(aaRSs):
            c [i,j] = stats.hmean(kd [i,:]/kd [i,j])/aaRSs
    return c,kd

def compute_fitness_given_coding_matrix(c,kd,coords,rate,square):
    f     = 1
    for i in range(tRNAs):
        fc = 0
        for j in range(aaRSs):
            fij  = phi**(abs(coords[i] - coords[j]))
            if rate:
                fc  += c [i,j] * fij
            else:
                fc  += c [i,j] * fij
        f  *= fc

    if rate:
        avg_kd  = stats.hmean(np.diagonal(kd))
        if square:
            f      *= 1.0 - exp(-1.505764 * avg_kd / 9680) # curve fit to data of Paulander et al (2007)
        else:
            f      *= 1.0 - exp(-1.505764 * avg_kd / 44)   # curve fit to data of Paulander et al (2007)

    return f


# this is the main target function for the multiprocessing pool during computation of the landscape
def compute_fitness(args):
    #input m is the width-th cartesian power of site-block-match-matrices
    (m,d,o),kdmax,epsilon,coords,rate = args
    
    ## if nsites:
    ##     m = np.clip(m,None,nsites)
    ## compute energies and kinetic off-rates between tRNAs and aaRSs from the match matrix
    #kd = 1/((1 / kdmax) * np.exp(m * epsilon)) # # K_ij = (kdmax)^-1 * exp [m_ij * epsilon], and k_d = (1/K_ij)
    kd = kdmax / np.exp(m * epsilon) 
    kd2 = kd**2

    ## compute the "code matrix" of conditional decoding probabilities 
    c  = np.zeros((tRNAs,aaRSs),dtype=float)
    c2 = np.zeros((tRNAs,aaRSs),dtype=float)
    for i in range(tRNAs):
        for j in range(aaRSs):
            c [i,j] = stats.hmean(kd [i,:]/kd [i,j])/aaRSs
            c2[i,j] = stats.hmean(kd2[i,:]/kd2[i,j])/aaRSs

            
    ## compute fitness bounds (f is in dead cells at equilibrium and f2 is with maximum kinetic proofreading)              
    f     = 1
    f2    = 1

    if rate:
        max_rate2 = kdmax**2
        max_rate  = kdmax
    
    for i in range(tRNAs):
        fc = fc2 = 0
        for j in range(aaRSs):
            fij  = phi**(abs(coords[i] - coords[j]))
            if rate:
                fc  += c [i,j] * fij * (kd[i,j]/max_rate) 
                fc2 += c2[i,j] * fij * (kd[i,j]/max_rate2) 
            else:
                fc  += c [i,j] * fij
                fc2 += c2[i,j] * fij
        f  *= fc
        f2 *= fc2

    if rate:
        avg_kd  = stats.hmean(np.diagonal(kd))
        avg_kd2 = stats.hmean(np.diagonal(kd2))
        f      *=  1.0 - exp(-1.505764 * avg_kd / 44)   # from curve fit to ref Paulander
        f2     *=  1.0 - exp(-1.505764 * avg_kd / 9680) # from curve fit to ref Paulander
    
    return m,d,o,f,f2


if __name__ == '__main__':
    starttime = time.time()
    version = 0.6
    prog = 'atinflat'
    usage = '''usage: %prog [options]

    atINFLAT: aaRS-tRNA-Interaction-Network-Fitness-LAndscape-Topographer

    Stationary frequencies and fitnesses over a N x N match landscape
    representing co-evolving interaction-determining features in a
    feed-forward macromolecular interaction network of N species of
    tRNA and N species of aminoacyl-tRNA-synthetase (aaRS).

    The network is evolving to express N amino acids in N >= 2 message
    site-types distributed evenly over a unit interval
    amino-acid/site-type space with one amino-acid site-type at
    coordinate 0 and one at coordinate 1. To each amino acid
    corresponds one aaRS that charges it exclusively and without
    error, and one site-type of equal frequency in which it attains
    perfect fitness. To each site-type corresponds one distinct
    species of codon that is read by a distinct species of
    tRNA. Networks evolve to match corresponding pairs of tRNA and
    aaRS species so that every codon produces that amino acid which
    best fits its site-type. The fitness of every amino acid in its
    matched site-type is the maximal fitness 1, while other amino
    acids contribute fitness phi^(|coord(a)-coord(b)|) with
    missense-per-site cost 0 < phi < 1. Fitness is arithmetically
    averaged within site-types and multiplied over site types.
    
    
    This program computes the complete fitness landscape over all
    binary genotype sequences of length L = <width> * (2 * <pairs>) =
    (2wN) representing the interaction-determining genotypes of ,
    where

        These 2^L genotypes represent every genetic state available within
    the interaction channel of length <width> shared across all (N^2)
    species of tRNA and aaRSs respectively assuming aaRS-tRNA
    interactions occur exclusively within each of <width>
    "site-blocks." Interaction matching and energy is additive over
    site-blocks. 
    
    assuming additivity of interaction energies over sites
    with parametrizable
     - interaction interfaces (width/number of potentially interacting sites and energy gap increments)
     - matching rules (site-symmetrical interaction rules of features across molecules)
     - interaction kinetics (tunable off-rate energy gaps and increments between aaRSs and tRNAs)
    -  evolutionary fixation process (strength of selection, constant effective population size)
    
    Copyright (2018) David H. Ardell
    All Wrongs Reversed.
    
    Please cite Collins-Hed et al. 2018 in published works using this software.
    
    Examples:
    '''
    parser = OptionParser(usage=usage,version='{:<3s} version {:3.1f}'.format(prog,version))
    parser.disable_interspersed_args()
    
    parser.add_option("-w","--width",
                      dest="width", type="int", default=2,
                      help="set interaction interface width/max matches, Default: %default")

    parser.add_option("-n","--nsites",
                      dest="nsites", type="int", default=None,
                      help="set number of matches to reach dissociation rate kdnsites.  Default: <width>")

    parser.add_option("-r","--rate",
                      dest="rate", action="store_true",
                      help="make fitness depend not only on accuracy but also rate of dissociation.  Default: False")

    parser.add_option("-p","--pairs",
                      dest="pairs", type="int", default=2,
                      help="set equal number of aaRS/tRNA pairs\n Default: %default")
    
    parser.add_option("-m","--mask", action="store_true",
                      dest="mask",
                      help="use evolveable mask bits as per-site interaction modifiers, Default: False")

    parser.add_option("-B","--beta",
                      dest="beta", type="int", default=100,
                      help="set beta, a function of the constant population size. See Sella (2009). Default: %default")

    parser.add_option("--phi",
                      dest="phi", type="float", default=0.99,
                      help="set phi, the maximum missense fitness-per-site penalty, 0 < phi < 1. See Sella and Ardell (2001). Default: %default")
    
    parser.add_option("--kdmax",
                      dest="kdmax", type="float", default=10000,
                      help="set Kdmax in sec^-1, the maximum dissociation rate constant (weakest binding)). Default: %default")
    
    parser.add_option("--kdnsites",
                      dest="kdnsites", type="float", default=220,
                      help="set Kdnsites in sec^-1, the dissociation rate constant reached at nsites. Default: %default")

    parser.add_option("--verbose",
                      dest="verbose",  action="store_true",
                      help="print more output about site blocks etc. Default: False")

    parser.add_option("-g","--genotypes",
                      dest="genotypes", type="string", default=None,
                      help="compute match and code matrices and fitness for binary string genotypes in file and exits. Assumes proofreading. If mask is True, genotype format is t11..t1w.a11..a1w...tP1..tPw.aP1..aPw.m11..m1w.n11..n1w...nP1..nPw, where mij is the maskbit for tij, nij is the maskbit for aij, w is <width> and P is <pairs>. You must set other parameters manually to match your genotype format. Default: %default")

    parser.add_option("--chunk",
                      dest="chunk", type="int", default=5,
                      help="set chunksize for Pool workers\n Default: %default")

    parser.add_option("--pool",
                      dest="pool", type="int", default=multiprocessing.cpu_count(),
                      help="set poolsize, number of Pool workers, default is the detected #CPUs\n Default: %default")

    parser.add_option("--cache",
                      dest="cache", action="store_true",
                      help="memcache numpy array dictionaries with joblib\n Default: False")
    
    myargv = sys.argv
    (options, args) = parser.parse_args()
    if len(args) != 0:
        parser.error("expects zero arguments")
        
    maxf        = 0
    maxf2       = 0
    fitb        = {}
    fitb2       = {}

    beta        = options.beta
    phi         = options.phi
    width       = options.width
    nsites      = options.nsites
    rate        = options.rate
    pairs       = options.pairs
    mask        = options.mask
    kdmax       = options.kdmax
    kdnsites    = options.kdnsites
    chunksize   = options.chunk
    poolsize    = options.pool
    cache       = options.cache

    verbose     = options.verbose
    genotypes   = options.genotypes


    ##     epsilon     = (log (kdmax) - log (kdnsites)) / nsites
    ## else:
    if not nsites:
        nsites = width
    epsilon     = (log (kdmax) - log (kdnsites)) / nsites

    #if pairs:
    aaRSs = pairs
    tRNAs = pairs
    length      = width * (tRNAs + aaRSs)
    if mask:
        length *= 2

        
    sfb         = 0
    sffb        = 0
    sfb2        = 0
    sffb2       = 0

    # calculate site-type space
    cuts  = pairs - 1
    segment = 1.0 / cuts
    coords = []
    for a in range(aaRSs):
        coords.append(segment * a)
        
    print('# {:<3s} version {:3.1f}'.format(prog,version))
    print('# Copyright (2018) David H. Ardell.')
    print('# All Wrongs Reversed.')
    print('#')
    print('# Please cite Collins-Hed (2018) in published works using this software.')
    print('#')
    print('# execution command:')
    print('# '+' '.join(myargv))
    print('#')
    print('# pairs     :  {}'.format(pairs))
    print('# tRNAs     :  {}'.format(pairs))
    print('# aaRSs     :  {}'.format(pairs))
    print('# width     :  {}'.format(width))
    print('# nsites    :  {}'.format(nsites))
    print('# rate      :  {}'.format(rate))
    print('# mask      :  {}'.format(mask))
    print('# length    :  {}'.format(length))
    print('# kdmax     :  {}'.format(kdmax))
    print('# kdnsites  :  {}'.format(kdnsites))
    print('# epsilon   :  {}'.format(epsilon))
    print('# phi       :  {}'.format(phi))
    print('# beta      :  {}'.format(beta))
    print('# coords    :  {}'.format(coords))
    print('#')
    print('# verbose   :  {}'.format(verbose))
    print('# genotypes :  {}'.format(genotypes))
    print('# pool-size :  {}'.format(poolsize))
    print('# chunk-size:  {}'.format(chunksize))
    print('# cache     :  {}'.format(cache))
    print('#')
    if genotypes:
        print('#')
        print('# analyzing binary genotypes in file {}, and exiting.'.format(genotypes))
        b = ''
        g = None
        # this will be updated to print the off-mask statistic
        with open(genotypes) as f:
            for line in f:
                strip = re.sub('\s+','',line)
                match = re.search('^[01]+', strip)
                if match:
                    b  = match.group(0)
                    g = Bits(bin=b)
                    m = compute_match_matrix(g,width,pairs,mask)
                    c,kd = compute_coding_matrix(m,kdmax,epsilon,square=True)
                    mstring = printline(m)
                    cstring = printline(np.round(c,2))
                    f = compute_fitness_given_coding_matrix(c,kd,coords,rate,square=True)

                    print ('genotype: {} | match: {} | code: {} | fitness: {}'.format(b,mstring,cstring,f))
        os._exit(1)

    print('# pre-computing site-block match matrices and degeneracies...')
    print('#')

    compute_degeneracy(tRNAs,aaRSs,mask,cache)
    
    if verbose:
        for key,number in sorted(degeneracy.items(),reverse=True,key=operator.itemgetter(1)):
            print('# degeneracy: {}'.format(number))
            print(re.sub(r'( |^)\[','# \g<1>[',np.array2string(matrix[key]),count=0)) # this could be updated to print single lines per matrix

    print('#')
    print('# computing landscape for interface width {}...'.format(width))
    print('#')

    def sb_keys():
        for key in sbmmd:
            yield key

    def key_tuples():
        keys = sb_keys()
        for key_tuple in combinations_with_replacement(keys,width):#product(keys,repeat=width):#
            yield key_tuple

    def get_match_matrix(tRNAs,aaRSs,key_tuple):
        m = np.zeros((tRNAs,aaRSs),dtype=np.int16)
        return reduce(lambda x,y:x+sbmmd[y], key_tuple, m)

    def get_match_matrix_cache(tRNAs,aaRSs,key_tuple):
        m = np.zeros((tRNAs,aaRSs),dtype=np.int16)
        return reduce(lambda x,y:x+sbm_matrix(y), key_tuple, m)

    def get_degeneracy(key_tuple):
        return reduce(lambda x,y:x*degeneracy[y], key_tuple, 1)

    def get_accuracy(c):
        return np.round(stats.hmean(np.diagonal(c)),6)

    def get_offmask(key_tuple):
        return reduce(lambda x,y:x+(off[y]), key_tuple, 0)

    def match_matrices_gen(tRNAs,aaRSs,cache):
        keyss = key_tuples()
        if cache:
            getmm = get_match_matrix_cache
        else:
            getmm = get_match_matrix
        for key_tuple in keyss:
            m  = getmm(tRNAs,aaRSs,key_tuple)
            d  = get_degeneracy(key_tuple)
            if mask:
                o  = get_offmask(key_tuple)
            else:
                o  = 0
            c  = Counter(key_tuple)
            nu = len(c)
            k  = tuple(c.values())
            mc = multinomial_coefficients(nu,width)
            d *= mc[k]
            o /= d
            o /= width
            yield m,d,o

    match_matrices = match_matrices_gen(tRNAs,aaRSs,cache)

    pool = multiprocessing.Pool(processes=poolsize)
    args = zip(match_matrices,repeat(kdmax),repeat(epsilon),repeat((coords)),repeat(rate))

    ## this also needs to become a diskcache if pairs > 4 (?)
    # mm = dict()

    dd = dict()
    oo = dict()
    fit = dict()
    fit2 = dict()

    #for arg in args:
    #    m,d,o,f,f2    = compute_fitness(arg)

    ## if (meso):
    ##     for arg in args:
    ##         print_stochkit2(arg)  
    #    else:

    
    for m,d,o,f,f2,g,g2 in pool.imap(compute_fitness,args,chunksize=chunksize):
        
        fk  = round(f,10) ## THIS IS A GIANT HACK
        if rate:
            f = g
            f2 = g2
            fk = round(f,10)
        
        if fk in dd:
            dd[fk]        += d
            oo[fk]        += o
            fb            =  fitb[fk]
            fb2           =  fitb2[fk]
        else:
            dd  [fk]      =  d
            oo  [fk]      =  o
            fit2[fk]      =  f2
            fb            =  f**beta
            fitb[fk]      =  fb
            fb2           =  f2**beta
            fitb2[fk]     =  fb2
            if cache:
                mmatrix(fk,m)  # mm[fk]   =  m
            else:
                mmd[fk] = m
            
        sfb               +=  (fb * d)
        sffb              += ((fb * f) * d)
        if f > maxf:
            maxf = f
                
        sfb2              +=  (fb2 * d)
        sffb2             += ((fb2 * f2) * d)
        if f2 > maxf2:
            maxf2 = f2

    avgf         = sffb / sfb
    load         = (maxf - avgf)/maxf
    avgf2        = sffb2 / sfb2
    load2        = (maxf2 - avgf2)/maxf2
    print ('{} < maxf < {}'.format(maxf,maxf2))
    print ('{} < avgf < {}'.format(avgf,avgf2))
    print ('{} > load > {}'.format(load,load2))
    print ('')

    for f,dd in sorted(dd.items(),key=operator.itemgetter(0)):
        m        = mmatrix(f)
        mstring  = printline(m)
        c,kd     = compute_coding_matrix(m,kdmax,epsilon,square=False)

        acc      = get_accuracy(c)
        c2,kd    = compute_coding_matrix(m,kdmax,epsilon,square=True)
        acc2     = get_accuracy(c2)
        cstring  = printline(np.round(c,3))
        print ('degen: {:>10} | off: {:<5.3e} | {:<8.6e} < accuracy < {:<8.6e} | {:<11.9e} < fitness < {:<11.9e} | {:<11.9e} < frequency < {:>11.9e} | match:{} | code(proofread):{:<}'.format(dd,oo[f],acc,acc2,f,fit2[f],(fitb2[f]/sfb2),(fitb[f]/sfb),mstring,cstring))
        
    print("# Run time (minutes): ",round((time.time()-starttime)/60,3))
                    
