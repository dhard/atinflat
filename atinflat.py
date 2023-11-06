from __future__ import division 
from __future__ import print_function
from optparse import OptionParser, OptionValueError
#from types import FloatType
from scipy import stats
from bitstring import Bits, BitArray
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
    return  re.sub('\[ ','[',re.sub(' \[','[',re.sub('  ',' ',re.sub('\n',',',np.array2string(m)))))

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
        for sn in powerset(range(aaRSs)):
            setn = set(sn)
            #if (bool(setm) and bool(setn) and bool(setm & setn)):
            for st in powerset(range(tRNAs)):
                sett     = set(st)
                #if bool(sett & setm):
                for sa in powerset(range(aaRSs)):
                    seta     = set(sa)
                    #if bool(seta & setn):
                    yield setm,setn,sett,seta
                            
                            
def genotypes_gen(tRNAs,aaRSs):
    for st in powerset(range(tRNAs)):
        sett     = set(st)
        for sa in powerset(range(aaRSs)):
            seta     = set(sa)
            yield sett,seta

from tempfile import mkdtemp
cachedir = mkdtemp()
print(cachedir)
from joblib import Memory
memory = Memory(location=cachedir, mmap_mode='r', verbose=0)

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

mmd     = dict() # match matrix dictionary keyed by joblib hash 
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
ips        = dict()
# this could be rewritten to take advantage of pool.imap().
def compute_degeneracy(tRNAs,aaRSs,mask,cache):
    """
    This function computes all possible site-block-match-matrices
    and their encodable genetic degeneracies
    """
    uni_t       = set(range(tRNAs))
    uni_a       = set(range(aaRSs))
    ## if mask:
    ##     zeros   = 2**(2*(tRNAs+aaRSs))
    ## else:
    ##     zeros   = 2**(tRNAs+aaRSs)

    if mask:
        genotypes = masked_genotypes_gen(tRNAs,aaRSs)
        for setm,setn,sett,seta in genotypes:
            offm      = uni_t - setm
            offn      = uni_a - setn
            eoff      = len(offm) + len(offn) # eoff is ultimately the expected fraction of sites masked per genotype 
            eips      = (2 * len(setm) * len(setn))  # eips is expected number of unmasked interactions per site 0 <= eips <= P or N+M/2 
            if ( eips > 0 ):
                eips /= ( len(setm) + len(setn))  
            settc     = uni_t - sett
            sett     &= setm
            settc    &= setm
            setac     = uni_a - seta
            seta     &= setn
            setac    &= setn


            m = np.zeros((tRNAs,aaRSs),dtype=np.int16)
            for match in chain(product(sett,seta),product(settc,setac)):
                m[match] += 1
                # if (m==0).all():
                 ##print ('# huh! in compute_degeneracy') # why do we get here?
                # continue
            key = joblib.hash(m)
            #if key == '3d364cbacfad5c8c2be9dc4314aec17c':
            #    pdb.set_trace()
            if key in degeneracy:
                degeneracy[key] += 1
                off[key]        += eoff / (2 * pairs * width)
                ips[key]        += eips / width
            else:
                degeneracy[key] = 1
                off[key]        = eoff / (2 * pairs * width)
                ips[key]        = eips / width
                if cache:
                     sbm_matrix(key,m) # THIS IS NOT TESTED
                else:
                    sbmmd[key] = m
                #zeros -= 1
 
        for key in off:
            off[key] /= degeneracy[key]
            ips[key] /= degeneracy[key]
        #pdb.set_trace()
    else:
        genotypes = genotypes_gen(tRNAs,aaRSs)
        for sett,seta in genotypes:
            settc     = uni_t - sett
            setac     = uni_a - seta        
            m = np.zeros((tRNAs,aaRSs),dtype=np.int16)
            for match in chain(product(sett,seta),product(settc,setac)):
                m[match] += 1
            key = joblib.hash(m)
            
            if key in degeneracy:
                degeneracy[key] += 1
                #zeros -= 1
            else:
                degeneracy[key] = 1
                if cache:
                    sbm_matrix(m) # THIS IS NOT TESTED
                else:
                    sbmmd[key] = m

                #zeros -= 1
    ## if zeros:
    ##     m0  = np.zeros((tRNAs,aaRSs),dtype=np.int16)
    ##     key = joblib.hash(m0)
    ##     degeneracy[key] = zeros
    ##     off[key]        = 0
    ##     if cache:
    ##         sbm_matrix(key,m0) # matrix[key] = mmatrix[key] = m0
    ##     else:
    ##         sbmmd[key] = m0

def compute_match_matrix(g,width,pairs,mask):
    # this is for genotype files
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
    o    = 0 # eoff
    eips = 0 # eips
    for i in range(pairs):
        if mask:
            o           +=  mt[i].count(0) + ma[i].count(0)
        for j in range(pairs): 
            taxnor       = ~(t[i] ^ a[j])
            if mask:
                maskand  =  mt[i] & ma[j]
                m[i,j]   = (maskand & taxnor).count(1)
            else:
                m[i,j]   = (taxnor).count(1)
    ## compute off mask
    o /=  2 * width * (pairs)

    ei  = 0
    eid = 0
    ## compute interactions per site 
    if mask:
        for k in range (width):
            for j in range(pairs):
                eid  +=  ma[j][k]
            for i in range(pairs):
                eid  +=  mt[i][k]
                if mt[i][k]:
                    for j in range(pairs): 
                        if (ma[j][k]):
                            ei  += 2
    ei /= eid        
             

    for i in range(pairs):
        if mask:
            o       +=  mt[i].count(0) + ma[i].count(0)
        for j in range(pairs): 
            taxnor       = ~(t[i] ^ a[j])
            if mask:
                maskand  =  mt[i] & ma[j]
                m[i,j]   = (maskand & taxnor).count(1)
            else:
                m[i,j]   = (taxnor).count(1)


    return m,o,ei

def compute_coding_matrix(m,kdnc,epsilon,square):
    ## compute energies and kinetic off-rates between tRNAs and aaRSs from the match matrix
    #kd = 1/((1 / kdnc) * np.exp(m * epsilon)) # # K_ij = (kdnc)^-1 * exp [m_ij * epsilon], and k_d = (1/K_ij)

    kd = kdnc / np.exp(m * epsilon)
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
            fc  += c [i,j] * fij
        f  *= fc

    if rate:
        #avg_kd  = stats.hmean(np.diagonal(kd))
        avg_kd   = stats.hmean(kd,axis=None)
        if square:
            f      *= 1.0 - exp(-1.505760 * avg_kd / 9680) # curve fit to data of Paulander et al (2007): dividing by 9680 results in a catalytic rate of 5 when the avg_kd is the square of the cognate rate
        else:
            f      *= 1.0 - exp(-1.505760 * avg_kd / 44)   # curve fit to data of Paulander et al (2007), assuming avg_k_cat = avg_kd / 44

    return f

# this is defined for debugging purposes



# this is the main target function for the multiprocessing pool during computation of the landscape
def compute_fitness(args):
    #input m is the width-th cartesian power of site-block-match-matrices
    (m,d,o,ei),kdnc,epsilon,coords,rate = args
    
    ## compute energies and kinetic off-rates between tRNAs and aaRSs from the match matrix
    #kd = 1/((1 / kdnc) * np.exp(m * epsilon)) # # K_ij = (kdnc)^-1 * exp [m_ij * epsilon], and k_d = (1/K_ij)
    kd = kdnc / np.exp(m * epsilon) 
    kd2 = kd**2

    ## compute the "code matrix" of conditional decoding probabilities 
    c  = np.zeros((tRNAs,aaRSs),dtype=float)
    c2 = np.zeros((tRNAs,aaRSs),dtype=float)
    for i in range(tRNAs):
        for j in range(aaRSs):
            c [i,j] = stats.hmean(kd [i,:]/kd [i,j])/aaRSs
            c2[i,j] = stats.hmean(kd2[i,:]/kd2[i,j])/aaRSs

            
    ## compute fitness bounds (f is in dead cells at equilibrium and f2 is with maximum kinetic proofreading)              
    fa     = 1
    fa2    = 1

    for i in range(tRNAs):
        fc = fc2 = 0
        for j in range(aaRSs):
            fij  = phi**(abs(coords[i] - coords[j]))
            fc  += c [i,j] * fij
            fc2 += c2[i,j] * fij
        fa  *= fc
        fa2 *= fc2

    f = fa
    f2 = fa2
    if rate:
        #avg_kd   = stats.hmean(np.diagonal(kd))
        #avg_kd2  = stats.hmean(np.diagonal(kd2))
        #avg_kd   = stats.hmean(np.sum(np.multiply(kd,c),axis=1))
        #avg_kd2  = stats.hmean(np.sum(np.multiply(kd2,c2),axis=1))
        avg_kd   = stats.hmean(kd,axis=None)
        avg_kd2  = stats.hmean(kd2,axis=None)
        f2      *= 1.0 - exp(-1.505760 * avg_kd2 / 9680) # curve fit to data of Paulander et al (2007)
        f       *= 1.0 - exp(-1.505760 * avg_kd / 44)    # curve fit to data of Paulander et al (2007)
    
    return m,d,o,ei,f,f2,fa,fa2


# version 1.1:
#              - fixes an ouput error in which the frequencies of the
#                non-proofread and proofread genotype class were switched.

# version 1.2:
#              - revised the definition of rate-dependent fitness factor to include both
#                cognate and non-cognate aminoacylations.
#              - changed name of "iota" to "epsilon"
#              - corrected curve-fit coefficient from -1.505764 to -1.505760
# version 1.3:
#              - added "--logscale" option to compute frequencies for multiple values of beta
#              - added "accfit" to output, so now with --rate, the rate-independent fitness is also calculated
#
# version 1.4:
#              - added ips-statistic, main data-structure now keyed by match-matrix keys
#                output is now factored by match-matrices and not by rounded fitness any more
#               
# version 1.5:
#              - changed --rate option to --accuracy option. By default now, both fitnesses are computed
#              - the output for --accuracy was not developed or test (in which accuracy-only fitness is the only one computed)
#            
# version 1.6:
#              - added summary stats and stationary frequency information for accuracy fitness on default output
# 

if __name__ == '__main__':
    starttime = time.time()
    version = 1.6
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
    
    parser.add_option("-n","--width",
                      dest="width", type="int", default=2,
                      help="set interaction interface width/max matches, Default: %default")

    parser.add_option("-k","--matches",
                      dest="matches", type="int", default=None,
                      help="set number of matches required to reach the dissociation rate of cognate pairs, kdc.  Default: <width>")

    parser.add_option("-a","--accuracy",
                      dest="accuracy", action="store_true",
                      help="make fitness depend only on accuracy but also rate of dissociation.  Default: False")

    parser.add_option("-P","--pairs",
                      dest="pairs", type="int", default=2,
                      help="set equal number of aaRS/tRNA pairs\n Default: %default")
    
    parser.add_option("--mask", action="store_true",
                      dest="mask",
                      help="use evolveable mask bits as per-site interaction modifiers, Default: False")

    parser.add_option("-B","--beta",
                      dest="beta", type="int", default=100,
                      help="set beta, one minus the haploid Moran population size.  Default: %default")

    parser.add_option("--logscale",
                      dest="logscale", type="string", default=None,
                      help="with string argument \"<start>:<stop>:<num>\" (three colon-separated positive integers): evaluate <num> values of beta from <beta>**<start> to <beta>**<stop> on a log scale.  Default: %default")
    
    parser.add_option("--phi",
                      dest="phi", type="float", default=0.99,
                      help="set phi, the maximum missense fitness-per-site penalty, 0 < phi < 1. See Sella and Ardell (2001). Default: %default")
    
    parser.add_option("--kdnc",
                      dest="kdnc", type="float", default=10000,
                      help="set kdnc in sec^-1, the maximum dissociation rate constant (weakest binding) for non-cognate pairs). Default: %default")
    
    parser.add_option("--kdc",
                      dest="kdc", type="float", default=220,
                      help="set kdc in sec^-1, the dissociation rate constant reached at nsites. Default: %default")

    parser.add_option("--verbose",
                      dest="verbose",  action="store_true",
                      help="print more output about site blocks etc. Default: False")

    parser.add_option("-g","--genotypes",
                      dest="genotypes", type="string", default=None,
                      help="compute match and code matrices and fitness for binary string genotypes in file and exits. Assumes proofreading. If mask is True, genotype format is t11..t1w.a11..a1w...tP1..tPw.aP1..aPw.m11..m1w.n11..n1w...nP1..nPw, where mij is the maskbit for tij, nij is the maskbit for aij, w is <width> and P is <pairs>. You must set other parameters manually to match your genotypes. Default: %default")

    parser.add_option("--mutate",
                      dest="mutate", action="store_true",
                      help="also compute results for all single- and double-mutants of genotypes input with -g. Default: False")

    parser.add_option("--show-dissociation",
                      dest="show_dissociation", action="store_true",
                      help="show dissociation rate matrix in output for genotypes. Default: False")


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

    beta        = options.beta
    logscale    = options.logscale
    phi         = options.phi
    width       = options.width
    matches     = options.matches
    accuracy    = options.accuracy
    pairs       = options.pairs
    mask        = options.mask
    kdnc        = options.kdnc
    kdc         = options.kdc
    chunksize   = options.chunk
    poolsize    = options.pool
    cache       = options.cache

    verbose     = options.verbose
    genotypes   = options.genotypes
    mutate      = options.mutate
    show_dissociation = options.show_dissociation

    rate = not accuracy;
    
    if not matches:
        matches = width
    epsilon     = (log (kdnc) - log (kdc)) / matches

    aaRSs = pairs
    tRNAs = pairs
    length      = width * (tRNAs + aaRSs)
    if mask:
        length *= 2

    if logscale:
        words     = logscale.split(':')
        startpow  = int(words[0])
        endpow    = int(words[1])
        numbeta   = int(words[2])
        beta      = np.floor(np.logspace(startpow, endpow, num=numbeta, base=beta))

    # calculate site-type space
    cuts  = pairs - 1
    segment = 1.0 / cuts
    coords = []
    for a in range(aaRSs):
        coords.append(segment * a)
        
    print('# {:<3s} version {:3.1f}'.format(prog,version))
    print('# Copyright (2019) David H. Ardell.')
    print('# All Wrongs Reversed.')
    print('#')
    print('# Please cite Collins-Hed and Ardell (2019) in published works using this software.')
    print('#')
    print('# execution command:')
    print('# '+' '.join(myargv))
    print('#')
    print('# pairs         :  {}'.format(pairs))
    #  print('# tRNAs      :  {}'.format(pairs))
    #  print('# aaRSs      :  {}'.format(pairs))
    print('# width (n)     :  {}'.format(width))
    print('# matches(k)    :  {}'.format(matches))
    print('# mask          :  {}'.format(mask))
    print('# accuracy-only :  {}'.format(accuracy))    
    print('# kdnc          :  {}'.format(kdnc))
    print('# kdc           :  {}'.format(kdc))
    print('# epsilon       :  {}'.format(epsilon))
    print('# phi           :  {}'.format(phi))
    print('# beta          :  {}'.format(beta))
    print('# length        :  {}'.format(length))
    print('# coords        :  {}'.format(coords))
    print('#')
    print('# verbose       :  {}'.format(verbose))
    print('# genotypes     :  {}'.format(genotypes))
    print('# mutate        :  {}'.format(mutate))
    print('# show-dissociation:  {}'.format(show_dissociation))
    print('# pool-size     :  {}'.format(poolsize))
    print('# chunk-size    :  {}'.format(chunksize))
    print('# cache         :  {}'.format(cache))
    print('#')
    if genotypes:
        print('#')
        print('# analyzing binary genotypes in file {}, and exiting.'.format(genotypes))
        b = ''
        g = None
        # this will be updated to print the off-mask statistic
        with open(genotypes) as f:
            for line in f:
                strip1 = re.sub('#.*','',line)
                strip2 = re.sub('\s+','',strip1)
                match  = re.search('^[01]+', strip2)
                if match:
                    b        = match.group(0)
                    g        = Bits(bin=b)
                    m,o,ei   = compute_match_matrix(g,width,pairs,mask)
                    c,kd     = compute_coding_matrix(m,kdnc,epsilon,square=True)
                    mstring  = printline(m)
                    cstring  = printline(np.round(c,2))
                    f        = compute_fitness_given_coding_matrix(c,kd,coords,rate,square=True)
                    if show_dissociation:
                        kdstring = printline(kd)
                        print ('genotype: {} | off: {:<5.3e} | ips: {:<5.3e} | fitness: {: <11.9e} | match: {} | dissociation: {} | proofread code: {}'.format(b,o,ei,f,mstring,kdstring,cstring))
                        #print ('genotype: {} | fitness: {: <11.9e} | match: {} | dissociation: {} | proofread code: {}'.format(b,f,mstring,kdstring,cstring))
                    else:
                        print ('genotype: {} | off: {:<5.3e} | ips: {:<5.3e} | fitness: {: <11.9e} | match: {} | proofread code: {}'.format(b,o,ei,f,mstring,cstring))                        
                    ## if mutate: ## DHA032918. THIS WAS DRAFTED BUT NEVER USED OR TESTED 
                    ##     gf = f
                    ##     d = dict()
                    ##     a = BitArray(g)
                    ##     for i in range(g.len):
                    ##         a.invert(i)
                    ##         g = Bits(a)
                    ##         b = g.bin
                    ##         m = compute_match_matrix(g,width,pairs,mask)
                    ##         c,kd = compute_coding_matrix(m,kdnc,epsilon,square=True)
                    ##         mstring  = printline(m)
                    ##         cstring  = printline(np.round(c,2))
                    ##         f = compute_fitness_given_coding_matrix(c,kd,coords,rate,square=True)
                    ##         d[i] = f
                    ##         if show_dissociation:
                    ##             kdstring = printline(kd)
                    ##             print ('genotype: {} | mutate: {} | fitness: {: <11.9e} | match: {} | dissociation: {} | proofread code: {}'.format(b,i,f,mstring,kdstring,cstring))
                    ##         else:
                    ##             print ('genotype: {} | mutate: {} | fitness: {} | match: {} | proofread code: {}'.format(b,i,f,mstring,cstring))  
                    ##         a.invert(i)
                    ##     for i in range(g.len):
                    ##         for j in range(i+1,g.len):
                    ##             a.invert(i)
                    ##             a.invert(j)
                    ##             g = Bits(a)
                    ##             b = g.bin
                    ##             m = compute_match_matrix(g,width,pairs,mask)
                    ##             c,kd = compute_coding_matrix(m,kdnc,epsilon,square=True)
                    ##             mstring  = printline(m)
                    ##             cstring  = printline(np.round(c,2))
                    ##             f = compute_fitness_given_coding_matrix(c,kd,coords,rate,square=True)
                    ##             e = ((gf*f) - (d[i]*d[j]))
                    ##             if show_dissociation:
                    ##                 kdstring = printline(kd)
                    ##                 print ('genotype: {} | mutate: {} | {} | fitness: {: <11.9e} | epistasis: {} | match: {} | dissociation: {} | proofread code: {}'.format(b,i,j,f,e,mstring,kdstring,cstring))
                    ##             else:
                    ##                 print ('genotype: {} | mutate: {} | {} | fitness: {: <11.9e} | epistasis: {} | match: {} | proofread code: {}'.format(b,i,j,f,e,mstring,cstring))  
                    ##             a.invert(i)
                    ##             a.invert(j)
                    
    else:
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
            return reduce(lambda x,y:x+off[y], key_tuple, 0)            

        def get_eips(key_tuple):
            return reduce(lambda x,y:x+ips[y], key_tuple, 0)
        
        def match_matrices_gen(tRNAs,aaRSs,cache):
            keyss = key_tuples()
            if cache:
                getmm = get_match_matrix_cache
            else:
                getmm = get_match_matrix
            for key_tuple in keyss:
                #pdb.set_trace()
                m  = getmm(tRNAs,aaRSs,key_tuple)
                d  = get_degeneracy(key_tuple)
                if mask:
                    o  = get_offmask(key_tuple)
                    ei = get_eips(key_tuple)
                else:
                    o  = 0
                    ei = 0
                c  = Counter(key_tuple)
                nu = len(c)
                k  = tuple(c.values())
                mc = multinomial_coefficients(nu,width)
                d *= mc[k]
                yield m,d,o,ei

        match_matrices = match_matrices_gen(tRNAs,aaRSs,cache)

        pool = multiprocessing.Pool(processes=poolsize)
        args = zip(match_matrices,repeat(kdnc),repeat(epsilon),repeat((coords)),repeat(rate))

        dd   = dict()
        oo   = dict()
        eeii = dict()

        fit  = dict()
        fit2 = dict()
        fita = dict()
        fita2= dict()

        fitb        = {}
        fitb2       = {}
        fitab       = {}
        fitab2      = {}
        
        maxf        = 0
        maxf2       = 0
        maxfa       = 0
        maxfa2      = 0
        
        sfb         = 0
        sfb2        = 0
        sfab        = 0
        sfab2       = 0
        
        sfbf        = 0
        sfbf2       = 0
        sfabfa      = 0
        sfabfa2     = 0        
        
        #for arg in args:
        #    m,d,o,f,f2    = compute_fitness(arg)

        for m,d,o,ei,f,f2,fa,fa2 in pool.imap(compute_fitness,args,chunksize=chunksize):

            key = joblib.hash(m)
            if key in dd:
                dd[key]        +=   d
                oo[key]        +=  (d * o)
                fb             =    fitb[key]
                fb2            =    fitb2[key]
                fab            =    fitab[key]
                fab2           =    fitab2[key]
            else:
                dd  [key]      =  d
                oo  [key]      =  (d * o)
                eeii[key]      =  ei

                fit [key]      =  f
                fit2[key]      =  f2
                fita[key]      =  fa
                fita2[key]     =  fa2

                fb             =  f**beta
                fb2            =  f2**beta                
                fab            =  fa**beta
                fab2           =  fa2**beta
                
                fitb[key]      =  fb
                fitb2[key]     =  fb2
                fitab[key]     =  fab
                fitab2[key]    =  fab2

                #if cache:
                mmatrix(key,m)  # mm[fk]   =  m
                #else:
                #    mmd[key] = m

            ## REWRITE THIS TO PRINT BOTH THE ACCURACY AND RATE SUMMARIES

            sfb               +=  (fb * d)
            sfb2              +=  (fb2 * d)
            sfab              +=  (fab * d)
            sfab2             +=  (fab2 * d)

            sfbf              += ((fb * f) * d)
            sfbf2             += ((fb2 * f2) * d)
            sfabfa            += ((fab * fa) * d)
            sfabfa2           += ((fab2 * fa2) * d)
            
            if f > maxf:
                maxf = f
            if f2 > maxf2:
                maxf2 = f2
            if fa > maxfa:
                maxfa = fa
            if fa2 > maxfa2:
                maxfa2 = fa2
                
        for key in dd:
            oo[key] /= dd[key]

        avgf         = sfbf    /  sfb
        avgf2        = sfbf2   /  sfb2
        avgfa        = sfabfa  /  sfab
        avgfa2       = sfabfa2 /  sfab2
        
        load         = ( maxf   - avgf   ) / maxf
        load2        = ( maxf2  - avgf2  ) / maxf2
        loada        = ( maxfa  - avgfa  ) / maxfa
        loada2       = ( maxfa2 - avgfa2 ) / maxfa2            

        print ('{}  <  max. fitness (accuracy and rate) < {} (proofread)'.format(maxf,maxf2))
        print ('{}  <  max. fitness (accuracy only)     < {} (proofread)'.format(maxfa,maxfa2))        

        print ('{}  <  avg. fitness (accuracy and rate) < {} (proofread)'.format(avgf,avgf2))
        print ('{}  <  avg. fitness (accuracy only)     < {} (proofread)'.format(avgfa,avgfa2))

        print ('{}  >  load         (accuracy and rate) > {} (proofread)'.format(load,load2))
        print ('{}  >  load         (accuracy only)     > {} (proofread)'.format(loada,loada2))
        print ('')
        print ('match'.format(loada,loada2))
        
        for key,f in sorted(fit.items(),key=operator.itemgetter(1)):
            m         = mmatrix(key)
            ddd       = dd[key]
            f2        = fit2[key]
            fa        = fita[key]
            fa2       = fita2[key]
            mstring   = printline(m)
            c,kd      = compute_coding_matrix(m,kdnc,epsilon,square=False)
            acc       = get_accuracy(c)
            c2,kd2    = compute_coding_matrix(m,kdnc,epsilon,square=True)
            acc2      = get_accuracy(c2)
            cstring   = printline(np.round(c,3))
            c2string  = printline(np.round(c2,3))
            fr        = (fitb[key]/sfb)    # frequencies
            fr2       = (fitb2[key]/sfb2)
            fra       = (fitab[key]/sfab)
            fra2      = (fitab2[key]/sfab2)

            if show_dissociation:
                kdstring = printline(kd2)
                print ('match: {} degen: {} off: {:<5.3e} ips: {:<5.3f} | {:<8.6e} <_acc_< {:<8.6e} | {: <11.9e} <_fit_< {: <11.9e} | {} <_freq_< {} | {: <11.9e} <_accfit_< {: <11.9e} | {} <_accfreq_< {} code:proof_code: {:<}:{:<} dissoc: {}'.format(mstring,ddd,oo[key],eeii[key],acc,acc2,f,f2,fr,fr2,fa,fa2,fra,fra2,cstring,c2string,kdstring))
            else:
                print ('match: {} degen: {} off: {:<5.3e} ips: {:<5.3f} | {:<8.6e} <_acc_< {:<8.6e} | {: <11.9e} <_fit_< {: <11.9e} | {} <_freq_< {} | {: <11.9e} <_accfit_< {: <11.9e} | {} <_accfreq_< {} code:proof_code: {:<}:{:<}'.format(mstring,ddd,oo[key],eeii[key],acc,acc2,f,f2,fr,fr2,fa,fa2,fra,fra2,cstring,c2string))

    print("# Run time (minutes): ",round((time.time()-starttime)/60,3))
                    
