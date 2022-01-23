import os
import numpy as np
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
from scipy.special import logsumexp

# find the current path
path = os.path.dirname(os.path.abspath(__file__))
# create the folders if not existed
if not os.path.exists(f'{path}/sims'):
    os.mkdir(f'{path}/sims')
if not os.path.exists(f'{path}/figures'):
    os.mkdir(f'{path}/figures')

# define some color 
Blue    = .85 * np.array([   9, 132, 227]) / 255
Green   = .85 * np.array([   0, 184, 148]) / 255
Red     = .85 * np.array([ 255, 118, 117]) / 255
Yellow  = .85 * np.array([ 253, 203, 110]) / 255
Purple  = .85 * np.array([ 108,  92, 231]) / 255
colors    = [ Blue, Red, Green, Yellow, Purple]
sns.set_style("whitegrid", {'axes.grid' : False})

# image dpi
dpi = 250

#------------------
#     Functions
#------------------

def save_cluster( outcomes, fname):
    '''Save the simulated cluster 

    Args:
        outcomes: the cluster waited to be saved 
        fname: file name 
    '''
    dim1 = len(outcomes)
    dim2 = 0 
    for o in outcomes:
        if len(o) > dim2: dim2 = len(o) 
    mat = np.zeros( [dim1, dim2])
    for i, o in enumerate(outcomes):
        mat[i, :len(o)] = o 
    col = [ f'c{i}' for i in range(dim2)]
    sname = f'{path}/sims/{fname}.csv'
    pd.DataFrame( mat, columns=col).to_csv(sname)
    
#-------------------------------------------
#     Chinese restaurant process cluster
#------------------------------------------

class CRPCluster:

    def __init__( self, clusters=[], alpha=1/np.e, rng=None):
        '''Chinese Restaurant Process Cluster

        Args:
            clusters: init the cluster
            alpha:  concencration parameter
            rng: random state generator
        '''
        self.alpha = alpha 
        self.N   = np.sum(clusters) 
        self.clusters = clusters + [alpha]
        self.C   = len(self.clusters)-1
        self.rng = np.random.RandomState(42) if rng is None else rng 

    def prob( self):
        '''Normalized clusters 
        '''
        return np.array(self.clusters) / (self.N + self.alpha)

    def push( self, x):
        '''Push new sample to the CRP

        Args:
            x: number of new object 
        '''
        for _ in range(x):
            # get the clsuster idx of the new object
            j = self.rng.choice( range(self.C+1), p=self.prob())
            if j < self.C:
                self.clusters[j] += 1
            else: 
                self.clusters[j] = 1 
                self.clusters += [self.alpha]
                self.C += 1 
            self.N += 1
    
    def export( self):
        return sorted( self.clusters[:-1], reverse=True)

#----------------------------------
#     Entropy based cluster 
#----------------------------------
            
class EnCluster:

    def __init__( self, clusters=[], rng=None):
        '''Entropy-based Cluster

        Args:
            clusters: init the cluster
            rng: random state generator
        '''
        self.clusters = [] if clusters is None else clusters
        self.N   = np.sum(self.clusters) 
        self.clusters = clusters
        self.C   = len(self.clusters)
        self.rng = np.random.RandomState(42) if rng is None else rng 

    def push( self, x):
        '''Push new sample to the entropy-based cluster

        Args:
            x: number of new object 
        '''
        for _ in range(x):
            if self.C:
                # Entropy of assign to the old cluster
                temp = np.array(self.clusters).reshape( [1, -1])
                prob = ( np.eye(self.C) + temp) / (self.N+1)
                Hj = np.sum( -prob * np.log( prob), axis=1)
                # Entorpy of assign to a new cluster 
                prob0 = np.array(self.clusters + [1]) / (self.N+1)
                H0 = np.sum( -prob0 * np.log( prob0))
                # append them to indicate the entropy 
                # for all possible conditions 
                Hs = np.hstack( [ Hj, H0])
                # the probability of the new choice
                # P(pi) \propto exp(-NH) 
                P_pi = np.exp( -self.N*Hs - logsumexp( -self.N*Hs))
                # get the clsuster idx of the new object
                j = self.rng.choice( range(self.C+1), p=P_pi)
                if j < self.C:
                    self.clusters[j] += 1
                else: 
                    self.clusters += [1]
                    self.C += 1 
            else:
                self.clusters = [1]
                self.C = 1 
            self.N += 1

    def export( self):
        return sorted( self.clusters, reverse=True)

def sim( M=20, M2=int(1e6), round=20, seed=42):

    ## Define some hyper value 
    En_outcomes, CRP_outcomes = [], []
    
    ## Start simulation!!! 
    for _ in range(round):
        seed += 1
        # init the cluster  
        rng = np.random.RandomState( seed)
        init_cluster = CRPCluster( rng=rng)
        init_cluster.push( M)
        init_C = init_cluster.export()
        # CRP clusting
        rng = np.random.RandomState( seed)
        CRP_cluster = CRPCluster( clusters=init_C, rng=rng)
        CRP_cluster.push( M2) 
        CRP_outcomes.append( CRP_cluster.export())
        # entropy clustering 
        rng = np.random.RandomState( seed)
        En_cluster = EnCluster( clusters=init_C, rng=rng)
        En_cluster.push( M2)
        En_outcomes.append( En_cluster.export())
    
    ## save simulated result 
    save_cluster( En_outcomes, f'En_outcomes-M={M}')
    save_cluster( CRP_outcomes, f'CRP_outcomes-M={M}')

def fig1A():

    ## Define some hyper value 
    M, M2 = 20, int(1e6)
    al, lw, mz, fz = .1, 1, 3, 16

    ## Load simulated results
    En_outcomes  = pd.read_csv( f'{path}/sims/En_outcomes-M={M}.csv').to_numpy()[:, 1:]
    CRP_outcomes = pd.read_csv( f'{path}/sims/CRP_outcomes-M={M}.csv').to_numpy()[:, 1:]
    # normalized 
    En_outcomes /= M2
    CRP_outcomes /= M2
    x_CRP = np.arange( 0, CRP_outcomes.shape[1])
    x_En  = np.arange( 0, En_outcomes.shape[1])
    
    plt.figure( figsize=( 5, 4))
    ## plot mean
    plt.plot( x_CRP, CRP_outcomes.mean(0), '-',
                    linewidth=lw*3, 
                    color=Blue, label='CRP')
    plt.plot( x_En, En_outcomes.mean(0), '-',
                    linewidth=lw*3, 
                    color=Red, label='Entropy')
    ## For all samples 
    for i in range( En_outcomes.shape[0]):
        plt.plot( x_CRP, CRP_outcomes[ i, :], 'o-',
                    linewidth=lw, markersize=mz,
                    color=Blue, alpha=al)
        plt.plot( x_En, En_outcomes[ i, :], 'o-',
                    linewidth=lw, markersize=mz,
                    color=Red, alpha=al)
    plt.legend( ['CRP', 'Entropy'], fontsize=fz)
    plt.xlabel( 'Cluster rank', fontsize=fz)
    plt.ylabel( 'Fraction of objects', fontsize=fz)
    plt.tight_layout()
    plt.savefig( f'{path}/figures/fig1A.png', dpi=dpi)
        
if __name__ == '__main__':

    ## STEP0: SIMULATE 
    sim()

    ## STEP1: DRAW A FIGURE 
    fig1A()
        