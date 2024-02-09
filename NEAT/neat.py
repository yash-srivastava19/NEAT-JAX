import jax
import copy
import nsga_sort
import jax.numpy as jnp

from utils import rankArray
from base_NE import NEAlgorithm
from ind import Ind 

# Now, given the Google Brain NEAT release, and the EvoJAx implementation, we can kind of create this from scratch. 

class NEATJax(NEAlgorithm):
    def __init__(self,hyp) :
        self.p = hyp 
        self.pop = []
        self.species = []
        self.innov = []
        self.gen = 0

    # Followed from the actual EvoJAX implementations of the algorithms.
    def ask(self) -> jnp.ndarray:
        if len(self.pop) == 0:
            self._initPop()

        else:
            self._probMoo() 
            self._speciate()
            self._evolvePop()

        return self.pop  


    def tell(self, reward) -> None:
        for i in range(jnp.shape(reward)[0]):
            self.pop[i].fitness = reward[i]
            self.pop[i].nConn = self.pop[i].nConn 


    def _initPop(self):
        p = self.p 

        nodeId = jnp.arange(0, p['ann_nInput'] + p['ann_nOutput'] + 1, 1)
        node = jnp.empty((3, len(nodeId)))
        node[0, :] = nodeId 

        # Types of Nodes -> 1: input, 2: hidden, 3:bias, 4: output
        node[1, 0] = 4
        node[1, 1:p['ann_nInput'] + 1] = 1
        node[1,(p['ann_nInput']+1):(p['ann_nInput']+p['ann_nOutput']+1)]  = 2

        # Node Activations : 
        node[2, :] = p['ann_initAct']

        # Create connection
        nConn = (p['ann_nInput']+1) * p['ann_nOutput']
        ins  = jnp.arange(0, p['ann_nOutput']+1, 1)
        outs = (p['ann_nInput']+1) + jnp.arange(0, p['ann_nOutput'])

        conn = jnp.empty((5,nConn,))
        conn[0,:] = jnp.arange(0,nConn,1)       # Connection Id
        conn[1,:] = jnp.tile(ins, len(outs))    # Source Nodes
        conn[2,:] = jnp.repeat(outs,len(ins) )  # Destination Nodes
        conn[3,:] = jnp.nan                     # Weight Values
        conn[4,:] = 1                           # Enabled?

        # Create population of individuals with varied weights.

        pop = []
        for i in range(p['popSize']):
            newInd = Ind(conn, node)
            newInd.conn[3,:] = (2*(jnp.random.rand(1,nConn)-0.5))*p['ann_absWCap']
            newInd.conn[4,:] = jnp.random.rand(1,nConn) < p['prob_initEnable']
            newInd.express()
            newInd.birth = 0
            pop.append(copy.deepcopy(newInd)) 
        
        innov = jnp.zeros([5, nConn])
        innov[0:3, :] = pop[0].conn[0:3, :]
        innov[3, :] = -1

        self.pop = pop 
        self.innov = innov 

    def _probMoo(self):
        """ Rank population according to Pareto dominance. """
        
        meanFit = jnp.asarray([ind.fitness for ind in self.pop])
        nConns  = jnp.asarray([ind.nConn   for ind in self.pop])
        nConns[nConns==0] = 1 # No connections is pareto optimal but boring...
        objVals = jnp.c_[meanFit,1/nConns] # Maximize

        # Alternate between two objectives and single objective
        if self.p['alg_probMoo'] < jnp.random.rand():
            rank = nsga_sort(objVals[:,[0,1]])
        else: # Single objective
            rank = rankArray(-objVals[:,0])

        # Assign ranks
        for i in range(len(self.pop)):
            self.pop[i].rank = rank[i]
