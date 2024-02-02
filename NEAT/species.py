import jax.numpy as jnp
from utils import *

class Species():

    def __init__(self,seed):
        self.seed = seed      # Seed is type Ind
        self.members = [seed] # All inds in species
        self.bestInd = seed
        self.bestFit = seed.fitness
        self.lastImp = 0
        self.nOffspring = []

def speciate(self):  
    p = self.p
    pop = self.pop
    species = self.species

    if p['alg_speciate'] == 'neat':
        if len(species) > p['spec_target']:
            p['spec_thresh'] += p['spec_compatMod']

        if len(species) < p['spec_target']:
            p['spec_thresh'] -= p['spec_compatMod']

        if p['spec_thresh'] < p['spec_threshMin']:
            p['spec_thresh'] = p['spec_threshMin']

        species, pop = self.assignSpecies  (species, pop, p)
        species = self.assignOffspring(species, pop, p)

    elif p['alg_speciate'] == "none" : 
        species = [Species(pop[0])]
        species[0].nOffspring = p['popSize']
        for ind in pop:
            ind.species = 0
        species[0].members = pop

    self.p = p
    self.pop = pop
    self.species = species

def assignSpecies(self, species, pop, p):

    if len(self.species) == 0:
        species = [Species(pop[0])]
        species[0].nOffspring = p['popSize']
        iSpec = 0
    else:
        for iSpec in range(len(species)):
            species[iSpec].members = []

    for i in range(len(pop)):
        assigned = False
        for iSpec in range(len(species)):
            ref = jnp.copy(species[iSpec].seed.conn)
            ind = jnp.copy(pop[i].conn)
            cDist = self.compatDist(ref,ind)
            if cDist < p['spec_thresh']:
                pop[i].species = iSpec
                species[iSpec].members.append(pop[i])
                assigned = True
                break

    if not assigned:
        pop[i].species = iSpec+1
        species.append(Species(pop[i]))

    return species, pop

# Check once if there are any errors.
def assignOffspring(self, species, pop, p):

    nSpecies = len(species)
    
    if nSpecies == 1:
        species[0].offspring = p['popSize']
    
    else:
        popFit = jnp.asarray([ind.fitness for ind in pop])
        popRank = tiedRank(popFit)
        
        if p['select_rankWeight'] == 'exp':
            rankScore = 1/popRank
        
        elif p['select_rankWeight'] == 'lin':
            rankScore = 1+abs(popRank-len(popRank))
        
        else:
            print("Invalid rank weighting (using linear)")
            rankScore = 1+abs(popRank-len(popRank))
        specId = jnp.asarray([ind.species for ind in pop])

        speciesFit = jnp.zeros((nSpecies,1))
        speciesTop = jnp.zeros((nSpecies,1))
        for iSpec in range(nSpecies):
            if not jnp.any(specId==iSpec):
                speciesFit[iSpec] = 0
            else:
                speciesFit[iSpec] = jnp.mean(rankScore[specId==iSpec])
                speciesTop[iSpec] = jnp.max(popFit[specId==iSpec])

                # Did the species improve?
                if speciesTop[iSpec] > species[iSpec].bestFit:
                    species[iSpec].bestFit = speciesTop[iSpec]
                    bestId = jnp.argmax(popFit[specId==iSpec])
                    species[iSpec].bestInd = species[iSpec].members[bestId]
                    species[iSpec].lastImp = 0
                else:
                    species[iSpec].lastImp += 1

                # Stagnant species don't recieve species fitness
                if species[iSpec].lastImp > p['spec_dropOffAge']:
                    speciesFit[iSpec] = 0
            
        if sum(speciesFit) == 0:
            speciesFit = jnp.ones((nSpecies,1))
            print("WARN: Entire population stagnant, continuing without extinction")
            
        offspring = bestIntSplit(speciesFit, p['popSize'])
        
        for iSpec in range(nSpecies):
            species[iSpec].nOffspring = offspring[iSpec]
    
    species[:] = [s for s in species if s.nOffspring != 0]

    return species

def compatDist(self, ref, ind):

    # Find matching genes
    IA, IB = quickINTersect(ind[0,:].astype(int),ref[0,:].astype(int))          

    # Calculate raw genome distances
    ind[3,jnp.isnan(ind[3,:])] = 0
    ref[3,jnp.isnan(ref[3,:])] = 0
    
    weightDiff = abs(ind[3,IA] - ref[3,IB])
    geneDiff   = sum(jnp.invert(IA)) + sum(jnp.invert(IB))

    # Normalize and take weighted sum
    nInitial = self.p['ann_nInput'] + self.p['ann_nOutput']
    longestGenome = max(len(IA),len(IB)) - nInitial
    weightDiff = jnp.mean(weightDiff)
    geneDiff   = geneDiff   / (1+longestGenome)

    dist = geneDiff   * self.p['spec_geneCoef']  + weightDiff * self.p['spec_weightCoef']  
    return dist
