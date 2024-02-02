import jax.numpy as jnp

def getNodeOrder(nodeG,connG):
    conn = jnp.copy(connG)
    node = jnp.copy(nodeG)
    nIns = len(node[0,node[1,:] == 1]) + len(node[0,node[1,:] == 4])
    nOuts = len(node[0,node[1,:] == 2])
  
    # Create connection and initial weight matrices
    conn[3,conn[4,:]==0] = jnp.nan # disabled but still connected
    src  = conn[1,:].astype(int)
    dest = conn[2,:].astype(int)

    lookup = node[0,:].astype(int)
    for i in range(len(lookup)): # Can we vectorize this?
        src[jnp.where(src==lookup[i])] = i
        dest[jnp.where(dest==lookup[i])] = i

    wMat = jnp.zeros((jnp.shape(node)[1],jnp.shape(node)[1]))
    wMat[src,dest] = conn[3,:]
    connMat = wMat[nIns+nOuts:,nIns+nOuts:]
    connMat[connMat!=0] = 1

    # Topological Sort of Hidden Nodes
    edge_in = jnp.sum(connMat,axis=0)
    Q = jnp.where(edge_in==0)[0]  # Start with nodes with no incoming connections
    for i in range(len(connMat)):
        if (len(Q) == 0) or (i >= len(Q)):
            Q = []
            return False, False # Cycle found, can't sort
        edge_out = connMat[Q[i],:]
        edge_in  = edge_in - edge_out # Remove nodes' conns from total
        nextNodes = jnp.setdiff1d(jnp.where(edge_in==0)[0], Q)
        Q = jnp.hstack((Q,nextNodes))

        if sum(edge_in) == 0:
            break
  
    # Add In and outs back and reorder wMat according to sort
    Q += nIns+nOuts
    Q = jnp.r_[lookup[:nIns], Q, lookup[nIns:nIns+nOuts]]
    wMat = wMat[jnp.ix_(Q,Q)]

    return Q, wMat

def getLayer(wMat):

    wMat[jnp.isnan(wMat)] = 0  
    wMat[wMat!=0]=1
    nNode = jnp.shape(wMat)[0]
    layer = jnp.zeros((nNode))
    while (True): # Loop until sorting is stable
        prevOrder = jnp.copy(layer)
        for curr in range(nNode):
            srcLayer=jnp.zeros((nNode))
            for src in range(nNode):
                srcLayer[src] = layer[src]*wMat[src,curr]   
            layer[curr] = jnp.max(srcLayer)+1    
        if all(prevOrder==layer):
            break
    return layer-1

# Activation : 

def act(weights, aVec, nIjnput, nOutput, ijnpattern):
    if jnp.ndim(weights) < 2:
        nNodes = int(jnp.sqrt(jnp.shape(weights)[0]))
        wMat = jnp.reshape(weights, (nNodes, nNodes))
    else:
        nNodes = jnp.shape(weights)[0]
        wMat = weights
    wMat[jnp.isnan(wMat)]=0

    if jnp.ndim(ijnpattern) > 1:
        nSamples = jnp.shape(ijnpattern)[0]
    else:
        nSamples = 1

    nodeAct  = jnp.zeros((nSamples,nNodes))
    nodeAct[:,0] = 1 # Bias activation
    nodeAct[:,1:nIjnput+1] = ijnpattern

    iNode = nIjnput+1
    for iNode in range(nIjnput+1,nNodes):
        rawAct = jnp.dot(nodeAct, wMat[:,iNode]).squeeze()
        nodeAct[:,iNode] = applyAct(aVec[iNode], rawAct) 
      
    output = nodeAct[:,-nOutput:]   
    return output

def applyAct(actId, x):

    if actId == 1:   # Linear
        value = x

    if actId == 2:   # Unsigned Step Function
        value = 1.0*(x>0.0)
    #value = (jnp.tanh(50*x/2.0) + 1.0)/2.0

    elif actId == 3: # Sin
        value = jnp.sin(jnp.pi*x) 

    elif actId == 4: # Gaussian with mean 0 and sigma 1
        value = jnp.exp(-jnp.multiply(x, x) / 2.0)

    elif actId == 5: # Hyperbolic Tangent (signed)
        value = jnp.tanh(x)     

    elif actId == 6: # Sigmoid (unsigned)
        value = (jnp.tanh(x/2.0) + 1.0)/2.0

    elif actId == 7: # Inverse
        value = -x

    elif actId == 8: # Absolute Value
        value = abs(x)   

    elif actId == 9: # Relu
        value = jnp.maximum(0, x)   

    elif actId == 10: # Cosine
        value = jnp.cos(jnp.pi*x)

    elif actId == 11: # Squared
        value = x**2

    else:
        value = x

    return value


# -- Action Selection ---------------------------------------------------- -- #

def selectAct(action, actSelect):  
    if actSelect == 'softmax':
        action = softmax(action)
    elif actSelect == 'prob':
        action = weightedRandom(jnp.sum(action,axis=0))
    else:
        action = action.flatten()
    return action

def softmax(x):   
    if x.ndim == 1:
        e_x = jnp.exp(x - jnp.max(x))
        return e_x / e_x.sum(axis=0)
    else:
        e_x = jnp.exp(x.T - jnp.max(x,axis=1))
        return (e_x / e_x.sum(axis=0)).T

def weightedRandom(weights):
  minVal = jnp.min(weights)
  weights = weights - minVal # handle negative vals
  cumVal = jnp.cumsum(weights)
  pick = jnp.random.uniform(0, cumVal[-1])
  for i in range(len(weights)):
    if cumVal[i] >= pick:
      return i
        

def exportNet(filename,wMat, aVec):
    indMat = jnp.c_[wMat,aVec]
    jnp.savetxt(filename, indMat, delimiter=',',fmt='%1.2e')

def importNet(fileName):
    ind = jnp.loadtxt(fileName, delimiter=',')
    wMat = ind[:,:-1]     # Weight Matrix
    aVec = ind[:,-1]      # Activation functions

    # Create weight key
    wVec = wMat.flatten()
    wVec[jnp.isnan(wVec)]=0
    wKey = jnp.where(wVec!=0)[0] 

    return wVec, aVec, wKey
