### example structures to work with
from jax.config import config
config.update("jax_enable_x64", True)

import jax.numpy as np

regName = ["Bris", "Town", "Darl"]
numReg = len(regName)
numTime = 4
numPol = 1
LV = numReg * numTime * numPol
vec = np.arange(0, LV, 1)

#-----------the dicts
RId = dict()
for i in range(len(regName)):
    RId[regName[i]] = i

RD = dict()
RDV = dict()
RDL = dict()
for i in range(numReg):
    RD[regName[i]] = slice(i, LV, numReg)
    RDV[regName[i]] = vec[i : LV : numReg]
    RDL[regName[i]] = list(vec[i : LV : numReg])
print(RD)
print(RDV)

# the first year is the "0th" year, this can of course be changed
TD = dict()
TDV = dict()
TDL = dict()

for i in range(numTime):
    TD[i] = slice(i * numReg, (i+1) * numReg, 1)
    TDV[i] = vec[i * numReg : (i+1) * numReg : 1]
    TDL[i] = list(vec[i * numReg : (i+1) * numReg : 1])
print(TD)
print(TDV)
#print(TDL)

# could enter keys as a vector for more expandable function
def xInd(timeKey,
         regKey,
         reg_dict=RD,
         time_dict=TD,
         totInds = numTime*numReg
         ):
    reg = range(totInds)[reg_dict[regKey]]
    time = range(totInds)[time_dict[timeKey]]
    return list(set(reg) & set(time))[0]

def xIndVTF(timeKey,
         regKey,
         reg_dict=RDV,
         time_dict=TDV,
         totInds = numTime*numReg
          ):
    # regKey enters before timeKey as RDV[timeKey] is a vec (would need a region index)
    return TDV[timeKey][RId[regKey]]

def xIndV(timeKey,
         regKey,
         reg_dict=RDV,
         time_dict=TDV,
         totInds = numTime*numReg
          ):
    # regKey enters before timeKey as RDV[timeKey] is a vec (would need a region index)
    return RDV[regKey][timeKey]

def xIndL(timeKey,
         regKey,
         reg_dict=RDL,
         time_dict=TDL,
         totInds = numTime*numReg
         ):
    reg = range(totInds)[reg_dict[regKey]]
    time = range(totInds)[time_dict[timeKey]]
    return list(set(reg) & set(time))[0]
### more testing
print('the index for ', regName[0], 'at time ', 3, 'is ', vec[xInd(3,"Bris")])
print('the index for ', regName[0], 'at time ', 3, 'is ', xIndV(3,"Bris"))
print('the index for ', regName[0], 'at time ', 3, 'is ', xIndVTF(3,"Bris"))
#print('the index for ', regName[0], 'at time ', 3, 'is ', vec[xIndL(3,"Bris")])

