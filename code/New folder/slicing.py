### example structures to work with
regName = ["Bris", "Town", "Darl"]
numReg = len(regName)
numTime = 4

vec=[1,2,3,4,5,6,7,8,9,10,11,12]

### functions below are what we want to keep

regDict = dict()
for i in range(numReg):
    regDict[regName[i]] = slice(i*numTime, (i+1)*numTime)

# the first year is the "0th" year, this can of course be changed
timeDict = dict()
for i in range(numTime):
    timeDict[i] = slice(i, numTime*numReg, numTime)

# could enter keys as a vector for more expandable function, but probably not worth the time
def xInd(timeKey, regKey, regDict=regDict, timeDict=timeDict, totInds = numTime*numReg):
    reg = range(totInds)[regDict[regKey]]
    time = range(totInds)[timeDict[timeKey]]
    return list(set(reg) & set(time))[0]

### more testing
print(vec[xInd(3,"Bris")])