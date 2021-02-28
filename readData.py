def loadData(path):

    rawData = []
    normalizedData = []
    curSent = []

    for line in open(path):

        tok = line.strip().split('\t')

        if tok == [''] or tok == []:
            rawData.append([x[0] for x in curSent])
            normalizedData.append([x[1] for x in curSent])
            curSent = []

        else:
            if len(tok) == 1:
                tok.append('')

            curSent.append(tok)

    if curSent != []:
        rawData.append([x[0] for x in curSent])
        normalizedData.append([x[1] for x in curSent])

    return rawData, normalizedData

rawData, normalizedData = loadData('train.norm')

for i in range(len(rawData)):

    print('Raw Data        : ',' '.join(rawData[i]))
    print('Normalized Data : ',' '.join(normalizedData[i]))
    print('\n')