import numpy as np


### GENERATOR POLYNOMIAL

def findMinimalPolynomialDegree(i, p, q):
    # znajdujemy stopien wielomianu minimalnego dla danego elementu algebraicznego
    # korzystajac z wzoru z ksiazki dr. Mochnackiego
    k = 1
    while ((i * pow(p, k)) % (q - 1) != i % (q - 1)):
        k += 1
    return k


def findAlgebraicElement(exponent, primPoly, baseDegree):
    divPoly = np.zeros(exponent + 1)
    divPoly[0] = 1
    result = np.rint(np.absolute(np.polydiv(divPoly, primPoly)[1]))
    if (baseDegree - result.size > 0):
        result = np.rint([x % 2 for x in np.pad(result, (baseDegree - result.size, 0), "constant")])
    else:
        result = np.rint([x % 2 for x in result])
    return result


def findAlgebraicElementsTable(baseDegree, primPoly):
    algebraicElementsTable = []
    # tworzymy tabele elementow algebraicznych, których jest zawsze (2^m) - 2 dla ciał binarnych
    for x in range(pow(2, baseDegree) - 1):
        algebraicElementsTable.append(findAlgebraicElement(x, primPoly, baseDegree))
    return algebraicElementsTable


def findMinimalPolynomialForAlgebraicElement(algebraicElementDegree, baseDegree, algebraicElementsTable):
    # znajdujemy tabele i stopień wielomianu
    k = findMinimalPolynomialDegree(algebraicElementDegree, 2, pow(2, baseDegree))
    vectorCoefficients = []
    for x in range(k, -1, -1):
        index = (x * algebraicElementDegree) % (pow(2, baseDegree) - 1)
        vectorCoefficients.append(algebraicElementsTable[index])
    b = vectorCoefficients[0]
    A = np.transpose(vectorCoefficients[1:])
    try:
        result = np.rint(np.absolute(np.append(np.array([1]), np.linalg.solve(A, b))))
    except:
        result = np.rint(np.absolute(np.append(np.array([1]), np.linalg.lstsq(A, b, rcond=None)[0])))
    result1 = np.ceil([x % 2 for x in result])
    result2 = np.floor([x % 2 for x in result])
    result3 = np.rint([x % 2 for x in result])
    print(f"result for {algebraicElementDegree}, ceil: ", result1)
    print(f"result for {algebraicElementDegree}, floor:", result2)
    print(f"result for {algebraicElementDegree}, rint: ", result3)
    return result3


def findGeneratorPolynomial(baseDegree, t, algebraicElementsTable):
    generatorPolynomial = np.array([1])
    minimalPolynomials = []
    print(algebraicElementsTable)
    for x in range(1, 2 * t + 1, 2):
        minimalPolynomials.append(findMinimalPolynomialForAlgebraicElement(x, baseDegree, algebraicElementsTable))
    # our simulation of LCM
    uniqueMinimalPolynomials = {array.tobytes(): array for array in minimalPolynomials}.values()
    for minimalPolynomial in uniqueMinimalPolynomials:
        generatorPolynomial = np.rint([x%2 for x in np.polymul(generatorPolynomial, minimalPolynomial)])
    return generatorPolynomial


#### ENCODING

def getMessage(message, k):
    result = []
    for letter in [char for char in message]:
        for byte in (format(ord(letter), '08b')):
            result.append(int(byte))
    for x in range(k - len(result)):
        result.append(0)
    return result


def getEncodedMessage(message, n, k, genPoly):
    # shift to left
    shiftedMessage = np.roll(message, n - k + 1)
    print(len(shiftedMessage), ''.join([str(x) for x in shiftedMessage]))
    remainder = np.mod(np.polydiv(shiftedMessage, genPoly)[1], 2)
    print('reszta: ', len(remainder), remainder)
    encodedMessage = [int(x % 2) for x in np.polyadd(np.absolute(shiftedMessage), np.absolute(remainder))]
    test = ''.join([str(x) for x in encodedMessage])
    print('result: ', len(test), test)


### DECODING

def corruptMessage(message, errorsNumber):
    return


def calculateSyndrome(message):
    return


message = getMessage('abcd', 255)
print(''.join([str(x) for x in message]))
#algebraicElementsTable = findAlgebraicElementsTable(4, [1, 0, 0, 1, 1])
#genPoly = findGeneratorPolynomial(4, 3, algebraicElementsTable)
algebraicElementsTable = findAlgebraicElementsTable(8, [1, 0, 0, 0, 1, 1, 1, 0, 1])
genPoly = findGeneratorPolynomial(8, 10, algebraicElementsTable)
print('genPoly: ', genPoly)

# getEncodedMessage(message, 255, 179, genPoly)
