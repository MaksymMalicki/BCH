import random

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
        generatorPolynomial = np.rint([x % 2 for x in np.polymul(generatorPolynomial, minimalPolynomial)])
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
    for x in range(errorsNumber):
        message[random.randint(0, len(message) - 1)] = 1 if message[random.randint(0, len(message) - 1)] else 0


def getSyndrome(message, genPoly):
    return np.mod(np.polydiv(message, genPoly)[1], 2)


def getHammingWeight(syndrome):
    return np.count_nonzero(syndrome == 1)

def add_GF2(arr1, arr2):
    if len(arr1) > len(arr2):
        arr2 = np.pad(arr2, (len(arr1) - len(arr2), 0), "constant")
    else:
        arr1 = np.pad(arr1, (len(arr2) - len(arr1), 0), "constant")
    return np.array([1 if x else 0 for x in np.logical_xor(arr1, arr2)])

def multiply_GF2(arr1, arr2):
    return

def decode(corruptedMessage, genPoly, t):
    syndrome = getSyndrome(corruptedMessage, genPoly)
    print("syndrome", syndrome)
    # sprawdzenie czy wszystkie elementy to zero
    if not np.any(syndrome):
        print("Nie ma bledow")
    else:
        if getHammingWeight(syndrome) <= t:
            add_GF2(corruptedMessage, syndrome)
        else:
            rollsCount = 0
            print("message b4 rolling: ", corruptedMessage)
            while True:
                corruptedMessage = np.roll(corruptedMessage, 1)
                print(f"roll: ${rollsCount}, message: {corruptedMessage}, syndrome: {syndrome}, hamming weight: {getHammingWeight(syndrome)}")
                rollsCount += 1
                syndrome = getSyndrome(corruptedMessage, genPoly)
                if getHammingWeight(syndrome) <= t:
                    break
            print(syndrome, corruptedMessage)
            correctMessage = add_GF2(corruptedMessage, syndrome)
            for x in range(rollsCount):
                correctMessage = np.roll(correctMessage, -1)
            print(correctMessage)


# message = getMessage('abcd', 255)
# print(''.join([str(x) for x in message]))
# # algebraicElementsTable = findAlgebraicElementsTable(4, [1, 0, 0, 1, 1])
# # genPoly = findGeneratorPolynomial(4, 3, algebraicElementsTable)
# algebraicElementsTable = findAlgebraicElementsTable(8, [1, 0, 0, 0, 1, 1, 1, 0, 1])
# genPoly = findGeneratorPolynomial(8, 10, algebraicElementsTable)
# print('genPoly: ', genPoly)
#
# # getEncodedMessage(message, 255, 179, genPoly)


decode(np.array([1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0]), np.array([1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1]), 2)


