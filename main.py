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
    # dzielimy element algebraiczny (np. alfa^4 -> [1,0,0,0,0]), przez wielomian pierwotny, reszta jest naszym elementem
    # algebraicznym w formie wielomianu, stąd [1], bo np.polydiv zwraca dwa elementy - wynik i reszte
    result = np.absolute(np.polydiv(divPoly, primPoly)[1])
    # każdy element musimy dać modulo 2, bo współczynniki mogą być tylko 0-1
    if (baseDegree - result.size > 0):
        result = [x % 2 for x in np.pad(result, (baseDegree - result.size, 0), "constant")]
    else:
        result = [x % 2 for x in result]
    return result


def findAlgebraicElementsTable(baseDegree, primPoly):
    algebraicElementsTable = []
    # tworzymy tabele elementow algebraicznych, których jest zawsze (2^m) - 2 dla ciał binarnych
    for x in range(pow(2, baseDegree) - 1):
        algebraicElementsTable.append(findAlgebraicElement(x, primPoly, baseDegree))
    return algebraicElementsTable


def findMinimalPolynomialForAlgebraicElement(algebraicElementDegree, baseDegree, primPoly):
    # znajdujemy tabele i stopień wielomianu
    algebraicElementsTable = findAlgebraicElementsTable(baseDegree, primPoly)
    print(algebraicElementsTable, len(algebraicElementsTable))
    k = findMinimalPolynomialDegree(algebraicElementDegree, 2, pow(2, baseDegree))
    vectorCoefficients = []
    # dla elementu algebraicznego o zadanym stopniu znajdujemy wielomian minimalny, podstawiając
    # zadany element algebraiczny do template'u rownania, przeprowadzając operacje mod 2^m - 1, która sprawi,
    # że współczynnik nie wyjdzie poza zakres tabeli, a następnie odczytując formę wektorową elementu po modularyzacji
    for x in range(k, -1, -1):
        index = (x * algebraicElementDegree) % (pow(2, baseDegree) - 1)
        vectorCoefficients.append(algebraicElementsTable[index])
    # następnie rozwiązujemy układ n równań, w zależności od stopnia ciała podstawowego
    b = vectorCoefficients[0]
    A = np.transpose(vectorCoefficients[1:])
    try:
        result = np.absolute(np.append(np.array([1]), np.linalg.solve(A, b)))
    except:
        result = np.absolute(np.append(np.array([1]), np.linalg.lstsq(A, b, rcond=None)[0]))
    result = np.floor([x % 2 for x in result])
    print(f"result for {algebraicElementDegree}:", result)
    return result


def findGeneratorPolynomial(baseDegree, primPoly, t):
    generatorPolynomial = np.array([1])
    minimalPolynomials = []
    for x in range(1, 2 * t + 1, 2):
        minimalPolynomials.append(findMinimalPolynomialForAlgebraicElement(x, baseDegree, primPoly))
    # our simulation of LCM
    uniqueMinimalPolynomials = {array.tobytes(): array for array in minimalPolynomials}.values()
    print("uniques: ", uniqueMinimalPolynomials)
    for minimalPolynomial in uniqueMinimalPolynomials:
        generatorPolynomial = np.polymul(generatorPolynomial, minimalPolynomial)
    generatorPolynomial = np.mod(generatorPolynomial, 2)
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
    #shift to left
    shiftedMessage = np.roll(message, n-k+1)
    print(len(shiftedMessage), ''.join([str(x) for x in shiftedMessage]))
    print(np.polydiv(shiftedMessage, genPoly))
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
genPoly = findGeneratorPolynomial(8, [1,0,0,0,1,1,1,0,1], 10)
print('genPoly: ', genPoly)
getEncodedMessage(message, 255, 179, genPoly)
