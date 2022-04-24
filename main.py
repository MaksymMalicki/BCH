import numpy as np

def findMinimalPolynomialDegree(i,p,q):
    #znajdujemy stopien wielomianu minimalnego dla danego elementu algebraicznego
    #korzystajac z wzoru z ksiazki dr. Mochnackiego
    k = 1
    while((i*pow(p, k)) % (q-1) != i%(q-1)):
        k += 1
    return k

def findAlgebraicElementsExponents(i, p, q):
    if i == 0:
        return [1]
    k = findMinimalPolynomialDegree(i, p, q)
    exponents = []
    for x in range(0, k-1):
        exponents.append(i*pow(p, x) % (q-1))
    exponents.append(i*pow(p, k-1) % (q-1))
    return exponents

def findAlgebraicElement(exponent, primPoly, baseDegree):
    divPoly = np.zeros(exponent+1)
    divPoly[0] = 1
    #dzielimy element algebraiczny (np. alfa^4 -> [1,0,0,0,0]), przez wielomian pierwotny, reszta jest naszym elementem
    #algebraicznym w formie wielomianu, stąd [1], bo np.polydiv zwraca dwa elementy - wynik i reszte
    result = np.absolute(np.polydiv(divPoly, primPoly)[1])
    #każdy element musimy dać modulo 2, bo współczynniki mogą być tylko 0-1
    if(baseDegree - result.size):
        result = [x%2 for x in np.pad(result, (baseDegree - result.size, 0), "constant")]
    else:
        result = [x%2 for x in result]
    return result

def findAlgebraicElementsTable(baseDegree, primPoly):
    algebraicElementsTable = []
    #tworzymy tabele elementow algebraicznych, których jest zawsze (2^m) - 2 dla ciał binarnych
    for x in range(pow(2, baseDegree)-1):
        algebraicElementsTable.append(findAlgebraicElement(x, primPoly, baseDegree))
    return algebraicElementsTable

def findMinimalPolynomialForAlgebraicElement(algebraicElementDegree, baseDegree, primPoly):
    #znajdujemy tabele i stopień wielomianu
    algebraicElementsTable = findAlgebraicElementsTable(baseDegree, primPoly)
    #print(algebraicElementsTable, len(algebraicElementsTable))
    k = findMinimalPolynomialDegree(algebraicElementDegree, 2, pow(2, baseDegree))
    vectorCoefficients = []
    #dla elementu algebraicznego o zadanym stopniu znajdujemy wielomian minimalny, podstawiając
    #zadany element algebraiczny do template'u rownania, przeprowadzając operacje mod 2^m - 1, która sprawi,
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
    return result

def findGeneratorPolynomial(baseDegree, primPoly, t):
    generatorPolynomial = np.array([1])
    for x in range(1, 2*t+1, 2):
        generatorPolynomial = np.polymul(generatorPolynomial,findMinimalPolynomialForAlgebraicElement(x, baseDegree, primPoly))
    generatorPolynomial = np.mod(generatorPolynomial, 2)
    return generatorPolynomial

def getMessage(message):
    result = []
    for letter in [char for char in message]:
        for byte in (format(ord(letter), '08b')):
            result.append(int(byte))
    return result

def getEncodedMessage(message, n,k,genPoly):
    shiftVector = [0]*(n-k)
    shiftVector[0] = 1
    shiftedMessage = np.polymul(message, shiftVector)
    remainder = np.polydiv(shiftedMessage, genPoly)[1]
    encodedMessage = [int(x%2) for x in np.absolute(np.append(shiftedMessage, remainder))]
    test = ''.join([str(x) for x in encodedMessage])
    print(test)

message = getMessage('abcd')
genPoly = findGeneratorPolynomial(4, [1, 0, 0, 1, 1], 2)
getEncodedMessage(message, 255, 179, genPoly)