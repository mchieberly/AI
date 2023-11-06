import numpy as np
import pickle

class Perceptron:

    def __init__(self, n, m, h):
            self.n = n
            self.m = m
            self.h = h
            self.wih = .3 * np.random.random((n + 1, h)) - 0.05
            self.who = .3 * np.random.random((h + 1, m)) - 0.05

    def f(self, x):
        return 1 / (1 + np.exp(-x))

    def df(self, x):
        y = self.f(x)
        return y * (1 - y)

    def save(self, filename):
        file = open(filename, 'wb')
        pickle.dump((self.wih, self.who), file)
        file.close()

    def load(self, filename):
        file = open(filename, 'rb')
        a = pickle.load(file)
        self.wih = a[0]
        self.who = a[1]

    def __str__(self):
            return "A perceptron with " +  str(self.n) + " inputs and " + str(self.m) + " outputs."

    def augment(self, x):
        return np.hstack((x, np.ones(1)))

    def test(self, a):
            Hnet = np.dot(self.augment(a), self.wih)
            H = self.f(Hnet)
            Onet = np.dot(self.augment(H), self.who)
            return self.f(Onet)

    def train(self, I, T, showInterval = True, niter = 10000, reportInt = 100, eta = 0.5):
        for i in range(niter):
            if showInterval:
                if i % reportInt == 0:
                    print(i, '/', niter)
            dwih = np.zeros((self.n + 1, self.h))
            dwho = np.zeros((self.h + 1, self.m))
            for j in range(len(I)):
                Ij = I[j]
                Hnet = np.dot(self.augment(Ij), self.wih)
                H = self.f(Hnet)
                Onet = np.dot(self.augment(H), self.who)
                O = self.f(Onet)
                eO = T[j] - O
                dO = eO * self.df(Onet)
                #Backprop
                eH = np.dot(dO, self.who.T)[:-1]
                dH = eH * self.df(Hnet)
                dwih += np.outer(self.augment(Ij), dH)
                dwho += np.outer(self.augment(H), dO)
            self.wih += eta * dwih / len(I)
            self.who += eta * dwho / len(I)

def testBoolean(targets, label):
        inputs = [[0,0], [0,1], [1,0], [1,1]]
        p = Perceptron(2, 1, 3)
        p.train(inputs, targets, showInterval = False)
        print(label)
        for I in inputs:
            print(I, p.test(I))
        print()

def readDigits(filename):
    words = open(filename).read().split()
    floats = list(filter(lambda word: '.' in word, words))
    values = [float(f) for f in floats]
    return np.array(values).reshape((2500, 196))

def main():

    print("\nPart 1: Learning Boolean function XOR --------------\n")
    targets = [[0], [1], [1], [0]]
    testBoolean(targets, "XOR:")
    '''
    print("\nPart 2: 2-not-2 -------------------------------------\n")

    inputs = readDigits("digits_train.txt")
    targets = np.zeros((2500))
    targets[500:750] = 1
    
    p = Perceptron(196, 1, 5)

    p.train(inputs, targets, niter = 100)

    p.save("part2w.dat")
    p.load("part2w.dat")


    correct = 0
    false_positives = 0
    false_negatives = 0

    inputs = readDigits("digits_test.txt")

    for k in range(2500):
        out = p.test(inputs[k])
        if k in range(730, 870):
            print(targets[k], out[0])
        if targets[k] == 0.0 and out[0] > 0.35:
            false_positives += 1
        if targets[k] == 1.0 and out[0] < 0.35:
            false_negatives += 1
        else:
            correct += 1

    print("\nCorrect rate: ", round(correct / 25, 2), "%", sep = "")
    print("False positive rate: ", round(false_positives / 22.5, 2), "%", sep = "")
    print("False negatives rate: ", round(false_negatives / 2.5, 2), "%", sep = "")


    '''
    print("\nPart 3: Classifying -----------------------------------------\n")
    #np.argmax

    inputs = readDigits("digits_train.txt")
    targets = np.zeros((2500,10))
    targets[0:250] = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    targets[250:500] = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    targets[500:750] = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    targets[750:1000] = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    targets[1000:1250] = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
    targets[1250:1500] = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
    targets[1500:1750] = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
    targets[1750:2000] = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
    targets[2000:2250] = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
    targets[2250:2499] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]


    a = np.zeros((10, 10))

    p = Perceptron(196, 10, 5)

    p.train(inputs, targets)

    p.save("part3w.dat")
    p.load("part3w.dat")

    correct = 0
    false_positives = 0
    false_negatives = 0

    inputs = readDigits("digits_test.txt")

    for k in range(2500):
        result = p.test(inputs[k])
        result = np.argmax(result)


if __name__ == "__main__":
    main()