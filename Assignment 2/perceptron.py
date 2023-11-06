import numpy as np

class Perceptron:

    def __init__(self, n, m):
            self.n = n
            self.m = m
            self.w = .1 * np.random.random((n + 1, m)) - 0.05

    def __str__(self):
            return "A perceptron with " +  str(self.n) + " inputs and " + str(self.m) + " outputs."

    def augment(self, x):
        return np.append(x, [1])

    def test(self, a):
        return np.dot(self.augment(a), self.w) > 0

    def train(self, I, T, niter = 1000, reportInterval = False):
        for i in range(niter):
            if reportInterval:
                if i % 100 == 0:
                    print(i, '/', niter)
            for j in range(len(I)):
                Ij = self.augment(I[j])
                Oj = np.dot(Ij, self.w) > 0
                Dj = T[j] - Oj
                self.w = self.w + np.outer(Ij, Dj)

def testBoolean(targets, label):
        inputs = [[0,0], [0,1], [1,0], [1,1]]
        p = Perceptron(2, 1)
        p.train(inputs, targets)
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

    print("\nPart 1: Learning Boolean functions AND and OR --------------\n")
    targets = [[0], [0], [0], [1]]
    testBoolean(targets, "AND:")
    targets = [[1], [1], [0], [1]]
    testBoolean(targets, "OR:")

    print("\nPart 2: Reading in handwritten digits --------------------\n")

    inputs = readDigits("digits_train.txt")

    print("Finished reading in handwritten digits")

    print("\nPart 3: Training -----------------------------------------\n")

    #For targets, we can ignore the actual target vectors and generate 
    # a 2500x10 matrix with 1s in the appropriate places. Here we put 
    targets = np.zeros((2500, 10))
    targets[500:750] = 1
    
    p = Perceptron(196, 1)

    p.train(inputs, targets, 1000, True)

    correct = 0
    false_positives = 0
    false_negatives = 0

    inputs = readDigits("digits_test.txt")

    for k in range(2500):
        out = p.test(inputs[k])
        if targets[k].all() == 0 and out.all() == 1:
            false_positives += 1
        if targets[k].all() == 1 and out.all() == 0:
            false_negatives += 1
        else:
            correct += 1

    print("\nCorrect rate: ", round(correct / 25, 2), "%", sep = "")
    print("False positive rate: ", round(false_positives / 22.5, 2), "%", sep = "")
    print("False negatives rate: ", round(false_negatives / 2.5, 2), "%", sep = "")


if __name__ == "__main__":
    main()