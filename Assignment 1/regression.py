import numpy as np
import matplotlib.pyplot as plt

def main():
    x1 = np.loadtxt("assign1_data.txt", usecols = 0, skiprows = 2, dtype = float)
    x2 = np.loadtxt("assign1_data.txt", usecols = 1, skiprows = 2, dtype = float)
    y = np.loadtxt("assign1_data.txt", usecols = 2, skiprows = 2, dtype = float)
    z = np.loadtxt("assign1_data.txt", usecols = 3, skiprows = 2, dtype = int)

    x1Mean = np.mean(x1)
    x2Mean = np.mean(x2)
    yMean = np.mean(y)
    zMean = np.mean(z)

    x1Dif = x1 - x1Mean
    x2Dif = x2 - x2Mean
    yDif = y - yMean
    zDif = z - zMean

    m1 = sum(x1Dif * yDif) / sum(x1Dif * x1Dif)
    m2 = sum(x2Dif * yDif) / sum(x2Dif * x2Dif)
    b1 = yMean - (m1 * x1Mean)
    b2 = yMean - (m2 * x2Mean)

    print("")
    print("PART 1:")
    print("y = mx1 + b ---------> ", "y = ", m1, "x1 + ", b1, sep = "")
    print("y = mx2 + b ---------> ", "y = ", m2, "x2 + ", b2, sep = "")
    print("")

    plt.plot(x1, y, 'o')
    plt.plot(x1, m1 * x1 + b1)
    plt.title("Part 1: x1")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

    plt.plot(x2, y, 'o')
    plt.plot(x2, m2 * x2 + b2)
    plt.title("Part 1: x2")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


    A = np.vstack([x1, x2, np.ones(len(x1))]).T
    w1, w2, b = np.linalg.lstsq(A, y, rcond = -1)[0]

    print("PART 2:")
    print("y = w1x1 + w2x2 + b ---------> ", "y = ", w1, "x1 + ", w2, "x2 + ", b, sep = "")
    print("")

    fig = plt.figure()
    ax = plt.axes(projection = '3d')
    ax.scatter3D(x1, x2, w1 * x1 + w2 * x2 + b)
    ax.set_title("Part 2")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel('y')
    plt.show()

    def successRate(n, baseline = False):
        x1Range = np.array([])
        x2Range = np.array([])
        yRange = np.array([])
        zRange = np.array([])
        if baseline:
            w1 = 0
            w2 = 0
            b = 0
        else:
            for i in range(n):
                x1Range = np.append(x1Range, x1[i])
                x2Range = np.append(x2Range, x2[i])
                yRange = np.append(yRange, y[i])
                zRange = np.append(zRange, z[i])
            A = np.vstack([x1Range, x2Range, np.ones(n)]).T
            w1, w2, b = np.linalg.lstsq(A, yRange, rcond = -1)[0]
        success = 0
        if n == 100:
            for i in range(100):
                if (w1 * x1[i]) + (w2 * x2[i]) + b > 0 and z[i] == 1:
                    success += 1
                elif (w1 * x1[i]) + (w2 * x2[i]) + b <= 0 and z[i] == 0:
                    success += 1
            return str((success / 100) * 100) + "%"
        else:
            for i in range(n, 100):
                if (w1 * x1[i]) + (w2 * x2[i]) + b > 0 and z[i] == 1:
                    success += 1
                elif (w1 * x1[i]) + (w2 * x2[i]) + b < 0 and z[i] == 0:
                    success += 1
            return str((success / (100 - n)) * 100) + "%"


    print("PART 3:")
    print("Success Rate: ", end = "")
    print(successRate(100))
    print("")


    # Part 4
    print("PART 4:")
    print("Success rate for 25:", successRate(25))
    print("Success rate for 50:", successRate(50))
    print("Success rate for 75:", successRate(75))
    print("Baseline rate:", successRate(100, True))
    print("")

if __name__ == "__main__":
    main()