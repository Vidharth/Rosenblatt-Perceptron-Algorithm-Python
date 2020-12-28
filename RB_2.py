"""
NNCI Project 1
Author: Ashwin Vaidya, S3911888
Author: Hari Vidharth, S4031180
"""

import numpy as np
from matplotlib import pyplot as plt

plt.style.use("ggplot")
CValues = [0]
CValuesNValuesVsQls = {}
NValues = [50, 100, 200]
RandomDatasetsNd = 100
MeanMu = 0
StandardDeviationSigma = 1
EpochsNmax = 1000
PlotAlphaValues = []
y = []

for CValue in CValues:
    C = CValue
    print("C Value {0}/{1}".format(C, CValues[-1]))
    NValuesVsQls = {}
    for NValue in NValues:
        N = NValue
        print("N Value {0}/{1}".format(N, NValues[-1]))
        Qls = []
        Alpha = 0.0
        for AlphaValue in range(0, 20):
            Alpha += 0.25
            print("Alpha Value {0}".format(Alpha), end="\r")
            AccuracyValues = []
            for Dataset in range(0, RandomDatasetsNd):
                print("Dataset {0}/{1}".format(Dataset + 1, RandomDatasetsNd), end="\r")
                P = int(Alpha * N)
                X = np.random.normal(MeanMu, StandardDeviationSigma, (P, N))
                Y = np.random.choice([-1, +1], size=(P, 1), p=[1/2, 1/2])
                W = np.zeros((1, N))
                for Epoch in range(0, EpochsNmax):
                    print("Epoch {0}/{1}".format(Epoch + 1, EpochsNmax), end="\r")
                    LocalPotentialEValues = []
                    for Location, Feature in enumerate(X):
                        LocalPotentialE = np.dot(W, Feature) * Y[Location]
                        if LocalPotentialE[0] <= C:
                            W = W + ((1 / N) * Feature * Y[Location])
                        LocalPotentialEValues.append(LocalPotentialE[0])
                    if np.min(LocalPotentialEValues) > 0:
                        break
                CorrectlyClassified = 0.0
                for Location, Feature in enumerate(X):
                    if np.sign(np.dot(W, Feature)) == Y[Location]:
                        CorrectlyClassified += 1.0
                Accuracy = CorrectlyClassified/P
                AccuracyValues.append(Accuracy)
            LinearSeparability = 0.0
            for Location, AccuracyValue in enumerate(AccuracyValues):
                if AccuracyValue == 1.0:
                    LinearSeparability += 1.0
            FinalAccuracy = LinearSeparability / len(AccuracyValues)
            Qls.append(FinalAccuracy)
            NValuesVsQls[N] = Qls
    CValuesNValuesVsQls[C] = NValuesVsQls

print(CValuesNValuesVsQls)

Alpha = 0.0
for Value in range(0, 20):
    Alpha += 0.25
    PlotAlphaValues.append(Alpha)

for Items, Values in CValuesNValuesVsQls.items():
    for Item, Value in Values.items():
        y.append(Value)

plt.figure(0)
plt.plot(PlotAlphaValues, y[0], "-r", label="N = 50")
plt.plot(PlotAlphaValues, y[1], "-g", label="N = 100")
plt.plot(PlotAlphaValues, y[2], "-b", label="N = 200")
plt.xlabel("Alpha Values")
plt.ylabel("Ql.s.")
plt.title("Ql.s. of Linearly Separable Functions vs Alpha")
plt.legend(loc="upper right")
plt.show()

"""
plt.figure(1)
plt.plot(PlotAlphaValues, y[3], "-r", label="N = 50")
plt.plot(PlotAlphaValues, y[4], "-g", label="N = 100")
plt.plot(PlotAlphaValues, y[5], "-b", label="N = 200")
plt.xlabel("Alpha Values")
plt.ylabel("Ql.s.")
plt.title("Ql.s. of Linearly Separable Functions vs Alpha")
plt.legend(loc="upper right")
"""
