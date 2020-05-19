import matplotlib.pyplot as plt

def MyPlot(coordinates):

    xs = [x[0] for x in coordinates]

    ys = [x[1] for x in coordinates]

    plt.plot(xs, ys, '*')

    plt.plot(xs, ys)

    plt.show()
