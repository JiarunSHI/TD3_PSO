import matplotlib.pyplot as plt

def plot_map(obstacles, path=None, start=None, goal=None):

    plt.figure()

    for ox, oy, r in obstacles:
        circle = plt.Circle((ox, oy), r)
        plt.gca().add_patch(circle)

    if path is not None:
        plt.plot(path[:,0], path[:,1], marker='o')

    if start is not None:
        plt.scatter(start[0], start[1])

    if goal is not None:
        plt.scatter(goal[0], goal[1])

    plt.xlim(0,50)
    plt.ylim(0,50)
    plt.gca().set_aspect('equal')
    plt.show()