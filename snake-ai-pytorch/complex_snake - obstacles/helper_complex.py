import matplotlib.pyplot as plt
from IPython import display

plt.ion()

def plot(scores, mean_scores,save=False,filename="figure"):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Traditional Snake Traning: Score vs Number of games')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores, label="scores")
    plt.plot(mean_scores, label="average")
    plt.legend()
    plt.ylim(ymin=0)
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores)-1.25, mean_scores[-1], str(mean_scores[-1]))
    plt.show(block=False)
    plt.pause(.1)

    if (save):
        plt.savefig("figures/"+str(filename)+".svg", format="svg")
