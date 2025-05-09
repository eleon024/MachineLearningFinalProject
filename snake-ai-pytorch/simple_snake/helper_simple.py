import matplotlib.pyplot as plt
from IPython import display

plt.ion()

def plot(scores, mean_scores, title, filename, filepath, save=True):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title(title)
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

fig, axs = plt.subplots(1, 3, figsize=(12, 5))
display.display(fig)

def plot_combined(scores, mean_scores, times, speeds):
    axs[0].cla()
    axs[0].set_title('Score')
    axs[0].plot(scores, label='Score')
    axs[0].plot(mean_scores, label='Mean')
    axs[0].legend()
    axs[0].set_ylim(ymin=0)

    axs[1].cla()
    axs[1].set_title('Survival Time')
    axs[1].plot(times, color='orange')
    axs[1].set_ylim(ymin=0)

    axs[2].cla()
    axs[2].set_title('Average Number of Moves')
    axs[2].plot(speeds, color='green')
    axs[2].set_ylim(ymin=0)

    # Redraw without creating a new figure
    display.clear_output(wait=True)
    display.display(fig)
    plt.pause(0.001)

    plt.savefig("figures/"+str("combined_plot_simple")+".svg", format="svg")