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
        plt.savefig("snake-ai-pytorch/figures/"+str(filename)+".svg", format="svg")


fig, axs = plt.subplots(1, 3, figsize=(12, 5))
display.display(fig)
def plot_combined(plot_scores, plot_mean_scores, plot_times, plot_mean_time, speeds_per_game, plot_mean_speed):
    axs[0].cla()
    axs[0].set_title('Score')
    axs[0].plot(plot_scores, label='Score')
    axs[0].plot(plot_mean_scores, label='Mean')
    axs[0].legend()
    axs[0].set_ylim(ymin=0)
    
    x_m, y_m = len(plot_mean_scores)-1, plot_mean_scores[-1]
    axs[0].text(x_m, y_m, str(y_m),
            ha='left', va='bottom')
    
    axs[1].cla()
    axs[1].set_title('Survival Time')
    axs[1].plot(plot_times, label='Time')
    axs[1].plot(plot_mean_time, label='Mean')
    axs[1].legend()
    axs[1].set_ylim(ymin=0)
    
    x_n, y_n = len(plot_mean_time)-1, plot_mean_time[-1]
    axs[1].text(x_n, y_n, str(y_n),
            ha='left', va='bottom')

    axs[2].cla()
    axs[2].set_title('Number of Moves')
    axs[2].plot(speeds_per_game, label='Moves')
    axs[2].plot(plot_mean_speed, label='Mean')
    axs[2].legend()
    axs[2].set_ylim(ymin=0)
    
    x_o, y_o = len(plot_mean_speed)-1, plot_mean_speed[-1]
    axs[2].text(x_o, y_o, str(y_o),
            ha='left', va='bottom')


    # Redraw without creating a new figure
    display.clear_output(wait=True)
    display.display(fig)
    plt.pause(0.001)

    plt.savefig("snake-ai-pytorch/figures/"+str("combined_plot_complex_obs")+".svg", format="svg")

