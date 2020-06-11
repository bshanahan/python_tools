import numpy as np
import matplotlib.pyplot as plt

def line_plot(independent, dependent, hline=None, vline=None, labels=None, fname=None, xlabel=' ', ylabel=' ', vline_label=None, hline_label=None, AR=(8,4.5)):
    plt.rc('font', family='Serif')
    plt.figure(figsize=AR)

    linewidth = 2.5

    n_dep = len(dependent)
    if labels==None:
        labels = np.full(n_dep,None)
    
    for k in np.arange(0,n_dep):
        plt.plot(independent, dependent[k], linewidth=linewidth, label=labels[k])

    ## plot any straight lines    
    if vline is not None:
        for i in np.arange(0,len(vline)):
            plt.axvline(vline[i], color='black', linestyle='dashed', linewidth=linewidth-0.5, label=vline_label[i])

    if hline is not None:
        for i in np.arange(0,len(hline)):
            plt.axhline(hline[i], color='black', linestyle='dashed', linewidth=linewidth-0.5, label=hline_label[i])


    plt.xlabel(str(xlabel), fontsize=18)
    plt.ylabel(str(ylabel), fontsize=18)
    if labels is not None:
        plt.legend(loc='best')
    plt.tick_params('both', labelsize=14)
    plt.tight_layout()
    plt.grid(alpha=0.5)
    if fname is not None:
        plt.savefig(str(fname)+'.png', dpi=300)
    plt.show()
