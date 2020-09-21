from scipy.interpolate import interp1d
import numpy as np

import shlex
import subprocess as sp

from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter
import io
from pandas import read_csv
def plot_git_commits(path='.'):
    plt.rc('font', family='Serif')

    cmd = "git log --no-merges --date=short --pretty='format:%ad\t%H\t%aN'"
    p = sp.Popen(shlex.split(cmd), stdout=sp.PIPE)
    # p.wait() will likely just hang if the log is long enough because the
    # stdout buffer will fill up
    stdout, _ = p.communicate()

    table = read_csv(io.StringIO(stdout.decode('utf-8')), sep='\t',
                     names=['date', 'hash', 'author'], index_col=0,
                     parse_dates=True)
    table = table.to_period(freq='W')
    commits_per_period = table.hash.groupby(level=0).aggregate(len)

    dates = [p.start_time.date() for p in commits_per_period.index]
    ncommits = np.asarray(commits_per_period.values)
    total_commits= np.zeros(ncommits.shape[0])
    for i in np.arange(0,ncommits.shape[0]):
        total_commits[i] = np.sum(ncommits[:i])
    
    fn = interp1d(range(len(dates)), total_commits, 'cubic', bounds_error=False)

    fig, ax = plt.subplots(1)
    fig.set_size_inches((8, 4.5))
    x = np.linspace(0, len(dates), 1000)
    plt.fill_between(x, 0, fn(x), color='royalblue', alpha=0.75)
    ax.set_xlim(0, len(dates))
    ax.set_ylim(0, max(total_commits) + 0.1 * max(total_commits))
    ax.xaxis.set_ticks(np.linspace(0, len(dates) - 1, 8)[1:-1])
    def formatter(x, p):
        if x >= len(dates):
            return ''
        return dates[int(x)].strftime('%b %Y')

    formatter = FuncFormatter(formatter)
    ax.xaxis.set_major_formatter(formatter)
    ax.xaxis.tick_bottom()
    ax.yaxis.tick_left()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(linestyle='dotted', alpha=1.0)
    plt.tick_params('both', labelsize=16)
    plt.ylabel("Total commits", fontsize=22)
    
    plt.tight_layout()
    # There must be a btter way to do this...
    plt.setp(plt.xticks()[1], rotation=30)
    fig.savefig('git-commits.png', bbox_inches='tight')


