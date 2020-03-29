import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np
import subprocess

state_url = 'https://covidtracking.com/api/states/daily.csv'
us_url = 'https://covidtracking.com/api/us/daily.csv'
df_state = pd.read_csv(state_url)
df_usa = pd.read_csv(us_url)
SUFFIXES = {1: 'st', 2: 'nd', 3: 'rd'}
starting_caseload = 100


def ordinal(num):
    # I'm checking for 10-20 because those are the digits that
    # don't follow the normal counting scheme.
    if 10 <= num % 100 <= 20:
        suffix = 'th'
    else:
        # the second parameter is a default.
        suffix = SUFFIXES.get(num % 10, 'th')
    return str(num) + suffix


def modify_df(df, initial_caseload, plot_stat):
    pd.set_option("mode.chained_assignment", None)
    df['NewDate'] = pd.to_datetime(
        df['date'].copy(), format='%Y%m%d')
    start_date = df[(df[plot_stat]) >
                    initial_caseload]['NewDate'].min()
    df['date_zero'] = (df['NewDate'] -
                       start_date).astype('timedelta64[D]')
    pd.set_option("mode.chained_assignment", 'warn')
    return df


df_grid = pd.DataFrame(index=np.arange(0, 100, 0.25))
df_grid['daily'] = 2 ** df_grid.index
df_grid['2 days'] = 2 ** (df_grid.index / 2)
df_grid['3 days'] = 2 ** (df_grid.index / 3)
df_grid['7 days'] = 2 ** (df_grid.index / 7)

for stat in ['positive', 'death']:
    fig, ax = plt.subplots()

    for col in df_grid.columns:
        ax.semilogy(df_grid.index, df_grid[col], ':k')
    x_max, y_max = 0, 0
    for state in ['VA', 'NY', 'WA', 'CA', 'LA']:
        df_plot = df_state[df_state['state'] == state]
        df_plot = modify_df(df_plot, starting_caseload, stat)
        ax.semilogy(df_plot['date_zero'], df_plot[stat], label=state)
        if df_plot[stat].max() > y_max:
            y_max = df_plot[stat].max()
        if df_plot['date_zero'].max() > x_max:
            x_max = df_plot['date_zero'].max()
    df_usa = modify_df(df_usa, starting_caseload, stat)
    ax.semilogy(df_usa['date_zero'], df_usa[stat],
                'k', linewidth=2, label='Total US')
    if df_usa[stat].max() > y_max:
        y_max = df_usa[stat].max()
    if df_usa['date_zero'].max() > x_max:
        x_max = df_usa['date_zero'].max()

    ax.set_ylim(top=y_max, bottom=starting_caseload)
    ax.set_xlim(left=0, right=x_max)
    ax.legend()
    str_xaxis_label = 'Days since {:s} {:s}'.format(
        ordinal(starting_caseload), stat)
    ax.set_xlabel(str_xaxis_label)
    str_yaxis_label = 'Total number of {:s}s'.format(stat)
    ax.set_ylabel(str_yaxis_label)
    ax.yaxis.set_major_formatter(ScalarFormatter())
    fname = 'covid_' + stat + '.pdf'
    plt.savefig(fname)
    subprocess.run(['open', fname])
# plt.show()
