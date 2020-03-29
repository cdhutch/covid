import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np
state_url = 'https://covidtracking.com/api/states/daily.csv'
us_url = 'https://covidtracking.com/api/us/daily.csv'
df = pd.read_csv(state_url)
SUFFIXES = {1: 'st', 2: 'nd', 3: 'rd'}


def ordinal(num):
    # I'm checking for 10-20 because those are the digits that
    # don't follow the normal counting scheme.
    if 10 <= num % 100 <= 20:
        suffix = 'th'
    else:
        # the second parameter is a default.
        suffix = SUFFIXES.get(num % 10, 'th')
    return str(num) + suffix


df_grid = pd.DataFrame(index=np.arange(0, 100, 0.25))
df_grid['daily'] = 2 ** (df_grid.index)
df_grid['2 days'] = 2 ** (df_grid.index / 2)
df_grid['3 days'] = 2 ** (df_grid.index / 3)
df_grid['7 days'] = 2 ** (df_grid.index / 7)

for stat in ['positive', 'death']:
    fig, ax = plt.subplots()

    for col in df_grid.columns:
        ax.semilogy(df_grid.index, df_grid[col], ':k')
    x_max, y_max = 0, 0
    for state in ['VA', 'NY', 'WA', 'CA']:
        df_plot = df[df['state'] == state]
        pd.set_option("mode.chained_assignment", None)
        df_plot['NewDate'] = pd.to_datetime(
            df_plot['date'].copy(), format='%Y%m%d')
        starting_caseload = 1
        start_date = df_plot[(df_plot[stat]) >
                             starting_caseload]['NewDate'].min()
        df_plot['date_zero'] = (df_plot['NewDate'] -
                                start_date).astype('timedelta64[D]')
        pd.set_option("mode.chained_assignment", 'warn')
        ax.semilogy(df_plot['date_zero'], df_plot[stat], label=state)
        if df_plot[stat].max() > y_max:
            y_max = df_plot[stat].max()
        if df_plot['date_zero'].max() > x_max:
            x_max = df_plot['date_zero'].max()
    ax.set_ylim(top=y_max, bottom=1)
    ax.set_xlim(left=0, right=x_max)
    ax.legend()
    str_xaxis_label = 'Days since {:s} {:s}'.format(
        ordinal(starting_caseload), stat)
    ax.set_xlabel(str_xaxis_label)
    ax.yaxis.set_major_formatter(ScalarFormatter())
plt.show()
