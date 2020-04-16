import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np
import subprocess
import sys


class CovidDataset(object):

    def __init__(self):
        self.url = 'https://covidtracking.com/api/states/daily.csv'
        self.starting_caseload = 1
        self.l_localites = ['VA', 'NY', 'WA', 'CA', 'LA', 'FL', 'NJ', 'GA']
        self.d_remap = {'state': 'locality'}
        self.source_date_format = '%Y%m%d'

    def return_ordinal(num):
        suffixes = {1: 'st', 2: 'nd', 3: 'rd'}
        if 10 <= num % 100 <= 20:
            suffix = 'th'
        else:
            suffix = suffixes.get(num % 10, 'th')
        return str(num) + suffix

    def load(self, l_d_datasets=None):
        if l_d_datasets is not None:
            self.df = pd.DataFrame()
            for d_dataset in l_d_datasets:
                df = pd.read_csv(d_dataset['url'])
                if 'd_remap' in d_dataset.keys():
                    df.rename(columns=d_dataset['d_remap'], inplace=True)
                if 'l_assignments' in d_dataset.keys():
                    for assignment in d_dataset['l_assignments']:
                        df[assignment[0]] = assignment[1]
                self.df = self.df.append(df, sort=False)
        else:
            df = pd.read_csv(self.url)
            df.rename(columns=self.d_remap, inplace=True)
            self.df = df
        self.df['NewDate'] = pd.to_datetime(
            self.df['date'].copy(), format=self.source_date_format)
        # print(self.df)


d_us = {'url': 'https://covidtracking.com/api/us/daily.csv',
        'd_remap': {'states': 'locality'},
        'source_date_format': '%Y%m%d',
        'l_assignments': [('locality', 'US')]
        }
d_state = {'url': 'https://covidtracking.com/api/states/daily.csv',
           'd_remap': {'state': 'locality'},
           'source_date_format': '%Y%m%d',
           }

state_url = 'https://covidtracking.com/api/states/daily.csv'
us_url = 'https://covidtracking.com/api/us/daily.csv'
df_state = pd.read_csv(state_url)
df_usa = pd.read_csv(us_url)
SUFFIXES = {1: 'st', 2: 'nd', 3: 'rd'}
starting_caseload = 1


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


def open_pdf(fname):
    if sys.platform == 'darwin':
        open_cmd = 'open'
    elif sys.platform == 'linux':
        open_cmd = 'xdg-open'
    elif sys.platform == 'win32':
        open_cmd = 'explorer'
    subprocess.run([open_cmd, fname])


def old_code():
    d_grids = {
        'daily': (1, (0, (5, 1))),
        '2 days': (2, (0, (5, 5))),
        '3 days': (3, (0, (5, 10))),
        'weekly': (7, (0, (1, 10)))}
    d_stats = {
        'US_states_cases': ('positive', 100),
        'US_states_deaths': ('death', 10)}
    for key in d_stats:
        stat = d_stats[key][0]
        starting_caseload = d_stats[key][1]
        fig, ax = plt.subplots()

        df_grid = pd.DataFrame(index=np.arange(0, 100, 0.25))
        for grid in d_grids:

            df_grid[grid] = 2 ** (df_grid.index / d_grids[grid]
                                  [0]) * starting_caseload

            ax.semilogy(df_grid.index, df_grid[grid], label=grid,
                        color='darkgray', linestyle=d_grids[grid][1])
        x_max, y_max = 0, 0
        for state in ['VA', 'NY', 'WA', 'CA', 'LA', 'FL', 'NJ', 'GA']:
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
        open_pdf(fname)


if __name__ == '__main__':

    # state = CovidDataset()
    # state.load()
    # us = CovidDataset()
    # us.url = 'https://covidtracking.com/api/us/daily.csv'
    # us.d_remap = {'states': 'locality'}
    # us.load()
    # us.df['locality'] = 'US'
    # print(us.df)
    us_states = CovidDataset()
    us_states.load([d_us, d_state])
    # us_states.load([d_us])
    # us_states.load()
    print(us_states.df)
