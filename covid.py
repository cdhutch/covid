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
        self.df = pd.DataFrame()

    @staticmethod
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
                if 'l_localities' in d_dataset.keys():
                    df = df[df['locality'].isin(d_dataset['l_localities'])]
                self.df = self.df.append(df, sort=False)
        else:
            df = pd.read_csv(self.url)
            df.rename(columns=self.d_remap, inplace=True)
            self.df = df
        self.df['NewDate'] = pd.to_datetime(
            self.df['date'].copy(), format=self.source_date_format)

    def create_ft_figure(self, stat='positive', starting_caseload=100):

        def _modify_df(df, initial_caseload, plot_stat):
            pd.set_option("mode.chained_assignment", None)
            df['NewDate'] = pd.to_datetime(
                df['date'].copy(), format='%Y%m%d')
            start_date = df[(df[plot_stat]) >
                            initial_caseload]['NewDate'].min()
            df['date_zero'] = (df['NewDate'] -
                               start_date).astype('timedelta64[D]')
            pd.set_option("mode.chained_assignment", 'warn')
            return df

        ft_fig, ax = plt.subplots()
        df_grid = pd.DataFrame(index=np.arange(0, 100, 0.25))
        d_grids = {
            'daily': (1, (0, (5, 1))),
            '2 days': (2, (0, (5, 5))),
            '3 days': (3, (0, (5, 10))),
            'weekly': (7, (0, (1, 10)))}
        for grid in d_grids:
            df_grid[grid] = 2 ** (df_grid.index / d_grids[grid]
                                  [0]) * starting_caseload
            ax.semilogy(df_grid.index, df_grid[grid], label=grid,
                        color='darkgray', linestyle=d_grids[grid][1])
        x_max, y_max = 0, 0
        for locality in self.df['locality'].unique():
            df_plot = self.df[self.df['locality'] == locality]
            df_plot = _modify_df(df_plot, starting_caseload, stat)
            ax.semilogy(df_plot['date_zero'], df_plot[stat], label=locality)
            if df_plot[stat].max() > y_max:
                y_max = df_plot[stat].max()
            if df_plot['date_zero'].max() > x_max:
                x_max = df_plot['date_zero'].max()

        ax.set_ylim(top=y_max, bottom=starting_caseload)
        ax.set_xlim(left=0, right=x_max)
        ax.legend()
        str_xaxis_label = 'Days since {:s} {:s}'.format(
            self.return_ordinal(starting_caseload), stat)
        ax.set_xlabel(str_xaxis_label)
        str_yaxis_label = 'Total number of {:s}s'.format(stat)
        ax.set_ylabel(str_yaxis_label)
        ax.yaxis.set_major_formatter(ScalarFormatter())
        return ft_fig


d_us = {'url': 'https://covidtracking.com/api/us/daily.csv',
        'd_remap': {'states': 'locality'},
        'source_date_format': '%Y%m%d',
        'l_assignments': [('locality', 'US')]
        }
d_state = {'url': 'https://covidtracking.com/api/states/daily.csv',
           'd_remap': {'state': 'locality'},
           'source_date_format': '%Y%m%d',
           'l_localities': ['VA', 'NY', 'WA', 'CA', 'LA', 'FL', 'NJ', 'GA'],
           }


def open_pdf(pdf_fname):
    open_cmd = 'open'
    if sys.platform == 'darwin':
        open_cmd = 'open'
    elif sys.platform == 'linux':
        open_cmd = 'xdg-open'
    elif sys.platform == 'win32':
        open_cmd = 'explorer'
    subprocess.run([open_cmd, pdf_fname])


if __name__ == '__main__':

    us_states = CovidDataset()
    us_states.load([d_us, d_state])
    d_stats = {
        'cases': ('positive', 100),
        'deaths': ('death', 10)
    }
    for key in d_stats:
        fig = us_states.create_ft_figure(d_stats[key][0], d_stats[key][1])
        fname = 'covid_' + d_stats[key][0] + '.pdf'
        plt.savefig(fname)
        open_pdf(fname)
