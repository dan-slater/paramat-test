"""Module for reading points of interest (pois) on stress-strain curves and producing plots and tables of the pois."""
from dataclasses import dataclass

import numpy as np
from matplotlib import pyplot as plt

from src.paramat_test.data_chief import DataSet, DataItem

PROCESSED_DATA = r'../data/03 processed data'
PROCESSED_INFO = r'../info/03 processed info.xlsx'


@dataclass
class POIs:
    dataitem: DataItem = None
    pmax: float = None
    pmax_i: int = None
    smax: float = None
    smax_i: float = None
    s01: float = None
    s03: float = None
    efin: float = None

    def read_off(self, dataitem):
        data = dataitem.data
        self.dataitem = dataitem
        p = data['Force(kN)']
        e = data['Strain']
        s = data['Stress(MPa)']
        self.pmax = p.max()
        self.pmax_i = p.idxmax()
        self.smax = s.max()
        self.smax_i = s.idxmax()
        self.s01 = np.interp(0.1, e, s)
        self.s03 = np.interp(0.3, e, s)
        self.efin = e.values[-1]

    def plot_pois(self, ax):
        e = self.dataitem.data['Strain']
        s = self.dataitem.data['Stress(MPa)']
        self.dataitem.data.plot(x='Strain', y='Stress(MPa)', ax=ax)
        ax.scatter(e[self.pmax_i], s[self.pmax_i], label=f'$\sigma_{{F_{{max}} = {self.pmax} \\, \\text{{kN}}}}$')
        ax.scatter(e[self.smax_i], self.smax, label=f'$\\sigma_{{max}} = {self.smax}$ MPa')
        ax.scatter(0.1, self.s01, label=f'$\\sigma_{{\\varepsilon = 0.1}} = {self.s01}$ MPa')
        ax.scatter(0.3, self.s03, label=f'$\\sigma_{{\\varepsilon = 0.3}} = {self.s03}$ MPa')
        ax.legend()

def main():
    processed_set = DataSet()
    processed_set.load(PROCESSED_DATA, PROCESSED_INFO)
    for dataitem in processed_set.datamap:
        pois = POIs()
        pois.read_off(dataitem)
        fig, ax = plt.subplots(1, 1, figsize=(9, 6))
        pois.plot_pois(ax)
        plt.show()


if __name__ == '__main__':
    main()
