from typing import Dict, List

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from src.paramat_test.data_chief import DataItem, DataSet

STRAIN_KEY = 'Strain'
STRESS_KEY = 'Stress(MPa)'

SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12
FONT = 11

# plt.style.use('pastel')

# sns.set_theme()
font = {'family': 'sans-serif',
        'size': MEDIUM_SIZE}
plt.rc('font', **font)
# plt.rc('font', size=FONT)  # controls default text sizes
plt.rc('axes', titlesize=FONT)  # fontsize of the axes title
plt.rc('axes', labelsize=FONT)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)  # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath} \usepackage{amssymb}'


def make_plots(cfg: Dict):
    plots = [
        before_after_processing,
        fitted_curves,
        report_plots,
        easy_view_receipts
    ]

    for plot in plots:
        if plot.__name__ in cfg['plots']:
            print(f'{".": <10}Plotting {plot.__name__}')
            plot(**cfg[plot.__name__])


def three_spine_plot(
        ax,
        x_data: np.ndarray,
        y_data: np.ndarray,
        y_data_label: str,
        y_data_color: str,
        x_axis_label: str,
        y_axis_label: str,
        y_data_2: np.ndarray = np.array([None]),
        y_data_2_label: str = '',
        y_data2_color: str = '',
        y_axis_2_label: str = '',
        y_data_3: np.ndarray = np.array([None]),
        y_data_3_label: str = '',
        y_data3_color: str = '',
        y_axis_3_label: str = '',
        alpha1: float = 1.0,
        alpha2: float = 1.0,
        alpha3: float = 1.0,
):
    twin1 = ax.twinx()
    twin2 = ax.twinx()
    twin2.spines.right.set_position(("axes", 1.15))

    p1, = ax.plot(x_data, y_data, y_data_color, label=y_data_label, alpha=alpha1)
    p2, = twin1.plot(x_data, y_data_2, y_data2_color, label=y_data_2_label, alpha=alpha2)
    p3, = twin2.plot(x_data, y_data_3, y_data3_color, label=y_data_3_label, alpha=alpha3)

    # ax.set_xlim(0, 2)
    # ax.set_ylim(0, 2)
    # twin1.set_ylim(0, 4)
    # twin2.set_ylim(1, 65)

    ax.set_xlabel(x_axis_label)
    ax.set_ylabel(y_axis_label)
    twin1.set_ylabel(y_axis_2_label)
    twin2.set_ylabel(y_axis_3_label)

    ax.yaxis.label.set_color(p1.get_color())
    twin1.yaxis.label.set_color(p2.get_color())
    twin2.yaxis.label.set_color(p3.get_color())

    ax.ticklabel_format(useOffset=False, style='plain')
    twin1.ticklabel_format(style='plain')
    twin2.ticklabel_format(style='plain')

    # tkw = dict(size=4, width=1.5)
    # ax.tick_params(axis='y', colors=p1.get_color(), **tkw)
    # twin1.tick_params(axis='y', colors=p2.get_color(), **tkw)
    # twin2.tick_params(axis='y', colors=p3.get_color(), **tkw)
    # ax.tick_params(axis='x', **tkw)
    # ax.legend(handles=[p1, p2, p3])
    twin2.legend(handles=[p1, p2, p3])
    return ax


def report_plots(
        plot_dir: str,
        pdf_name: str,
        data_path: str,
        info_path: str,
        filters: Dict[str, List[str]],
) -> None:
    dataset = DataSet()
    dataset.load(data_path, info_path, filters)
    with PdfPages(f'{plot_dir}/{pdf_name}') as pdf:
        for dataitem in dataset.datamap:
            test_id = dataitem.test_id
            df = dataitem.data
            df['rate'] = np.gradient(df['Strain'], df['Time(sec)'])
            fig, axes = plt.subplots(3, 2, sharex='col')  # super-plot setup
            df.plot(ax=axes[0, 0], x='Strain', y='Stress(MPa)')
            df.plot(ax=axes[1, 0], x='Strain', y='TC1(C)')
            df.plot(ax=axes[2, 0], x='Strain', y='rate')
            df.plot(ax=axes[0, 1], x='Time(sec)', y='TC1(C)')
            df.plot(ax=axes[1, 1], x='Time(sec)', y='Force(kN)')
            df.plot(ax=axes[2, 1], x='Time(sec)', y='Strain')
            plt.suptitle(f'{" ".join(test_id.split("_"))} report plots.')
            pdf.savefig(dpi=100)
            plt.close()


def before_after_processing(
        plot_dir: str,
        pdf_name: str,
        before_dir: str,
        before_info: str,
        after_dir: str,
        after_info: str,
        filters: Dict[str, List[str]],
        strain_key: str = 'Strain',
        stress_key: str = 'Stress(MPa)',
        min_strain: float = 0.0,
        max_strain: float = 2.0
):
    before_set = DataSet()
    after_set = DataSet()
    before_set.load(before_dir, before_info, filters)
    after_set.load(after_dir, after_info, filters)
    # assert (before_set.info_table.index == after_set.info_table.index).all() ?

    with PdfPages(f'{plot_dir}/{pdf_name}') as pdf:
        for test_id in before_set.info_table.index:
            fig, (ax) = plt.subplots(1, 1, figsize=(6, 4))  # super-plot setup
            plt.title(f'{" ".join(test_id.split("_"))}', loc='left')
            ax.set_xlabel(strain_key)
            ax.set_ylabel(stress_key)
            try:
                for data_dir, label, color in zip([before_dir, after_dir], ['before', 'after'], ['pink', 'b']):
                    df = pd.read_csv(f'{data_dir}/{test_id}.csv')
                    df = df[df[strain_key] >= min_strain]  # trim data
                    df = df[df[strain_key] <= max_strain]
                    strain = df[strain_key].values
                    stress = df[stress_key].values
                    ax.plot(strain, stress, color=color, label=label)
                    # ax.plot(strain, stress, color='k', lw=0, marker='x')
            except TypeError or KeyError as e:
                add_stamp_to(ax, f"{type(e)}\n{e.__str__()}")
            ax.legend()
            pdf.savefig(dpi=100)
            plt.close()


def fitted_curves(
        plot_dir: str,
        pdf_name: str,
        fitted_data: str,
        fitted_info: str,
        filters: Dict[str, List[str]],
        models: List[str],
        colors: List[str],
):
    fitted_set = DataSet()
    fitted_set.load(fitted_data, fitted_info, filters)
    with PdfPages(f'{plot_dir}/{pdf_name}') as pdf:
        for dataitem in fitted_set.datamap:
            fig, (ax) = plt.subplots(1, 1, figsize=(10, 4))  # super-plot setup
            id = " ".join(dataitem.test_id.split("_"))
            info = dataitem.info_row
            hmn = info['experimentalist']
            typ = info['test type'][:3]
            mat = info['material']
            rate = info['rate']
            temp = info['temperature']
            plt.title(f'{id} {hmn} {typ} {mat} S{rate} T{temp}', loc='left')
            try:
                df = pd.read_csv(f'{fitted_data}/{dataitem.test_id}.csv')
                for model, color in zip(models, colors):
                    strain = df[f'model strain'].values
                    stress = df[f'{model} stress'].values
                    ax.plot(strain, stress, color=color, label=model)
                df.plot(x='Strain', y='Stress(MPa)', ax=ax, color='pink', label='data')
                df.plot(x='sampled strain', y='sampled stress', ax=ax, lw=0, color='k', marker='x',
                        label='sampled points', markersize=1.2)
            except Exception as e:
                add_stamp_to(ax, f"{type(e)}\n{e.__str__()}")
            ax.set_xlabel('Strain (mm/mm)')
            ax.set_ylabel('Stress (MPa)')
            ax.legend()
            plt.tight_layout()
            pdf.savefig(dpi=100)
            plt.close()


def add_stamp_to(ax: plt.Axes, stamp: str) -> plt.Axes:
    ax.set_facecolor('lightgray')
    ax.annotate(stamp, xy=(0.05, 0.8), xycoords='axes fraction', color='r', size=16)
    return ax


def make_combined_subplot_of(data_list: List[DataItem],
                             save_path: str,
                             unit_width: float = 3.,
                             unit_height: float = 3.,
                             cols: int = 3,
                             xmin: float = None,
                             xmax: float = None,
                             ymin: float = None,
                             ymax: float = None,
                             trimmed: bool = False,
                             sharex: str = 'all',
                             sharey: str = 'all'
                             ) -> None:
    """Unpacks list of data and makes subplot of all data."""
    nr_units = len(data_list)
    rows = int(np.ceil(nr_units / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(cols * unit_width, rows * unit_height), sharex=sharex, sharey=sharey)
    for ax, data in zip(axs.flat, data_list):
        strain, stress = data.get_stress_strain(trimmed=trimmed)
        name = data.name
        ax.plot(strain, stress)
        ax.set_title(name)
        ax.set_xlim(xmin=xmin, xmax=xmax)
        ax.set_ylim(ymin=ymin, ymax=ymax)
        ax.xaxis.set_tick_params(which='both', labelbottom=True)
        ax.yaxis.set_tick_params(which='both', labelbottom=True)
        ax.grid()

    plt.tight_layout()
    plt.savefig(save_path, dpi=900)


def make_raw_plot(ax: plt.Axes, dataitem: DataItem, name: str, mode='strain'):
    data = dataitem.data
    id = dataitem.test_id
    if mode == 'strain':
        three_spine_plot(
            ax, data['Strain'], data['Stress(MPa)'], 'Stress', 'b', 'Strain (mm/mm)', 'Stress (MPa)',
            data['TC1(C)'], 'Temperature', 'r', 'T $^{\circ}$C',
            np.gradient(data['Strain'], data['Time(sec)']), 'Strain-rate', 'g', 'Strain-rate (s$^{-1}$)', alpha3=0.5,
            alpha2=0.5
        )
        ax.set_ylim(ymin=-10)
    elif mode == 'time':
        three_spine_plot(
            ax, data['Time(sec)'], data['Force(kN)'], 'Force', 'b', 'Time (s)', 'Force (kN)',
            data['TC1(C)'], 'Temperature', 'r', 'T $^{\circ}$C',
            data['Strain'], 'Strain', 'g', 'Strain (mm/mm)', alpha1=0.6, alpha2=0.5
        )
    else:
        print('Error: make_raw_plot mode')
    plt.tight_layout()
    plt.savefig(f'plots/receipts/tests/{id}/{name}')


def make_processing_plot(ax: plt.Axes, dataitem: DataItem, name: str, mode='strain'):
    data = dataitem.data
    id = dataitem.test_id
    if mode == 'strain':
        three_spine_plot(
            ax, data['Strain'], data['Stress(MPa)'], 'Stress', 'b', 'Strain (mm/mm)', 'Stress (MPa)',
            data['TC1(C)'], 'Temperature', 'r', 'T $^{\circ}$C',
            np.gradient(data['Strain'], data['Time(sec)']), 'Strain-rate', 'g', 'Strain-rate (s$^{-1}$)', alpha3=0.5, alpha2=0.5
        )
        ax.set_ylim(ymin=-10)
    elif mode == 'time':
        three_spine_plot(
            ax, data['Time(sec)'], data['Force(kN)'], 'Force', 'b', 'Time (s)', 'Force (kN)',
            data['TC1(C)'], 'Temperature', 'r', 'T $^{\circ}$C',
            data['Strain'], 'Strain', 'g', 'Strain (mm/mm)', alpha1=0.6, alpha2=0.5
        )
    else:
        print('Error: make_raw_plot mode')
    plt.tight_layout()
    plt.savefig(f'plots/receipts/tests/{id}/{name}')


def make_fitted_plot(ax: plt.Axes, dataitem: DataItem, name: str, mode: str):
    df = dataitem.data
    id = dataitem.test_id
    info = dataitem.info_row

    if mode == 'data':
        z = 0
        for model, color in zip(['perfect', 'linear', 'voce', 'ramberg'],
                                ['grey', 'limegreen', 'darkcyan', 'darkmagenta']):
            strain = df[f'model strain'].values
            stress = df[f'{model} stress'].values
            z += 1
            ax.plot(strain, stress, color=color, label=f'{model.title()} fitted model', zorder=z)
        df.plot(x='Strain', y='Stress(MPa)', ax=ax, color='b', label='Measured stress data', alpha=0.65)
        df.plot(x='sampled strain', y='sampled stress', ax=ax, lw=0, color='k', marker='x',
                label='Sampled points for fitting', markersize=2.5)
        twin2 = ax.twinx()
        twin2.spines.right.set_position(("axes", 1.21))
        twin2.spines.right.set_visible(False)
        twin2.yaxis.set_ticks([])
        twin2.set_ylabel('a')
        twin2.yaxis.label.set_color('w')
    elif mode == 'repr':
        count = 1
        for model, color in zip(['perfect', 'linear', 'voce', 'ramberg'],
                                ['grey', 'limegreen', 'darkcyan', 'darkmagenta']):
            count += 1
            strain = df[f'model strain'].values
            stress = df[f'{model} stress'].values
            ax.plot(strain, stress, color=color, label=model.title(), zorder=count)
            error = info[f'{model} error']
            ax.fill_between(strain, stress - error, stress + error, alpha=0.2, color=color, zorder=count)

    ax.set_xlabel('Strain (mm/mm)')
    ax.set_ylabel('Stress (MPa)')
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'plots/receipts/tests/{id}/{name}')


def make_error_histogram(ax: plt.Axes, dataitem: DataItem, name: str):
    max_error = 10
    import matplotlib.patches as mpatches
    id = dataitem.test_id
    info = dataitem.info_row
    model_names = ['perfect', 'linear', 'voce', 'ramberg']
    errors_mpa = [info[f'{s} error'] for s in model_names]
    max_stress = np.max(dataitem.data['Stress(MPa)'].values)
    errors = np.array(errors_mpa)*100/max_stress
    colours = ['grey', 'limegreen', 'darkcyan', 'darkmagenta']
    bar_positions = np.arange(4)
    ax.bar(bar_positions, errors, color=colours, label=model_names, alpha=0.5)
    ax.set_xticks(bar_positions)
    # ax.set_xticks(bar_positions, labels=[s.title() for s in model_names])
    thresh_line, = ax.plot([-1, 6], [max_error] * 2, lw=2, linestyle='--', color='k', label='Flag threshold',
                           alpha=0.5)
    if not any(x < max_error for x in errors):
        add_stamp_to(ax, 'Max error threshold exceeded for all models!')
    ax.set_xticklabels([s.title() for s in model_names])
    patches = [mpatches.Patch(color=c, label=f'{m.title()} error', alpha=0.5) for c, m in zip(colours, model_names)]
    patches.append(thresh_line)
    ax.legend(handles=patches)
    ax.set_xlabel('Fitted models')
    ax.set_ylabel('Error (MPa)')
    twin2 = ax.twinx()
    twin2.spines.right.set_position(("axes", 1.15))
    twin2.spines.right.set_visible(False)
    twin2.yaxis.set_ticks([])
    twin2.set_ylabel('a')
    twin2.yaxis.label.set_color('w')
    plt.tight_layout()
    plt.savefig(f'plots/receipts/tests/{id}/{name}')