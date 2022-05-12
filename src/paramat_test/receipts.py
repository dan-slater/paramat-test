import copy
import os
import shutil
import subprocess
from typing import Dict
import pandas as pd
from jinja2 import Environment, FileSystemLoader, meta
from matplotlib import pyplot as plt
from src.paramat_test.data_chief import DataSet, DataItem
from src.paramat_test.plotting import make_raw_plot, make_processing_plot, make_fitted_plot, make_error_histogram

ENV = Environment(
    variable_start_string=r'\VAR{',
    variable_end_string='}',
    autoescape=False,
    loader=FileSystemLoader(os.path.abspath('../..'))
)

PREPARED_DATA = r'../data/02 prepared data'
PREPARED_INFO = r'../info/02 prepared info.xlsx'

PROCESSED_DATA = r'../data/03 processed data'
PROCESSED_INFO = r'../info/03 processed info.xlsx'

FITTED_DATA = r'../data/04 fitted data'
FITTED_INFO = r'../info/04 fitted info.xlsx'

TEMPLATE = r'src/plots/receipts/template.tex'
OUTPUT = r'plots/receipts/tests'


def easy_view_receipts(cfg):
    empty_vars = parse_vars_from_template(TEMPLATE)

    prepped_set = DataSet()
    prepped_set.load(PREPARED_DATA, PREPARED_INFO)
    for dataitem in prepped_set.datamap:
        # break
        id = dataitem.test_id
        make_output_folder(OUTPUT, id)

        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        make_raw_plot(ax, dataitem, 'gleeble-out-strain.pdf', mode='strain')
        plt.close()
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        make_raw_plot(ax, dataitem, 'gleeble-out-time.pdf', mode='time')
        plt.close()
        # break

    processed_set = DataSet()
    processed_set.load(PROCESSED_DATA, PROCESSED_INFO)
    for dataitem in processed_set.datamap:
        # break
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        make_processing_plot(ax, dataitem, 'processing-strain.pdf', mode='strain')
        plt.close()
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        make_processing_plot(ax, dataitem, 'processing-time.pdf', mode='time')
        plt.close()
        # break

    fitted_set = DataSet()
    fitted_set.load(FITTED_DATA, FITTED_INFO)
    for dataitem in fitted_set.datamap:
        id = dataitem.test_id

        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        make_fitted_plot(ax, dataitem, 'fitted-plot.pdf', mode='data')
        plt.close()
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        make_error_histogram(ax, dataitem, 'fitted-error-repr-curve.pdf')
        plt.close()

        filled_vars = fill_vars(empty_vars, dataitem)
        write_output_file(TEMPLATE, f'{OUTPUT}/{id}/{id}_receipt.tex', filled_vars)
        make_latex_pdf(id, f'{OUTPUT}/{id}')
        # break


def fill_vars(empty_vars: Dict[str, str], dataitem: DataItem) -> Dict[str, str]:
    filled_vars = copy.deepcopy(empty_vars)
    info = dataitem.info_row
    id = dataitem.test_id
    filled_vars['id'] = id.replace('_', ' ')
    infotable_df = pd.DataFrame(info).loc[
        ['experimentalist', 'test type', 'material', 'direction', 'rate', 'temperature']
    ]
    infotable_df.index = infotable_df.index.map(str.title)
    infotable_df.index = infotable_df.index.str.replace('Rate', 'Rate (s$^{-1}$)')
    infotable_df.index = infotable_df.index.str.replace('Uniaxial compressionSH', 'UC')
    infotable_df.index = infotable_df.index.str.replace('PSC BigSamples', 'PSC big')
    infotable_df.index = infotable_df.index.str.replace('PSC SmallSamples', 'PSC small')
    infotable_df.loc['Test Type'] = infotable_df.loc['Test Type'].str.replace('_', ' ')
    infotable_df.loc['Test Type'] = infotable_df.loc['Test Type'].str.replace('Uniaxial compressionSH', 'UC')
    infotable_df.loc['Test Type'] = infotable_df.loc['Test Type'].str.replace('PSC BigSamples', 'PSC big')
    infotable_df.loc['Test Type'] = infotable_df.loc['Test Type'].str.replace('PSC SmallSamples', 'PSC small')
    infotable_df.index = infotable_df.index.str.replace('Temperature', 'Temperature ($^{\circ}$C)', regex=False)
    # filled_vars['infotable'] = re.sub(  # make column header bold
    #     '<([^>]*)>', '\\\\textbf{\g<1>}', infotable_df.rename(columns=lambda x: f'<{x}>').to_latex(bold_rows=False)
    # )
    filled_vars['infotable'] = infotable_df.to_latex(header=False, escape=False, longtable=True, caption='Test info')
    # filled_vars['infotable'] = infotable_df.to_latex()
    processedtable_df = pd.DataFrame(info).loc[
        ['raw data indices', 'sampling rate trim indices', 'max force trim indices', 'cluster trim indices']
    ]
    processedtable_df.index = processedtable_df.index.str.replace('sampling rate trim indices', 'Rate trim indices')
    processedtable_df.index = processedtable_df.index.map(str.capitalize)
    filled_vars['processedtable'] = processedtable_df.to_latex(header=False, longtable=True, caption='Processing info')
    fittedtable_df = pd.DataFrame(info).loc[
        ['nr of points sampled',
         'perfect error', 'E perfect opt val', 's_y perfect opt val',
         'linear error', 'E linear opt val', 's_y linear opt val', 'H linear opt val',
         'voce error', 'E voce opt val', 's_y voce opt val', 's_u voce opt val', 'd voce opt val',
         'ramberg error', 'E ramberg opt val', 's_y ramberg opt val', 'C ramberg opt val', 'q ramberg opt val']
    ]
    fittedtable_df.index = fittedtable_df.index.map(str.capitalize)
    fittedtable_df.index = fittedtable_df.index.str.replace('Perfect error', 'Perfect error')
    fittedtable_df.index = fittedtable_df.index.str.replace('Linear error', 'Linear error')
    fittedtable_df.index = fittedtable_df.index.str.replace('Voce error', 'Voce error')
    fittedtable_df.index = fittedtable_df.index.str.replace('Ramberg error', 'Ramberg error')

    fittedtable_df.index = fittedtable_df.index.str.replace('E perfect opt val', 'Perfect $E$')
    fittedtable_df.index = fittedtable_df.index.str.replace('S_y perfect opt val', r'Perfect $\sigma_y$',
                                                            regex=False)
    fittedtable_df.index = fittedtable_df.index.str.replace('E linear opt val', 'Linear $E$')
    fittedtable_df.index = fittedtable_df.index.str.replace('S_y linear opt val', 'Linear $\sigma_y$',
                                                            regex=False)
    fittedtable_df.index = fittedtable_df.index.str.replace('H linear opt val', 'Linear $H$')
    fittedtable_df.index = fittedtable_df.index.str.replace('E voce opt val', 'Voce $E$')
    fittedtable_df.index = fittedtable_df.index.str.replace('S_y voce opt val', 'Voce $\sigma_y$', regex=False)
    fittedtable_df.index = fittedtable_df.index.str.replace('S_u voce opt val', 'Voce $\sigma_u$', regex=False)
    fittedtable_df.index = fittedtable_df.index.str.replace('D voce opt val', 'Voce $d$')
    fittedtable_df.index = fittedtable_df.index.str.replace('E ramberg opt val', 'Ramberg $E$')
    fittedtable_df.index = fittedtable_df.index.str.replace('S_y ramberg opt val', 'Ramberg $\sigma_y$',
                                                            regex=False)
    fittedtable_df.index = fittedtable_df.index.str.replace('C ramberg opt val', 'Ramberg $C$')
    fittedtable_df.index = fittedtable_df.index.str.replace('Q ramberg opt val', 'Ramberg $q$')

    fittedtable_df['Units'] = [' '] + ['MPa']*11 + [' '] + ['MPa']*4 + [' ']

    filled_vars['fittedtable'] = fittedtable_df.to_latex(header=False, escape=False, longtable=True, caption='Fitting info')
    return filled_vars


def parse_vars_from_template(template_path):
    template_source = ENV.loader.get_source(ENV, template_path)
    parsed_content = ENV.parse(template_source)
    vars = {key: None for key in meta.find_undeclared_variables(parsed_content)}
    return vars


def make_output_folder(output_dir, filename):
    test_folder_path = f'{output_dir}/{filename}'
    if not os.path.exists(test_folder_path):
        os.makedirs(test_folder_path)
    else:
        shutil.rmtree(test_folder_path)
        os.makedirs(test_folder_path)


def write_output_file(template_path: str, out_path: str, vars: Dict[str, str]):
    template = ENV.get_template(template_path)
    document = template.render(**vars)
    with open(out_path, 'w') as out_file:
        out_file.write(document)


def make_latex_pdf(id, output_path):
    src_wd = os.getcwd()
    os.chdir(output_path)
    cmd = ['pdflatex', '-interaction', 'nonstopmode', f'{id}_receipt.tex']
    proc = subprocess.Popen(cmd)
    proc.communicate()
    retcode = proc.returncode
    if not retcode == 0:
        os.unlink(f'{id}_receipt.pdf')
        raise ValueError('Error {} executing command: {}'.format(retcode, ' '.join(cmd)))
    os.chdir(src_wd)


def combine_receipts():
    from PyPDF2 import PdfFileMerger

    pdfs = []

    test_receipts_dir = '../src/plots/receipts/tests'
    for folder in os.listdir(test_receipts_dir):
        for file in os.listdir(f'{test_receipts_dir}/{folder}'):
            if file.startswith('testID') and file.endswith('.pdf'):
                # print(f'{test_receipts_dir}/{folder}/{file}')
                pdfs.append(f'{test_receipts_dir}/{folder}/{file}')

    merger = PdfFileMerger()

    for pdf in pdfs:
        merger.append(pdf)

    merger.write(f"../src/plots/receipts/combined-receipts.pdf")
    merger.close()

