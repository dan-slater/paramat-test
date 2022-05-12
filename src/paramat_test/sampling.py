"""For sampling data to pass into fitting. [danslater, 7apr22]"""
from src.paramat_test.data_chief import DataItem


def sample(dataitem: DataItem, sample_size: int):
    dataitem.info_row['nr of points sampled'] = sample_size
    df = dataitem.data
    sampling_stride = int(len(df) / sample_size)
    if sampling_stride < 1:
        sampling_stride = 1
    x_data = df['Strain'].values[::sampling_stride]
    y_data = df['Stress(MPa)'].values[::sampling_stride]
    return x_data, y_data

# todo: sample data with minimum in given area (i.e both x and y tolerance)
# todo: variable sampling runtime src
# todo: full elastic range with plastic sampling only
# todo: equidistant strain sampling
# todo: if strain decreasing, omit point