""" Fitting imported constitutive models to sampled stress-strain data. [danslater, 2march2022] """
from typing import Dict

import numpy as np

import src.paramat_test.models as hm
from src.paramat_test.data_chief import DataItem
from src.paramat_test.sampling import sample


model_funcs = [
    hm.perfect,
    hm.linear,
    hm.voce,
    hm.quadratic,
    hm.ramberg
]


def fit_models(dataitem: DataItem, cfg: Dict):
    # sample data
    x_data, y_data = sample(dataitem, cfg['sampling']['sample_size'])  # sample data
    strain_vec = np.linspace(0, x_data[-1], len(dataitem.data))  # make strain monotonically increasing

    dataitem.data['model strain'] = strain_vec
    dataitem.data['sampled strain'] = np.hstack([x_data, np.array([None] * (len(dataitem.data) - len(x_data)))])
    dataitem.data['sampled stress'] = np.hstack([y_data, np.array([None] * (len(dataitem.data) - len(y_data)))])

    # setup models
    models = []
    for model_name in cfg['models']:
        for model_func in model_funcs:
            if model_name == model_func.__name__:
                model = hm.IsoReturnMapModel(
                    name=model_name,
                    func=model_func,
                    param_names=(cfg['bounds'][model_name].keys()),
                    bounds=[eval(bounds) for bounds in cfg['bounds'][model_name].values()],
                    constraints=cfg['constraints'][model_name],
                    # x_data=x_data,  # actual strain data
                    x_data=strain_vec,  # artificial strain data
                    y_data=y_data
                )
                models.append(model)

    # fit models
    for model in models:
        print(f'{".": <10}Fitting "{model.name}" to "{dataitem.test_id}".')
        model.fit()

        dataitem.data[f'{model.name} stress'] = model.predict()
        dataitem.data[f'{model.name} plastic strain'] = model.predict_plastic_strain()
        dataitem.data[f'{model.name} accumulated plastic strain'] = model.predict_accumulated_plastic_strain()
        dataitem.info_row[f'{model.name} error'] = round(model.opt_res.fun, 2)  # store residual error

        for param_name, param_bounds, param_val in zip(model.param_names, model.bounds, list(model.opt_res.x)):
            dataitem.info_row[f'{param_name} {model.name} bounds'] = param_bounds  # store bounds
            dataitem.info_row[f'{param_name} {model.name} opt val'] = round(param_val, 2)  # store optimised value

    return dataitem
