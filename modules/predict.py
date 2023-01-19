import glob
import os

import pandas as pd
import dill
import json

from datetime import datetime


path = os.environ.get('PROJECT_PATH', '.')


def predict():
    with open(f'{path}/data/models/cars_pipe.pkl', 'rb') as file:
        model = dill.load(file)

    df_pred = pd.DataFrame(columns=['car_id', 'pred'])
    for jsonfile in glob.glob(f'{path}/data/test/*.json'):
        with open(jsonfile, 'r') as j:
            form = json.load(j)
            df = pd.DataFrame([form])
            y = model.predict(df)
            x = {'car_id': df.id, 'pred': y}
            df1 = pd.DataFrame(x)
            df_pred = pd.concat([df_pred, df1], axis=0)

    df_pred.to_csv(f'{path}/data/predictions/preds_{datetime.now().strftime("%Y%m%d%H%M")}.csv')


if __name__ == '__main__':
    predict()
