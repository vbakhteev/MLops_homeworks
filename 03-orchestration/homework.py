from datetime import datetime
from pathlib import Path

import dateutil.relativedelta
import mlflow
import pandas as pd
from joblib import dump
from prefect import flow, task, get_run_logger
from prefect.deployments import DeploymentSpec
from prefect.flow_runners import SubprocessFlowRunner
from prefect.orion.schemas.schedules import CronSchedule
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

mlflow.set_tracking_uri('sqlite:///mlflow.db')
mlflow.set_experiment('hw3')


@task()
def get_paths(date: str) -> tuple[Path, Path]:
    date = datetime.strptime(date, '%Y-%m-%d')

    prev_month = date - dateutil.relativedelta.relativedelta(months=1)
    prev_prev_month = prev_month - dateutil.relativedelta.relativedelta(months=1)

    data_dir = Path('../data/fhv/')
    train_path = data_dir / f'fhv_tripdata_2021-{str(prev_prev_month.month).zfill(2)}.parquet'
    valid_path = data_dir / f'fhv_tripdata_2021-{str(prev_month.month).zfill(2)}.parquet'

    return train_path, valid_path


@task()
def read_data(path):
    df = pd.read_parquet(path)
    return df


@task()
def prepare_features(df, categorical, train=True):
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    mean_duration = df.duration.mean()

    logger = get_run_logger()
    if train:
        logger.info(f"The mean duration of training is {mean_duration}")
    else:
        logger.info(f"The mean duration of validation is {mean_duration}")

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    return df


@task()
def train_model(df, categorical):
    train_dicts = df[categorical].to_dict(orient='records')
    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts)
    y_train = df.duration.values

    logger = get_run_logger()
    logger.info(f"The shape of X_train is {X_train.shape}")
    logger.info(f"The DictVectorizer has {len(dv.feature_names_)} features")

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_train)
    mse = mean_squared_error(y_train, y_pred, squared=False)
    logger.info(f"The MSE of training is: {mse}")
    return lr, dv


@task()
def run_model(df, categorical, dv, lr):
    val_dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(val_dicts)
    y_pred = lr.predict(X_val)
    y_val = df.duration.values

    mse = mean_squared_error(y_val, y_pred, squared=False)
    logger = get_run_logger()
    logger.info(f"The MSE of validation is: {mse}")
    return


@flow()
def main(date=None):
    if date is None:
        date = datetime.now().strftime('%Y-%m-%d')

    train_path, val_path = get_paths(date).result()

    categorical = ['PUlocationID', 'DOlocationID']

    df_train = read_data(train_path).result()
    df_train_processed = prepare_features(df_train, categorical).result()

    df_val = read_data(val_path).result()
    df_val_processed = prepare_features(df_val, categorical).result()

    # train the model
    lr, dv = train_model(df_train_processed, categorical).result()
    run_model(df_val_processed, categorical, dv, lr)

    out_dir = Path('out')
    if not out_dir.exists():
        out_dir.mkdir()

    dump(lr, out_dir / f'model-{date}.bin')
    dump(dv, out_dir / f'dv-{date}.bin')

    mlflow.log_artifacts(str(out_dir))

# prefect agent start ae1ee06f-71d4-4643-9578-9d29b60e2384
DeploymentSpec(
    flow=main,
    name="model_training",
    schedule=CronSchedule(cron='0 9 15 * *'),
    flow_runner=SubprocessFlowRunner(),
    tags=["ml"]
)

if __name__ == '__main__':
    main(date="2021-08-15")
