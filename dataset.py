import pandas as pd

class Dataset():

    def __init__(self, dataset: str, datetime_column_name: str):

        self.dataset = dataset
        self.datetime_column_name = datetime_column_name

    def dataset_preparation(self):

        # dataset
        df = pd.read_csv(self.dataset)

        # order by Datatime column ascending
        df = df.sort_values(by=[self.datetime_column_name], ascending=True)

        # set 'Datetime' column as index of dataset
        df.Datetime = pd.to_datetime(df.Datetime)
        df = df.set_index(self.datetime_column_name)

        return df

    def dataset_split(self):

        day_mean = self.dataset_preparation()

        # split data in train and test
        train_size = int(len(day_mean) * 0.75)
        test_size = len(day_mean) - train_size

        train = day_mean[:train_size]
        test = day_mean[train_size:]

        return train, test