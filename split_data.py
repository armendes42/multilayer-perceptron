import numpy as np
import pandas as pd
import sys
import csv

def main():
    if len(sys.argv) != 2:
        print("usage: split_data.py [dataset]")
    else:
        try:
            data = pd.read_csv(sys.argv[1])
            data.dropna(inplace=True)
            data = data.sample(frac=1).reset_index(drop=True)

            train_size = int(len(data) * 0.8)
            data_train = data[:train_size]
            data_test = data[train_size:]

            with open("dataset_train.csv", 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["Index", "Id", "Result"] + ["Feature " + str(i) for i in range(data.shape[1] - 2)])
                for index, row in data_train.iterrows():
                    writer.writerow([index] + list(row))

            with open("dataset_test.csv", 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["Index", "Id", "Result"] + ["Feature " + str(i) for i in range(data.shape[1] - 2)])
                for index, row in data_test.iterrows():
                    writer.writerow([index - len(data_train)] + list(row))

        except FileNotFoundError:
            print(sys.argv[1] + " not found.")


if __name__ == "__main__":
    main()