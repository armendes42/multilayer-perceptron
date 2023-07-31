import matplotlib.pyplot as plt
import pandas as pd
import sys
import tkinter as tk

def feature_plot(features, result):
    plt.close()
    _, ax = plt.subplots(nrows=6, ncols=5, figsize=(15, 18))
    plt.subplots_adjust(wspace=0.5, hspace=0.5, left=0.026, right=0.983, top=0.933, bottom=0.036)

    for i, column in enumerate(features.columns):
        row = i // 5
        col = i % 5
        ax[row, col].scatter(result, features[column], label=column)
        ax[row, col].set_title(column)

    plt.suptitle('Scatter Plots of Features vs. Result', fontsize=16)
    plt.show()

def plot(features_M, features_B, name):
    plt.close()
    _, ax = plt.subplots(nrows=6, ncols=5, figsize=(15, 18))
    plt.subplots_adjust(wspace=0.5, hspace=0.5, left=0.026, right=0.983, top=0.933, bottom=0.036)
    for i, column in enumerate(features_M.columns):
        row = i // 5
        col = i % 5
        ax[row, col].scatter(features_M[name], features_M[column], label=column, alpha=0.5, color="red", s=20)
        ax[row, col].scatter(features_B[name], features_B[column], label=column, alpha=0.5, color="blue", s=20)
        ax[row, col].set_title(column)
        ax[row, col].tick_params(labelsize=6)
    plt.suptitle("Scatter Plots of " + name + " vs. All Features", fontsize=16)
    plt.show()


def main():
    if len(sys.argv) != 2:
        print("usage: analyze_data.py [dataset]")
    else:
        try:
            root = tk.Tk()
            root.title("Multiple Pages")

            data = pd.read_csv(sys.argv[1])
            features_M = data.loc[data['Result'] == 'M', ["Feature " + str(i) for i in range(data.shape[1] - 3)]]
            features_B = data.loc[data['Result'] == 'B', ["Feature " + str(i) for i in range(data.shape[1] - 3)]]
            features = data[["Feature " + str(i) for i in range(data.shape[1] - 3)]]
            result = data.loc[:, "Result"]

            page_names = ["Page 01", "Page 02", "Page 03", "Page 04", "Page 05", "Page 06", "Page 07", "Page 08", "Page 09", "Page 10",
                        "Page 11", "Page 12", "Page 13", "Page 14", "Page 15", "Page 16", "Page 17", "Page 18", "Page 19", 
                        "Page 20", "Page 21", "Page 22", "Page 23", "Page 24", "Page 25", "Page 26", "Page 27", "Page 28", 
                        "Page 29", "Page 30"]

            other_button = tk.Button(root, text="Features vs Result", command=lambda : feature_plot(features, result))
            other_button.pack()

            for i in range(0, 30):
                button = tk.Button(root, text=page_names[i], command=lambda i=i: plot(features_M, features_B, "Feature " + str(i)))
                button.pack()

            root.mainloop()
        except FileNotFoundError:
            print(sys.argv[1] + " not found.")


if __name__ == "__main__":
    main()