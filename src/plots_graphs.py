import matplotlib.pyplot as plt
import pandas as pd

def plot_epochs(metric1, metric2, ylab):
    plt.plot(metric1, label='CAISA2')
    plt.plot(metric2, label='NC16')
    plt.ylabel(ylab)
    plt.xlabel("Epoch")
    plt.legend(loc='lower right')
    plt.show()

if __name__ == "__main__":
    df1 = pd.read_csv(filepath_or_buffer="../data/output/SRM_accuracy_Full_NoRot_LR0005_b200_nodrop.csv")
    df2 = pd.read_csv(filepath_or_buffer="../data/output/nc16_no_rot_results_lr0.001_batch32_accuracy.csv")
    df3 = pd.read_csv(filepath_or_buffer="../data/output/SRM_loss_Full_NoRot_LR0005_b200_nodrop.csv")
    df4 = pd.read_csv(filepath_or_buffer="../data/output/nc16_no_rot_results_lr0.001_batch32_loss.csv")
    plot_epochs(df1.iloc[:, 1], df2.iloc[:, 1], 'Training Accuracy')
    plot_epochs(df3.iloc[:, 1], df4.iloc[:, 1], 'Training Loss')
