import numpy as np
import matplotlib.pyplot as plt

def plot_multiple_sequences(sequences, labels=None, title='Multiple Sequence Plot'):
    """
    Plots multiple 1D sequences on the same figure.
    
    Parameters:
    sequences (list of np.ndarray): A list of 1D numpy arrays to plot.
    labels (list of str, optional): Labels for each sequence.
    title (str, optional): Title of the plot.
    """
    plt.figure(figsize=(10, 5))
    
    for i, seq in enumerate(sequences):
        label = labels[i] if labels is not None else f'Sequence {i+1}'
        plt.plot(seq, label=label)
    
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig('test.png')
    plt.show()

# Example usage:
if __name__ == "__main__":
    seq1 = np.load("/home/server35/hyeongwon_workspace/Time-Series-XAI/results_pred/mimic3_state_timex++_result_0_42.npy")
    seq2 = np.load("/home/server35/hyeongwon_workspace/Time-Series-XAI/results_pred/mimic3_state_timing_sample50_seg50_min10_max48_result_0_42.npy")
    seq3 = np.load("/home/server35/hyeongwon_workspace/Time-Series-XAI/results_pred/mimic3_state_winit_result_0_42.npy")
    
    sequences = [seq1, seq2, seq3]
    labels = ['TimeX++', 'Timing', 'WinIT']
    
    plot_multiple_sequences(sequences, labels)
