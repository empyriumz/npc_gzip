import numpy as np
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    average_precision_score,
    roc_auc_score,
)
from npc_gzip.compressors.base import BaseCompressor
from npc_gzip.compressors.gzip_compressor import GZipCompressor
from npc_gzip.knn_classifier import KnnClassifier


def one_hot_encode(sequence):
    # Define the 20 standard amino acids
    amino_acids = list("ACDEFGHIKLMNPQRSTVWY")

    # Create a dictionary mapping amino acids to their one-hot encodings
    aa_to_onehot = {
        aa: [1 if i == j else 0 for j in range(20)] for i, aa in enumerate(amino_acids)
    }

    # One-hot encode the sequence
    one_hot_sequence = [aa_to_onehot.get(aa, [0] * 20) for aa in sequence]

    return one_hot_sequence


# Function for padding sequences
def pad_sequences(sequences, maxlen=None, padding="post", value="X"):
    if maxlen is None:
        maxlen = max(len(seq) for seq in sequences)

    # Initialize a matrix of zeros having dimensions (len(sequences), maxlen)
    padded_sequences = [[value] * maxlen for _ in range(len(sequences))]

    for i, seq in enumerate(sequences):
        if padding == "post":
            padded_sequences[i][: len(seq)] = seq
        elif padding == "pre":
            padded_sequences[i][-len(seq) :] = seq

    return padded_sequences


def get_data(train=True):
    # Lists to store the sequences and labels
    sequences = []
    labels = []
    if train:
        fpath = "mib_train.txt"
    else:
        fpath = "mib_test.txt"
    # Open the file and read line by line
    with open(fpath, "r") as f:
        for line in f:
            # Split the line by tab to separate sequence and label
            seq, label = line.strip().split("\t")
            # Append to respective lists
            sequences.append(seq)
            labels.append(int(label))  # Convert label to integer
    sequences = np.array(sequences)
    # sequence_encoded = [one_hot_encode(seq) for seq in sequences]
    # sequence_encoded = pad_sequences(sequence_encoded)
    # sequence_encoded = np.stack(sequence_encoded)

    # sequence_encoded = np.array(sequence_encoded)
    labels = np.array(labels)

    return sequences, labels


def get_esm_data(train=True):
    # Lists to store the sequences and labels
    sequences = []
    labels = []
    if train:
        fpath = "mib_train.txt"
        embed_path = "precomputed_embeddings/esm_small/train/"
    else:
        fpath = "mib_test.txt"
        embed_path = "precomputed_embeddings/esm_small/test/"
    # Open the file and read line by line
    with open(fpath, "r") as f:
        for i, line in enumerate(f):
            # Split the line by tab to separate sequence and label
            seq, label = line.strip().split("\t")
            # Append to respective lists
            esm_embedding = np.load(embed_path + seq[0] + str(i) + "_avg" + ".npy")
            sequences.append(esm_embedding)
            labels.append(int(label))  # Convert label to integer
    sequences = np.stack(sequences)
    labels = np.array(labels)

    return sequences, labels


def fit_model(
    train_text: np.ndarray, train_labels: np.ndarray, distance_metric: str = "ncd"
) -> KnnClassifier:
    """
    Fits a Knn-GZip compressor on the train
    data and returns it.

    Arguments:
        train_text (np.ndarray): Training dataset as a numpy array.
        train_labels (np.ndarray): Training labels as a numpy array.

    Returns:
        KnnClassifier: Trained Knn-Compressor model ready to make predictions.
    """

    compressor: BaseCompressor = GZipCompressor()
    model: KnnClassifier = KnnClassifier(
        compressor=compressor,
        training_inputs=train_text,
        training_labels=train_labels,
        distance_metric=distance_metric,
    )

    return model


def main() -> None:
    print(f"Fetching data...")
    train_text, train_labels = get_data(train=True)
    test_text, test_labels = get_data(train=False)
    # train_text, train_labels = get_esm_data(train=True)
    # test_text, test_labels = get_esm_data(train=False)
    print(f"Fitting model...")
    model = fit_model(train_text, train_labels, distance_metric="ncd")
    print("# of training and test samples:", len(train_labels), len(test_labels))
    # random_indices = np.random.choice(test_text.shape[0], 1000, replace=False)
    # sample_test_text = test_text[random_indices]
    # sample_test_labels = test_labels[random_indices]

    print(f"Generating predictions...")
    top_k = 1

    # Here we use the `sampling_percentage` to save time
    # at the expense of worse predictions. This
    # `sampling_percentage` selects a random % of training
    # data to compare `sample_test_text` against rather
    # than comparing it against the entire training dataset.
    distances, labels, similar_samples = model.predict(
        test_text, top_k, sampling_percentage=0.25
    )

    print(classification_report(test_labels, labels.reshape(-1)))

    print("accuracy:", accuracy_score(test_labels, labels.reshape(-1)))
    print("auprc:", average_precision_score(test_labels, labels.reshape(-1)))
    print("auroc:", roc_auc_score(test_labels, labels.reshape(-1)))


if __name__ == "__main__":
    main()
