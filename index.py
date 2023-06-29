import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import LambdaCallback

# Load and preprocess the text data
text = open("corpus.txt", "r").read()  # Replace "corpus.txt" with the path to your text file
text = text.lower()  # Convert text to lowercase

# Create a set of unique characters in the text
chars = sorted(list(set(text)))
num_chars = len(chars)

# Create dictionaries to map characters to indices and vice versa
char_to_idx = {c: i for i, c in enumerate(chars)}
idx_to_char = {i: c for i, c in enumerate(chars)}

# Define parameters for the LSTM model
sequence_length = 40  # Number of characters to consider as input for each training example
step = 3  # Step size to create overlapping sequences
num_units = 128  # Number of LSTM units
batch_size = 128  # Number of training examples in each batch
epochs = 50  # Number of training iterations

# Prepare the training data
input_sequences = []
output_sequences = []

for i in range(0, len(text) - sequence_length, step):
    input_seq = text[i:i + sequence_length]
    output_seq = text[i + sequence_length]
    input_sequences.append([char_to_idx[char] for char in input_seq])
    output_sequences.append(char_to_idx[output_seq])

num_sequences = len(input_sequences)

# Prepare the input and output arrays
X = np.zeros((num_sequences, sequence_length, num_chars), dtype=np.bool)
y = np.zeros((num_sequences, num_chars), dtype=np.bool)

for i, seq in enumerate(input_sequences):
    for t, char in enumerate(seq):
        X[i, t, char] = 1
    y[i, output_sequences[i]] = 1

model_path = "path/to/save/model.h5"

# Build the LSTM model
model = Sequential()
model.add(LSTM(num_units, input_shape=(sequence_length, num_chars)))
model.add(Dense(num_chars, activation="softmax"))
model.compile(loss="categorical_crossentropy", optimizer="adam")

# Function to sample the next character based on the model's predictions
def sample(preds):
    preds = np.asarray(preds)
    preds = np.log(preds) / 0.5  # Adjust the temperature parameter for more or less randomness
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

# Function to generate text at the end of each training iteration
def generate_text(epoch, _):
    if epoch % 10 == 0:
        start_index = np.random.randint(0, len(text) - sequence_length - 1)
        generated = ""
        seed_text = text[start_index:start_index + sequence_length]
        generated += seed_text
        print("----- Generating text after Epoch: %d" % epoch)
        for _ in range(400):  # Adjust the number of characters to generate
            x_pred = np.zeros((1, sequence_length, num_chars))
            for t, char in enumerate(seed_text):
                x_pred[0, t, char_to_idx[char]] = 1.
            preds = model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds)
            next_char = idx_to_char[next_index]
            generated += next_char
            seed_text = seed_text[1:] + next_char
        print(generated)

# Train the LSTM model
model.fit(X, y, batch_size=batch_size, epochs=epochs, callbacks=[LambdaCallback(on_epoch_end=generate_text)])
model.save(model_path)