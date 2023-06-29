# import numpy as np
# from keras.models import Sequential
# from keras.layers import LSTM, Dense
# from keras.callbacks import LambdaCallback
#
# # Load and preprocess the text data
# text = open("corpus.txt", "r").read()  # Replace "corpus.txt" with the path to your text file
# text = text.lower()  # Convert text to lowercase
#
# # Create a set of unique characters in the text
# chars = sorted(list(set(text)))
# num_chars = len(chars)
#
# # Create dictionaries to map characters to indices and vice versa
# char_to_idx = {c: i for i, c in enumerate(chars)}
# idx_to_char = {i: c for i, c in enumerate(chars)}
#
# # Define parameters for the LSTM model
# sequence_length = 40  # Number of characters to consider as input for each training example
# step = 3  # Step size to create overlapping sequences
# num_units = 128  # Number of LSTM units
# batch_size = 128  # Number of training examples in each batch
# epochs = 50  # Number of training iterations
#
# # Prepare the training data
# input_sequences = []
# output_sequences = []
#
# for i in range(0, len(text) - sequence_length, step):
#     input_seq = text[i:i + sequence_length]
#     output_seq = text[i + sequence_length]
#     input_sequences.append([char_to_idx[char] for char in input_seq])
#     output_sequences.append(char_to_idx[output_seq])
#
# num_sequences = len(input_sequences)
#
# # Prepare the input and output arrays
# X = np.zeros((num_sequences, sequence_length, num_chars), dtype=np.bool)
# y = np.zeros((num_sequences, num_chars), dtype=np.bool)
#
# for i, seq in enumerate(input_sequences):
#     for t, char in enumerate(seq):
#         X[i, t, char] = 1
#     y[i, output_sequences[i]] = 1
#
# # Build the LSTM model
# model = Sequential()
# model.add(LSTM(num_units, input_shape=(sequence_length, num_chars)))
# model.add(Dense(num_chars, activation="softmax"))
# model.compile(loss="categorical_crossentropy", optimizer="adam")
#
# # Function to sample the next character based on the model's predictions
# def sample(preds):
#     preds = np.asarray(preds)
#     preds = np.log(preds) / 0.5  # Adjust the temperature parameter for more or less randomness
#     exp_preds = np.exp(preds)
#     preds = exp_preds / np.sum(exp_preds)
#     probas = np.random.multinomial(1, preds, 1)
#     return np.argmax(probas)
#
# # Function to generate text at the end of each training iteration
# def generate_text(epoch, _):
#     if epoch % 10 == 0:
#         start_index = np.random.randint(0, len(text) - sequence_length - 1)
#         generated = ""
#         seed_text = text[start_index:start_index + sequence_length]
#         generated += seed_text
#         print("----- Generating text after Epoch: %d" % epoch)
#         for _ in range(400):  # Adjust the number of characters to generate
#             x_pred = np.zeros((1, sequence_length, num_chars))
#             for t, char in enumerate(seed_text):
#                 x_pred[0, t, char_to_idx[char]] = 1.
#             preds = model.predict(x_pred, verbose=0)[0]
#             next_index = sample(preds)
#             next_char = idx_to_char[next_index]
#             generated += next_char
#             seed_text = seed_text[1:] + next_char
#         print(generated)
#
# # Train the LSTM model
# model.fit(X, y, batch_size=batch_size, epochs=epochs, callbacks=[LambdaCallback(on_epoch_end=generate_text)])
#





# import numpy as np
# from keras.models import Sequential
# from keras.layers import LSTM, Dense
# from keras.callbacks import LambdaCallback
#
# # Load and preprocess the text data
# text = open("corpus.txt", "r").read()  # Replace "corpus.txt" with the path to your text file
# text = text.lower()  # Convert text to lowercase
#
# # Create a set of unique characters in the text
# chars = sorted(list(set(text)))
# num_chars = len(chars)
#
# # Create dictionaries to map characters to indices and vice versa
# char_to_idx = {c: i for i, c in enumerate(chars)}
# idx_to_char = {i: c for i, c in enumerate(chars)}
#
# # Define parameters for the LSTM model
# sequence_length = 50  # Number of characters to consider as input for each training example
# step = 3  # Step size to create overlapping sequences
# num_units = 256  # Number of LSTM units
# batch_size = 128  # Number of training examples in each batch
# epochs = 100  # Number of training iterations
#
# # Prepare the training data
# input_sequences = []
# output_sequences = []
#
# for i in range(0, len(text) - sequence_length, step):
#     input_seq = text[i:i + sequence_length]
#     output_seq = text[i + sequence_length]
#     input_sequences.append([char_to_idx[char] for char in input_seq])
#     output_sequences.append(char_to_idx[output_seq])
#
# num_sequences = len(input_sequences)
#
# # Prepare the input and output arrays
# X = np.zeros((num_sequences, sequence_length, num_chars), dtype=np.bool)
# y = np.zeros((num_sequences, num_chars), dtype=np.bool)
#
# for i, seq in enumerate(input_sequences):
#     for t, char in enumerate(seq):
#         X[i, t, char] = 1
#     y[i, output_sequences[i]] = 1
#
# # Build the LSTM model
# model = Sequential()
# model.add(LSTM(num_units, input_shape=(sequence_length, num_chars), return_sequences=True))
# model.add(LSTM(num_units))
# model.add(Dense(num_chars, activation="softmax"))
# model.compile(loss="categorical_crossentropy", optimizer="adam")
#
# # Function to sample the next character based on the model's predictions
# def sample(preds, temperature=0.5):
#     preds = np.asarray(preds).astype('float64')
#     preds = np.log(preds) / temperature
#     exp_preds = np.exp(preds)
#     preds = exp_preds / np.sum(exp_preds)
#     probas = np.random.multinomial(1, preds, 1)
#     return np.argmax(probas)
#
# # Function to generate text at the end of each training iteration
# def generate_text(epoch, _):
#     if epoch % 10 == 0:
#         start_index = np.random.randint(0, len(text) - sequence_length - 1)
#         generated = ""
#         seed_text = text[start_index:start_index + sequence_length]
#         generated += seed_text
#         print("----- Generating text after Epoch: %d" % epoch)
#         for _ in range(400):  # Adjust the number of characters to generate
#             x_pred = np.zeros((1, sequence_length, num_chars))
#             for t, char in enumerate(seed_text):
#                 x_pred[0, t, char_to_idx[char]] = 1.
#             preds = model.predict(x_pred, verbose=0)[0]
#             next_index = sample(preds, temperature=0.5)
#             next_char = idx_to_char[next_index]
#             generated += next_char
#             seed_text = seed_text[1:] + next_char
#         print(generated)
#
# # Train the LSTM model
# model.fit(X, y, batch_size=batch_size, epochs=epochs, callbacks=[LambdaCallback(on_epoch_end=generate_text)])


# import numpy as np
# from keras.models import Sequential
# from keras.layers import LSTM, Dense
# from keras.callbacks import LambdaCallback
#
# # Load and preprocess the text data
# text = open("corpus.txt", "r").read()  # Replace "corpus.txt" with the path to your text file
# text = text.lower()  # Convert text to lowercase
#
# # Create a set of unique characters in the text
# chars = sorted(list(set(text)))
# num_chars = len(chars)
#
# # Create dictionaries to map characters to indices and vice versa
# char_to_idx = {c: i for i, c in enumerate(chars)}
# idx_to_char = {i: c for i, c in enumerate(chars)}
#
# # Define parameters for the LSTM model
# sequence_length = 50  # Number of characters to consider as input for each training example
# step = 3  # Step size to create overlapping sequences
# num_units = 256  # Number of LSTM units
# batch_size = 128  # Number of training examples in each batch
# epochs = 100  # Number of training iterations
#
# # Prepare the training data
# input_sequences = []
# output_sequences = []
#
# for i in range(0, len(text) - sequence_length, step):
#     input_seq = text[i:i + sequence_length]
#     output_seq = text[i + sequence_length]
#     input_sequences.append([char_to_idx[char] for char in input_seq])
#     output_sequences.append(char_to_idx[output_seq])
#
# num_sequences = len(input_sequences)
#
# # Prepare the input and output arrays
# X = np.zeros((num_sequences, sequence_length, num_chars), dtype=np.bool)
# y = np.zeros((num_sequences, num_chars), dtype=np.bool)
#
# for i, seq in enumerate(input_sequences):
#     for t, char in enumerate(seq):
#         X[i, t, char] = 1
#     y[i, output_sequences[i]] = 1
#
# # Build the LSTM model
# model = Sequential()
# model.add(LSTM(num_units, input_shape=(sequence_length, num_chars), return_sequences=True))
# model.add(LSTM(num_units))
# model.add(Dense(num_chars, activation="softmax"))
# model.compile(loss="categorical_crossentropy", optimizer="adam")
#
# # Function to sample the next character based on the model's predictions
# def sample(preds, temperature=0.5):
#     preds = np.asarray(preds).astype('float64')
#     preds = np.log(preds) / temperature
#     exp_preds = np.exp(preds)
#     preds = exp_preds / np.sum(exp_preds)
#     probas = np.random.multinomial(1, preds, 1)
#     return np.argmax(probas)
#
# # Function to generate text at the end of each training iteration
# def generate_text(epoch, _):
#     if epoch % 10 == 0:
#         print("----- Generating text after Epoch: %d" % epoch)
#         while True:
#             user_input = input("Enter the starting text (at least %d characters): " % sequence_length)
#             if len(user_input) >= sequence_length:
#                 break
#             else:
#                 print("Starting text should be at least %d characters." % sequence_length)
#
#         generated = user_input.lower()
#         for _ in range(400):  # Adjust the number of characters to generate
#             x_pred = np.zeros((1, sequence_length, num_chars))
#             for t, char in enumerate(user_input):
#                 x_pred[0, t, char_to_idx[char]] = 1.
#             preds = model.predict(x_pred, verbose=0)[0]
#             next_index = sample(preds, temperature=0.5)
#             next_char = idx_to_char[next_index]
#             generated += next_char
#             user_input = user_input[1:] + next_char
#         print(generated)
#
# # Train the LSTM model
# model.fit(X, y, batch_size=batch_size, epochs=epochs, callbacks=[LambdaCallback(on_epoch_end=generate_text)])


# import numpy as np
# from keras.models import Sequential
# from keras.layers import LSTM, Dense
# from keras.callbacks import LambdaCallback
#
# # Get the text input from the user
# text = input("Enter the text corpus: ")
# text = text.lower()  # Convert text to lowercase
#
# # Create a set of unique characters in the text
# chars = sorted(list(set(text)))
# num_chars = len(chars)
#
# # Create dictionaries to map characters to indices and vice versa
# char_to_idx = {c: i for i, c in enumerate(chars)}
# idx_to_char = {i: c for i, c in enumerate(chars)}
#
# # Define parameters for the LSTM model
# sequence_length = 50  # Number of characters to consider as input for each training example
# step = 3  # Step size to create overlapping sequences
# num_units = 256  # Number of LSTM units
# batch_size = 128  # Number of training examples in each batch
# epochs = 100  # Number of training iterations
#
# # Prepare the training data
# input_sequences = []
# output_sequences = []
#
# for i in range(0, len(text) - sequence_length, step):
#     input_seq = text[i:i + sequence_length]
#     output_seq = text[i + sequence_length]
#     if all(char in char_to_idx for char in input_seq + output_seq):
#         input_sequences.append([char_to_idx[char] for char in input_seq])
#         output_sequences.append(char_to_idx[output_seq])
#
# num_sequences = len(input_sequences)
#
# # Prepare the input and output arrays
# X = np.zeros((num_sequences, sequence_length, num_chars), dtype=np.bool)
# y = np.zeros((num_sequences, num_chars), dtype=np.bool)
#
# for i, seq in enumerate(input_sequences):
#     for t, char in enumerate(seq):
#         X[i, t, char] = 1
#     y[i, output_sequences[i]] = 1
#
# # Build the LSTM model
# model = Sequential()
# model.add(LSTM(num_units, input_shape=(sequence_length, num_chars), return_sequences=True))
# model.add(LSTM(num_units))
# model.add(Dense(num_chars, activation="softmax"))
# model.compile(loss="categorical_crossentropy", optimizer="adam")
#
# # Function to sample the next character based on the model's predictions
# def sample(preds, temperature=0.5):
#     preds = np.asarray(preds).astype('float64')
#     preds = np.log(preds) / temperature
#     exp_preds = np.exp(preds)
#     preds = exp_preds / np.sum(exp_preds)
#     probas = np.random.multinomial(1, preds, 1)
#     return np.argmax(probas)
#
# # Function to generate text at the end of each training iteration
# def generate_text(epoch, _):
#     if epoch % 10 == 0:
#         print("----- Generating text after Epoch: %d" % epoch)
#         while True:
#             user_input = input("Enter the starting text (at least %d characters): " % sequence_length)
#             if len(user_input) >= sequence_length and all(char in char_to_idx for char in user_input):
#                 break
#             else:
#                 print("Starting text should be at least %d characters and contain only valid characters." % sequence_length)
#
#         generated = user_input.lower()
#         for _ in range(400):  # Adjust the number of characters to generate
#             x_pred = np.zeros((1, sequence_length, num_chars))
#             for t, char in enumerate(user_input):
#                 x_pred[0, t, char_to_idx[char]] = 1.
#             preds = model.predict(x_pred, verbose=0)[0]
#             next_index = sample(preds, temperature=0.5)
#             next_char = idx_to_char[next_index]
#             generated += next_char
#             user_input = user_input[1:] + next_char
#         print(generated)
#
# # Train the LSTM model
# model.fit(X, y, batch_size=batch_size, epochs=epochs, callbacks=[LambdaCallback(on_epoch_end=generate_text)])


# import streamlit as st
# import numpy as np
# from keras.models import Sequential
# from keras.layers import LSTM, Dense
# from keras.callbacks import LambdaCallback
#
#
# # Define the necessary variables
# sequence_length = 50
# num_units = 256
# num_chars = 0
# char_to_idx = {}
# idx_to_char = {}
#
#
# # Function to sample the next character based on the model's predictions
# def sample(preds, temperature=0.5):
#     preds = np.asarray(preds).astype("float64")
#     preds = np.log(preds) / temperature
#     exp_preds = np.exp(preds)
#     preds = exp_preds / np.sum(exp_preds)
#     probas = np.random.multinomial(1, preds, 1)
#     return np.argmax(probas)
#
#
# # Function to generate text based on user input
# def generate_text(user_input):
#     generated = user_input.lower()
#     for _ in range(400):  # Adjust the number of characters to generate
#         x_pred = np.zeros((1, sequence_length, num_chars))
#         for t, char in enumerate(user_input):
#             x_pred[0, t, char_to_idx[char]] = 1.
#         preds = model.predict(x_pred, verbose=0)[0]
#         next_index = sample(preds, temperature=0.5)
#         next_char = idx_to_char[next_index]
#         generated += next_char
#         user_input = user_input[1:] + next_char
#     return generated
#
#
# # Create the Streamlit app and define the user interface
# def main():
#     st.title("Text Generation with LSTM")
#
#     # Get the text corpus from the user
#     text_corpus = st.text_area("Enter the text corpus:", height=200)
#
#     if st.button("Train Model"):
#         global num_chars, char_to_idx, idx_to_char
#
#         text = text_corpus.lower()
#         chars = sorted(list(set(text)))
#         num_chars = len(chars)
#         char_to_idx = {c: i for i, c in enumerate(chars)}
#         idx_to_char = {i: c for i, c in enumerate(chars)}
#
#         # Prepare the training data
#         input_sequences = []
#         output_sequences = []
#
#         for i in range(0, len(text) - sequence_length):
#             input_seq = text[i:i + sequence_length]
#             output_seq = text[i + sequence_length]
#             if all(char in char_to_idx for char in input_seq + output_seq):
#                 input_sequences.append([char_to_idx[char] for char in input_seq])
#                 output_sequences.append(char_to_idx[output_seq])
#
#         num_sequences = len(input_sequences)
#
#         # Prepare the input and output arrays
#         X = np.zeros((num_sequences, sequence_length, num_chars), dtype=np.bool)
#         y = np.zeros((num_sequences, num_chars), dtype=np.bool)
#
#         for i, seq in enumerate(input_sequences):
#             for t, char in enumerate(seq):
#                 X[i, t, char] = 1
#             y[i, output_sequences[i]] = 1
#
#         # Define the model architecture
#         model = Sequential()
#         model.add(LSTM(num_units, input_shape=(sequence_length, num_chars), return_sequences=True))
#         model.add(LSTM(num_units))
#         model.add(Dense(num_chars, activation="softmax"))
#         model.compile(loss="categorical_crossentropy", optimizer="adam")
#
#         # Train the LSTM model
#         model.fit(X, y, batch_size=128, epochs=100, verbose=1)
#
#         st.success("Model trained successfully!")
#
#     if st.button("Generate Text"):
#         user_input = st.text_input("Enter the starting text (at least {} characters):".format(sequence_length))
#         if len(user_input) < sequence_length or not all(char in char_to_idx for char in user_input):
#             st.warning("Starting text should be at least {} characters and contain only valid characters.".format(
#                 sequence_length))
#         else:
#             # Load the trained model and weights
#             model = Sequential()
#             model.add(LSTM(num_units, input_shape=(sequence_length, num_chars), return_sequences=True))
#             model.add(LSTM(num_units))
#             model.add(Dense(num_chars, activation="softmax"))
#             model.compile(loss="categorical_crossentropy", optimizer="adam")
#             model.load_weights(r"C:\Users\Pratima\Documents\GitHub\LSTM-model-for-text-generation\path\to\save\model.h5")  # Replace with your actual model weights file path
#
#             generated_text = generate_text(user_input)
#             st.write(generated_text)
#
#
# if __name__ == "__main__":
#     main()


# import streamlit as st
# import numpy as np
# from keras.models import Sequential
# from keras.layers import LSTM, Dense
# from keras.callbacks import LambdaCallback
#
#
# # Define the necessary variables
# sequence_length = 50
# num_units = 256
# num_chars = 0
# char_to_idx = {}
# idx_to_char = {}
#
#
# # Function to sample the next character based on the model's predictions
# def sample(preds, temperature=0.5):
#     preds = np.asarray(preds).astype("float64")
#     preds = np.log(preds) / temperature
#     exp_preds = np.exp(preds)
#     preds = exp_preds / np.sum(exp_preds)
#     probas = np.random.multinomial(1, preds, 1)
#     return np.argmax(probas)
#
#
# # Function to generate text based on user input
# def generate_text(user_input):
#     generated = user_input.lower()
#     for _ in range(400):  # Adjust the number of characters to generate
#         x_pred = np.zeros((1, sequence_length, num_chars))
#         for t, char in enumerate(user_input):
#             x_pred[0, t, char_to_idx[char]] = 1.
#         preds = model.predict(x_pred, verbose=0)[0]
#         next_index = sample(preds, temperature=0.5)
#         next_char = idx_to_char[next_index]
#         generated += next_char
#         user_input = user_input[1:] + next_char
#     return generated
#
#
# # Create the Streamlit app and define the user interface
# def main():
#     st.title("Text Generation with LSTM")
#
#     # Get the text corpus from the user
#     text_corpus = st.text_area("Enter the text corpus:", height=200)
#
#     if st.button("Train Model"):
#         global num_chars, char_to_idx, idx_to_char, model
#
#         text = text_corpus.lower()
#         chars = sorted(list(set(text)))
#         num_chars = len(chars)
#         char_to_idx = {c: i for i, c in enumerate(chars)}
#         idx_to_char = {i: c for i, c in enumerate(chars)}
#
#         # Prepare the training data
#         input_sequences = []
#         output_sequences = []
#
#         for i in range(0, len(text) - sequence_length):
#             input_seq = text[i:i + sequence_length]
#             output_seq = text[i + sequence_length]
#             if all(char in char_to_idx for char in input_seq + output_seq):
#                 input_sequences.append([char_to_idx[char] for char in input_seq])
#                 output_sequences.append(char_to_idx[output_seq])
#
#         num_sequences = len(input_sequences)
#
#         # Prepare the input and output arrays
#         X = np.zeros((num_sequences, sequence_length, num_chars), dtype=np.bool)
#         y = np.zeros((num_sequences, num_chars), dtype=np.bool)
#
#         for i, seq in enumerate(input_sequences):
#             for t, char in enumerate(seq):
#                 X[i, t, char] = 1
#             y[i, output_sequences[i]] = 1
#
#         # Define the model architecture
#         model = Sequential()
#         model.add(LSTM(num_units, input_shape=(sequence_length, num_chars), return_sequences=True))
#         model.add(LSTM(num_units))
#         model.add(Dense(num_chars, activation="softmax"))
#         model.compile(loss="categorical_crossentropy", optimizer="adam")
#
#         # Train the LSTM model
#         model.fit(X, y, batch_size=128, epochs=100, verbose=1)
#
#         st.success("Model trained successfully!")
#
#     if st.button("Generate Text"):
#         user_input = st.text_input("Enter the starting text (at least {} characters):".format(sequence_length))
#         if len(user_input) < sequence_length or not all(char in char_to_idx for char in user_input):
#             st.warning("Starting text should be at least {} characters and contain only valid characters.".format(
#                 sequence_length))
#         else:
#             generated_text = generate_text(user_input)
#             st.write(generated_text)
#
#
# if __name__ == "__main__":
#     main()


import streamlit as st
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import LambdaCallback

# Define the necessary variables
sequence_length = 50
num_units = 256
num_chars = 0
char_to_idx = {}
idx_to_char = {}
model = None

# Function to sample the next character based on the model's predictions
def sample(preds, temperature=0.5):
    preds = np.asarray(preds).astype("float64")
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

# Function to generate text based on user input
def generate_text(user_input):
    generated = user_input.lower()
    for _ in range(400):  # Adjust the number of characters to generate
        x_pred = np.zeros((1, sequence_length, num_chars))
        for t, char in enumerate(user_input):
            x_pred[0, t, char_to_idx[char]] = 1.
        preds = model.predict(x_pred, verbose=0)[0]
        next_index = sample(preds, temperature=0.5)
        next_char = idx_to_char[next_index]
        generated += next_char
        user_input = user_input[1:] + next_char
    return generated

# Function to generate random text
def generate_random_text():
    global model, char_to_idx, idx_to_char
    starting_text = ""
    generated_text = ""

    # Generate random starting text
    for _ in range(sequence_length):
        random_char = np.random.choice(list(char_to_idx.keys()))
        starting_text += random_char

    generated_text += starting_text

    # Generate text based on the starting text
    for _ in range(400):  # Adjust the number of characters to generate
        x_pred = np.zeros((1, sequence_length, num_chars))
        for t, char in enumerate(starting_text):
            x_pred[0, t, char_to_idx[char]] = 1.
        preds = model.predict(x_pred, verbose=0)[0]
        next_index = sample(preds, temperature=0.5)
        next_char = idx_to_char[next_index]
        generated_text += next_char
        starting_text = starting_text[1:] + next_char

    return generated_text

# Create the Streamlit app and define the user interface
def main():
    st.title("Text Generation with LSTM")

    # Get the text corpus from the user
    text_corpus = st.text_area("Enter the text corpus:", height=200)

    if st.button("Train Model"):
        global num_chars, char_to_idx, idx_to_char, model

        text = text_corpus.lower()
        chars = sorted(list(set(text)))
        num_chars = len(chars)
        char_to_idx = {c: i for i, c in enumerate(chars)}
        idx_to_char = {i: c for i, c in enumerate(chars)}

        # Prepare the training data
        input_sequences = []
        output_sequences = []

        for i in range(0, len(text) - sequence_length):
            input_seq = text[i:i + sequence_length]
            output_seq = text[i + sequence_length]
            if all(char in char_to_idx for char in input_seq + output_seq):
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

        # Define the model architecture
        model = Sequential()
        model.add(LSTM(num_units, input_shape=(sequence_length, num_chars), return_sequences=True))
        model.add(LSTM(num_units))
        model.add(Dense(num_chars, activation="softmax"))
        model.compile(loss="categorical_crossentropy", optimizer="adam")

        # Train the LSTM model
        model.fit(X, y, batch_size=128, epochs=100, verbose=1)

        st.success("Model trained successfully!")

        # Generate random text
        generated_text = generate_random_text()
        st.write(generated_text)

if __name__ == "__main__":
    main()


# import streamlit as st
# import numpy as np
# from keras.models import Sequential
# from keras.layers import LSTM, Dense
# from keras.callbacks import LambdaCallback
#
#
# # Define the necessary variables
# sequence_length = 50
# num_units = 256
# num_chars = 0
# char_to_idx = {}
# idx_to_char = {}
#
#
# # Function to sample the next character based on the model's predictions
# def sample(preds, temperature=0.2):
#     preds = np.asarray(preds).astype("float64")
#     preds = np.log(preds) / temperature
#     exp_preds = np.exp(preds)
#     preds = exp_preds / np.sum(exp_preds)
#     probas = np.random.multinomial(1, preds, 1)
#     return np.argmax(probas)
#
#
# # Function to generate text based on user input
# def generate_text(user_input):
#     generated = user_input.lower()
#     for _ in range(400):  # Adjust the number of characters to generate
#         x_pred = np.zeros((1, sequence_length, num_chars))
#         for t, char in enumerate(user_input):
#             x_pred[0, t, char_to_idx[char]] = 1.
#         preds = model.predict(x_pred, verbose=0)[0]
#         next_index = sample(preds, temperature=0.2)  # Lower temperature for more focused text
#         next_char = idx_to_char[next_index]
#         generated += next_char
#         user_input = user_input[1:] + next_char
#     return generated
#
#
# # Create the Streamlit app and define the user interface
# def main():
#     st.title("Text Generation with LSTM")
#
#     # Get the text corpus from the user
#     text_corpus = st.text_area("Enter the text corpus:", height=200)
#
#     if st.button("Train Model"):
#         global num_chars, char_to_idx, idx_to_char, model
#
#         text = text_corpus.lower()
#         chars = sorted(list(set(text)))
#         num_chars = len(chars)
#         char_to_idx = {c: i for i, c in enumerate(chars)}
#         idx_to_char = {i: c for i, c in enumerate(chars)}
#
#         # Prepare the training data
#         input_sequences = []
#         output_sequences = []
#
#         for i in range(0, len(text) - sequence_length):
#             input_seq = text[i:i + sequence_length]
#             output_seq = text[i + sequence_length]
#             if all(char in char_to_idx for char in input_seq + output_seq):
#                 input_sequences.append([char_to_idx[char] for char in input_seq])
#                 output_sequences.append(char_to_idx[output_seq])
#
#         num_sequences = len(input_sequences)
#
#         # Prepare the input and output arrays
#         X = np.zeros((num_sequences, sequence_length, num_chars), dtype=np.bool)
#         y = np.zeros((num_sequences, num_chars), dtype=np.bool)
#
#         for i, seq in enumerate(input_sequences):
#             for t, char in enumerate(seq):
#                 X[i, t, char] = 1
#             y[i, output_sequences[i]] = 1
#
#         # Define the model architecture
#         model = Sequential()
#         model.add(LSTM(num_units, input_shape=(sequence_length, num_chars), return_sequences=True, name="lstm_1"))
#         model.add(LSTM(num_units, name="lstm_2"))
#         model.add(Dense(num_chars, activation="softmax", name="dense"))
#         model.compile(loss="categorical_crossentropy", optimizer="adam")
#
#         # Train the LSTM model
#         model.fit(X, y, batch_size=128, epochs=100, verbose=1)
#
#         st.success("Model trained successfully!")
#
#     if st.button("Generate Text"):
#         user_input = st.text_input("Enter the starting text (at least {} characters):".format(sequence_length))
#         if len(user_input) < sequence_length or not all(char in char_to_idx for char in user_input):
#             st.warning("Starting text should be at least {} characters and contain only valid characters.".format(
#                 sequence_length))
#         else:
#             generated_text = generate_text(user_input)
#             st.write(generated_text)
#
#
# if __name__ == "__main__":
#     main()
#
