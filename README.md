# LSTM-model-for-text-generation
## Text Generation with LSTM

This project demonstrates how to generate text using a Long Short-Term Memory (LSTM) neural network. The LSTM model is trained on a given text corpus, and then it is used to generate new text based on user input.

### Dependencies

The code uses the following dependencies:

- `Streamlit`: A Python library for creating web applications.
- `numpy`: A library for numerical computations in Python.
- `keras`: A high-level neural networks API.
- `Tensorflow`: A deep learning framework.

To install the dependencies, run the following command:

``` shell
pip install streamlit numpy keras tensorflow
```

### Usage

1. Clone the repository:

``` shell
git clone https://github.com/your-username/your-repository.git
```

2. Navigate to the project directory:

``` shell
cd your-repository
```

3. Run the Streamlit app:

``` shell
streamlit run text_generation_lstm.py
```

4. Open the provided URL in your web browser to access the Streamlit app.

5. Enter the desired text corpus in the text area.

6. Click the "Train Model" button to train the LSTM model based on the provided text corpus.

7. The model will be trained, and once completed, the generated text will be displayed.

### How It Works

1. The text corpus is provided by the user in the Streamlit app.

2. The LSTM model is defined with the specified number of units.

3. The model is compiled with the "adam" optimizer and "categorical_crossentropy" loss function.

4. The training data is prepared by creating input sequences and corresponding output sequences.

5. The input sequences are one-hot encoded, and the output sequences are converted to categorical values.

6. The model is trained on the input and output sequences for the specified number of epochs.

7. The trained model is used to generate text based on user input. The `generate_text` function generates text by predicting the next character using the model's predictions.

8. The generated text is displayed in the Streamlit app.

### Customization

- You can adjust the `sequence_length` variable to change the length of the input sequences.
- Modify the `num_units` variable to change the number of units in the LSTM layers.
- Adjust the number of characters to generate by changing the range in the `generate_text` function.
- You can experiment with different temperature values in the `sample` function to control the randomness of the generated text.
