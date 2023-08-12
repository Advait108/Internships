import tensorflow as tf
import numpy as np
# Define the input and target data
input_texts = []
target_texts = []
#https://www.kaggle.com/datasets/grafstor/simple-dialogs-for-chatbot
with open('data.txt', 'r', encoding='utf-8') as f:
    for line in f:
        # Read in each line of the file and split it into input and target
        # texts
        input_text, target_text = line.rstrip().split('\t')
        #print(input_text)
        #print(target_text)
        input_texts.append(input_text)
        target_texts.append(target_text)




#check if data is splitting properly
#check if inputs of my own messages work with this code, then we know it is a data problem


# Tokenize
tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
tokenizer.fit_on_texts(input_texts + target_texts)
input_sequences = tokenizer.texts_to_sequences(input_texts)
target_sequences = tokenizer.texts_to_sequences(target_texts)


# Pad the input and target sequences
max_seq_len = max(len(seq) for seq in input_sequences + target_sequences)
input_sequences = tf.keras.preprocessing.sequence.pad_sequences(input_sequences, maxlen=max_seq_len, padding='post')
target_sequences = tf.keras.preprocessing.sequence.pad_sequences(target_sequences, maxlen=max_seq_len, padding='post')




#check if data is padded properly
#play around with optimizer


# Create a sequential model: embedding layer, LSTM layer, and dense layer
model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(len(tokenizer.word_index) + 1, 256),
    #dense vector representations of words

    tf.keras.layers.LSTM(2048, return_sequences=True), # LSTM layer to learn
    #sequence patterns in the data

    tf.keras.layers.Dense(len(tokenizer.word_index) + 1, activation='softmax')
    # output probabilities for each word
])

# Compile the model
model.compile(optimizer='Adam', loss='sparse_categorical_crossentropy')

# Train the model
model.fit(input_sequences, target_sequences, epochs=40)

# Chat with the bot
while True:
    input_text = input('You: ')

    # input to sequence of integers w/ tokenizer
    input_sequence = tokenizer.texts_to_sequences([input_text])
    input_sequence = tf.keras.preprocessing.sequence.pad_sequences(input_sequence, maxlen=max_seq_len, padding='post')
    #print(input_sequence)
    #generate sequence with model
    predicted_sequence = model.predict(input_sequence)[0]
    #print(predicted_sequence) #to check if a predicted sequence was actually made
    # Convert the predicted sequence to text using the tokenizer
    predicted_text = ' '.join([tokenizer.index_word.get(int(i)) for i in
                               tf.argmax(predicted_sequence, axis=-1) if
                               tokenizer.index_word.get(int(i)) is not None])

    if not predicted_text:
        predicted_text = "I'm sorry, I don't understand. Can you please " \
                         "rephrase your question?"
    print('Bot:', predicted_text)


