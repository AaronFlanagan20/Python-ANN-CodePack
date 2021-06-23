from numpy import argmax
from numpy import array
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from keras.utils import to_categorical


# credit: https://machinelearningmastery.com/how-to-one-hot-encode-sequence-data-in-python/

# MANUAL
# input values i.e possible classes
alphabet = 'abcdefghijklmnopqrstuvwxyz '

# create the integer encoding for each class
# integer will refer to the binary vector index
char_to_int = dict((c, i) for i, c in enumerate(alphabet))
int_to_char = dict((i, c) for i, c in enumerate(alphabet))

# encode a random input, access dict using the class label as a "key"
integer_encoded = [char_to_int[char] for char in "random string"]

# list of binary vectors
onehot_encoded = list()

# do one-hot encoding
for value in integer_encoded:
    # create binary vector with all zeros
    letter = [0 for _ in range(len(alphabet))]

    # set index of letter to 1
    letter[value] = 1

    # store vector
    onehot_encoded.append(letter)

# pull the 1 from the vector and print out the corresponding class label
inverted = int_to_char[argmax(onehot_encoded[0])]

# SCIKIT

# define example
data = ['cold', 'cold', 'warm', 'cold', 'hot', 'hot', 'warm', 'cold', 'warm', 'hot']
values = array(data)

# integer encode
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(values)

# binary encode
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

# KERAS

data = [1, 3, 2, 0, 3, 2, 2, 1, 0, 1]
data = array(data)

# one hot encode
encoded = to_categorical(data)
print(encoded)

# invert encoding
inverted = argmax(encoded[0])
print(inverted)
