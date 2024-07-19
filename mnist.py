import numpy as np
import math
import pickle as pkl

ARCHITECTURE = [784, 300, 100, 10]
EPOCHS = 2
N = 3
LAMBDA = .1  # Learning Rate
TRAIN_NEW = True

# Sigmoid function for input num
def sigmoid(num):
    return 1 / (1 + math.exp(-num))

# First derivative of the sigmoid function
def deriv_sigmoid(num):
    return sigmoid(num) * (1 - sigmoid(num))

# Generates weights and biases with randoms between -1 and 1 according to the network structure provided
def generate_randoms(structure: list):
    w = [None]
    b = [None]
    for i in range(1, len(structure)):
        w.append(np.random.uniform(-1,1,(structure[i-1], structure[i])))
        b.append(np.random.uniform(-1,1,(1, structure[i])))
    return w, b

# Identifies most likely classification from output matrix
def rounded(a):
    output = a.tolist()[0]
    m = max(output)
    max_index = output.index(m)
    for i in range(len(output)):
        if i==max_index:
            output[i]=1
        else:
            output[i]=0
    return np.array([output])

# Preprocesses train and test sets
def ingest_train_test(filepath: str, train_or_test_set: list):
    with open(filepath) as f:
        for line in f:
            l = line.strip()
            arr = list(map(int, l.split(",")))
            x = np.array([arr[1:]])
            x = np.divide(x, 255)
            label = int(arr[0])
            y = np.zeros((1,10))
            y[0,label] = 1
            train_or_test_set.append((x,y))
    return train_or_test_set

def train(training_set, dest_filename):
    weights, biases = generate_randoms(ARCHITECTURE)
    for i in range(EPOCHS):
        print(f"Epoch {i + 1}:")
        for x, y in training_set:
            # 
            dot_list = [None]
            a_list = [x]  # a^0 = x
            delta_list = [np.zeros(1)] * (N + 1)
            # Forward Propagation:
            for L in range(1, N + 1):
                dot_list.append(a_list[L - 1] @ weights[L] + biases[L])  # dot^L = (a^L-1 dotted with w^L) + b^L
                a_list.append(ACTIVATION(dot_list[L]))  # a^L = sigmoid(dot^L)
            # Back Propagation:
            delta_N = ACTIVATION_DERIV(dot_list[N]) * (y - a_list[N])  # delta^N = A'(dot^N) * (y-a^N)
            delta_list[N] = delta_N
            for L in range(N - 1, 0, -1):
                delta_L = ACTIVATION_DERIV(dot_list[L]) * (delta_list[L + 1] @ weights[L+1].T)  # delta^L = A'(dot^L) * (delta^L+1 @ weights^L+1 transposed)
                delta_list[L] = delta_L
            for L in range(1, N + 1):
                biases[L] = biases[L] + LAMBDA * delta_list[L]
                weights[L] = weights[L] + LAMBDA * a_list[L - 1].T @ delta_list[L]
        # filename = "weights_biases"
        pkl.dump([weights, biases], open(dest_filename, 'wb'))
    return weights, biases

ACTIVATION = np.vectorize(sigmoid)
ACTIVATION_DERIV = np.vectorize(deriv_sigmoid)

# Load in train/test sets
training_set = ingest_train_test("Unit 7/MNIST/mnist_train.csv", [])
test_set = ingest_train_test("Unit 7/MNIST/mnist_test.csv", [])

# Train new weights/biases or load existing from pkl file
pkl_filename = "weights_biases"
if TRAIN_NEW:
    weights, biases = train(training_set, pkl_filename)
else:
    l = pkl.load(open("weights_biases", 'rb'))
    weights = l[0]
    biases = l[1]

# Calculate accuracy on the test set
misclassified = 0
for x,y in test_set:
    dot_list = [None]
    a_list = [x]  # a^0 = x
    # Forward Propagation:
    for L in range(1, N + 1):
        dot_list.append(a_list[L - 1] @ weights[L] + biases[L])  # dot^L = (a^L-1 dotted with w^L) + b^L
        a_list.append(ACTIVATION(dot_list[L]))  # a^L = sigmoid(dot^L)
    # Error Calculation:
    output = a_list[N]
    output = rounded(output)
    if not np.array_equal(output, y):
        misclassified += 1
print(f"{misclassified} points misclassified from the test set out of 10,000")
error = 100*(misclassified/10000)
print(f"{100 - error}% accuracy on the test set")