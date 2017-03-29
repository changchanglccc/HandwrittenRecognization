import math
import struct


""" 

    Pre-processing 
    
    The following code will process the MNIST training and label 
    files and store the data portined as training, validation and
    testing data each containing 60%, 20% and 20% of the MNIST data

"""
class MNIST_Processing():
    # Initializing significant variables
    MNIST_TRAIN_FILE = 'data/mnist-train'
    MNIST_TRAIN_LABELS_FILE = 'data/mnist-train-labels'

    TRAINING_PERCENTAGE = 60
    VALIDATION_PERCENTAGE = 20
    TESTING_PERCENTAGE = 20

    training_list = []
    validation_list = []
    testing_list = []

    training_labels = []
    validation_labels = []
    testing_labels = []

    #
    def __init__(self):
        self.read_data()
        self.read_labels()

    # This retrieves and stores a list of vectors for the training, validation and testing list
    def read_data(self):
        image_file = open(self.MNIST_TRAIN_FILE, 'r+b')

        # Locates the beginning of the file and retrieves the "magic number"
        image_file.seek(0)

        magic_number = image_file.read(4)
        magic_number = struct.unpack('>i', magic_number)[0]

        # Records the number of images in the file
        num_images = image_file.read(4)
        num_images = struct.unpack('>i', num_images)[0]

        # Calculates requires size for the training, validation and testing data
        num_training = int(round((num_images * (self.TRAINING_PERCENTAGE / 100.0)), 0))
        num_validation = int(round((num_images * (self.VALIDATION_PERCENTAGE / 100.0)), 0))
        num_testing = int(round((num_images * (self.TESTING_PERCENTAGE / 100.0)), 0))

        print('Vectors in training: ' + str(num_training),
              '\nVectors in validation: ' + str(num_validation),
              '\nVectors in testing: ' + str(num_testing),
              '\nTotal vectors: ' + str((num_training + num_validation + num_testing)),
              '\n')

        # Records number of rows and columns
        rows = image_file.read(4)
        rows = struct.unpack('>i', rows)[0]

        columns = image_file.read(4)
        columns = struct.unpack('>i', columns)[0]

        # Initializes the training list
        print('Reading & parsing training data...')
        temp_training = image_file.read(rows * columns * num_training)
        self.training_list = self.normalize_features(temp_training, rows, columns)

        print('Reading & parsing validation data...')
        temp_validation = image_file.read(rows * columns * num_validation)
        self.validation_list = self.normalize_features(temp_validation, rows, columns)

        print('Reading & parsing testing data...\n')
        temp_testing = image_file.read(rows * columns * num_testing)
        self.testing_list = self.normalize_features(temp_testing, rows, columns)

        image_file.close()
    ### END OF read_data() ###

    #
    def read_labels(self):
        label_file = open(self.MNIST_TRAIN_LABELS_FILE, 'r+b')

        # Locates the beginning of the file and retrieves the "magic number"
        label_file.seek(0)

        magic_number = label_file.read(4)
        magic_number = struct.unpack('>i', magic_number)[0]

        # Records the number of labels in the file
        num_labels = label_file.read(4)
        num_labels = struct.unpack('>i', num_labels)[0]

        # Initializing the labels lists for the training, validation and testing data
        print('Reading & storing the labels...\n')
        self.training_labels = list(label_file.read(len(self.training_list)))
        self.validation_labels = list(label_file.read(len(self.validation_list)))
        self.testing_labels = list(label_file.read(len(self.testing_list)))

        label_file.close()
    ### END OF read_labels() ###

    #
    def normalize_features(self, feature, rows, columns):
        feature = list(map(lambda x: 0 if x <100 else 1, feature))
        return [feature[i: i + (rows * columns)] for i in range(0, len(feature), (rows * columns))]
    ### END OF normalize_featues() ###
""" END OF MNIST_Processing() CLASS """


""" """
class MNIST_NaiveBayes():
    # Initializing significant variables
    class_prob = [0 for x in range(10)]
    pixel_prob = []

    training_list = []
    validation_list = []
    testing_list = []

    training_labels = []
    validation_labels = []
    testing_labels = []

    nb_training_labels = []
    nb_validation_labels = []
    nb_testing_labels = []

    #
    def __init__(self):
        # Initializes and prepares necessary components to run Bayes
        self.__initNB()
        self.train_bayes()

        # The following runs the Bayes algoritm and records the predictions
        self.nb_training_labels = self.run_bayes(self.training_list)
        self.nb_validation_labels = self.run_bayes(self.validation_list)
        self.nb_testing_labels = self.run_bayes(self.testing_list)

        # Displays accuracy results     # Uncomment to display results
        self.test_accuracy('Training', self.nb_training_labels, self.training_labels)
        self.test_accuracy('Validation', self.nb_validation_labels, self.validation_labels)
        self.test_accuracy('Testing', self.nb_testing_labels, self.testing_labels)

        # The following will display all vectors where the label matches the value passed
        #self.print_number(0)


    #
    def __initNB(self):
        self.MNIST_OBJECT = MNIST_Processing()

        self.training_list = self.MNIST_OBJECT.training_list
        self.validation_list = self.MNIST_OBJECT.validation_list
        self.testing_list = self.MNIST_OBJECT.testing_list

        self.training_labels = self.MNIST_OBJECT.training_labels
        self.validation_labels = self.MNIST_OBJECT.validation_labels
        self.testing_labels = self.MNIST_OBJECT.testing_labels
    ### END OF __initNB() ###

    #
    def train_bayes(self):
        print('Training Naive Bayes algorithm with training list...\n')
        self.pixel_prob = [[0 for x in range(len(self.training_list))] for y in range(10)]

        # Essential variables
        class_occ = [0 for x in range(10)]
        pixel_occ_per_class = [[0 for x in range(len(self.training_list))] for y in range(10)]
        pixel_per_class = [0 for x in range(10)]
        num_occ = 0

        # The following records the number of pixels present and their independent occurances
        for x in range(len(self.training_list)):
            class_occ[self.training_labels[x]] += 1

            for y in range(len(self.training_list[x])):
                if self.training_list[x][y] == 1:
                    pixel_occ_per_class[self.training_labels[x]][y] += 1
                    pixel_per_class[self.training_labels[x]] += 1

        # The following creates a list of probability for each pixel in each class
        for x in range(len(pixel_occ_per_class)):
            for y in range(len(pixel_occ_per_class[x])):
                self.pixel_prob[x][y] = float((pixel_occ_per_class[x][y] + 1) / (pixel_per_class[x] + len(self.training_list[0])))


        for value in class_occ:
            num_occ += value

        for x in range(len(class_occ)):
            self.class_prob[x] = float(class_occ[x] / num_occ)
    ### END OF training_bayes() ###

    #
    def run_bayes(self, list_to_test):
        output_labels = []

        for vector in list_to_test:
            score = [math.log10(x) for x in self.class_prob]

            for x in range(len(vector)):
                if vector[x] == 1:
                    for y in range(len(score)):
                        score[y] += math.log10(self.pixel_prob[y][x])

            output_labels.append(score.index(max(score)))

        return output_labels
    ### END OF run_bayes() ###

    #
    def test_accuracy(self, list_name, nm_labels, labels):
        correct = 0
        incorrect = 0
        accuracy = 0.0

        for x in nm_labels:
            if nm_labels[x] == labels[x]:
                correct += 1
            else:
                incorrect += 1

        accuracy = int((correct / (correct + incorrect)) * 100)
        print(list_name + ' results: ')
        print('Correct: ' + str(correct),
              '\nIncorrect: ' + str(incorrect),
              '\nAccuracy: ' + str(accuracy) + '\n')
    ### END OF test_accuracy() ###

    #
    def print_number(self, num_to_display):
        for x in range(len(self.training_labels)):
            if self.training_labels[x] == num_to_display:

                print('Index: ' + str(x))
                output = ''

                for y in range(len(self.training_list[x])):
                    if self.training_list[x][y] == 1:
                        output += str(num_to_display)
                    else:
                        output += ' '

                    if y % 28 == 0:
                        print(output)
                        output = ''

                print('\n')
    ### END OF print_number() ###
""" END OF MNIST_NaiveBayes() CLASS """

test = MNIST_NaiveBayes()
