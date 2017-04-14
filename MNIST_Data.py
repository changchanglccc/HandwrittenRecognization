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

    neural_data_set = []
    neural_label_set = []
    pre_neural_label_set = []

    #
    def __init__(self):
        self.read_data()
        self.read_labels()
        self.merge_sets()
        self.adapt_labels()

    #
    def merge_sets(self):
        self.pre_neural_label_set = self.training_labels + self.validation_labels + self.testing_labels
        self.neural_data_set = self.training_list + self.validation_list + self.testing_list


    #
    def adapt_labels(self):
        label_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        for value in self.pre_neural_label_set:
            label_list[value] = 1

            self.neural_label_set.append(list(label_list))
            label_list[value] = 0

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
    def normalize_features(self, feature, rows, columns,option = 'pixelsPerRow'):
        feature = list(map(lambda x: 0 if x < 100 else 1, feature))
        data = [feature[i: i + (rows * columns)] for i in range(0, len(feature), (rows * columns))]
        if option == 'pixelsPerRow':
            data = self.pixelsPerRowFeatExtraction(data)
        return data

    def pixelsPerRowFeatExtraction(self,data):
        totalData = []
        rowData = []
        row = 0
        for d in data:
            rowData = []
            for i in range(28):
                row = 0
                for j in range(28):
                    if (d[i * 28 + j] == 1):
                        row += 1
                rowData.append(row)
            totalData.append(rowData)
        return totalData

    ### END OF normalize_featues() ###
""" END OF MNIST_Processing() CLASS """

"""
test = MNIST_Processing()

print(test.neural_label_set[0])
print(len(test.neural_label_set))
print(test.neural_data_set[0])
"""