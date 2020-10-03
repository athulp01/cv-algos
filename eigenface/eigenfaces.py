import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os


class EigenFaces():
    # Path to the training data folder
    def __init__(self, height, width, path):
        self.height = height
        self.width = width
        self.img_name_map, self.matrix = self.__prepare_data(path)

    # k is the number of eigen faces
    def train(self, k):
        self.k = k
        self.avg_face, self.matrix = self.__normalize()
        self.eigen_values, self.eigen_vectors = self.__eigen_vector()
        self.eigen_faces, self.coef = self.__eigen_faces(k)

        # Return the predicted image name
    def predict(self, path, threshold):
        img = Image.open(path)
        flattened = np.array(img).flatten()
        normalized = np.subtract(flattened, self.avg_face)
        test_coeff, _, _, _ = np.linalg.lstsq(self.eigen_faces, normalized)
        # Make the error vector
        err = list()
        for face_coeff in np.transpose(self.coef):
            err.append(np.linalg.norm(np.subtract(test_coeff, face_coeff)))
        # return the image name corresponding to minumum error
        print(self.img_name_map[np.argmin(err)])
        print(np.min(err))
        if np.min(err) > threshold:
            return "No matching face"
        return self.img_name_map[np.argmin(err)]

    def accuracy(self, path):
        correct = 0
        total = 0
        for file in os.listdir(path):
            full_path = os.path.join(path, file)
            if os.path.isfile(full_path):
                try:
                    img = Image.open(os.path.join(path, file))
                except IOError:
                    continue
                else:
                    total += 1
                    print(file.split('.')[0])
                    if self.predict(full_path, 1) == file.split('.')[0]:
                        correct += 1
        return correct/total

    # Prepares training data for processing
    def __prepare_data(self, path):
        tmp_list = list()
        # stores mapping from idx->filename
        img_map = dict()
        size = 0
        for file in os.listdir(path):
            full_path = os.path.join(path, file)
            if os.path.isfile(full_path):
                try:
                    img = Image.open(os.path.join(path, file))
                except IOError:
                    continue
                else:
                    img_map[size] = file.split('.')[0]
                    size += 1
                    flattened = np.array(img).flatten()
                    tmp_list.append(flattened)
        return img_map, np.transpose(np.array(tmp_list))

    # Normalize with the average face
    def __normalize(self):
        average_face = np.sum(self.matrix, 1)/np.size(self.matrix, 0)
        plt.show()
        # subtract each face from average face
        for face in range(np.size(self.matrix, 1)):
            self.matrix[:, face] = np.subtract(
                self.matrix[:, face], average_face)
        return (average_face, self.matrix)

    # return eigen values and vectors from the training data
    def __eigen_vector(self):
        cov = np.matmul(np.transpose(self.matrix), self.matrix)
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        # take absolute of imaginary
        abs_eigenvalues = np.array([np.abs(x) for x in eigenvalues])
        abs_eigenvectors = np.array([np.abs(x) for x in eigenvectors])
        return (abs_eigenvalues, np.matmul(self.matrix, abs_eigenvectors))

    # Return the eigen faces from the training data
    def __eigen_faces(self, k):
        # Take first k largest eigen values
        k_max_e_value = np.argpartition(self.eigen_values, -k)[-k:]

        eigen_faces = np.ndarray((self.height*self.width, k_max_e_value.size))
        # Store the corresponding eigen vector
        for idx, i in enumerate(k_max_e_value):
            eigen_faces[:, idx] = self.eigen_vectors[:, i]

        coef = np.ndarray((k, np.size(self.matrix, 1)))
        # Calculate the coefficients
        for i in range(np.size(self.matrix, 1)):
            coef[:, i], _, _, _ = np.linalg.lstsq(
                eigen_faces, self.matrix[:, i])
        return eigen_faces, coef


if __name__ == "__main__":
    yale = EigenFaces(320, 243, "./data")
    yale.train(10)
    print(yale.accuracy("./data/testing"))
