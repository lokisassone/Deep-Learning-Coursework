"""
K Nearest Neighbours Model
"""
import numpy as np


class KNN(object):
    def __init__(
        self,
        num_class: int
    ):
        self.num_class = num_class

    def train(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        k: int
    ):
        """
        Train KNN Classifier

        KNN only need to remember training set during training

        Parameters:
            x_train: Training samples ; np.ndarray with shape (N, D)
            y_train: Training labels  ; snp.ndarray with shape (N,)
        """
        self._x_train = x_train
        self._y_train = y_train
        self.k = k

    def predict(
        self,
        x_test: np.ndarray,
        k: int = None,
        loop_count: int = 1
    ):
        """
        Use the contained training set to predict labels for test samples

        Parameters:
            x_test    : Test samples                                     ; np.ndarray with shape (N, D)
            k         : k to overwrite the one specificed during training; int
            loop_count: parameter to choose different knn implementation ; int

        Returns:
            predicted labels for the data in X_test; a 1-dimensional array of
                length N, where each element is an integer giving the predicted
                class.
        """
        k_test = k if k is not None else self.k

        if loop_count == 1:
            distance = self.calc_dis_one_loop(x_test)
        elif loop_count == 2:
            distance = self.calc_dis_two_loop(x_test)
        prediction=np.zeros((500,), dtype=int)
        kpredictions=np.zeros([10,500], dtype=int)
        for x in range(k_test):
            for y in range(500):
                minim=distance[y][0]
                index=0
                label=0
                for z in range(5000):
                    if (distance[y][z]<minim):
                        minim=distance[y][z]
                        index=z
                        label=self._y_train[z]
                kpredictions[label][y]=kpredictions[label][y]+1
                distance[y][index]=10000000
        for s in range(500):
            vote=kpredictions[0][s]
            label=0
            for t in range(10):
                if (kpredictions[t][s]>vote):
                    vote=kpredictions[t][s]
                    label=t
            prediction[s]=label
        return prediction

    def calc_dis_one_loop(
        self,
        x_test: np.ndarray
    ):
        """
        Calculate distance between training samples and test samples

        This function could one for loop

        Parameters:
            x_test: Test samples; np.ndarray with shape (N, D)
        """
        dist=np.zeros([500,5000])
        
        for x in range(500):
            dist[x]=np.sum((x_test[x]-self._x_train)**2,axis=1)
        return dist

    def calc_dis_two_loop(
        self,
        x_test: np.ndarray
    ):
        """
        Calculate distance between training samples and test samples

        This function could contain two loop

        Parameters:
            x_test: Test samples; np.ndarray with shape (N, D)
        """
        d=np.zeros([500,5000])
        for x in range(500):
            for y in range(5000):
                d[x][y]=np.sum((np.subtract(x_test[x][:],self._x_train[y][:]))**2)
        return d