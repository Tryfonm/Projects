import os
import time
import numpy as np
import DecisionTree
import multiprocessing
from multiprocessing import Pool, Lock, cpu_count


class BaggingTree():
    def __init__(self, n_trees: int, n_samples: int, depth: int):
        """
        Parameters
        ----------
        n_trees:    Number of trees to grow
        n_samples:  Number of data points / instances / samples to consider for each tree
        depth:      The maximum depth of the trees
        """
        if n_samples == 0:
            raise ValueError(f'Invalid requested number of samples: n_samples={n_samples}.')
        self._n_trees = n_trees
        self._n_samples = n_samples
        self._depth = depth

    @property
    def trees_list(self):
        """
        Returns
        -------
        A list of the tree objects that have been grown.
        """
        return self._trees

    @staticmethod
    def grow_a_tree(x, y, d):
        """
        Used in a pool of processes, to use as many resources (cpu cores) as possible.
        Parameters
        ----------
        x:  numpy array of x_train
        y:  numpy array of y_train
        d:  Max depth for the trees
        Returns
        -------
        A tree object.
        """
        mytree = DecisionTree.DecisionTree(max_depth=d)
        mytree.fit(x, y, n_features_to_consider=None, print_while_growing=True)  # n_features_to_consider=None ignores the argument
        return mytree

    def fit(self, x_train, y_train, n_workers=None):
        """
        Parameters
        ----------
        x_train:    Expects a numpy array of x_train
        y_train:    Expects a numpy array of y_train
        n_workers:  Number of cores to use

        Returns
        -------
        None.
        """
        if self._n_samples > x_train.shape[0]:
            raise ValueError("The requested number of samples is larger than the available data points")

        if n_workers > cpu_count():
            raise ValueError('Requested number of workers is higher than the cpu count.')

        self._x = x_train
        self._y = y_train
        available_cpus = cpu_count()
        if n_workers != None and n_workers != available_cpus:
            print(f'Using {n_workers} out of {available_cpus} cpu cores...')
        else:
            print(f'Using all {available_cpus} available cpu cores...')

        start = time.perf_counter()
        pool = Pool(processes=n_workers)
        processes = []
        for process in range(self._n_trees):
            # The second tuple returns the data instances that were not sample and could be used as a validation set for the current tree
            # Theoretically speaking, the unsampeld part of the dataset is expeted to be 1/3 of the initial dataset.
            (x_bs, y_bs), (_, _) = BaggingTree.bootstrap_dataset(self._x, self._y, self._n_samples)
            processes.append(pool.apply_async(BaggingTree.grow_a_tree, args=(x_bs, y_bs, self._depth)))
        pool.close()
        pool.join()
        finish = time.perf_counter()
        print(f'Finished growing {self._n_trees} trees with depth {self._depth} in {round(finish - start, 2)} sec(s)')

        # Get the decision trees out of the async processes
        self._trees = [p.get() for p in processes]
        # print(trees_list)

    @staticmethod
    def bootstrap_dataset(x, y, n_samples):
        """

        Parameters
        ----------
        x:  Expects a numpy array of x_train
        y:  Expects a numpy array of y_train

        Returns
        -------

        """
        x_shape = x.shape[0]
        indices_sampled = np.random.choice(range(0, x_shape), n_samples, replace=False)
        indices_isin_sampled = np.in1d(range(0, x_shape), indices_sampled)

        x_bs = x[indices_sampled, :]
        y_bs = y[indices_sampled]
        x_oob = x[~indices_isin_sampled, :]
        y_oob = y[~indices_isin_sampled]

        return (x_bs, y_bs), (x_oob, y_oob)

    def predict(self, input_vector):
        """
        Predict method for Bagging Trees
        Parameters
        ----------
        input_vector: Expects a numpy array of size (num_features,) ~ zero-dimensional

        Returns
        -------
        The  predicted class of the input data point

        """
        vote_for_prediction = []

        for tree in self._trees:
            vote_for_prediction.append(tree.predict(input_vector))
        classes, counts = np.unique(np.array(vote_for_prediction), return_counts=True)
        class_counts = list(zip(classes, counts))

        # Although it's not really calibrated, a probability can be computed also:
        probability = {}
        for i, t in enumerate(class_counts):
            probability[t[0]] = t[1] * 100 / self._n_trees
        index = np.argmax(np.array([val[1] for val in class_counts]))
        prediction = class_counts[index][0]

        return prediction, probability


if __name__ == '__main__':
    bt = BaggingTree(n_trees=1, n_samples=5, depth=3)
    x_train, y_train = DecisionTree.create_dataset()
    bt.fit(x_train=x_train, y_train=y_train, n_workers=11)

    test_point = np.array([-42., -55., -59., -43., -71., -77., 35.], dtype=np.float32)
    class_prediction, predicted_probabilities = bt.predict(test_point)
    print(f'The predicted class is : {class_prediction}')
    [print(f'Predicted probability for class {x} is {predicted_probabilities[x]}%') for x in predicted_probabilities.keys()]
