import os
import time
import numpy as np
import DecisionTree
from BagginTrees import BaggingTree
import multiprocessing
from multiprocessing import Pool, Lock, cpu_count


class RandomForest(BaggingTree):
    """
    A simple RandomForest implementation that uses the DecisionTree from my DecisionTree module as the base classifier
    """

    def __init__(self, n_trees: int, n_samples: int, depth: int, n_features: int):
        """
        Parameters
        ----------
        n_trees:    Number of trees to grow
        n_samples:  Number of data points / instances / samples to consider for each tree
        depth:      The maximum depth of the trees
        n_features: Number of features to consider in each split. (randomly chosen with 'np.random.choice')
        """
        super(RandomForest, self).__init__(n_trees, n_samples, depth)
        self._n_features_to_consider = n_features

    @staticmethod
    def grow_a_tree(x, y, d, n_features):
        """
        Overrides the BaggingTrees.grow_a_tree() method. Used in a pool of processes, to use as many resources (cpu cores) as possible.
        Parameters
        ----------
        x:  numpy array of x_train
        y:  numpy array of y_train
        d:  Max depth for the trees
        n_features: Number of features to consider in each split. (randomly chosen with 'np.random.choice')

        Returns
        -------
        A tree object.
        """
        mytree = DecisionTree.DecisionTree(max_depth=d)
        mytree.fit(x, y, n_features_to_consider=n_features, print_while_growing=False)  # n_features_to_consider=None ignores the argument
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
        if self._n_features_to_consider > x_train.shape[1]:
            raise ValueError("The requested number of features to consider in each split is larger than the dataset's input space")
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
            processes.append(pool.apply_async(RandomForest.grow_a_tree, args=(x_bs, y_bs, self._depth, self._n_features_to_consider)))
        pool.close()
        pool.join()
        finish = time.perf_counter()
        print(f'Finished growing {self._n_trees} trees with depth {self._depth} in {round(finish - start, 2)} sec(s)')

        # Get the decision trees out of the async processes
        self._trees = [p.get() for p in processes]


if __name__ == '__main__':
    rf = RandomForest(n_trees=10, n_samples=100, depth=3, n_features=4)
    x_train, y_train = DecisionTree.create_dataset()
    rf.fit(x_train=x_train, y_train=y_train, n_workers=11)

    test_point = np.array([-42., -55., -59., -43., -71., -77., 35.], dtype=np.float32)
    class_prediction, predicted_probabilities = rf.predict(test_point)
    print(f'The predicted class is : {class_prediction}')
    [print(f'Predicted probability for class {x} is {predicted_probabilities[x]}%') for x in predicted_probabilities.keys()]
