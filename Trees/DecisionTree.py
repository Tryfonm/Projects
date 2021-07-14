import time
import pandas as pd
import numpy as np
from scipy.fft._pocketfft.setup import pre_build_hook


class DecisionTree():
    """
    My simple Decision Tree (classification) implementation.
    """

    def __init__(self, max_depth):
        self._max_depth = max_depth

    class Node():
        """
        A Node class to hold all the important information for the Tree's splits (=nodes) and track parent and
        children (if they exist) nodes, for easier traversal.
        """

        def __init__(self, subset, parent_node=None):
            """
            Parameters
            ----------
            subset:         Current node's subset of the dataset
            parent_node:    Current node's parent node
            """
            self._subset = subset
            self.parent_node = parent_node
            if parent_node == None:
                self.parent_nodes_list = [None]
            else:
                self.parent_nodes_list = self.parent_node.parent_nodes_list + [self.parent_node]
            self.depth = len(self.parent_nodes_list) - 1
            self._is_a_leaf_node = False

            self._total_instances = subset.shape[0]
            self._num_features = subset.shape[1] - 1
            self._gini = DecisionTree.gini(self._subset)
            # The following is used for RandomForests

        @property
        def subset(self):
            return self._subset

        @property
        def gini(self):
            return np.round(self._gini, 3)

        @property
        def samples(self):
            return self._total_instances

        @property
        def is_a_leaf_node(self):
            return self._is_a_leaf_node

        @is_a_leaf_node.setter
        def is_a_leaf_node(self, status: bool):
            """
            Parameters
            ----------
            status: -True if the current node is a leaf node
                    -False otherwise
            """
            self._left_child = None
            self._right_child = None
            self.optimal_split_value = None
            self.optimal_feature = None
            self._is_a_leaf_node = status

        @property
        def split(self):
            if not self.is_a_leaf_node:
                return self.optimal_split_value, self.optimal_feature

        @property
        def left_child(self):
            if not self.is_a_leaf_node:
                return self._left_child

        @left_child.setter
        def left_child(self, left_node):
            self._left_child = left_node

        @property
        def right_child(self):
            if not self.is_a_leaf_node:
                return self._right_child

        @right_child.setter
        def right_child(self, right_child):
            self._right_child = right_child

        @property
        def class_distribution(self):
            """
            Returns
            -------
            A list of tuples containing (class, value_count of the class)
            """
            vals, counts = np.unique(self.subset[:, -1].astype(np.int16), return_counts=True)
            return list(zip(vals.astype(np.int16), counts))

        def find_optimal_split(self, n_features_to_consider):
            """
            Find the optimal split for the current node and  creates two subsets of the dataset (left and right child)
            based on the optimal split that can be accessed by 'get_subset_children'. If for any reason, the current node
            is found to be a leaf node, then None is returned by the method and the current node's is_a_leaf_node status is
            set to True.

            Parameters
            -------
            n_features_to_consider: Number of features to consider for the split. Only used for RandomForests (otherwise ignored).
            Returns
            -------
            A tuple containing:
            optimal_split_value
            optimal_feature
            np.round(optimal_weighted_gini_split, 3)
            """
            optimal_weighted_gini_split = 1

            # In case there is no impurity in the node, avoid doing the rest of cause this is a leaf node
            if self.gini == 0:
                self.is_a_leaf_node = True
                return None

            # If the following is not None, then a RandomForest is grown
            if n_features_to_consider is not None:
                # Randomly pick indices of features that will be considered for the current node / split.
                sampled_features = list(np.random.choice(range(0, self._num_features), n_features_to_consider, replace=False))

            for m in range(self._num_features):
                if n_features_to_consider is not None:
                    if m not in sampled_features:
                        continue

                subset_sorted = self._subset[self._subset[:, m].argsort()]
                # A starting value on the left of the split, so that it includes all instances
                first_split = subset_sorted[0, m] + subset_sorted[0, m] - (subset_sorted[0, m] + subset_sorted[1, m]) / 2

                splits = (subset_sorted[1:, m] + subset_sorted[:-1, m]) / 2
                splits = np.concatenate((first_split.reshape(1), splits), axis=0)

                for index, split_value in enumerate(splits):

                    left_subset = subset_sorted[:index, :]
                    right_subset = subset_sorted[index:, :]
                    left_gini = DecisionTree.gini(left_subset)
                    right_gini = DecisionTree.gini(right_subset)

                    # IMPORTANT SHIT (that took me too long to figure out) !
                    try:
                        # This one makes sure that the split value that is currently investigated, does not appear in both the left and right split
                        if subset_sorted[index - 1, m] == subset_sorted[index, m]:
                            continue
                    except:
                        # For the first iteration, though, condition does not hold and the 'index-1' expression throws an error.
                        pass

                    # print(f'Iteration: {index}')
                    # print(f'Current split vale: {split_value}')
                    # print(f'Gini left: {left_gini}')
                    # print(f'Gini right: {right_gini}')

                    gini_index_split = ((left_subset.shape[0] * left_gini) + (right_subset.shape[0] * right_gini)) / self._total_instances
                    # Floating point precision affects the result

                    if gini_index_split < optimal_weighted_gini_split:
                        optimal_split_value = split_value
                        optimal_value_index = index
                        optimal_weighted_gini_split = gini_index_split
                        optimal_feature = m

            # Now that the optimal split is found we need to resort the subset with respect to the optimal feature
            subset_sorted = self._subset[self._subset[:, optimal_feature].argsort()]
            self._left_child = subset_sorted[:optimal_value_index, :]
            self._right_child = subset_sorted[optimal_value_index:, :]
            self.optimal_split_value = optimal_split_value
            self.optimal_feature = optimal_feature
            # If weighted gini index does not get any better after the optimal split, then we have a leaf node and thus, should not split
            if gini_index_split >= self._gini:
                self._is_a_leaf_node = True
                return None

            return optimal_split_value, optimal_feature, np.round(optimal_weighted_gini_split, 3)

        def get_subset_children(self):
            """
            Returns
            -------
            A tuple of (LEFT / RIGHT) subset children (not nodes).
            """
            return self._left_child, self._right_child

    def fit(self, x, y, n_features_to_consider=None, print_while_growing=False):
        """
        Fit the data
        Parameters
        ----------
        x:  numpy_array of shape (number_of_training_instances, number_of_features) of the training data
        y:  numpy array of shape (number_of_training_instances, ) of the targets
        n_features_to_consider: Number of features to consider for the split. Only used for RandomForests (otherwise ignored).
        print_while_growing: If set, the growing procedure is printed out while it takes place
        """
        self._x = x
        self._y = y
        self.dataset = np.concatenate((x, y.reshape(-1, 1)), axis=1)
        self._root_node = DecisionTree.Node(self.dataset)
        DecisionTree.dfs_grow(self._root_node, max_depth=self._max_depth, n_features_to_consider=n_features_to_consider, pwg=print_while_growing)

    @staticmethod
    def dfs_grow(root_node, max_depth, n_features_to_consider, pwg=True):
        """
        Grow a tree in a depth-first-search manner. If for any reason any node is found to be a leaf node, either by reaching the maximum requested depth
        or by not splitting further (due to no improvement in gini_index of the children nodes or not having enough points to split) the traversal stops and
        the procedure backtracks to a previous node where growing further is possible.
        Parameters
        ----------
        root_node:  The root node of the tree to grow
        max_depth:  The requested depth of the tree
        n_features_to_consider: Number of features to consider for the split. Only used for RandomForests (otherwise ignored).
        pwg:    Print while growing. Argument passed in by the 'fit' method.

        Returns
        -------
        None. Called inside the 'fit' method in order to create the tree nodes.
        """
        queue = [root_node]
        while queue:

            current_node = queue.pop(-1)

            if current_node.is_a_leaf_node or current_node.depth >= max_depth:
                current_node.is_a_leaf_node = True
                if pwg: DecisionTree.print_node_info(current_node)
                continue
            current_node.find_optimal_split(n_features_to_consider=n_features_to_consider)

            if not current_node.is_a_leaf_node:

                left_subset, right_subset = current_node.get_subset_children()
                if pwg: DecisionTree.print_node_info(current_node)
                left_node = DecisionTree.Node(left_subset, current_node)
                right_node = DecisionTree.Node(right_subset, current_node)

                # Also update the path
                current_node.left_child = left_node
                current_node.right_child = right_node

                queue = queue + [left_node, right_node]

    def predict(self, input_vector):
        """
        Parameters
        ----------
        data_point: Expects a numpy array of size (num_features,)

        Returns
        -------
        The  predicted class of the input data point
        """
        if input_vector.ndim != 1 or input_vector.shape[0] != self._x.shape[1]:
            temp = np.zeros((self._x.shape[1], 0))
            raise ValueError(f"Wrong input shape. Expected numpy array of size: {temp.shape}")

        current_node = self._root_node
        while not current_node.is_a_leaf_node:
            if input_vector[current_node.optimal_feature] < current_node.optimal_split_value:
                current_node = current_node.left_child
            else:
                current_node = current_node.right_child
        # This one will find the tuple for which the 2nd value is max
        class_tuple = max(current_node.class_distribution, key=lambda x: x[1])
        return class_tuple[0]

    @staticmethod
    def print_node_info(node):
        if not node.is_a_leaf_node:
            val, feature = node.split
            print(
                f'-----------\nNode depth: {node.depth}\nOptimal split feature {feature} < {val}\nGini: {node.gini}\nSamples: {node.samples}\nClass distribution: {node.class_distribution}\n')
        else:
            print(
                f'-----------\nNode depth: {node.depth}, Leaf node\nGini: {node.gini}\nSamples: {node.samples}\nClass distribution: {node.class_distribution}\n')

    @staticmethod
    def gini(subset):
        """
        Function used to find the given subset's gini index (Impurity)
        Parameters
        ----------
        subset: A numpy array

        Returns
        -------
        The subset's Gini index
        """
        target_feature = subset[:, -1]
        total_instances = target_feature.shape[0]
        vals, counts = np.unique(target_feature, return_counts=True)
        return 1 - (np.square(counts / total_instances)).sum()


def create_dataset():
    dataset = pd.read_csv('./random_dataset.txt', sep='\t', header=None)
    dataset.columns = ['wifi ' + str(x) for x in range(1, 8)] + ['room']
    dataset_as_np = np.array(dataset, dtype=np.float32)
    return dataset_as_np[:, :-1], dataset_as_np[:, -1]


def prepare_mnist():
    import idx2numpy

    images_path = './MNIST/raw/train-images-idx3-ubyte'
    img = idx2numpy.convert_from_file(images_path)

    labels_path = './MNIST/raw/train-labels-idx1-ubyte'
    lbl = idx2numpy.convert_from_file(labels_path)

    return img, lbl


if __name__ == '__main__':
    start = time.perf_counter()
    x_train, y_train = create_dataset()
    mytree = DecisionTree(max_depth=10)
    mytree.fit(x_train, y_train, print_while_growing=False)
    finish = time.perf_counter()
    print(f'Finished in: {round(finish - start, 2)} sec(s)')

    test_point = np.array([-42., -55., -59., -43., -71., -77., 35.], dtype=np.float32)
    print(f'The datapoint belongs to class: {mytree.predict(test_point)}.')
