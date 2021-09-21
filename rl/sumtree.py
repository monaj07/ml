"""
This code implements a sumTree data structure,
to be used in Prioritized Experience Replay (PER) sampling algorithm.
It has O(log(n)) complexity for samplings, insertions, and updates.
Great explanation in:
https://adventuresinmachinelearning.com/sumtree-introduction-python/
This method is an alternative for straightforward inverse probabilistic sampling,
where we compute the CDF of the array probabilities,
and then take a uniform sample and map it to the CDF to retrieve our sample.
This easy method, despite having O(1) complexity for insertion and update,
has a complexity of O(n) for sampling, which is not desirable,
given that sampling is very frequent in the learning process.
That is why we take a refuge in the sumTree approach.
Basically the main idea is that we mimic the CDF in form of a Binary Tree,
where the sum of each two nodes is assigned to their parent node.
"""
import numpy as np
import random


class SumTree:
    def __init__(self, capacity):
        # Capacity is the maximum size of the experience replay memory
        self.capacity = capacity
        # The filled size of the memory so far
        self.size = 0
        # Experience memory data
        self.replay_memory = np.empty(capacity, dtype=object)
        # The sumTree tree array
        self.priority_tree = np.zeros(2 * capacity - 1)

    def __len__(self):
        return self.size

    def add(self, transition, priority):
        """
        If the memory still has vacancy, append the new experience to it;
        otherwise compare its priority with the minimum priority,
        and if it is greater than that minimum value, replace it.
        """
        if self.size >= self.capacity:
            # If the replay memory is out of capacity, start filling it from the begining
            self.size = 0
        self.replay_memory[self.size] = transition
        # Filling the last row (leaf nodes) of the binary tree starting from (capacity - 1) index
        tree_index = self.size + self.capacity - 1
        self.priority_tree[tree_index] = priority
        # Add the new priority to all upstream nodes up to the root
        self._update_upstream_priority_tree(tree_index, priority)
        self.size += 1

    def _update_upstream_priority_tree(self, idx, change):
        self.priority_tree[idx] += change
        while idx > 0:
            parent_index = (idx - 1) // 2
            self.priority_tree[parent_index] += change
            idx = parent_index

    def _search_downstream_priority_tree(self):
        z = random.uniform(0, self.priority_tree[0])
        # Start the search from the root node
        idx = 0
        while 1:
            # Move to the left or right child depending on the random value
            # If we have to go to the right direction,
            # then subtract the value of the left child
            if z <= self.priority_tree[2 * idx + 1]:
                child_idx = 2 * idx + 1
            else:
                z -= self.priority_tree[2 * idx + 1]
                child_idx = 2 * idx + 2

            # If we are at a leaf node, return the index of the item and its priority
            # Otherwise go one level deeper.
            if child_idx >= len(self.priority_tree):
                # child_idx is beyond the size of the tree,
                # Hence 'idx' is a leaf node, so stop and return the current value
                # We return the normalised indices of the last row that refers to the indices of the memory array
                return (idx - (self.capacity - 1)), self.priority_tree[idx]
            else:
                idx = child_idx

    def sample(self, n):
        samples_indices = []
        mini_batch_priorities = []
        for _ in range(n):
            # Retrieve a sample given the current tree structure
            sample_idx, sample_priority = self._search_downstream_priority_tree()
            samples_indices.append(sample_idx)
            mini_batch_priorities.append(sample_priority)
        mini_batch_data = self.replay_memory[samples_indices]
        return mini_batch_data, mini_batch_priorities


if __name__ == "__main__":
    st = SumTree(4)
    print()
