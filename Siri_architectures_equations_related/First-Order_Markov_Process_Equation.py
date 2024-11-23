import numpy as np
import random

class MarkovChain:
    def __init__(self, states, transition_matrix, initial_state_distribution=None):
        """
        Initialize the Markov Chain.

        :param states: List of states.
        :param transition_matrix: Square matrix of state transition probabilities.
        :param initial_state_distribution: Probabilities for the initial state.
        """
        self.states = states
        self.state_index = {state: idx for idx, state in enumerate(states)}
        self.num_states = len(states)
        self.transition_matrix = np.array(transition_matrix)
        self._validate_transition_matrix()

        if initial_state_distribution is not None:
            self.initial_state_distribution = np.array(initial_state_distribution)
            self._validate_initial_state_distribution()
        else:
            # If no initial distribution is provided, assume uniform distribution
            self.initial_state_distribution = np.ones(self.num_states) / self.num_states

    def _validate_transition_matrix(self):
        """
        Validates the transition matrix.
        """
        if self.transition_matrix.shape != (self.num_states, self.num_states):
            raise ValueError("Transition matrix must be square with size equal to number of states.")

        for row in self.transition_matrix:
            if not np.isclose(np.sum(row), 1):
                raise ValueError("Each row of the transition matrix must sum to 1.")
            if np.any(row < 0):
                raise ValueError("Transition probabilities cannot be negative.")

    def _validate_initial_state_distribution(self):
        """
        Validates the initial state distribution.
        """
        if len(self.initial_state_distribution) != self.num_states:
            raise ValueError("Initial state distribution must have the same length as number of states.")
        if not np.isclose(np.sum(self.initial_state_distribution), 1):
            raise ValueError("Initial state distribution must sum to 1.")
        if np.any(self.initial_state_distribution < 0):
            raise ValueError("Initial state probabilities cannot be negative.")

    def next_state(self, current_state):
        """
        Returns the next state given the current state.

        :param current_state: The current state.
        :return: Next state.
        """
        idx = self.state_index[current_state]
        probabilities = self.transition_matrix[idx]
        next_state_idx = np.random.choice(self.num_states, p=probabilities)
        return self.states[next_state_idx]

    def generate_states(self, current_state, no=10):
        """
        Generates a sequence of states.

        :param current_state: The initial state.
        :param no: Number of future states to generate.
        :return: List of states.
        """
        future_states = [current_state]
        for _ in range(no):
            next_state = self.next_state(future_states[-1])
            future_states.append(next_state)
        return future_states

    def sequence_probability(self, state_sequence):
        """
        Calculates the probability of a sequence of states.

        :param state_sequence: List of states.
        :return: Probability of the sequence.
        """
        if not state_sequence:
            raise ValueError("State sequence must contain at least one state.")

        # Initial state probability
        first_state = state_sequence[0]
        if first_state not in self.states:
            raise ValueError(f"State '{first_state}' is not a valid state.")
        idx = self.state_index[first_state]
        prob = self.initial_state_distribution[idx]

        # Transition probabilities
        for (current_state, next_state) in zip(state_sequence[:-1], state_sequence[1:]):
            current_idx = self.state_index[current_state]
            next_idx = self.state_index[next_state]
            trans_prob = self.transition_matrix[current_idx][next_idx]
            prob *= trans_prob

        return prob

    def is_absorbing_state(self, state):
        """
        Checks if a state is an absorbing state.

        :param state: State to check.
        :return: True if absorbing, False otherwise.
        """
        idx = self.state_index[state]
        return np.isclose(self.transition_matrix[idx][idx], 1.0)

    def reachable_states(self, state):
        """
        Finds all states reachable from a given state.

        :param state: The state from which to find reachable states.
        :return: Set of reachable states.
        """
        visited = set()
        to_visit = [state]
        while to_visit:
            current = to_visit.pop()
            if current not in visited:
                visited.add(current)
                idx = self.state_index[current]
                next_states = [self.states[i] for i, prob in enumerate(self.transition_matrix[idx]) if prob > 0]
                to_visit.extend(next_states)
        return visited
