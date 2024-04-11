# automaton class

class DFA:
    """
    Deterministic Finite-state Automaton

    ...

    Attributes
    ----------
    initial_automaton_state : int
        initial automaton state (default 0)
    accepting_set : list
        the set of accepting states

    Methods
    -------
    reset()
        resets the automaton state
    step(label)
        changes and returns the state of the automaton (self.automaton_state) upon reading a label (any un-accepting sink state is "-1")
    accepting(automaton_state)
        checks if the next automaton state is in the accepting set
    """

    def __init__(self, initial_automaton_state=0, accepting_set=None):
        self.initial_automaton_state = initial_automaton_state
        self.automaton_state = self.initial_automaton_state
        self.accepting_set = accepting_set

    def reset(self):
        self.automaton_state = self.initial_automaton_state

    def step(self, label):
        pass

    def accepting(self, next_automaton_state):
        # return a positive flag if the next state is in the accepting set
        if next_automaton_state in self.accepting_set:
            # self.accepting_set = []
            return 1
        elif next_automaton_state == -1:
            return -1
        # return zero otherwise
        else:
            return 0
