class minecraft_1:
    def __init__(self, initial_automaton_state=0, accepting_set=None):
        self.initial_automaton_state = initial_automaton_state
        self.automaton_state = self.initial_automaton_state
        self.accepting_set = accepting_set

    def reset(self):
        self.automaton_state = self.initial_automaton_state

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

    def step(self, label):
        # state 0
        if self.automaton_state == 0:
            if ('creeper' in label) or ('basalt_lava' in label):
                self.automaton_state = -1
            else:
                self.automaton_state = 0
        # step function returns the new automaton state
        return self.automaton_state
