#TODO don't forget to set unavailable policy actions to 0 and rescale

class FullyConv:
    def __init__(self, num_actions, use_lstm):
        self.num_actions = num_actions
        self.use_lstm = use_lstm

    #
    def build(self, screen_input, minimap_input, nonspatial_input):
        print('Build FullyConv')

    #
    def sample_action(self, policy):
        print('Sample action')
