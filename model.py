class FullyConv:
    def __init__(self, num_actions, use_lstm):
        self.num_actions = num_actions
        self.use_lstm = use_lstm

    def build(self, screen_input, minimap_input, nonspatial_input):
        print('Build FullyConv')