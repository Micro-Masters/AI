class RewardModifier:
    def __init__(self, config):
        # TODO: this will be replaced by something more specific
        self.config = config

    def modify(self, new_observation, reward, old_observation):
        #zerglings_killed = old_observation[2] - obs.observation['player'][8]
        print("reward modifier")

        if old_observation is None:
            return 0

        ## damage taken
        ## damage dealt
        ## enemy units killed
            # need to check if last action = camera moved??

        ##should we reward more units beomcing visible?
        ##reward more units visible? (for moving camera)
        #   but we don't want to penalize units dying?

        damage = 0
        if old_observation is not None and old_observation[1] is not None: #isinstance(old_observation, list): # is not None:
            if (len(old_observation[1]) == len(new_observation[1])):
                #calculate damage done
                for i in range(len(old_observation[1])):
                    damage += old_observation[1][i][0] - new_observation[1][i][0]  #health change of ith enemy unit
                print("dealt damage: ", damage)
            elif len(old_observation[1]) > len(new_observation[1]):
                print("enemy units died: ", len(old_observation[1]) - len(new_observation[1]))


        if old_observation is not None and (new_observation[2] < old_observation[2]):
            zerglings_lost = old_observation[2] - new_observation[2]
            print("zerglings lost: ", zerglings_lost)

        return reward
