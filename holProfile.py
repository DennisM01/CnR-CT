from copy import deepcopy


class HolProfile:
    def __init__(
        self, name, preference, upper_bound, lower_bound, flex_capacity, flex_SoC
    ):
        self.name = name
        self.preference = deepcopy(preference)
        self.allocated = deepcopy(preference)
        if len(upper_bound) == 1:
            self.upper_bound = [i + upper_bound[0] for i in deepcopy(self.preference)]
        else:
            self.upper_bound = upper_bound
        if len(lower_bound) == 1:
            self.lower_bound = [i - lower_bound[0] for i in deepcopy(self.preference)]
        else:
            self.lower_bound = lower_bound
        self.deficit = 0
        self.priority = 0
        self.flex_capacity = flex_capacity
        self.flex_SoC = flex_SoC

    def update(self, power, time):
        """
        Update the data in the holon

        Args:
            power (float): new allocated power at specific time
            time (int): current time
        """
        difference = power - self.allocated[time]
        self.deficit += difference
        self.allocated[time] = power
        for i in range(time + 1, len(self.flex_SoC)):
            self.flex_SoC[i] += difference
        self.priority = sum(
            (pow((self.allocated[t] - self.preference[t]), 2) for t in range(24))
        )
        pass
