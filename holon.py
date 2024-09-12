from copy import deepcopy
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

from holProfile import HolProfile
import csv
from statistics import mean, stdev

plt.rcParams["axes.formatter.use_mathtext"] = True
plt.rcParams["xtick.labelsize"] = 22  # Adjust as needed for x-axis tick labels
plt.rcParams["ytick.labelsize"] = 22  # Adjust as needed for y-axis tick labels
plt.rcParams["axes.labelsize"] = 22  # Adjust as needed for axes labels
plt.rcParams["axes.titlesize"] = 22  # Adjust as needed for subplot titles


class Holon:
    def __init__(
        self,
        name,
        preference,
        upper_bound,
        lower_bound,
        flex_capacity,
        flex_SoC,
        epsilon=0.01,
    ):
        self.name = name
        self.preference = deepcopy(preference)
        self.allocated = deepcopy(preference)
        if len(upper_bound) == 1:
            self.upper_bound = [
                preference + upper_bound[0] for preference in deepcopy(self.preference)
            ]
        else:
            self.upper_bound = [
                preference[index] + upper_bound[index]
                for index in range(len(upper_bound))
            ]
        if len(lower_bound) == 1:
            self.lower_bound = [
                preference - lower_bound[0] for preference in deepcopy(self.preference)
            ]
        else:
            self.lower_bound = [
                preference[index] - lower_bound[index]
                for index in range(len(lower_bound))
            ]
        self.deficit = 0
        self.priority = (
            abs(self.deficit) / sum([abs(i) for i in self.allocated])
            if sum([abs(i) for i in self.allocated]) != 0
            else abs(self.deficit)
        )
        self.flex_capacity = flex_capacity
        self.flex_SoC = flex_SoC
        self.epsilon = epsilon
        self.profiles = {}
        self.allocated_profiles = {}
        self.exit_timer = 0
        self.holProfile = HolProfile(
            name=self.name,
            preference=self.preference,
            upper_bound=self.upper_bound,
            lower_bound=self.lower_bound,
            flex_capacity=self.flex_capacity,
            flex_SoC=self.flex_SoC,
        )
        self.TIMESTEPS = 24
        self.exit_loop = False
        self.flex_violation = []

    def CnR(self, sub_limit):
        """
        Main function for the Curtailment and Reallocation algorithm

        Args:
            sub_limit (list(int)): available capacity at the subtation
        """
        exit = False
        while not exit:
            self.flex_violation.append(self.calc_metrics()["flexibility_violation"])

            if all(
                abs(self.calc_total_power(time=t)) < sub_limit[t]
                for t in range(self.TIMESTEPS)
            ):
                print("limit never exceeded")
                break
            for time in range(self.TIMESTEPS):
                total_power = self.calc_total_power(time=time)
                violation = (
                    total_power
                    / abs(total_power)
                    * (sub_limit[time] - abs(total_power))
                )  # Calculate violation amount
                if (
                    abs(total_power) > sub_limit[time]
                ):  # If curtailment needs to be done
                    self.curtail(time, violation, sub_limit[time])
                elif abs(total_power) < sub_limit[time]:  # If reallocation is possible
                    self.reallocate(
                        time=time, total_power=total_power, sub_limit=sub_limit[time]
                    )
            retry = False
            for profile in self.profiles:
                if retry:
                    break
                if abs(self.profiles[profile].deficit) >= 0 + self.epsilon:
                    for t in range(self.TIMESTEPS):
                        if (
                            self.profiles[profile].deficit > 0
                            and self.profiles[profile].allocated[t]
                            > self.profiles[profile].lower_bound[t]
                            and abs(self.calc_total_power(t)) < sub_limit[t]
                        ):
                            self.exit_timer += 1
                            # Maximum of 10 revisions
                            if self.exit_timer < 10:
                                retry = True
                                break
                        elif (
                            self.profiles[profile].deficit < 0
                            and self.profiles[profile].allocated[t]
                            < self.profiles[profile].upper_bound[t]
                            and abs(self.calc_total_power(t)) < sub_limit[t]
                        ):
                            self.exit_timer += 1
                            # Maximum of 10 revisions
                            if self.exit_timer < 10:
                                retry = True
                                break
            if not retry:
                exit = True

    def calc_total_power(self, time):
        """
        Calculates the total amount of allocated power at a time slot

        Args:
            time (int): current time

        Returns:
            int: total amount of allocated power at current time
        """
        return sum(self.profiles[profile].allocated[time] for profile in self.profiles)

    def curtail(self, time, total_curtail, sub_limit):
        """
        Curtail capacity of holons

        Args:
            time (int): current time
            total_curtail (float): amount of capacity that can be reallocated
            sub_limit (float): maximum capacity of the substation
        """
        # Get priorities
        priorities_sorted = {
            name: profile.priority
            for name, profile in sorted(
                self.profiles.items(), key=lambda item: item[1].priority, reverse=False
            )
        }

        # If holons need to be curtailed beyond their lower bound
        if all(
            self.profiles[name].allocated[time] >= self.profiles[name].upper_bound[time]
            for name in priorities_sorted
        ) or all(
            self.profiles[name].allocated[time] <= self.profiles[name].lower_bound[time]
            for name in priorities_sorted
        ):
            curtailed = 0

            highest_prio = max(self.profiles[i].priority for i in self.profiles)
            total_priority = sum(
                [highest_prio - self.profiles[i].priority for i in self.profiles]
            )
            for name in priorities_sorted:
                profile = self.profiles[name]

                if total_priority == 0:
                    factor = 1 / len(self.profiles)
                else:
                    factor = (
                        highest_prio - self.profiles[name].priority
                    ) / total_priority

                amount_curtail = factor * total_curtail
                profile.update(
                    power=(profile.allocated[time] + amount_curtail), time=time
                )
                curtailed += amount_curtail
            total_curtail -= curtailed

        # If holons have to be curtailed, but can stay above their lower bound
        else:
            for name in priorities_sorted:
                profile = self.profiles[name]
                if total_curtail == 0:
                    continue
                elif total_curtail < 0:
                    amount_curtail = max(
                        total_curtail,
                        profile.lower_bound[time] - profile.allocated[time],
                    )  # is -
                else:
                    amount_curtail = min(
                        total_curtail,
                        profile.upper_bound[time] - profile.allocated[time],
                    )  # is +
                profile.update(
                    power=(profile.allocated[time] + amount_curtail), time=time
                )
                total_curtail -= amount_curtail
        if (
            abs(
                sum(self.profiles[profile].allocated[time] for profile in self.profiles)
            )
            > sub_limit + self.epsilon
        ):
            self.curtail(time, total_curtail, sub_limit)

    def reallocate(self, time, total_power, sub_limit):
        """
        Reallocate capacity to holons if possible

        Args:
            time (int): current time
            total_reallocate (float): amount of capacity that can be reallocated
        """
        # Get priorities
        priorities_sorted = {
            name: profile.priority
            for name, profile in sorted(
                self.profiles.items(), key=lambda item: item[1].priority, reverse=True
            )
        }
        for priority in priorities_sorted:
            profile = self.profiles[priority]
            if profile.deficit == 0:
                continue
            elif profile.deficit > 0:
                amount_reallocate = max(
                    -total_power - sub_limit,
                    profile.lower_bound[time] - profile.allocated[time],
                    -profile.deficit,
                )
            else:
                amount_reallocate = min(
                    sub_limit - total_power,
                    profile.upper_bound[time] - profile.allocated[time],
                    -profile.deficit,
                )
            # Update data
            profile.update(
                power=(profile.allocated[time] + amount_reallocate), time=time
            )
            total_power += amount_reallocate

    def create_window(self):
        """
        Creating trade window for the Capacity Trading algorithm
        """
        own_profile = self.holProfile
        t_begin = 0
        t_end = self.TIMESTEPS
        infraction = 0
        # Determine window
        for t in range(self.TIMESTEPS):
            if own_profile.flex_SoC[t] < 0 - self.epsilon:
                t_end = t
                limit = [
                    x
                    for x, item in enumerate(own_profile.flex_SoC[0:t_end])
                    if item >= own_profile.flex_capacity
                ]
                t_begin = 0 if len(limit) == 0 else limit[-1]
                infraction = own_profile.flex_SoC[t_end]
                break
            elif own_profile.flex_SoC[t] > own_profile.flex_capacity + self.epsilon:
                t_end = t
                limit = [
                    x
                    for x, item in enumerate(own_profile.flex_SoC[0:t_end])
                    if item <= 0
                ]
                t_begin = 0 if len(limit) == 0 else limit[-1]
                infraction = own_profile.flex_SoC[t_end] - own_profile.flex_capacity
                break
        return [t_begin, t_end, infraction]

    def respond_trade(self, t_begin, t_end, infraction):
        """
        Adjusting trade window and power profile of the Capacity Trading algorithm
        Also creates new trade window for returning capacity

        Args:
            t_begin (int): begin of the trading window
            t_end (int): end time of the trading window
            infraction (float): amount to be traded
        """
        own_profile = deepcopy(self.holProfile)
        change_profile = [0] * self.TIMESTEPS
        if infraction < 0:
            limit = [
                x + t_begin
                for x, item in enumerate(own_profile.flex_SoC[t_begin:t_end])
                if item <= 0
            ]
            t_begin = t_begin if len(limit) == 0 else limit[-1]
            for t in range(t_begin, t_end):
                delta = max(
                    infraction,
                    own_profile.lower_bound[t] - own_profile.allocated[t],
                    0 - min(own_profile.flex_SoC[t + 1 : t_end + 1]),
                )
                own_profile.update(power=own_profile.allocated[t] + delta, time=t)
                infraction -= delta
                change_profile[t] = delta
            if abs(infraction) >= 0 + self.epsilon:  # No plannning possible
                return [False] * 3

            t_begin = t_end
            t_end = self.TIMESTEPS
            limit = [
                x + t_begin
                for x, item in enumerate(own_profile.flex_SoC[t_begin:t_end])
                if item < 0
            ]
            t_end = self.TIMESTEPS if len(limit) == 0 else limit[0]
            limit = [
                x + t_begin
                for x, item in enumerate(own_profile.flex_SoC[t_begin:t_end])
                if item >= own_profile.flex_capacity
            ]
            t_begin = t_begin if len(limit) == 0 else limit[-1]
        else:
            limit = [
                x + t_begin
                for x, item in enumerate(own_profile.flex_SoC[t_begin:t_end])
                if item >= own_profile.flex_capacity
            ]
            t_begin = t_begin if len(limit) == 0 else limit[-1]
            for t in range(t_begin, t_end):
                delta = min(
                    infraction,
                    own_profile.upper_bound[t] - own_profile.allocated[t],
                    max(
                        0,
                        own_profile.flex_capacity
                        - max(own_profile.flex_SoC[t + 1 : t_end + 1]),
                    ),
                )
                own_profile.update(power=own_profile.allocated[t] + delta, time=t)
                infraction -= delta
                change_profile[t] = delta
            if abs(infraction) >= 0 + self.epsilon:  # No planning possible
                return [False] * 3

            # CREATE WINDOW FOR RETURNING CAPACITY
            t_begin = t_end
            t_end = self.TIMESTEPS
            limit = [
                x + t_begin
                for x, item in enumerate(own_profile.flex_SoC[t_begin:t_end])
                if item > own_profile.flex_capacity
            ]
            t_end = t_end if len(limit) == 0 else limit[0]
            limit = [
                x + t_begin
                for x, item in enumerate(own_profile.flex_SoC[t_begin:t_end])
                if item <= 0
            ]
            t_begin = t_begin if len(limit) == 0 else limit[-1]
        return [t_begin, t_end, change_profile]

    def check_for_trade(self, t_begin, t_end, infraction):
        """
        Adjusting window and power profile of the Capacity Trading algorithm

        Args:
            t_begin (int): begin of the trading window
            t_end (int): end time of the trading window
            infraction (float): amount to be traded
        """
        own_profile = deepcopy(self.holProfile)
        change_profile = [0] * self.TIMESTEPS
        if infraction < 0:
            limit = [
                x + t_begin
                for x, item in enumerate(own_profile.flex_SoC[t_begin:t_end])
                if item <= 0
            ]
            t_begin = t_begin if len(limit) == 0 else limit[-1]
            for t in range(t_begin, t_end):
                delta = max(
                    infraction,
                    own_profile.lower_bound[t] - own_profile.allocated[t],
                    -min(own_profile.flex_SoC[t + 1 : t_end + 1]),
                )
                own_profile.update(power=own_profile.allocated[t] + delta, time=t)
                infraction -= delta
                change_profile[t] = delta
            if abs(infraction) >= 0 + self.epsilon:  # No planning possible
                return False
        else:  # infraction > 0
            limit = [
                x + t_begin
                for x, item in enumerate(own_profile.flex_SoC[t_begin:t_end])
                if item >= own_profile.flex_capacity
            ]
            t_begin = t_begin if len(limit) == 0 else limit[-1]
            for t in range(t_begin, t_end):
                delta = min(
                    infraction,
                    own_profile.upper_bound[t] - own_profile.allocated[t],
                    max(
                        0,
                        own_profile.flex_capacity
                        - max(own_profile.flex_SoC[t + 1 : t_end + 1]),
                    ),
                )
                own_profile.update(power=own_profile.allocated[t] + delta, time=t)
                infraction -= delta
                change_profile[t] = delta
            if abs(infraction) >= 0 + self.epsilon:  # No planning possible
                return False

        return change_profile

    def plot(self):
        """
        Plot the current allocated power profiles,
        as well as the aggregate power profile
        """
        # ALLOCATED PROFILES
        fig, axs = plt.subplots(3, 1, figsize=(15, 14))
        fig.suptitle("Allocated profiles", fontsize=22)
        plt.rc("font", size=22)
        if len(self.profiles) == 3:
            for i, hol in enumerate(self.profiles):
                preference = deepcopy(self.profiles[hol].preference)
                allocated = deepcopy(self.profiles[hol].allocated)
                upper_bound = deepcopy(self.profiles[hol].upper_bound)
                lower_bound = deepcopy(self.profiles[hol].lower_bound)
                preference.insert(0, preference[0])
                allocated.insert(0, allocated[0])
                upper_bound.insert(0, upper_bound[0])
                lower_bound.insert(0, lower_bound[0])
                axs[i].step(
                    range(len(preference)),
                    preference,
                    "--",
                    color="orange",
                    label="$P_{p,%s}$" % hol,
                )
                axs[i].step(
                    range(len(allocated)),
                    allocated,
                    color="orange",
                    label="$P_{a,%s}$" % hol,
                )
                axs[i].step(
                    range(len(allocated)),
                    upper_bound,
                    color="red",
                    label="$P_{ub,%s}$" % hol,
                )
                axs[i].step(
                    range(len(allocated)),
                    lower_bound,
                    color="blue",
                    label="$P_{lb,%s}$" % hol,
                )
                axs[i].fill_between(
                    range(0, len(allocated), 1),
                    upper_bound,
                    lower_bound,
                    step="pre",
                    alpha=0.4,
                    color="yellow",
                    label="preference bound",
                )
                axs[i].step(range(len(allocated)), allocated, color="orange")
                axs[i].set_xticks(range(1, 25, 1))
                axs[i].set_yticks(range(-10000, 12000, 2000))
                axs[i].set_xticklabels(range(1, 25, 1), fontsize=22)
                axs[i].set_yticklabels(range(-10000, 12000, 2000), fontsize=22)
                axs[i].set_xlim(1, 24)
                axs[i].set_ylim(-12000, 12000)
                axs[i].set_xlabel("time [h]", fontsize=22)
                axs[i].set_ylabel("power [kW]", fontsize=22)
                axs[i].legend(loc="lower center", fontsize=15)
                axs[i].yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
                axs[i].ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
            plt.subplots_adjust(bottom=0.07, top=0.955, wspace=0.2, hspace=0.22)
            plt.show()
        # AGGREGATED ALLOCATED PROFILE
        plt.figure(figsize=(15, 7))
        agg_allocated = [0] * len(self.profiles["hol1"].allocated)
        for hol in self.profiles:
            profile = self.profiles[hol]
            agg_allocated = [x + y for x, y in zip(agg_allocated, profile.allocated)]
        agg_allocated.insert(0, agg_allocated[0])
        plt.step(range(len(agg_allocated)), agg_allocated, label="$P_{tot}$")
        plt.axhline(12000, color="red", label="$P_{s}$")
        plt.axhline(-12000, color="red")
        plt.xticks(range(1, 25, 1), fontsize=22)
        plt.yticks(range(-16000, 18000, 2000), fontsize=22)
        plt.xlim(1, 24)
        plt.ylim(-15000, 15000)
        plt.xlabel("time [h]", fontsize=22)
        plt.ylabel("power [kW]", fontsize=22)
        plt.legend(loc="center right", fontsize=18)
        plt.title("Aggregated power profile")
        plt.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
        plt.show()

    def calc_metrics(self, printThings=False):
        """
        Calculate the metrics
        """
        metrics = {
            "profile": [],
            "preference_met": [],
            "out_of_bounds": [],
            "relative_deficit": [],
            "absolute_deficit": [],
            "flexibility_violation": [],
        }

        # PREFERENCE MET
        for profile in self.profiles:
            totalDeviation = sum(
                abs(x - y)
                for x, y in zip(
                    self.profiles[profile].allocated, self.profiles[profile].preference
                )
            )
            total_preference = abs(sum(self.profiles[profile].preference))
            metric = (
                100 * (total_preference - totalDeviation) / total_preference
                if total_preference != 0
                else 100
            )
            metrics["profile"].append(profile)
            metrics["preference_met"].append(metric)
        if printThings:
            print(
                f"preference met mean: {mean(metrics['preference_met'])}, stdev: {stdev(metrics['preference_met'])}"
            )

        # RELATIVE OUT OF BOUNDS
        for profile in self.profiles:  # out of bounds
            profile = self.profiles[profile]
            out_of_bounds = 0
            for t in range(self.TIMESTEPS):
                if profile.allocated[t] > profile.upper_bound[t]:
                    out_of_bounds += abs(profile.allocated[t] - profile.upper_bound[t])
                elif profile.allocated[t] < profile.lower_bound[t]:
                    out_of_bounds += abs(profile.lower_bound[t] - profile.allocated[t])
            metric = (
                100 * out_of_bounds / sum(profile.preference)
                if sum(profile.preference) != 0
                else 0
            )
            metrics["out_of_bounds"].append(metric)
        if printThings:
            print(
                f"Out of bounds mean: {mean(metrics['out_of_bounds'])}, stdev: {stdev(metrics['out_of_bounds'])}"
            )

        # RELATIVE DEFICIT
        for profile in self.profiles:  # relative deficit
            metric = (
                100
                * self.profiles[profile].deficit
                / abs(sum(self.profiles[profile].preference))
                if sum(self.profiles[profile].preference) != 0
                else 0
            )
            metrics["relative_deficit"].append(metric)
        if printThings:
            print(
                f"relative deficit mean: {mean(metrics['relative_deficit'])}, stdev: {stdev(metrics['relative_deficit'])}"
            )

        # ABSOLUTE DEFICIT
        for profile in self.profiles:  # aboslute deficit
            metric = self.profiles[profile].deficit
            metrics["absolute_deficit"].append(metric)

        # FLEXIBILITY VIOLATION
        for profile in self.profiles:
            profile = self.profiles[profile]
            minimum = min(profile.flex_SoC)
            maximum = max(profile.flex_SoC)
            if minimum < 0:
                metrics["flexibility_violation"].append(minimum)
            elif maximum > profile.flex_capacity:
                metrics["flexibility_violation"].append(maximum)
            else:
                metrics["flexibility_violation"].append(minimum)

        # Save metrics to csv file if desired
        save = False
        if save:
            with open("metrics.csv", mode="a", newline="") as file:
                writer = csv.writer(file)
                if file.tell() == 0:
                    writer.writerow(metrics.keys())

                rows = zip(*metrics.values())
                writer.writerows(rows)
        return metrics
