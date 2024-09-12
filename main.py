from holon import Holon
from copy import deepcopy
import csv
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

# Settings start here

converter = 1000  # Factor to convert from kW to MW, 1=kW, 1000=MW
TIMESTEPS = 24  # Amount of timesteps in algorithm
sub_limit = [12 * 1000] * TIMESTEPS  # Capacity available at the substation
file_path = "data/Small-scale"  # Path to the input files
plot = True  # If desired to plot the images
save = False  # If desired to save the images
save_path = "Figures/"  # Path to save figures
print_things = True  # If printing relevant info, i.e., metrics

# Parameters for plotting
plt.rcParams["axes.formatter.use_mathtext"] = True
plt.rcParams["xtick.labelsize"] = 20  # Adjust as needed for x-axis tick labels
plt.rcParams["ytick.labelsize"] = 20  # Adjust as needed for y-axis tick labels
plt.rcParams["axes.labelsize"] = 20  # Adjust as needed for axes labels
plt.rcParams["axes.titlesize"] = 20  # Adjust as needed for subplot titles
plt.rcParams["legend.fontsize"] = 20  # Adjust as needed for subplot titles
plt.rcParams["figure.titlesize"] = 20  # Adjust as needed for title size
plt.rcParams["lines.linewidth"] = (
    2.5  # Adjust as needed for width of plot lines (orginal 1.5)
)


# Settings end here


agg_preference = [0] * 25
agg_lower = [0] * 25
agg_upper = [0] * 25

uppers = list(csv.reader(open(file_path + "/upper.csv", "r", encoding="utf-8-sig")))
averages = list(csv.reader(open(file_path + "/avg.csv", "r", encoding="utf-8-sig")))
lowers = list(csv.reader(open(file_path + "/lower.csv", "r", encoding="utf-8-sig")))
capacities = list(
    csv.reader(open(file_path + "/capacity.csv", "r", encoding="utf-8-sig"))
)
initial_SoCs = list(
    csv.reader(open(file_path + "/initial_soc.csv", "r", encoding="utf-8-sig"))
)

holons = []

for index, _ in enumerate(averages):
    upper = uppers[index]
    upper = [float(x) for x in upper]
    upper.append(upper[-1])
    agg_upper = [x + y for x, y in zip(agg_upper, upper)]

    avg = averages[index]
    avg = [float(x) for x in avg]
    avg.append(avg[-1])
    agg_preference = [x + y for x, y in zip(agg_preference, avg)]

    lower = lowers[index]
    lower = [float(x) for x in lower]
    lower.append(lower[-1])
    agg_lower = [x + y for x, y in zip(agg_lower, lower)]

    capacity = float(capacities[index][0])
    initial_SoC = float(initial_SoCs[index][0])

    holons.append(
        Holon(
            name="hol" + str(index + 1),
            preference=avg,
            upper_bound=upper,
            lower_bound=lower,
            flex_capacity=capacity,
            flex_SoC=[initial_SoC] * 25,
        )
    )


# PREFERENCE PROFILES
if plot:
    if len(holons) == 3:
        fig, axs = plt.subplots(3, 1, figsize=(15, 14))
        fig.suptitle("Preference profiles")
        for i, hol in enumerate(holons):
            preference = deepcopy(hol.preference)
            upper_bound = deepcopy(hol.upper_bound)
            lower_bound = deepcopy(hol.lower_bound)
            preference.insert(0, preference[0])
            upper_bound.insert(0, upper_bound[0])
            lower_bound.insert(0, lower_bound[0])
            preference = [x / converter for x in preference]
            lower_bound = [x / converter for x in lower_bound]
            upper_bound = [x / converter for x in upper_bound]
            axs[i].step(
                range(len(preference)),
                preference,
                color="orange",
                label="$P_{p,%s}$" % hol.name,
            )
            axs[i].step(
                range(len(preference)),
                upper_bound,
                color="red",
                label="$P_{ub,%s}$" % hol.name,
            )
            axs[i].step(
                range(len(preference)),
                lower_bound,
                color="blue",
                label="$P_{lb,%s}$" % hol.name,
            )
            axs[i].fill_between(
                range(0, len(preference), 1),
                upper_bound,
                lower_bound,
                step="pre",
                alpha=0.4,
                color="yellow",
                label="preference band",
            )
            axs[i].step(range(len(preference)), preference, color="orange")
            axs[i].set_xticks(range(1, 25, 1))
            axs[i].set_yticks(
                range(
                    int(-10000 / converter),
                    int(12000 / converter),
                    int(2000 / converter),
                )
            )
            axs[i].set_xticklabels(range(1, 25, 1))
            axs[i].set_yticklabels(
                range(
                    int(-10000 / converter),
                    int(12000 / converter),
                    int(2000 / converter),
                )
            )  # ,fontsize=22)
            axs[i].set_xlim(1, 24)
            axs[i].set_ylim(-12000 / converter, 12000 / converter)
            axs[i].set_xlabel("time [h]")
            if converter == 1:
                axs[i].set_ylabel("power [kW]")
                axs[i].yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
                axs[i].ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
            else:
                axs[i].set_ylabel("power [MW]")
            box = axs[i].get_position()
            axs[i].set_position([box.x0, box.y0, box.width * 0.8, box.height])
            axs[i].legend(loc="center left", bbox_to_anchor=(1, 0.5))
        plt.subplots_adjust(bottom=0.07, top=0.955, wspace=0.2, hspace=0.22)
        if save:
            plt.savefig(save_path + "preference_profiles.pdf", bbox_inches="tight")
        else:
            plt.show()

# AGGREGATED PREFERENCE PROFILES
if plot:
    fig, axs = plt.subplots(1, 1, figsize=(15, 7))
    fig.suptitle("Aggregated preference profile")
    agg_preference = [sum(x) for x in zip(*[hol.preference for hol in holons])]
    agg_upper_bound = [x + y for x, y in zip(agg_preference, agg_upper)]
    agg_lower_bound = [x - y for x, y in zip(agg_preference, agg_lower)]
    agg_preference.insert(0, agg_preference[0])
    agg_upper_bound.insert(0, agg_upper_bound[0])
    agg_lower_bound.insert(0, agg_lower_bound[0])
    agg_preference = [x / converter for x in agg_preference]
    agg_upper_bound = [x / converter for x in agg_upper_bound]
    agg_lower_bound = [x / converter for x in agg_lower_bound]
    axs.step(range(len(agg_preference)), agg_preference, label="$P_{tot}$")
    axs.step(range(len(agg_preference)), agg_upper_bound, label="$P_{ub}$")
    axs.step(range(len(agg_preference)), agg_lower_bound, label="$P_{lb}$")
    axs.fill_between(
        range(0, len(agg_preference), 1),
        agg_upper_bound,
        agg_lower_bound,
        step="pre",
        alpha=0.4,
        color="yellow",
        label="preference band",
    )
    axs.axhline(12000 / converter, color="red", label="$P_{s}$")
    axs.axhline(-12000 / converter, color="red")
    axs.set_xticks(range(1, 25, 1))
    axs.set_yticks(
        range(int(-16000 / converter), int(24000 / converter), int(2000 / converter))
    )
    axs.set_xlim(1, 24)
    axs.set_ylim(-17000 / converter, 24000 / converter)
    axs.set_xlabel("time [h]")
    if converter == 1:
        axs.set_ylabel("power [kW]")
        axs.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    else:
        axs.set_ylabel("power [MW]")
    box = axs.get_position()
    axs.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    axs.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    if save:
        plt.savefig(save_path + "aggregated_preference.pdf", bbox_inches="tight")
    else:
        plt.show()


for hol in holons:
    holons[0].profiles[hol.holProfile.name] = deepcopy(hol.holProfile)

# Curtailment and Reallocation algorithm
holons[0].CnR(sub_limit)

# Update the holons with the results from the Curtailment
#   and Reallocation algorithm
for hol in holons:
    hol.holProfile = holons[0].profiles[hol.holProfile.name]


holons[0].flex_violation.append(holons[0].calc_metrics()["flexibility_violation"])
if plot:
    # ALLOCATED PROFILES
    if len(holons[0].profiles) == 3:
        fig, axs = plt.subplots(3, 1, figsize=(15, 14))
        fig.suptitle(
            "Allocated profiles after the Curtailment and Reallocation algorithm"
        )
        for i, hol in enumerate(holons[0].profiles):
            preference = deepcopy(holons[0].profiles[hol].preference)
            allocated = deepcopy(holons[0].profiles[hol].allocated)
            upper_bound = deepcopy(holons[0].profiles[hol].upper_bound)
            lower_bound = deepcopy(holons[0].profiles[hol].lower_bound)
            preference.insert(0, preference[0])
            allocated.insert(0, allocated[0])
            upper_bound.insert(0, upper_bound[0])
            lower_bound.insert(0, lower_bound[0])
            preference = [x / converter for x in preference]
            allocated = [x / converter for x in allocated]
            upper_bound = [x / converter for x in upper_bound]
            lower_bound = [x / converter for x in lower_bound]
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
                label="preference band",
            )
            axs[i].step(range(len(allocated)), allocated, color="orange")
            axs[i].set_xticks(range(1, 25, 1))
            axs[i].set_yticks(
                range(
                    int(-10000 / converter),
                    int(12000 / converter),
                    int(2000 / converter),
                )
            )
            axs[i].set_xticklabels(range(1, 25, 1))
            axs[i].set_yticklabels(
                range(
                    int(-10000 / converter),
                    int(12000 / converter),
                    int(2000 / converter),
                )
            )
            axs[i].set_xlim(1, 24)
            axs[i].set_ylim(-12000 / converter, 12000 / converter)
            axs[i].set_xlabel("time [h]")
            if converter == 1:
                axs[i].set_ylabel("power [kW]")
                axs[i].yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
                axs[i].ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
            else:
                axs[i].set_ylabel("power [MW]")
            box = axs[i].get_position()
            axs[i].set_position([box.x0, box.y0, box.width * 0.8, box.height])
            axs[i].legend(loc="center left", bbox_to_anchor=(1, 0.5))
        plt.subplots_adjust(bottom=0.07, top=0.955, wspace=0.2, hspace=0.22)
        if save:
            plt.savefig(save_path + "allocated_profiles_CnR.pdf", bbox_inches="tight")
        else:
            plt.show()

    # AGGREGATED ALLOCATED PROFILE
    fig, axs = plt.subplots(1, 1, figsize=(15, 7))
    fig.suptitle(
        "Aggregated allocated profile after the\nCurtailment and Reallocation algorithm"
    )
    agg_allocated = [0] * len(holons[0].profiles["hol1"].allocated)
    for hol in holons[0].profiles:
        profile = holons[0].profiles[hol]
        agg_allocated = [x + y for x, y in zip(agg_allocated, profile.allocated)]
    agg_allocated.insert(0, agg_allocated[0])
    agg_allocated = [x / converter for x in agg_allocated]
    axs.axhline(12000 / converter, color="red", label="$P_{s}$")
    axs.axhline(-12000 / converter, color="red")
    axs.step(
        range(len(agg_preference)),
        agg_preference,
        linestyle="dashed",
        label="$P_{p,tot}$",
    )
    axs.step(range(len(agg_allocated)), agg_allocated, label="$P_{a,tot}$")
    axs.step(range(len(agg_upper_bound)), agg_upper_bound, label="$P_{ub,tot}$")
    axs.step(range(len(agg_lower_bound)), agg_lower_bound, label="$P_{lb,tot}$")
    axs.fill_between(
        range(0, len(agg_preference), 1),
        agg_upper_bound,
        agg_lower_bound,
        step="pre",
        alpha=0.4,
        color="yellow",
        label="preference band",
    )
    axs.set_xticks(range(1, 25, 1))
    axs.set_yticks(
        range(int(-16000 / converter), int(20000 / converter), int(2000 / converter))
    )
    axs.set_xlim(1, 24)
    axs.set_ylim(-15000 / converter, 23000 / converter)
    axs.set_xlabel("time [h]")
    if converter == 1:
        axs.set_ylabel("power [kW]")
        axs.set_ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    else:
        axs.set_ylabel("power [MW]")
    box = axs.get_position()
    axs.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    axs.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    if save:
        plt.savefig(save_path + "allocated_aggregated_CnR.pdf", bbox_inches="tight")
    else:
        plt.show()

    # BATTERY SOC BEFORE
    fig, axs = plt.subplots(1, 1, figsize=(15, 7))
    fig.suptitle("State of charge profiles before the\nCapacity Trading algorithm")
    for hol in holons:
        SoC = deepcopy(hol.holProfile.flex_SoC)
        SoC.insert(0, SoC[0])
        SoC = [x / converter for x in SoC]
        axs.plot(SoC, label="SoC: " + hol.name, linestyle="dashed")
        if print_things:
            print(
                f"min SoC value for holon {hol.holProfile.name}: {min(hol.holProfile.flex_SoC)}, max SoC value: {max(hol.holProfile.flex_SoC)}"
            )
    axs.hlines(0, xmin=0, xmax=25, color="red")
    axs.hlines(
        max((hol.flex_capacity for hol in holons)) / 1000, xmin=0, xmax=25, color="red"
    )
    axs.set_xlim(left=1, right=25)
    axs.set_ylim(
        bottom=-2000 / converter,
        top=(max(hol.flex_capacity for hol in holons) + 1000) / converter,
    )
    axs.set_xlabel("time [h]")
    axs.set_xticks(range(1, 25, 1))
    axs.set_yticks(
        range(
            int(-2500 / converter),
            int((max(hol.flex_capacity for hol in holons) + 1000) / converter),
            int(1000 / converter),
        )
    )
    if converter == 1:
        axs.set_ylabel("SoC [kWh]")
    else:
        axs.set_ylabel("SoC [MWh]")
    box = axs.get_position()
    axs.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    axs.legend(loc="center left", bbox_to_anchor=(1, 0.5))

    if save:
        plt.savefig(save_path + "battery_CnR.pdf", bbox_inches="tight")
    else:
        plt.show()

# Capacity Trading Algorithm
trades = []
infractionsss = False
for x in holons:
    # Creating Trade Window
    [t_begin, t_end, infraction_init] = x.create_window()
    f = 0
    infraction = infraction_init
    while infraction != 0:  # if there are flexbility violations
        if f >= 1:
            infractionsss = True
            break
        infraction = infraction * (1 - f)
        success = False
        for y in holons:
            if y.name == x.name:  # Holon cannot trade with itself
                continue

            # Adjusting Trade Window + Adjusting Power Profile
            [t_begin_new, t_end_new, change_profile1] = y.respond_trade(
                t_begin, t_end, infraction
            )
            if change_profile1 == False:
                continue
            for t in range(x.TIMESTEPS):
                x.holProfile.update(
                    power=x.holProfile.allocated[t] - change_profile1[t], time=t
                )
                y.holProfile.update(
                    power=y.holProfile.allocated[t] + change_profile1[t], time=t
                )
            change_profile2 = x.check_for_trade(t_begin_new, t_end_new, infraction)
            if change_profile2 == False:
                for t in range(x.TIMESTEPS):
                    x.holProfile.update(
                        power=x.holProfile.allocated[t] + change_profile1[t], time=t
                    )
                    y.holProfile.update(
                        power=y.holProfile.allocated[t] - change_profile1[t], time=t
                    )
                continue

            # Trade was successful
            # Adjusting Final Power Profile
            for t in range(x.TIMESTEPS):
                x.holProfile.update(
                    power=x.holProfile.allocated[t] + change_profile2[t], time=t
                )
                y.holProfile.update(
                    power=y.holProfile.allocated[t] - change_profile2[t], time=t
                )
                success = True
            break
        # Create new trade window for next iteration
        [t_begin, t_end, infraction] = x.create_window()
        if not success:
            f += 0.05  # Increase factor to decrease trade amount
        else:
            trades.append([x.name, y.name])
            f = 0  # Reset factor since trading was successful
        holons[0].flex_violation.append(
            holons[0].calc_metrics()["flexibility_violation"]
        )  # Add flexibility violations for convergence

# ALLOCATED PROFILES AFTER CAPACITY TRADING
if plot:
    if len(holons[0].profiles) == 3:
        fig, axs = plt.subplots(3, 1, figsize=(15, 14))
        fig.suptitle("Allocated profiles after the Capacity Trading algorithm")
        for i, hol in enumerate(holons[0].profiles):
            preference = deepcopy(holons[0].profiles[hol].preference)
            allocated = deepcopy(holons[0].profiles[hol].allocated)
            upper_bound = deepcopy(holons[0].profiles[hol].upper_bound)
            lower_bound = deepcopy(holons[0].profiles[hol].lower_bound)
            preference.insert(0, preference[0])
            allocated.insert(0, allocated[0])
            upper_bound.insert(0, upper_bound[0])
            lower_bound.insert(0, lower_bound[0])
            preference = [x / converter for x in preference]
            allocated = [x / converter for x in allocated]
            upper_bound = [x / converter for x in upper_bound]
            lower_bound = [x / converter for x in lower_bound]
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
                label="preference band",
            )
            axs[i].step(range(len(allocated)), allocated, color="orange")
            axs[i].set_xticks(range(1, 25, 1))
            axs[i].set_yticks(
                range(
                    int(-10000 / converter),
                    int(12000 / converter),
                    int(2000 / converter),
                )
            )
            axs[i].set_xticklabels(range(1, 25, 1))
            axs[i].set_yticklabels(
                range(
                    int(-10000 / converter),
                    int(12000 / converter),
                    int(2000 / converter),
                )
            )
            axs[i].set_xlim(1, 24)
            axs[i].set_ylim(-12000 / converter, 12000 / converter)
            axs[i].set_xlabel("time [h]")
            if converter == 1:
                axs[i].set_ylabel("power [kW]")
                axs[i].yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
                axs[i].ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
            else:
                axs[i].set_ylabel("power [MW]")
            box = axs[i].get_position()
            axs[i].set_position([box.x0, box.y0, box.width * 0.8, box.height])
            axs[i].legend(loc="center left", bbox_to_anchor=(1, 0.5))
        plt.subplots_adjust(bottom=0.07, top=0.955, wspace=0.2, hspace=0.22)
        if save:
            plt.savefig(save_path + "allocated_profiles_CT.pdf", bbox_inches="tight")
        else:
            plt.show()

    # AGGREGATED ALLOCATED AFTER CT
    fig, axs = plt.subplots(1, 1, figsize=(15, 7))
    fig.suptitle("Aggregated allocated profile after the\nCapacity Trading algorithm")
    agg_allocated = [0] * len(holons[0].profiles["hol1"].allocated)
    for hol in holons[0].profiles:
        profile = holons[0].profiles[hol]
        agg_allocated = [x + y for x, y in zip(agg_allocated, profile.allocated)]
    agg_allocated.insert(0, agg_allocated[0])
    agg_allocated = [x / converter for x in agg_allocated]
    axs.axhline(12000 / converter, color="red", label="$P_{s}$")
    axs.axhline(-12000 / converter, color="red")
    axs.step(
        range(len(agg_preference)),
        agg_preference,
        linestyle="dashed",
        label="$P_{p,tot}$",
    )
    axs.step(range(len(agg_allocated)), agg_allocated, label="$P_{a,tot}$")
    axs.step(range(len(agg_upper_bound)), agg_upper_bound, label="$P_{ub,tot}$")
    axs.step(range(len(agg_lower_bound)), agg_lower_bound, label="$P_{lb,tot}$")
    axs.fill_between(
        range(0, len(agg_preference), 1),
        agg_upper_bound,
        agg_lower_bound,
        step="pre",
        alpha=0.4,
        color="yellow",
        label="preference band",
    )
    axs.set_xticks(range(1, 25, 1))
    axs.set_yticks(
        range(int(-16000 / converter), int(30000 / converter), int(2000 / converter))
    )
    axs.set_xlim(1, 24)
    axs.set_ylim(-15000 / converter, 23000 / converter)
    axs.set_xlabel("time [h]")
    if converter == 1:
        axs.set_ylabel("power [kW]")
        axs.set_ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    else:
        axs.set_ylabel("power [MW]")
    box = axs.get_position()
    axs.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    axs.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    if save:
        plt.savefig(save_path + "allocated_aggregated_CT.pdf", bbox_inches="tight")
    else:
        plt.show()

    if not save:
        pass
        # holons[0].plot()

# SoC PROFILES AFTER CAPACITY TRADING
if plot:
    colors = ["blue", "orange", "green", "red", "purple", "brown", "pink", "gray"]
    fig, axs = plt.subplots(1, 1, figsize=(15, 7))
    fig.suptitle("State of charge profiles after the\nCapacity Trading algorithm")
    for index, hol in enumerate(holons):
        SoC = deepcopy(hol.holProfile.flex_SoC)
        SoC.insert(0, SoC[0])
        SoC = [x / converter for x in SoC]
        axs.plot(SoC, label="SoC: " + hol.name, linestyle="dashed")
        if print_things:
            print(
                f"min SoC value for holon {hol.holProfile.name}: {min(hol.holProfile.flex_SoC)}, max SoC value: {max(hol.holProfile.flex_SoC)}"
            )
    axs.hlines(0, xmin=0, xmax=25, color="red")
    axs.hlines(
        max(hol.flex_capacity for hol in holons) / converter,
        xmin=0,
        xmax=25,
        color="red",
    )
    axs.set_xlim(left=1, right=25)
    axs.set_ylim(
        bottom=-1000 / converter,
        top=(max(hol.flex_capacity for hol in holons) + 1000) / converter,
    )
    axs.set_xlabel("time [h]")
    axs.set_xticks(range(1, 25, 1))
    axs.set_yticks(
        range(
            int(-1500 / converter),
            int((max(hol.flex_capacity for hol in holons) + 1000) / converter),
            int(1000 / converter),
        )
    )
    if converter == 1:
        axs.set_ylabel("SoC [kWh]")
    else:
        axs.set_ylabel("SoC [MWh]")
    box = axs.get_position()
    axs.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    axs.legend(loc="center left", bbox_to_anchor=(1, 0.5), ncol=1)
    if save:
        plt.savefig(save_path + "battery_after.pdf", bbox_inches="tight")
    else:
        plt.show()

holons[0].flex_violation.append(holons[0].calc_metrics()["flexibility_violation"])
transposed = list(map(list, zip(*holons[0].flex_violation)))

# CONVERGENCE
if plot:
    fig, axs = plt.subplots(figsize=(15, 7))
    fig.suptitle("Convergence of the flexibility violation")
    axs.set_xlabel("iteration")
    axs.set_ylabel("flexibility violation [MWh]")
    axs.set_xticks(
        range(0, len(transposed[0]) + 2, 1)
    )  # set step size higher if # iterations becomes too high
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    axs.set_xlim(0, len(transposed[0]) - 1)
    labels = []
    lines = []
    for index, item in enumerate(transposed):
        item = [x / converter for x in item]
        (line,) = axs.plot(range(len(item)), item, label="hol" + str(index + 1))
        lines.append(line)
        labels.append(line.get_label())
    box = axs.get_position()
    axs.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    axs.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    if save:
        plt.savefig(save_path + "convergence.pdf", bbox_inches="tight")
    else:
        plt.show()

if print_things:
    print("Metrics:")
    print(holons[0].calc_metrics(print_things=True))
    print("Holon specific info")
    for hol in holons:
        print("Info for " + hol.name + ":")
        print(f"Allocated Power: {hol.holProfile.allocated}")
        print(f"State of charge: {hol.holProfile.flex_SoC}")
