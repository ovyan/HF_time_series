import numpy as np
from sortedcontainers import SortedList
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (17, 9)


def get_data():
    curr_price, curr_time = [], []
    tick_net, tick_time, tick_change = [], [], []
    ask_price, ask_time, ask_size = [], [], []
    bid_price, bid_time, bid_size = [], [], []

    with open('data_TQQQ.txt') as f:
        for _ in range(1_000_000):
            line = f.readline().strip().split("|")
            line = line[1:]  # skipping "D|"
            row = {}
            for elem in line:
                key, value = elem.split("=")
                if key == '16' and value != "":
                    row['16'] = float(value)
                elif key == '361' and value != "":
                    row['361'] = float(value)
                elif key == '10' and value != "":
                    row['10'] = float(value)
                elif key == '11' and value != "":
                    row['11'] = float(value)
                elif key == '12' and value != "":
                    row['12'] = float(value)
                elif key == '13' and value != "":
                    row['13'] = float(value)

            if '361' in row:
                tick_net.append(row['361'])
                tick_time.append(row['16'])
            if '10' in row and row['10'] < 100:  # TODO: just check outliers
                ask_price.append(row['10'])
                ask_time.append(row['16'])
            if '11' in row:
                ask_size.append(row['11'])
            if '12' in row and row['12'] > 50:
                bid_price.append(row['12'])
                bid_time.append(row['16'])
            if '13' in row:
                bid_size.append(row['13'])

    tick_net = np.array(tick_net)
    tick_time = np.array(tick_time)
    tick_change = tick_net[1:] - tick_net[:-1]
    tick_change = np.insert(tick_change, 0, 0)  # start from 0

    return ask_price, ask_time, ask_size, bid_price, bid_time, bid_size


def get_midpoint_spread(ask_price, ask_time, bid_price, bid_time):

    i = 0
    j = 0
    sorted_ask = SortedList([ask_price[i]])  # *int(ask_size[i]))
    sorted_bid = SortedList([bid_price[j]])  # *int(bid_size[j]))
    spread = []
    spread_t = []
    midpoint = []
    moved = 0

    curr_time = 0
    last_add_bid = 0

    while i < len(ask_time) and j < len(bid_time):
        if i > 150000:
            break

        if len(sorted_bid) == 0:
            j += 1  # move curr time
            sorted_bid += [bid_price[j]]
            continue
        elif len(sorted_ask) == 0:
            i += 1
            sorted_ask += [ask_price[i]]
            continue

        if bid_time[j] < ask_time[i]:  # + eps
            j += 1
            curr_time = bid_time[j]
            last_add_bid = bid_price[j]
            sorted_bid += [bid_price[j]]
            continue

        if bid_time[j] > ask_time[i]:  # + eps
            i += 1
            curr_time = ask_time[i]
            sorted_ask += [ask_price[i]]
            continue

        if sorted_bid[-1] < sorted_ask[0]:  # bid < ask
            spread.append(sorted_ask[0] - sorted_bid[-1])
            spread_t.append(curr_time)
            midpoint.append((sorted_ask[0] + sorted_bid[-1]) / 2)
            j += 1  # move time
            sorted_bid += [bid_price[j]]
            i += 1
            sorted_ask += [ask_price[i]]
            curr_time = max(ask_time[i], bid_time[j])

            continue

        if sorted_bid[-1] >= sorted_ask[0]:  # bid >= ask
            sorted_bid.pop(-1)
            sorted_ask.pop(0)

    return midpoint, spread, spread_t
