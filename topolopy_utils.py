import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, Counter

def extract_pairs(state):
    """
    Converts a flat list into a list of tuples representing front-tail pairs.
    Handles odd number of elements by ignoring the last element.
    """
    pairs = []
    for i in range(0, len(state) - (len(state) % 2), 2):
        pairs.append((state[i], state[i+1]))
    return pairs


def detect_bifurcations(prev_pairs, current_pairs, all_seen_pairs):
    """
    Detects bifurcations where a single existing pair splits into multiple new pairs.
    Returns a list of bifurcation events.
    """
    bifurcations = []
    disappeared = list((Counter(prev_pairs) - Counter(current_pairs)).elements())
    
    for pair in disappeared:
        related_new_pairs = [p for p in current_pairs if (p[0] == pair[0] or p[1] == pair[1])]
        related_new_pairs = [p for p in current_pairs if p in related_new_pairs]
        if len(related_new_pairs) >= 2:
            bifurcations.append({'from': pair, 'to': related_new_pairs})
    return bifurcations


def detect_mergers(prev_pairs, current_pairs):
    """
    Detects mergers where two or more existing pairs combine into a single new pair.
    Returns a list of merger events.
    """
    mergers = []
    new_pairs = list((Counter(current_pairs) - Counter(prev_pairs)).elements())
    
    for new_pair in new_pairs:
        contributing_pairs = [p for p in prev_pairs if (p[0] == new_pair[0] or p[1] == new_pair[1])]
        if len(contributing_pairs) >= 2:
            mergers.append({'from': contributing_pairs, 'to': new_pair})
    return mergers


def detect_new_waves(current_pairs, all_seen_pairs, prev_pairs):
    """
    Detects new waves which are completely new pairs not present in any previous state.
    Excludes pairs that were present in the previous state.
    Returns a list of new wave pairs.
    """
    new_waves = []
    for pair in current_pairs:
        if pair not in all_seen_pairs and pair not in prev_pairs:
            new_waves.append(pair)
            all_seen_pairs.add(pair)
    return new_waves


def detect_dissipations(prev_pairs, current_pairs):
    """
    Detects dissipations where pairs from the previous state have disappeared entirely.
    Returns a list of dissipated pairs.
    """
    return [pair for pair in prev_pairs if pair not in current_pairs]


def analyze_sequence(data):
    """
    Analyzes the sequence of states to detect bifurcations, mergers, new waves, and dissipations.
    Prioritizes Bifurcations > Mergers > New Waves > Dissipations.
    """
    analysis = []
    prev_pairs = []
    all_seen_pairs = set()

    for idx, state in enumerate(data):
        current_pairs = extract_pairs(state)

        if idx == 0:
            all_seen_pairs.update(current_pairs)
            analysis.append({
                'Step': idx,
                'Previous State': None,
                'State': current_pairs,
                'Events': 'Initial State'
            })
        else:
            bifurcations = detect_bifurcations(prev_pairs, current_pairs, all_seen_pairs)
            if bifurcations:
                events = {'Bifurcations': bifurcations}
            else:
                mergers = detect_mergers(prev_pairs, current_pairs)
                if mergers:
                    events = {'Mergers': mergers}
                else:
                    new_waves = detect_new_waves(current_pairs, all_seen_pairs, prev_pairs)
                    if new_waves:
                        events = {'New Waves': new_waves}
                    else:
                        dissipations = detect_dissipations(prev_pairs, current_pairs)
                        events = {'Dissipations': dissipations} if dissipations else 'No Significant Event'

            analysis.append({
                'Step': idx,
                'Previous State': prev_pairs,
                'State': current_pairs,
                'Events': events
            })

        prev_pairs = current_pairs

    return analysis