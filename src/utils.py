import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sys
import os
import scipy.cluster.hierarchy as sch
from typing import List
import scipy.stats as stats 
import itertools
import functools
from collections import Counter

######################################################
### Utility functions to make the notebook cleaner ###
######################################################


def count2percentage(category: int, total: int) -> float:
    return (category / total) * 100


def series2list(series: pd.Series) -> list:
    return series.tolist()


def default_plotting_params() -> None:
    sns.set()
    sns.set_style('white')
    params = {'legend.fontsize': 'xx-large',
            'text.usetex':False,
            'axes.labelsize': 'xx-large',
            'axes.titlesize':'xx-large',
            'xtick.labelsize':'x-large',
            'ytick.labelsize':'x-large'}
    plt.rcParams.update(params)


def column_titles() -> list:
    column_1_title = "Who do you defer to most on AI timelines?"
    column_2_title = "Who do you defer to second-most on AI timelines?"
    column_3_title = "Who do you defer to third-most on AI timelines?" 
    return [column_1_title, column_2_title, column_3_title]


def load_data():
    timelines_deference = pd.read_csv("data/timelines_deference_survey.csv")
    timelines_deference_cleaned = pd.read_csv("data/timelines_deference_cleaned_.csv")
    timelines_deference_cleaned_sam = pd.read_csv("data/timelines_deference_cleaned_sam_edit.csv") 
    timelines_deference_self_other = pd.read_csv("data/timelines_deferrence_self_other_cleaning_.csv")
    #timelines_deference_category = pd.read_csv("data/timelines_deferrence_category_cleaning_.csv")
    timelines_deference_category = pd.read_csv("data/timelines_deference_category_cleaning_revised.csv")

    return timelines_deference, timelines_deference_cleaned, timelines_deference_cleaned_sam, timelines_deference_self_other, timelines_deference_category


############################################################
### Plot the data for "inside view" vs "other" deference ###
############################################################

def generate_data_self_vs_other(data: pd.core.frame.DataFrame) -> list:
    #Here, the data is always "timelines_deference_self_other"
    # Self
    self_deference_first = np.sum(data[column_titles()[0]].str.count("s"))
    self_deference_second = np.sum(data[column_titles()[1]].str.count("s"))
    self_deference_third = np.sum(data[column_titles()[2]].str.count("s"))
    # Other
    other_deference_first = np.sum(data[column_titles()[0]].str.count("o"))
    other_deference_second = np.sum(data[column_titles()[1]].str.count("o"))
    other_deference_third = np.sum(data[column_titles()[2]].str.count("o"))
    # Collect across rankings
    self_deference = [self_deference_first, self_deference_second, self_deference_third]
    other_deference = [other_deference_first, other_deference_second, other_deference_third]
    total_respondents = self_deference + other_deference
    
    return self_deference, other_deference, total_respondents


def plot_inside_view_vs_other(data: pd.core.frame.DataFrame) -> None:
    #Here, the data is always "timelines_deference_self_other"
    plt.rcParams['figure.figsize'] = (12,9)
    self_deference, other_deference, total_respondents = generate_data_self_vs_other(data)
    
    plt.bar(x=[0,3,6], height=[count2percentage(c, sum(total_respondents)) for c in self_deference], \
        color='grey', alpha=0.45, label="inside view")

    plt.bar(x=[1,4,7], height=[count2percentage(c, sum(total_respondents)) for c in other_deference], \
            color='purple', alpha=0.45, label="other")

    plt.legend(labelcolor="linecolor")
    plt.ylabel("% of total respondents")

    x_labels = ["first-place \ndeference", "second-place \ndeference", "third-place \ndeference"]
    plt.xticks(np.arange(len(x_labels))*3.25, x_labels, rotation=0)
    plt.title("Deference – inside view vs. other")
    sns.despine() 

############################################
### Plot the data for deference category ###
############################################


def generate_data_category(data: pd.core.frame.DataFrame) -> list:
    #Here, data is always timelines_deference_category
    
    # Category == individual
    individual_deference_first = np.sum(data[column_titles()[0]].str.count("i"))
    individual_deference_second = np.sum(data[column_titles()[1]].str.count("i"))
    individual_deference_third = np.sum(data[column_titles()[2]].str.count("i"))
    # Category == group
    group_deference_first = np.sum(data[column_titles()[0]].str.count("g"))
    group_deference_second = np.sum(data[column_titles()[1]].str.count("g"))
    group_deference_third = np.sum(data[column_titles()[2]].str.count("g"))
    # Category == organisation
    organisation_deference_first = np.sum(data[column_titles()[0]].str.count("o"))
    organisation_deference_second = np.sum(data[column_titles()[1]].str.count("o"))
    organisation_deference_third = np.sum(data[column_titles()[2]].str.count("o"))
    # Collect
    individual_deference = [individual_deference_first, individual_deference_second, individual_deference_third]
    group_deference = [group_deference_first, group_deference_second, group_deference_third]
    organisation_deference = [organisation_deference_first, organisation_deference_second, organisation_deference_third]
    # Total
    total_response_category = individual_deference + group_deference + organisation_deference

    return individual_deference, group_deference, organisation_deference, total_response_category
    
    
def plot_deference_by_category_LEGACY(data: pd.core.frame.DataFrame) -> None:
    #Here, data is always timelines_deference_category
    plt.rcParams['figure.figsize'] = (12,9)
    individual_deference, group_deference, organisation_deference, total_response_category = generate_data_category(data)
    plt.bar(x=[0,4,8], height=[count2percentage(c, sum(total_response_category)) for c in individual_deference], \
       label="individual", color="grey", alpha=0.45)
    plt.bar(x=[1,5,9], height=[count2percentage(c, sum(total_response_category)) for c in group_deference], \
           label="group", color="purple", alpha=0.45)
    plt.bar(x=[2,6,10], height=[count2percentage(c, sum(total_response_category)) for c in organisation_deference], \
           label="organisation", color="darkgreen", alpha=0.45)
    plt.legend(labelcolor="linecolor")
    plt.ylabel("% of total respondents")
    x_labels = ["first-place \ndeference", "second-place \ndeference", "third-place \ndeference"]
    plt.xticks([1,5,9], x_labels, rotation=0)
    plt.title("Deference by category")
    sns.despine()
   

def plot_deference_by_category(data: pd.core.frame.DataFrame) -> None:
    #Here, data is always timelines_deference_category
    plt.rcParams['figure.figsize'] = (12,9)
    individual_deference, group_deference, organisation_deference, total_response_category = generate_data_category(data)
    plt.bar(x=[0,3,6], height=[count2percentage(c, sum(total_response_category)) for c in individual_deference], \
       label="individual", color="grey", alpha=0.45)
    #plt.bar(x=[1,5,9], height=[count2percentage(c, sum(total_response_category)) for c in group_deference], \
           #label="group", color="purple", alpha=0.45)
    plt.bar(x=[1,4,7], height=[count2percentage(c, sum(total_response_category)) for c in organisation_deference], \
           label="organisation", color="darkgreen", alpha=0.45)
    plt.legend(labelcolor="linecolor")
    plt.ylabel("% of total respondents")
    x_labels = ["first-place \ndeference", "second-place \ndeference", "third-place \ndeference"]
    plt.xticks([0.5, 3.5, 6.5], x_labels, rotation=0)
    plt.title("Deference by category")
    sns.despine()


##################################################
### Plot the responses for each deference rank ###
##################################################


def generate_deference_counts(data: pd.core.frame.DataFrame) -> dict:
    
    #Here, data is always timelines_deference_cleaned
    
    # Create an empty dictionary to store the counts
    counts = {}
    # Iterate through each column in the DataFrame
    for column in data:
        # Count the occurrences of each value in the column, ignoring missing values
        column_counts = data[column].value_counts(dropna=True)
        # Store the resulting counts in the dictionary
        counts[column] = column_counts
    #print(counts)
    return counts


def visualise_deference_responses(deference_rank: int, data: pd.core.frame.DataFrame, include_self_responses: bool) -> None:
    
    counts = generate_deference_counts(data)
    
    if deference_rank == 1:
        cc_1 = counts[column_titles()[0]]
        #cc_1 = lambda: cc_1 if include_self_responses else cc_1.drop("Self")
        plt.bar(x=np.arange(len(cc_1)), height=cc_1, edgecolor='k', linewidth=3, color='w')
        x_labels = list(cc_1.keys())
        plt.xticks(np.arange(len(x_labels)), x_labels, rotation=90)
        plt.ylabel("# responses")
        plt.title("Who do you defer to most on AI timelines?")

    elif deference_rank == 2:
        cc_2 = counts[column_titles()[1]]
        plt.bar(x=np.arange(len(cc_2)), height=cc_2, edgecolor='k', linewidth=3, color='w')
        x_labels = list(cc_2.keys())
        plt.xticks(np.arange(len(x_labels)), x_labels, rotation=90)
        plt.ylabel("# responses")
        plt.title("Who do you defer to second-most on AI timelines?")
        
    elif deference_rank == 3:
        cc_2 = counts[column_titles()[2]]
        plt.bar(x=np.arange(len(cc_2)), height=cc_2, edgecolor='k', linewidth=3, color='w')
        x_labels = list(cc_2.keys())
        plt.xticks(np.arange(len(x_labels)), x_labels, rotation=90)
        plt.ylabel("# responses")
        plt.title("Who do you defer to third-most on AI timelines?")
        
    else:
        raise ValueError("deference rank must be 1, 2, or 3")
    
    #plt.xlim()
    sns.despine()


##################################################
### Generate some weighted score for deference ###
##################################################


def combine_func(x, y):
    if pd.isnull(x):
        return y
    if pd.isnull(y):
        return x
    return x + y


def visualise_weighted_deference_(cc_1, cc_2, cc_3, include_self_responses: bool, xlim_: float):
    default_plotting_params()
    plt.rcParams['figure.figsize'] = (18,10)
    result_scaled = functools.reduce(lambda x, y: x.combine(y, combine_func), [(3*cc_1), (2*cc_2), cc_3])
    result_scaled_ = result_scaled.sort_values(ascending=False)
    
    if include_self_responses:
        result_scaled_drop_self = result_scaled_
    else:
        result_scaled_drop_self = result_scaled_.drop("Inside view")

    threshold = len(result_scaled_drop_self)

    plt.bar(x=np.arange(len(result_scaled_drop_self[:threshold])), height=result_scaled_drop_self[:threshold],\
           edgecolor='k', linewidth=2, color='w')

    x_labels = result_scaled_drop_self.index.tolist()[:threshold]
    plt.xticks([elem for elem in np.arange(threshold)], x_labels, rotation=90)
    plt.xlim(-1,(xlim_ + 0.5))
    plt.ylabel("deference score \n(arbitrary units)")

    sns.despine()


def visualise_unweighted_deference_(cc_1, cc_2, cc_3, include_self_responses: bool, xlim_: float):
    default_plotting_params()
    plt.rcParams['figure.figsize'] = (18,10)
    result_scaled = functools.reduce(lambda x, y: x.combine(y, combine_func), [(cc_1), (cc_2), (cc_3)])
    result_scaled_ = result_scaled.sort_values(ascending=False)
    
    if include_self_responses:
        result_scaled_drop_self = result_scaled_
    else:
        result_scaled_drop_self = result_scaled_.drop("Inside view")

    threshold = len(result_scaled_drop_self)

    plt.bar(x=np.arange(len(result_scaled_drop_self[:threshold])), height=result_scaled_drop_self[:threshold],\
           edgecolor='k', linewidth=2, color='w')

    x_labels = result_scaled_drop_self.index.tolist()[:threshold]
    plt.xticks([elem for elem in np.arange(threshold)], x_labels, rotation=90)
    plt.xlim(-1,(xlim_ + 0.5))
    plt.ylabel("deference responses across ranks")

    sns.despine()
    

############################################################################
### Plot clusters of individuals and organisations with correlated views ###
############################################################################


def plot_clusters(data: pd.core.series.Series, include_total: bool) -> None:
    #data is always result_ in the case of this dataset...
    # Specify the two clusters of people/organisations with correlated views
    # c.f. Sam Clarke's discussion
    results_group_1 = [data["Ajeya Cotra"], data["Paul Christiano"], data["Holden Karnofsky"],\
                  data["Bioanchors"]]
    results_group_2 = [data["Eliezer Yudkowsky"], data["MIRI"]]
    x_labels_group_1 = ["Ajeya Cotra", "Holden Karnofsky", "Paul Christiano", "Bioanchors"]
    x_labels_group_2 = ["Eliezer Yudkowsky", "MIRI"]
    
    if include_total:
        x_labels_group_1_total = ["Total"] + x_labels_group_1
        x_labels_group_2_total = ["Total"] + x_labels_group_2
        results_group_1_total = [sum(results_group_1)] + results_group_1
        results_group_2_total = [sum(results_group_2)] + results_group_2
        clrs = ["darkmagenta", "purple", "purple"]
        clrs_1 = ["grey", "lightgrey", "lightgrey", "lightgrey", "lightgrey", "lightgrey"]
        clrs_2 = ["goldenrod", "papayawhip", "papayawhip"]
        plt.bar(x=np.arange(len(x_labels_group_1_total)), height=results_group_1_total, color=clrs_1, alpha=1)
        plt.bar(x=np.arange(len(x_labels_group_2_total))+7, height=results_group_2_total, color=clrs_2, alpha=1)
        x_labels = x_labels_group_1_total + [" "] + x_labels_group_2_total
        plt.xticks(np.arange(len(x_labels)), x_labels, rotation=90)
        plt.ylabel("# responses across ranks")
        # Demarcate group 1
        plt.axhline(y=100, xmin=0.05, xmax=0.55, c='grey')
        plt.axvline(x=-0.35, ymin=0.9, ymax=0.95, c='grey')
        plt.axvline(x=5.05, ymin=0.9, ymax=0.95, c='grey')
        plt.axvline(x=2.35, ymin=0.9525, ymax=0.97, c='grey', label="cluster 1")
        # Demarcate group 2
        plt.axhline(y=100, xmin=0.65, xmax=0.95, c='goldenrod')
        plt.axvline(x=6.1, ymin=0.9, ymax=0.95, c='goldenrod')
        plt.axvline(x=9.35, ymin=0.9, ymax=0.95, c='goldenrod')
        plt.axvline(x=(9.35 + 6.1)/2, ymin=0.9525, ymax=0.97, c='goldenrod', label="cluster 2")
        # Label the clusters
        plt.annotate("cluster 1", (1.85, 103), xytext=None, fontsize=16, color='grey')
        plt.annotate("cluster 2", (7.3, 103), xytext=None, fontsize=16, color='goldenrod')
        sns.despine();
    
    else:
        plt.bar(x=np.arange(len(x_labels_group_1)), height=results_group_1, color="grey", alpha=0.5)
        plt.bar(x=np.arange(len(x_labels_group_2))+6, height=results_group_2, color="goldenrod", alpha=0.5)
        x_labels = x_labels_group_1 + [" "] + x_labels_group_2
        plt.xticks(np.arange(len(x_labels)), x_labels, rotation=90)
        # Demarcate group 1
        plt.axhline(y=60, xmin=0.05, xmax=0.5, c='grey')
        plt.axvline(x=-0.35, ymin=0.9, ymax=0.95, c='grey')
        plt.axvline(x=3.95, ymin=0.9, ymax=0.95, c='grey')
        plt.axvline(x=1.8, ymin=0.9525, ymax=0.97, c='grey', label="cluster 1")
        # Demarcate group 2
        plt.axhline(y=60, xmin=0.665, xmax=0.95, c='goldenrod')
        plt.axvline(x=5.525, ymin=0.9, ymax=0.95, c='goldenrod')
        plt.axvline(x=8.3, ymin=0.9, ymax=0.95, c='goldenrod')
        plt.axvline(x=(8.3 + 5.525)/2, ymin=0.9525, ymax=0.97, c='goldenrod', label="cluster 2")
        # Label the clusters
        plt.annotate("cluster 1", (1.4, 62), xytext=None, fontsize=16, color='grey')
        plt.annotate("cluster 2", (6.5, 62), xytext=None, fontsize=16, color='goldenrod')
        plt.ylabel("# responses across ranks")
        sns.despine();



def plot_clusters_(data: pd.core.series.Series, include_total: bool) -> None:
    #data is always result_ in the case of this dataset...
    # Specify the two clusters of people/organisations with correlated views
    # c.f. Sam Clarke's discussion
    results_group_1 = [data["Ajeya Cotra"], data["Paul Christiano"], data["Holden Karnofsky"],\
                  data["Bioanchors"]]
    results_group_2 = [data["Eliezer Yudkowsky"], data["MIRI"]]
    x_labels_group_1 = ["Ajeya Cotra", "Holden Karnofsky", "Paul Christiano", "Bioanchors"]
    x_labels_group_2 = ["Eliezer Yudkowsky", "MIRI"]
    
    if include_total:
        x_labels_group_1_total = ["Total"] + x_labels_group_1
        x_labels_group_2_total = ["Total"] + x_labels_group_2
        results_group_1_total = [sum(results_group_1)] + results_group_1
        results_group_2_total = [sum(results_group_2)] + results_group_2
        clrs = ["darkmagenta", "purple", "purple"]
        clrs_1 = ["grey", "lightgrey", "lightgrey", "lightgrey", "lightgrey", "lightgrey"]
        clrs_2 = ["goldenrod", "papayawhip", "papayawhip"]
        plt.bar(x=np.arange(len(x_labels_group_1_total)), height=results_group_1_total, color=clrs_1, alpha=1)
        plt.bar(x=np.arange(len(x_labels_group_2_total))+6, height=results_group_2_total, color=clrs_2, alpha=1)
        x_labels = x_labels_group_1_total + [" "] + x_labels_group_2_total
        plt.xticks(np.arange(len(x_labels)), x_labels, rotation=90)
        plt.ylabel("# responses across ranks")
        # Demarcate group 1
        plt.axhline(y=100, xmin=0.05, xmax=0.5, c='grey')
        plt.axvline(x=-0.35, ymin=0.9, ymax=0.95, c='grey')
        plt.axvline(x=4.475, ymin=0.9, ymax=0.95, c='grey')
        plt.axvline(x=((4.475 - 0.35)/2), ymin=0.9525, ymax=0.97, c='grey', label="cluster 1")
        # Demarcate group 2
        plt.axhline(y=100, xmin=0.6, xmax=0.95, c='goldenrod')
        plt.axvline(x=5.55, ymin=0.9, ymax=0.95, c='goldenrod')
        plt.axvline(x=9.34, ymin=0.9, ymax=0.95, c='goldenrod')
        plt.axvline(x=(9.34 + 5.55)/2, ymin=0.9525, ymax=0.97, c='goldenrod', label="cluster 2")
        # Label the clusters
        plt.annotate("cluster 1", (1.75, 103), xytext=None, fontsize=16, color='grey')
        plt.annotate("cluster 2", (7.15, 103), xytext=None, fontsize=16, color='goldenrod')
        sns.despine();
    
    else:
        plt.bar(x=np.arange(len(x_labels_group_1)), height=results_group_1, color="grey", alpha=0.5)
        plt.bar(x=np.arange(len(x_labels_group_2))+5, height=results_group_2, color="goldenrod", alpha=0.5)
        x_labels = x_labels_group_1 + [" "] + x_labels_group_2
        plt.xticks(np.arange(len(x_labels)), x_labels, rotation=90)
        # Demarcate group 1
        plt.axhline(y=60, xmin=0.05, xmax=0.525, c='grey')
        plt.axvline(x=-0.365, ymin=0.9, ymax=0.95, c='grey')
        plt.axvline(x=3.2, ymin=0.9, ymax=0.95, c='grey')
        plt.axvline(x=((3.2 - 0.365)/2), ymin=0.9525, ymax=0.97, c='grey', label="cluster 1")
        # Demarcate group 2
        plt.axhline(y=60, xmin=0.7, xmax=0.95, c='goldenrod')
        plt.axvline(x=4.5, ymin=0.9, ymax=0.95, c='goldenrod')
        plt.axvline(x=6.37, ymin=0.9, ymax=0.95, c='goldenrod')
        plt.axvline(x=(6.37 + 4.5)/2, ymin=0.9525, ymax=0.97, c='goldenrod', label="cluster 2")
        # Label the clusters
        plt.annotate("cluster 1 (Open Philanthropy cluster)", (1.2, 62), xytext=None, fontsize=16, color='grey')
        plt.annotate("cluster 2 (MIRI cluster)", (5.2, 62), xytext=None, fontsize=16, color='goldenrod')
        plt.ylabel("# responses across ranks")
        sns.despine();



def plot_group_clusters(data: pd.core.series.Series, sort: bool) -> None:
    
    plt.rcParams['figure.figsize'] = (13,8)
    # Define groups to be plotted
    # Groups: OpenPhil, MIRI, Self, Forecasting, EveryoneElse
    results_group_open_phil = [data["Ajeya Cotra"], data["Paul Christiano"], data["Holden Karnofsky"],\
                                data["Bioanchors"]]
    results_group_miri = [data["Eliezer Yudkowsky"], data["MIRI"]]
    results_group_self = [data["Inside view"]]
    results_group_forecasting = [data["Samotsvety"], data["Metaculus"]]
    results_group_danielk = [data["Daniel Kokotajlo"]]
    # Create a partially complete list[list[int]] data structure to check the number of deferences left over
    result_groups = [results_group_open_phil, results_group_miri, results_group_self, \
                     results_group_forecasting, results_group_danielk]
    results_group_everyone_else = [(sum(data)) - (sum([sum(group) for group in result_groups]))]
    # Collect everything into one list (i.e. OpenPhil, MIRI, Self, Forecasting, EverythingElse clusters)
    result_groups_with_everyone_else = result_groups + [results_group_everyone_else]
    
    # Group labels
    #x_labels_group_1 = ["Ajeya Cotra", "Holden Karnofsky", "Paul Christiano", "Bioanchors"]
    #x_labels_group_2 = ["Eliezer Yudkowsky", "MIRI"]
    result_group_open_phil_label = ["Open Philanthropy"]
    result_group_miri_label = ["MIRI"]
    result_group_self_label = ["Self"]
    result_group_forecasting_label = ["Forecasting"]
    result_group_everyone_else_label = ["Everyone else"]
    result_group_danielk = ["Daniel Kokotajlo"]
    # Collect all labels into a list[str] for convenience
    result_group_labels_all = ["Open Philanthropy \n cluster", "MIRI \n cluster", "Inside view", "Samotsvety \n & Metaculus", "Everyone else"]
    result_group_labels_all_sorted = ["Open\nPhilanthropy \n cluster", "Everyone\nelse", "Inside\nview", "Daniel\nKokotajlo","MIRI \n cluster", " Samotsvety \n & Metaculus"]
    
    # Total counts to plot for the five selected categories
    total_counts_to_plot = [sum(group) for group in result_groups] + results_group_everyone_else
    total_counts_to_plot_sorted = sorted(total_counts_to_plot, reverse=True)
    
    if sort:
        clrs = ["darkred", "firebrick", "indianred", "lightcoral", "papayawhip", "linen"]
        plt.bar(x=np.arange(len(total_counts_to_plot)), height=sorted(total_counts_to_plot, reverse=True), \
               color=clrs, alpha=0.8, edgecolor="k")
        x_labels = result_group_labels_all_sorted
        plt.xticks(np.arange(len(x_labels)), x_labels, rotation=0)
        # Add counts to the bars for readability
        for i in range(len(total_counts_to_plot)):
            plt.annotate(total_counts_to_plot_sorted[i], xy=((int(i) - 0.1), total_counts_to_plot_sorted[i] + 1), size=18)
        # Labels and title
        plt.ylim(0, 89)
        plt.ylabel("# responses across ranks")
        #plt.xlabel("major clusters")
        plt.title("Deference responses for some influential categories")
        sns.despine()
        
    else:
        plt.bar(x=np.arange(len(total_counts_to_plot)), height=total_counts_to_plot, color='grey', alpha=0.2)
        x_labels = result_group_labels_all
        plt.xticks(np.arange(len(x_labels)), x_labels, rotation=0)
        #plt.axvline(1.5, linestyle="--", c='k')
        # Add counts to the bars for readability
        for i in range(len(total_counts_to_plot)):
            plt.annotate(total_counts_to_plot[i], xy=((int(i) - 0.1), total_counts_to_plot[i] + 1), size=18)

        # Labels and title
        plt.ylim(0, 89)
        plt.ylabel("# responses across ranks")
        #plt.xlabel("major clusters")
        plt.title("Deference responses for some influential categories")
        sns.despine()



def plot_group_clusters_first_rank_deference(data: pd.core.series.Series) -> None:

    _, _, timelines_deference_cleaned_sam_edit, _, _ = load_data()
    first_deference = series2list(timelines_deference_cleaned_sam_edit["Who do you defer to most on AI timelines?"])
    first_deference_nan_filter = [elem for elem in first_deference if type(elem) == str]
    first_deference_counts = Counter(first_deference_nan_filter)
    
    first_deference_open_phil_cluster = first_deference_counts["Ajeya Cotra"] + \
                                    first_deference_counts["Ajeya Cotra and Holden Karnofsky "] + \
                                    first_deference_counts["Paul Christiano"] + \
                                    first_deference_counts["Holden Karnofsky"] + \
                                    first_deference_counts["Bioanchors"]

    first_deference_miri_cluster = first_deference_counts["MIRI"] + first_deference_counts["Eliezer Yudkowsky"]
    first_deference_inside_view = first_deference_counts["Inside view"]
    first_deference_forecasting = first_deference_counts["Samotsvety"] + first_deference_counts["Metaculus"]
    first_deference_danielk = first_deference_counts["Daniel Kokotajlo"]
    first_deference_everyone_else = len(first_deference_nan_filter) - (first_deference_open_phil_cluster + \
                                                                       first_deference_miri_cluster + \
                                                                       first_deference_inside_view + \
                                                                       first_deference_forecasting + \
                                                                       first_deference_danielk)

    first_deference_responses = [first_deference_open_phil_cluster, first_deference_miri_cluster, \
                                first_deference_inside_view, first_deference_forecasting, \
                                 first_deference_danielk, first_deference_everyone_else]

    assert sum(first_deference_responses) == len(first_deference_nan_filter)
    
    plt.rcParams['figure.figsize'] = (14,9)
    default_plotting_params()
    # Collect all labels into a list[str] for convenience
    result_group_labels_all_sorted = ["Open\nPhilanthropy \n cluster", "Inside\nview", "Everyone\nelse", \
                                      "MIRI \n cluster", " Samotsvety \n & Metaculus", "Daniel\nKokotajlo"]

    clrs = ["darkred", "firebrick", "indianred", "lightcoral", "papayawhip", "linen"]

    plt.bar(x=np.arange(len(first_deference_responses)), height=sorted(first_deference_responses, reverse=True), \
           color=clrs, alpha=0.8, edgecolor="k")
    x_labels = result_group_labels_all_sorted
    plt.xticks(np.arange(len(x_labels)), x_labels, rotation=0)

    # Add counts to the bars for readability
    for i in range(len(first_deference_responses)):
        plt.annotate(sorted(first_deference_responses, reverse=True)[i], xy=((int(i) - 0.1), sorted(first_deference_responses, reverse=True)[i] + 1), size=18)

    # Labels and title
    plt.ylim(0, 39)
    plt.ylabel("# responses – \n Who do you defer to most on AI timelines?")
    #plt.xlabel("major clusters")
    plt.title("Deference responses for some influential categories")
    sns.despine();


def plot_group_clusters_deference_score(data: pd.core.series.Series) -> None:

    _, _, timelines_deference_cleaned_sam_edit, _, _ = load_data()
    counts = generate_deference_counts(timelines_deference_cleaned_sam_edit)
    result_scaled = functools.reduce(lambda x, y: x.combine(y, combine_func), [(3*counts[column_titles()[0]]), (2*counts[column_titles()[1]]), counts[column_titles()[2]]])
    result_scaled_ = result_scaled.sort_values(ascending=False)
    result_scaled_open_philanthropy_cluster = result_scaled_["Ajeya Cotra"] + \
                                            result_scaled_["Holden Karnofsky"] + \
                                            result_scaled_["Bioanchors"] + \
                                            result_scaled_["Paul Christiano"]

    result_scaled_miri_cluster = result_scaled_["MIRI"] + result_scaled["Eliezer Yudkowsky"]
    result_scaled_inside_view = result_scaled["Inside view"]
    result_scaled_forecasting_cluster = result_scaled_["Metaculus"] + result_scaled_["Samotsvety"]
    result_scaled_danielk = result_scaled_["Daniel Kokotajlo"]
    result_scaled_everyone_else = sum(result_scaled_) - (result_scaled_open_philanthropy_cluster + result_scaled_miri_cluster + result_scaled_inside_view + result_scaled_forecasting_cluster + result_scaled_danielk)

    result_scaled_all = [result_scaled_open_philanthropy_cluster, result_scaled_miri_cluster, \
                        result_scaled_inside_view, result_scaled_forecasting_cluster, \
                        result_scaled_danielk, result_scaled_everyone_else]

    assert sum(result_scaled_) == sum(result_scaled_all)
    plt.rcParams['figure.figsize'] = (14,9)
    default_plotting_params()
    result_group_labels_all_sorted = ["Open\nPhilanthropy \n cluster", "Everyone\nelse", \
                                      "Inside\nview", "Daniel\nKokotajlo", "MIRI \n cluster", " Samotsvety \n & Metaculus"]
    clrs = ["darkred", "firebrick", "indianred", "lightcoral", "papayawhip", "linen"]

    plt.bar(x=np.arange(len(result_scaled_all)), height=sorted(result_scaled_all, reverse=True), \
           color=clrs, alpha=0.8, edgecolor="k")
    x_labels = result_group_labels_all_sorted
    plt.xticks(np.arange(len(x_labels)), x_labels, rotation=0)

    # Add counts to the bars for readability
    for i in range(len(result_scaled_all)):
        plt.annotate(sorted(result_scaled_all, reverse=True)[i], xy=((int(i) - 0.1), sorted(result_scaled_all, reverse=True)[i] + 1), size=18)

    # Labels and title
    #plt.ylim(0, )
    plt.ylabel("deference score \n (arbitrary units)")
    #plt.xlabel("major clusters")
    plt.title("Deference score for some influential categories")
    sns.despine();

