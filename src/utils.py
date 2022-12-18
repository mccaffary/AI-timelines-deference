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
    timelines_deference_self_other = pd.read_csv("data/timelines_deferrence_self_other_cleaning_.csv")
    timelines_deference_category = pd.read_csv("data/timelines_deferrence_category_cleaning_.csv")

    return timelines_deference, timelines_deference_cleaned, timelines_deference_self_other, timelines_deference_category


#####################################################
### Plot the data for "self" vs "other" deference ###
#####################################################

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



def plot_self_vs_other(data: pd.core.frame.DataFrame) -> None:
    #Here, the data is always "timelines_deference_self_other"
    plt.rcParams['figure.figsize'] = (12,9)
    self_deference, other_deference, total_respondents = generate_data_self_vs_other(data)
    
    plt.bar(x=[0,3,6], height=[count2percentage(c, sum(total_respondents)) for c in self_deference], \
        color='grey', alpha=0.45, label="self")

    plt.bar(x=[1,4,7], height=[count2percentage(c, sum(total_respondents)) for c in other_deference], \
            color='purple', alpha=0.45, label="other")

    plt.legend(labelcolor="linecolor")
    plt.ylabel("% of total respondents")

    x_labels = ["first-place \ndeference", "second-place \ndeference", "third-place \ndeference"]
    plt.xticks(np.arange(len(x_labels))*3.25, x_labels, rotation=0)
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
    
    
def plot_deference_by_category(data: pd.core.frame.DataFrame) -> None:
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



def visualise_weighted_deference(cc_1, cc_2, cc_3):
    plt.rcParams['figure.figsize'] = (18,10)
    result_scaled = functools.reduce(lambda x, y: x.combine(y, combine_func), [(3*cc_1), (2*cc_2), cc_3])
    result_scaled_ = result_scaled.sort_values(ascending=False)
    result_scaled_drop_self = result_scaled_.drop("Self")

    threshold = len(result_scaled_drop_self)

    plt.bar(x=np.arange(len(result_scaled_drop_self[:threshold])), height=result_scaled_drop_self[:threshold],\
           edgecolor='k', linewidth=1, color='w')

    x_labels = result_scaled_drop_self.index.tolist()[:threshold]
    plt.xticks([elem for elem in np.arange(threshold)], x_labels, rotation=90)

    sns.despine()









def void():
    pass







