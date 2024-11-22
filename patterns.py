import math
import itertools
from itertools import combinations

# DO NOT CHANGE THE FOLLOWING LINE
def apriori(itemsets, threshold):
    # DO NOT CHANGE THE PRECEDING LINE
    
    # calculate the minimum support count 
    # min_support = (threshold / 100) * len(itemsets)

    # find frequent 1-itemsets
    L = find_frequent_1_itemsets(itemsets, threshold)

    k = 2 # start with 2-itemsets (pairs of items)
    all_freq_items = L.copy()

    while True:
        Ck = apriori_gen(L, k) # generate k-itemsets from frequent (k-1) itemsets

        if not Ck: # no candidates, stop
            break

        # count support of candidate k-itemsets by checking their occurence in the itemsets
        freq_k = []
        for c in Ck:
            count = sum(1 for itemset in itemsets if set(c).issubset(itemset))
            support = count / len (itemsets)
            if support >= threshold:
                freq_k.append((c, support))
        
        # find frequent k-itemsets, add to L 
        if freq_k:
            L = freq_k
            all_freq_items.extend(freq_k)
        else: 
            break
        
        k+=1
    
    # Should return a list of pairs, where each pair consists of the frequent itemset and its support 
    # e.g. [(set(items), 0.7), (set(otheritems), 0.74), ...]
    return [(set(itemset), support) for itemset, support in all_freq_items]

# function to find frequent 1-itemsets
def find_frequent_1_itemsets(itemsets, min_support):
    item_counts = {} 
    for itemset in itemsets:
        for item in itemset:
            # count how many times each item appears in the dataset
            if item in item_counts:
                item_counts[item] += 1
            else:
                item_counts[item] = 1
    min_count = min_support * len(itemsets)
    return [(tuple([item]), count / len(itemsets)) for item, count in item_counts.items() if count >= min_count]

# function to generate candidate k-itemsets (k >= 2)
def apriori_gen(Lk_1, k):
    Ck = [] # list to store candidate k-itemsets (Ck)

    # compare every pair of (k-1) itemsets in Lk-1 to generate candidate k-itemsets
    for i in range(len(Lk_1)):
        for j in range(i+1, len(Lk_1)):
            l1, l2 = sorted(Lk_1[i][0]), sorted(Lk_1[j][0])

            if l1[:k-2] == l2[:k-2]:
                c = tuple(sorted(set(Lk_1[i][0]).union(Lk_1[j][0]))) # join two itemsets to form a candidate (c)

                if not has_infrequent_subset(c, Lk_1):
                    Ck.append(c)
    return Ck

# function to check if candidate k-itemset has any infrequent (k-1) itemsets
def has_infrequent_subset(c, Lk_1):
    for subset in combinations(c, len(c) - 1):
        if not any(set(subset) == set(itemset[0]) for itemset in Lk_1):
            return True
    return False
    
# DO NOT CHANGE THE FOLLOWING LINE
def association_rules(itemsets, frequent_itemsets, metric, metric_threshold):
    # DO NOT CHANGE THE PRECEDING LINE
    
    # Should return a list of triples: condition, effect, metric value 
    # Each entry (c,e,m) represents a rule c => e, with the matric value m
    # Rules should only be included if m is greater than the given threshold.    
    # e.g. [(set(condition),set(effect),0.45), ...]

    # initialize rules to be returned as a list
    rules = []

    # for each frequent itemset, generate all nonempty subsets of each frequent itemset
    for (itemset, support) in frequent_itemsets:
        # generate subsets of itemset to be converted into antecedents
        for r in range(1, len(itemset)):
            for subset in itertools.combinations(itemset, r):
                antecedent = set(subset)
                consequent = itemset - antecedent

                #check if consequent is empty
                if not consequent:
                    continue

                # calculate support of antecedent
                antecedent_support = find_support(antecedent, itemsets)

                if antecedent_support == 0:
                    continue;

                # calculate metrics
                confidence = support / antecedent_support
                lift = confidence / find_support(consequent, itemsets)
                kulczynski = 0.5 * (confidence + (support / find_support(consequent, itemsets)))
                cosine = support / math.sqrt(antecedent_support * find_support(consequent, itemsets))
                max_conf = max(confidence, (support / find_support(consequent, itemsets)))
                all_conf = support / max(antecedent_support, find_support(consequent, itemsets))

                # determine if the rule passes the metric threshold
                if metric == "confidence":
                    metric_value = confidence
                elif metric == "lift":
                    metric_value = lift
                elif metric == "all":
                    metric_value = all_conf;
                elif metric == "max":
                    metric_value = max_conf;
                elif metric == "kulczynski":
                    metric_value = kulczynski
                elif metric == "cosine":
                    metric_value = cosine
                
                if metric_value >= metric_threshold:
                    rules.append((set(antecedent), set(consequent), metric_value))
    return rules

# calculate support, represented as a percentage
def find_support(antecedent, itemsets):
    count = sum(1 for itemset in itemsets if set(antecedent).issubset(itemset))
    support = count / len(itemsets)
    return support