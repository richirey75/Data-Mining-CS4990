import math

# DO NOT CHANGE THE FOLLOWING LINE
def apriori(itemsets, threshold):
    # DO NOT CHANGE THE PRECEDING LINE
    
    # calculate the minimum support count 
    min_support = (threshold / 100) * len(itemsets)

    # implement the function here (?)
    
    # Should return a list of pairs, where each pair consists of the frequent itemset and its support 
    # e.g. [(set(items), 0.7), (set(otheritems), 0.74), ...]
    return []

# function to generate candidate k-itemsets (k >= 2)
def apriori_gen(Lk_1, k):
    Ck = set() # set to store candidate k-itemsets (Ck)

    # compare every pair of (k-1) itemsets in Lk-1 to generate candidate k-itemsets
    for i in range(len(Lk_1)):
        for j in range(i+1, len(Lk_1)):
            l1, l2 = Lk_1[i][0], Lk_1[j][0] 

            if list(l1)[:k-2] == list(l2)[:k-2]:
                c = l1.union(l2) # join two itemsets to form a candidate (c)

                if not has_infrequent_subset(c, Lk_1):
                    Ck.add(c)
    return Ck

# function to check if candidate k-itemset has any infrequent (k-1) itemsets
def has_infrequent_subset(c, Lk_1):
    for i in range(len(c)):
        subset = frozenset(c).difference([c[i]])
        if not any(subset == itemset[0] for itemset in Lk_1):
            return True
    return False
    
# DO NOT CHANGE THE FOLLOWING LINE
def association_rules(itemsets, frequent_itemsets, metric, metric_threshold):
    # DO NOT CHANGE THE PRECEDING LINE
    
    # Should return a list of triples: condition, effect, metric value 
    # Each entry (c,e,m) represents a rule c => e, with the matric value m
    # Rules should only be included if m is greater than the given threshold.    
    # e.g. [(set(condition),set(effect),0.45), ...]
    return []
