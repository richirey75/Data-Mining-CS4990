import math
from collections import defaultdict

# These are suggested helper functions
# You can structure your code differently, but if you have
# trouble getting started, this might be a good starting point

# Create the decision tree recursively
def make_node(previous_ys, xs, ys, columns):
    # WARNING: lists are passed by reference in python
    # If you are planning to remove items, it's better 
    # to create a copy first
    columns = columns[:]

    # First, check the three termination criteria:
    
    # If there are no rows (xs and ys are empty): 
    #      Return a node that classifies as the majority class of the parent
    if not xs or not ys: 
        return {"type": "class", "class": majority(previous_ys)}
    
    # If all ys are the same:
    #      Return a node that classifies as that class 
    if same(ys):
        return {"type": "class", "class": ys[0]}
    
    # If there are no more columns left:
    #      Return a node that classifies as the majority class of the ys
    if not columns:
        return {"type": "class", "class": majority(ys)}

    # Otherwise:
    # Compute the entropy of the current ys 
    curr_entropy = entropy(ys)
    best_gain = -1
    best_column = None
    best_splits = None

    # For each column:
    #     Perform a split on the values in that column 
    #     Calculate the entropy of each of the pieces
    #     Compute the overall entropy as the weighted sum 
    #     The gain of the column is the difference of the entropy before
    #        the split, and this new overall entropy 
    # Select the column with the highest gain, then:
    # Split the data along the column values and recursively call 
    #    make_node for each piece 
    # Create a split-node that splits on this column, and has the result 
    #    of the recursive calls as children.
    
    for column in columns:
        splits = {}
        for i, value in enumerate(xs):
            if value[column] not in splits:
                splits[value[column]] = {"xs": [], "ys": []}
            splits[value[column]]["xs"].append(xs[i])
            splits[value[column]]["ys"].append(ys[i])

        total = len(ys)
        split_entropy = 0
        for split in splits.values():
            weight = len(split["ys"]) / total
            split_entropy += weight * entropy(split["ys"])
        
        gain = curr_entropy - split_entropy
        if gain > best_gain:
            best_gain = gain
            best_column = column
            best_splits = splits

    if best_gain <= 0:
        return {"type": "class", "class": majority(ys)}
    
    columns.remove(best_column)

    node = {"type": "split", "split": best_column, "children": {}}
    for value, split_data in best_splits.items():
        node["children"][value] = make_node(
            ys, split_data["xs"], split_data["ys"], columns
        )

    return node
    

# Determine if all values in a list are the same 
# Useful for the second basecase above
def same(values):
    if not values: # handle case when input list values is empty []
        return True
    # if there are values:
    # pick the first, check if all other are the same 
    first = values[0]
    return all(value == first for value in values)

    
# Determine how often each value shows up 
# in a list; this is useful for the entropy
# but also to determine which values is the 
# most common
def counts(values):
    result = {}
    for value in values:
        if value not in result:
            result[value] = 0
        result[value] += 1
    return result
   

# Return the most common value from a list 
# Useful for base cases 1 and 3 above.
def majority(values):
    freq = counts(values)
    return max(freq, key=freq.get)
    

# Calculate the entropy of a set of values 
# First count how often each value shows up 
# When you divide this value by the total number 
# of elements, you get the probability for that element 
# The entropy is the negation of the sum of p*log2(p) 
# for all these probabilities.
def entropy(values):
    freq = counts(values)
    total = len(values)
    return -sum((count / total) * math.log2(count / total) for count in freq.values())

# This is the main decision tree class 
# DO NOT CHANGE THE FOLLOWING LINE
        # To classify using the tree:
        # Start with the root as the "current" node
        # As long as the current node is an interior node (type == "split"):
        #    get the value of the attribute the split is performed on 
        #    select the child corresponding to that value as the new current node 
        
        # NOTE: In some cases, your tree may not have a child for a particular value 
        #       In that case, return the majority value (self.majority) from the training set 
        
        # IMPORTANT: You have to perform this classification *for each* element in x 
        
        # Note that the result is a list of predictions, one for each x-value

    # DO NOT CHANGE THE FOLLOWING LINE
class DecisionTree:
# DO NOT CHANGE THE PRECEDING LINE
# replace all attributes after tree={} with None to test without pruning
 #Uncomment For Pruning
    def __init__(self, tree={}, pruning_type = 'yes', max_depth=2, min_entropy = 0.5):
        self.tree = tree
        self.max_depth = max_depth
        self.min_entropy = min_entropy
        self.pruning_type = pruning_type
  
    def prune(self, node, current_level, max_depth, min_entropy):
        if max_depth is None:
            return node
        if self.pruning_type == "yes":
            # if leaf node is found
            if node["type"] == "class":
                return node
            if current_level >= max_depth:
                return {"type": "class", "class": majority(node)}
            if entropy(node) <= min_entropy:
                return {"type": "class", "class": majority(node)}
        if "children" in node:
            for value, child in node["children"].items():
                node["children"][value] = self.prune(child, current_level + 1, max_depth, min_entropy)
        return node

#Comment For Pruning
   # def __init__(self, tree={}):
        #self.tree = tree

    # DO NOT CHANGE THE FOLLOWING LINE    0
    def fit(self, x, y):
    # DO NOT CHANGE THE PRECEDING LINE
    
        self.majority = majority(y)
        self.tree = make_node(y, x, y, list(range(len(x[0]))))
        self.tree = self.prune(self.tree, current_level=0, max_depth=self.max_depth, min_entropy=self.min_entropy) #Uncomment For Pruning
        
            
    # DO NOT CHANGE THE FOLLOWING LINE    
    def predict(self, x):
    # DO NOT CHANGE THE PRECEDING LINE    
        if not self.tree:
            return None

        predictions = []
        for row in x:
            node = self.tree
            while node["type"] == "split":
                column = node["split"]
                value = row[column]
                if value in node["children"]:
                    node = node["children"][value]
                else:
                    node = {"type": "class", "class": self.majority}
            predictions.append(node["class"])
        return predictions
       
    def to_dict(self):
    # DO NOT CHANGE THE PRECEDING LINE
        # change this if you store the tree in a different format
        return self.tree
        #Pruning version 




#The indicated calculate_performance function should take the actual and the predicted y-values 
# and compute the accuracy, precision and recall of the classifier and 
# print them (check the test framework for an example of such a function for the binary case).

def calculate_performance(actual, predicted):
    """
    Calculate and print accuracy, precision, and recall for the classifier.
    Handles both binary and multi-class scenarios.
    """
    # Initialize counts
    class_counts = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "tn": 0})
    
    # Identify all unique classes
    classes = set(actual)
    
    # Compute true positives, false positives, false negatives, and true negatives per class
    for i in range(len(actual)):
        for cls in classes:
            if actual[i] == cls and predicted[i] == cls:
                class_counts[cls]["tp"] += 1
            elif actual[i] != cls and predicted[i] == cls:
                class_counts[cls]["fp"] += 1
            elif actual[i] == cls and predicted[i] != cls:
                class_counts[cls]["fn"] += 1
            elif actual[i] != cls and predicted[i] != cls:
                class_counts[cls]["tn"] += 1
    
    # Compute metrics per class
    precisions, recalls = [], []
    print("Performance metrics per class:")
    for cls in classes:
        tp = class_counts[cls]["tp"]
        fp = class_counts[cls]["fp"]
        fn = class_counts[cls]["fn"]
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        precisions.append(precision)
        recalls.append(recall)
        
        print(f"Class {cls}:")
        print(f"  Precision: {precision:.2f}")
        print(f"  Recall: {recall:.2f}")
    
    # Compute overall metrics
    accuracy = sum(1 for i in range(len(actual)) if actual[i] == predicted[i]) / len(actual)
    weighted_precision = sum(precisions) / len(classes)
    weighted_recall = sum(recalls) / len(classes)
    
    print("\nOverall performance metrics:")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Weighted Precision: {weighted_precision:.2f}")
    print(f"Weighted Recall: {weighted_recall:.2f}")

# --- classification ---
# classifier = DecisionTree()
# classifier.fit(train_x, train_y)
# train_y_hat = classifier.predict(train_x)
# calculate_performance(train_y, train_y_hat)
# validation_y_hat = classifier.predict(validation_x)
#pull classifier decision tree 
# calculate_performance(validation_y, validation_y_hat)
