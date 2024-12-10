import math

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
class DecisionTree:
# DO NOT CHANGE THE PRECEDING LINE
    def __init__(self, tree={}):
        self.tree = tree
    
    # DO NOT CHANGE THE FOLLOWING LINE    
    def fit(self, x, y):
    # DO NOT CHANGE THE PRECEDING LINE
    
        self.majority = majority(y)
        self.tree = make_node(y, x, y, list(range(len(x[0]))))
        
    # DO NOT CHANGE THE FOLLOWING LINE    
    def predict(self, x):
    # DO NOT CHANGE THE PRECEDING LINE    
        if not self.tree:
            return None

        # To classify using the tree:
        # Start with the root as the "current" node
        # As long as the current node is an interior node (type == "split"):
        #    get the value of the attribute the split is performed on 
        #    select the child corresponding to that value as the new current node 
        
        # NOTE: In some cases, your tree may not have a child for a particular value 
        #       In that case, return the majority value (self.majority) from the training set 
        
        # IMPORTANT: You have to perform this classification *for each* element in x 
        
        # Note that the result is a list of predictions, one for each x-value
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


    # DO NOT CHANGE THE FOLLOWING LINE
    def to_dict(self):
    # DO NOT CHANGE THE PRECEDING LINE
        # change this if you store the tree in a different format
        return self.tree

# --- classification ---
# classifier = DecisionTree()
# classifier.fit(train_x, train_y)
# train_y_hat = classifier.predict(train_x)
# calculate_performance(train_y, train_y_hat)
# validation_y_hat = classifier.predict(validation_x)
# calculate_performance(validation_y, validation_y_hat)