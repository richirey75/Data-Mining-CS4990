import random
import csv
from classification import DecisionTree,calculate_performance

# Load data from CSV
def load_data(file_path):
    dataset = []
    with open(file_path, "r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            # Parse necessary fields
            dataset.append({
                "energy": float(row["energy"]),
                "valence": float(row["valence"]),
                "danceability": 1 if float(row["danceability"]) > 0.5 else 0,  # Danceable threshold
                # Add other fields if needed
            })
    return dataset
#change it to train test validate
# Split data into training and validation sets
def split_data(dataset, train_ratio=0.75):
    random.shuffle(dataset)
    split_point = int(len(dataset) * train_ratio)
    train_set = dataset[:split_point]
    validation_set = dataset[split_point:]
    return train_set, validation_set

# Prepare data for the decision tree
def prepare_data(dataset):
    xs = [[row["energy"], row["valence"]] for row in dataset]  # Use energy and valence
    ys = [row["danceability"] for row in dataset]
    return xs, ys

# Main workflow
def main():
    # Load and prepare data
    file_path = "combined_tracks.csv"  # Replace with your dataset file path
    dataset = load_data(file_path)
    
    # Split into training and validation sets
    train_set, validation_set = split_data(dataset)
    train_x, train_y = prepare_data(train_set)
    validation_x, validation_y = prepare_data(validation_set)
    
    # Train the decision tree classifier
    classifier = DecisionTree()
    classifier.fit(train_x, train_y)
    
    # Evaluate on training data
    train_y_hat = classifier.predict(train_x)
    print("Training Performance:")
    calculate_performance(train_y, train_y_hat)
    
    # Evaluate on validation data
    validation_y_hat = classifier.predict(validation_x)
    print("\nValidation Performance:")
    calculate_performance(validation_y, validation_y_hat)
    
    # Print decision tree
    print("\nDecision Tree Structure:")
    print(classifier.to_dict())

# Run the program
if __name__ == "__main__":
    main()
