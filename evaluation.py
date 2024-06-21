import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_accuracy(df):
    df['TITLE_MODIFIED'] = df['TITLE'].str.replace('I', 'L')
    correct_predictions = df[df['TITLE_MODIFIED'] == df['DENOVO']]
    accuracy = len(correct_predictions) / len(df)
    return accuracy

def plot_distributions(df):
    plt.figure(figsize=(12, 5))

    # Plot Score distribution
    plt.subplot(1, 2, 1)
    sns.histplot(df['Score'], kde=True, bins=100)
    plt.title('Score Distribution')
    plt.xlabel('Score')
    plt.ylabel('Frequency')

    # Plot PPM Difference distribution
    plt.subplot(1, 2, 2)
    sns.histplot(df['PPM Difference'], kde=True, bins=30)
    plt.title('PPM Difference Distribution')
    plt.xlabel('PPM Difference')
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Evaluate predictions and plot distributions.')
    parser.add_argument('--novorst', required=True, help='Path to the TSV file containing predictions')

    args = parser.parse_args()

    # Read the TSV file
    df = pd.read_csv(args.novorst, sep='\t')

    # Calculate accuracy
    accuracy = calculate_accuracy(df)
    print(f'Accuracy: {accuracy:.2f}')

    # Plot distributions
    plot_distributions(df)

if __name__ == "__main__":
    main()
