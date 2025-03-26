import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np # Make sure numpy is imported
from sklearn.decomposition import PCA


def load_byte_data(directory):
    data = []
    labels = []
    for root, _, files in os.walk(directory):
        for file in files:
            filepath = os.path.join(root, file)
            if not "mysql" in filepath.lower() and not "postgres" in filepath.lower() and not "sqlserver" in filepath.lower():
                continue
            print(filepath)
            try:
                with open(filepath, 'rb') as f:
                    byte_data = f.read()
                    data.append(byte_data)
                    engine = os.path.basename(root)
                    label = engine
                    
                    # label = "unknown"
                    # if "mysql" in engine.lower():
                    #     label = "mysql"
                    # if "postgres" in engine.lower():
                    #     label = "postgres"
                    # if "sqlserver" in engine.lower():
                    #     label = "sqlserver"
                    print(label)
                    labels.append(label)
            except Exception as e:
                print(f"Error reading file {filepath}: {e}")
    return data, labels


if __name__ == "__main__":
    print("Hello!")

    train_dir = "/Users/raresraf/code/project-martial/dataset/packets/train"
    train_data, train_labels = load_byte_data(train_dir)

    test_dir = "/Users/raresraf/code/project-martial/dataset/packets/test"
    test_data, test_labels = load_byte_data(test_dir)

    def byte_ngram_tokenizer(byte_sequence):
        ngrams = [byte_sequence[i:i + 2] for i in range(len(byte_sequence) - 2)]
        return ngrams

    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer(tokenizer=byte_ngram_tokenizer, encoding='cp437')
    X_train = vectorizer.fit_transform(train_data)
    X_test = vectorizer.transform(test_data)

    # Train SVM classifier
    classifier = SVC()
    classifier.fit(X_train, train_labels)
    predictions = classifier.predict(X_test)
    accuracy = accuracy_score(test_labels, predictions)
    print("Accuracy:", accuracy)
    report = classification_report(test_labels, predictions)
    print("\nClassification Report:\n", report)


    class_labels = sorted(list(set(train_labels)))
    cm = confusion_matrix(test_labels, predictions, labels=class_labels)
    plt.figure(figsize=(20, 24))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.savefig('myfig.png')
    plt.show()




    pca = PCA(n_components=2, random_state=42) 
    X_train_dense = X_train.toarray()
    pca.fit(X_train_dense)

    X_test_dense = X_test.toarray()
    X_test_pca = pca.transform(X_test_dense)
    print("PCA transformation successful.")

    # --- Define Color Mapping based on keywords ---
    color_map = {
        'mysql': 'red',
        'postgres': 'orange',
        'sqlserver': 'green'
    }
    default_color = 'gray' # Color for labels not matching known keywords

    def get_color_from_label(label, color_map):
        """Assigns color based on keywords in the label."""
        label_lower = label.lower()
        if 'mysql' in label_lower:
            return color_map['mysql']
        elif 'postgres' in label_lower:
            return color_map['postgres']
        elif 'sqlserver' in label_lower:
            return color_map['sqlserver']
        else:
            # Handle cases where the label might not contain the expected keywords
            print(f"Warning: Label '{label}' did not match known keywords. Assigning default color '{default_color}'.")
            return default_color

    true_colors = [get_color_from_label(label, color_map) for label in test_labels]
    predicted_colors = [get_color_from_label(label, color_map) for label in predictions]

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='MySQL versions',
            markerfacecolor=color_map['mysql'], markersize=10),
        Line2D([0], [0], marker='o', color='w', label='PostgreSQL versions',
            markerfacecolor=color_map['postgres'], markersize=10),
        Line2D([0], [0], marker='o', color='w', label='SQL Server versions',
            markerfacecolor=color_map['sqlserver'], markersize=10)
    ]
    unique_colors_used = set(true_colors + predicted_colors)
    if default_color in unique_colors_used:
        legend_elements.append(Line2D([0], [0], marker='o', color='w', label='Other/Unknown',
                            markerfacecolor=default_color, markersize=10))


    misclassified_idx = np.where(np.array(test_labels) != np.array(predictions))[0]

    if len(misclassified_idx) > 0:
        plt.figure(figsize=(15, 6))
        plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=true_colors, alpha=0.5, s=80, label='_nolegend_')

        plt.scatter(X_test_pca[misclassified_idx, 0], X_test_pca[misclassified_idx, 1],
                    marker='x', c='black', edgecolors='yellow',linewidths=2, s=40, label=f'Misclassified major version (correct engine)')

        plt.title('Misclassified Major Version based on the SVM classification (plotted in 2D PCA Space)')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        combined_legend_elements = legend_elements + plt.gca().get_legend_handles_labels()[0]
        plt.legend(handles=combined_legend_elements, title="Classes / Errors")
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.show()
    else:
        print("No misclassifications found on the test set!")
