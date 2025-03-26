import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
# import xgboost as xgb

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

    train_dir = "/Users/raf/code/project-martial/dataset/packets/train"
    train_data, train_labels = load_byte_data(train_dir)

    test_dir = "/Users/raf/code/project-martial/dataset/packets/test"
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

    # Train XGBoost classifier
    # classifier = xgb.XGBClassifier()
    # classifier.fit(X_train, train_labels)
    # predictions = classifier.predict(X_test)
    # accuracy = accuracy_score(test_labels, predictions)
    # print("Accuracy:", accuracy)



