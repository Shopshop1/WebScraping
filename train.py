from webscraping import WebScraping


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import pickle
import os


class Create_model:
    def __init__(self, csv_path, model_path, dataset_path, vectorizer_path):
        self.raw_db = pd.read_csv(csv_path)
        self.labels = []
        self.model_path = model_path
        self.dataset_path = dataset_path
        self.vectorizer_path = vectorizer_path

    def pre_process(self):
        # WebScraping
        url_lst = self.raw_db.url.to_list()
        paragraphs = []
        print('Starting webscraping')
        self.raw_db.url.apply(lambda x: paragraphs.append(WebScraping(x).url2paragraphs()))
        data = {
            'url': url_lst,
            'paragraphs': paragraphs,
        }

        # Create a DataFrame from the dictionary
        data_df = pd.DataFrame(data)

        # Explode the "paragraphs" column to create new rows for each paragraph
        df_exploded = data_df.explode('paragraphs').reset_index(drop=True)
        # Add a "label" column to the DataFrame
        df_exploded['label'] = df_exploded.paragraphs.apply(lambda x: 'ingredients' if x in self.raw_db.ingredients.to_list() else ('instructions' if x in self.raw_db.instructions.to_list() else 'None'))
        # save dataset
        # df_exploded.to_pickle(self.dataset_path)
        print(df_exploded.label.value_counts())  # data is not balanced
        num_of_cases = df_exploded.label.value_counts().min()
        data = df_exploded.groupby('label').apply(lambda x: x[:num_of_cases]).reset_index(drop=True)
        print('After balancing the data: \n', data.label.value_counts())
        return data

    def create_model(self, data):
        # Step 1: Prepare the data
        X = data['paragraphs']
        y = data['label']
        self.labels = y.unique()

        # Step 2: Text preprocessing
        # Lowercasing
        X = X.str.lower()

        # Step 3: Vectorize the text
        vectorizer = TfidfVectorizer()
        X_vectorized = vectorizer.fit_transform(X)

        # Step 4: Split the data into training and testing sets
        # For further analysis, I would consider splitting the data such that one url will appear only in one set
        X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

        # Step 5: Train a multi-class classification model
        model = MultiOutputClassifier(MultinomialNB())
        model.fit(X_train, pd.get_dummies(y_train))
        # Step 6: Evaluate the model
        probabilities = model.predict_proba(X_test)
        y_pred, df = prob_to_label(probabilities)
        self.evaluate_model(y_pred, y_test)
        self.save_model(model, vectorizer)

    def evaluate_model(self, y_pred, y_test):
        # Calculate model metrics
        metrics = {}
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        # another metrics can be added
        metrics['Overall'] = {
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1 Score': f1
            }

        # plot the confusion matrix
        cm_model = confusion_matrix(y_test, y_pred)
        ax = sns.heatmap(cm_model, annot=True, cmap="Blues", fmt="d", xticklabels=self.labels, yticklabels=self.labels)
        ax.set_xlabel('Predicted Labels')
        ax.set_ylabel('True Labels')
        ax.set_title('Confusion Matrix')  # Set the title using ax.set_title()

        for label in self.labels:
            true_binary = [1 if l == label else 0 for l in y_test]
            predicted_binary = [1 if l == label else 0 for l in y_pred]

            accuracy_per_label = accuracy_score(true_binary, predicted_binary)
            precision_per_label = precision_score(true_binary, predicted_binary)
            recall_per_label = recall_score(true_binary, predicted_binary)
            f1_per_label = f1_score(true_binary, predicted_binary)

            metrics[label] = {
                'Accuracy': accuracy_per_label,
                'Precision': precision_per_label,
                'Recall': recall_per_label,
                'F1 Score': f1_per_label
            }

        # Print the metrics for each label
        for label, metrics_dict in metrics.items():
            print('Label:', label)
            for metric, value in metrics_dict.items():
                print(metric + ':', value)

    def save_model(self, model, vectorizer):
        # Save the model to a file
        with open(self.model_path, 'wb') as file:
            pickle.dump(model, file)
        with open(self.vectorizer_path, 'wb') as file:
            pickle.dump(vectorizer, file)
        print('Model saved successfully')


def prob_to_label(probabilities):
    z = [array[:, 1].reshape(-1, 1) for array in probabilities]
    predictions_df = pd.DataFrame(
        {'None': z[0].flatten(), 'ingredients': z[1].flatten(), 'instructions': z[2].flatten()})
    # added a column with the meximal probability column name
    predictions_df['predicted_label'] = predictions_df.apply(lambda row: row.idxmax(), axis=1)
    y_pred = predictions_df['predicted_label'].to_list()
    return y_pred, predictions_df


if __name__ == '__main__':
    folder_path = os.path.dirname(os.path.abspath(__file__))
    # join folder path to csv file name
    csv_path = os.path.join(folder_path, 'loaveandlemons_dataset.csv')
    model_path = os.path.join(folder_path, 'model.pkl')
    dataset_path = os.path.join(folder_path, 'dataset.pkl')
    vectorizer_path = os.path.join(folder_path, 'vectorizer.pkl')
    create_model = Create_model(csv_path, model_path, dataset_path, vectorizer_path)
    data = create_model.pre_process()
    create_model.create_model(data)





