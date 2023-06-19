from webscraping import WebScraping
from train import prob_to_label

import argparse
import pickle
import json
import os


class Predict:

    def __init__(self, url, model_path, vectorizer_path):
        self.model_path = model_path
        self.vectorizer_path = vectorizer_path
        self.url = url

    def import_model(self):
        with open(self.model_path, 'rb') as file:
            model = pickle.load(file)
        with open(self.vectorizer_path, 'rb') as file:
            vectorizer = pickle.load(file)

        return model, vectorizer

    def predict(self):
        model, vectorizer = self.import_model()
        paragraphs = WebScraping(self.url).url2paragraphs()
        X = vectorizer.transform(paragraphs)
        probabilities = model.predict_proba(X)
        inx_dct = probability_to_label(probabilities)
        result_dct = {k: paragraphs[v] for k, v in inx_dct.items()}
        result_dct['ingredients'] = result_dct['ingredients'].split('\n')
        print('URL:' + self.url)
        print(json.dumps(result_dct, indent=4, ensure_ascii=False))

        return inx_dct


def probability_to_label(probabilities):
    """Converts a list of probabilities to a list of labels based on a threshold."""
    y_pred, predictions_df = prob_to_label(probabilities)
    max_lst = {}
    for label in ['ingredients', 'instructions']:
        label_rows = predictions_df[predictions_df['predicted_label'] == label]
        if not label_rows.empty:
            max_label_row = label_rows[label].idxmax()
            max_lst[label] = max_label_row
        else:
            max_lst[label] = ''
    return max_lst
    return y_pred


def get_cmd_args(cli_input=None):
    args = argparse.ArgumentParser('Train Gen2X')
    args.add_argument('url', type=str, help='URL to scrape and predict')
    return args.parse_args(cli_input)

if __name__ == '__main__':
    # import sys url
    args = get_cmd_args()
    url = args.url
    folder_path = os.path.dirname(os.path.abspath(__file__))
    # join folder path to csv file name
    model_path = os.path.join(folder_path, 'model.pkl')
    vectorizer_path = os.path.join(folder_path, 'vectorizer.pkl')
    test_url = url
    example = Predict(test_url, model_path, vectorizer_path)
    inx_dct = example.predict()
