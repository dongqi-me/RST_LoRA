import json
import os
import evaluate
import datasets

class BLEURTCalculator:
    def __init__(self):
        self.bleurt = datasets.load_metric("bleurt")

    def compute_bleurt(self, input_data):
        if not input_data:
            raise ValueError("Input data cannot be empty.")

        results = []
        for pair in input_data:
            try:
                reference = pair['reference'].strip()
                prediction = pair['prediction'].strip()
            except KeyError:
                raise ValueError("Each dictionary in input data should contain 'reference' and 'prediction' keys.")

            score = self.bleurt.compute(predictions=[prediction], references=[reference])
            results.append({'bleurt': round(score["scores"][0], 4)})

        average_score = round(sum([res['bleurt'] for res in results]) / len(results), 4)
        results.append({'corpus_level': {'bleurt': average_score}})
        return results

    def compute_bleurt_from_file(self, input_json_path, output_json_path):
        if not os.path.isfile(input_json_path):
            raise FileNotFoundError(f"{input_json_path} does not exist.")

        with open(input_json_path, 'r') as f:
            data = json.load(f)
        results = self.compute_bleurt(data)

        with open(output_json_path, 'w') as f:
            json.dump(results, f, indent=4)

if __name__ == '__main__':
    bleurt_calculator = BLEURTCalculator()

    input_data = [
        {"reference": "The cat sat on the mat. It was happy.", "prediction": "A cat was sitting on the mat. The cat seemed pleased."},
        {"reference": "The quick brown fox jumps over the lazy dog. The dog did not react.", "prediction": "A fast brown fox jumps above the lazy dog. The dog seemed indifferent."}
    ]

    input_json_path = 'input.json'
    with open(input_json_path, 'w') as f:
        json.dump(input_data, f)

    output_json_path = 'output.json'
    bleurt_calculator.compute_bleurt_from_file(input_json_path, output_json_path)