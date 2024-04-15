import re

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z]', '', text)
    return text

def text_to_vector(text):
    text = preprocess_text(text)
    vector = [0] * 26
    for char in text:
        if 'a' <= char <= 'z':
            index = ord(char) - ord('a')
            vector[index] += 1
    return vector


class Perceptron:
    def __init__(self):
        self.weights = [0.0] * 26
        self.bias = 0.0

    def predict(self, inputs):
        activation = sum(w * x for w, x in zip(self.weights, inputs)) + self.bias
        return 1.0 if activation >= 0.0 else 0.0

    def train(self, inputs, target, learning_rate):
        prediction = self.predict(inputs)
        error = target - prediction
        self.weights = [w + learning_rate * error * x for w, x in zip(self.weights, inputs)]
        self.bias += learning_rate * error

class LanguageClassifier:
    def __init__(self, languages):
        self.languages = languages
        self.perceptrons = {lang: Perceptron() for lang in languages}

    def predict(self, text):
        vector = text_to_vector(preprocess_text(text))
        predictions = {lang: perceptron.predict(vector) for lang, perceptron in self.perceptrons.items()}
        return max(predictions, key=predictions.get)


    def train(self, vector, language, learning_rate):
        for lang, perceptron in self.perceptrons.items():
            target = 1.0 if lang == language else 0.0
            perceptron.train(vector, target, learning_rate)

def read_training_data(file_path):
    training_data = []
    languages = set()

    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
        languages_texts = re.findall(r'==([^\n]+)==\n([^=]+)', text)
        for language, content in languages_texts:
            languages.add(language)
            text = preprocess_text(content)
            if text:
                vector = text_to_vector(text)
                training_data.append((vector, language))

    return training_data, list(languages)

def main():
    file_path = 'training_data.txt'
    training_data, languages = read_training_data(file_path)
    classifier = LanguageClassifier(languages)


    learning_rate = 0.1
    for vector, language in training_data:
        classifier.train(vector, language, learning_rate)

    while True:
        choice = input("Wybierz opcję:\n1. Wprowadź zdanie testowe\n2. Wczytaj plik\n3. Wyjście\n")

        if choice == "1":
            sentence = input("Wprowadź zdanie testowe: ")
            predicted_language = classifier.predict(sentence)
            print("Przewidywany język:", predicted_language)
        elif choice == "2":
            file_path = input("Podaj ścieżkę do pliku: ")
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
            predicted_language = classifier.predict(text)
            print("Przewidywany język:", predicted_language)
        elif choice == "3":
            break
        else:
            print("Niepoprawny wybór. Spróbuj ponownie.")


if __name__ == "__main__":
    main()
