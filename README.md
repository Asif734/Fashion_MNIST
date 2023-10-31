# Fashion MNIST Classification with Neural Network

This project demonstrates the classification of fashion items using a simple Neural Network model on the Fashion MNIST dataset.

## Dataset
The dataset used in this project is the Fashion MNIST dataset, which contains grayscale images of fashion items, each belonging to one of ten classes: t-shirt, trouser, pullover, dress, coat, sandal, shirt, sneaker, bag, and ankle boot. The dataset is divided into training and test sets.

## Code Structure

The project consists of the following main sections:

1. **Data Loading and Exploration**: We import the necessary libraries, load the training and test datasets, and explore the data. The classes are defined for visualization.

2. **Data Visualization**: We visualize sample images from the dataset to get a better understanding of the items.

3. **Data Preprocessing**: The dataset is preprocessed, including scaling the pixel values to the range [0, 1] and splitting it into a training and test set.

4. **Model Creation**: We define a Neural Network model with a Flatten layer followed by two Dense layers and an output layer. The model is compiled with the Adam optimizer and sparse categorical cross-entropy loss.

5. **Model Training**: The model is trained on the training data for 10 epochs.

6. **Model Evaluation**: The trained model is evaluated on the test data, and the accuracy is displayed.

7. **Prediction Visualization**: We visualize some test samples and their predictions made by the model.

## Requirements

Before running the code, ensure you have the following libraries and dependencies installed:

- Python 3
- TensorFlow
- NumPy
- Pandas
- Matplotlib
- Scikit-learn (for train-test splitting)
- Jupyter Notebook (optional)

You can install these dependencies using the provided `requirements.txt` file. Create a virtual environment and install the dependencies as follows:

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
pip install -r requirements.txt

## Model Evaluation

The results of the model evaluation can be found in the `output.txt` file. This file includes:

- Model architecture summary.
- Evaluation metric (accuracy).
- Additional insights and observations.

To view the detailed model evaluation results, you can open the [output.txt](https://github.com/Asif734/Fashion_MNIST/blob/master/output.txt) file.

