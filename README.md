# Definition of .py files 

1. preprocess_get_data :

	1.	Data Loading and Preprocessing:
	•	Loads a dataset (ZHVInew.txt) containing some data (likely related to housing prices, considering the column names), drops irrelevant columns, and performs linear interpolation for missing values.
	2.	Feature Scaling:
	•	The data is normalized using StandardScaler to standardize the features (i.e., mean = 0, standard deviation = 1). This step is essential for machine learning models to perform better.
	3.	Training and Test Data Split:
	•	The dataset is split into training and testing sets using an 80-20 split (80% for training, 20% for testing). For demonstration purposes, only a limited portion (1000 samples) is used.
	4.	Sliding Window Creation:
	•	The code creates sliding windows of time series data for the input features (X) and their corresponding target values (y).
	•	X: Input data for model training (shape: window_size x features).
	•	y: The target value, which is the next time step (the value after the sliding window).
	•	The sliding window method is crucial in time series forecasting tasks, where the model uses a sequence of previous values to predict the next value.
	5.	Data Reshaping:
	•	The X and y arrays are reshaped and transposed to match the format expected by machine learning models, where:
	•	X_prepared: Represents the input features (batch_size, sequence_length, features).
	•	y_prepared: Represents the target values.
	6.	Conversion to Tensors:
	•	The prepared data (X_prepared and y_prepared) is converted to PyTorch tensors, which are the required format for deep learning models in PyTorch.
	7.	Dataset and DataLoader Creation:
	•	A TensorDataset is created using the PyTorch tensors (X_tensor, y_tensor).
	•	A DataLoader is also created from the dataset for efficient batching during training.
	8.	Test Data Preparation:
	•	Similar steps are followed for preparing the test data. The test_data() function takes the test dataset, creates sliding windows, reshapes the data, and returns tensors (X_tensor_test, y_tensor_test) for testing.

Detailed Breakdown of the Functions:

	•	create_sliding_windows(data, window_size, stride):
	•	This function creates sliding windows for the given data by slicing it into smaller sequences of length window_size. Each window is used to predict the next time step (y).
	•	get_prepared_data():
	•	This function transposes X and y to match the format expected by PyTorch, where the shape is (batch_size, sequence_length, features) for the input data and (batch_size, sequence_length, 1) for the target data.
	•	convert_to_tensor():
	•	Converts the X_prepared and y_prepared data into PyTorch tensors (X_tensor and y_tensor).
	•	get_tensor_dataset():
	•	Uses convert_to_tensor() to get the prepared tensors, then creates a TensorDataset and a DataLoader. The DataLoader will be used for batching during training.
	•	test_data():
	•	This function performs the same sliding window transformation and reshaping steps for the test data and returns the test data tensors (X_tensor_test and y_tensor_test).

Flow of the Code:

	1.	Data Loading:
	•	The dataset ZHVInew.txt is loaded and columns that are not necessary for the task (such as region and state information) are dropped. The remaining data is interpolated to fill missing values.
	2.	Data Normalization:
	•	The StandardScaler is applied to normalize the dataset, ensuring that each feature has a mean of 0 and a standard deviation of 1. This helps machine learning models to perform more efficiently and converge faster.
	3.	Sliding Window Creation:
	•	The create_sliding_windows function splits the dataset into smaller windows. Each window will be used as input (X) to predict the next value (y). The sliding window approach is suitable for time series forecasting tasks where we predict future values based on past values.
	4.	Data Preparation for PyTorch:
	•	The X and y arrays are reshaped and transposed to fit the expected format of PyTorch models (i.e., batch_size, sequence_length, features). They are then converted into PyTorch tensors.
	5.	Dataset Creation:
	•	The prepared tensors are wrapped into a TensorDataset and a DataLoader. The DataLoader helps in batching the data during model training.
	6.	Test Data Preparation:
	•	Similar steps are followed for the test data to prepare it for model evaluation.

2. TransformerModel

Class: TransformerModel

The TransformerModel class inherits from torch.nn.Module and encapsulates the transformer architecture.

Components of the Transformer Model:

	1.	Embedding Layer (nn.Linear):
	•	Converts the input features (of shape input_dim) into a higher-dimensional space (embed_dim) to capture more complex patterns.
	•	Maps each input feature (e.g., time-series data) to an embedding of the specified embed_dim.
	2.	Positional Encoding (nn.Parameter):
	•	Since the transformer model does not inherently account for the sequential nature of the data (unlike RNNs or LSTMs), positional encoding is   added to provide information about the position of each element in the sequence.
	•	This is a learned parameter, represented as a tensor of shape (1, seq_length, embed_dim), where seq_length is the length of the sequence (number of time steps) and embed_dim is the dimension of the embedding.
	3.	Transformer Encoder (nn.TransformerEncoder):
	•	The core of the model is the TransformerEncoder, which uses the TransformerEncoderLayer. This layer consists of multi-head attention and a feedforward network. The TransformerEncoder applies multiple encoder layers (defined by num_layers) to the input sequence.
	•	This encoder processes the embedded input with positional encoding, enabling it to capture long-range dependencies in the sequence.
	4.	Fully Connected Output Layer (nn.Linear):
	•	After the encoder processes the input sequence, the output is passed through a fully connected (linear) layer (fc_out) that maps the final representation of the sequence to the desired output dimension (output_dim).
	5.	Forward Pass:
	•	In the forward method, the input data x is first passed through the embedding layer and then has positional encoding added.
	•	The encoded sequence is then processed by the transformer encoder.
	•	Finally, the output is passed through the fully connected layer to generate the final prediction.

3. train.py

	1.	Model Definition:
	•	The model defined here is a Transformer-based model that uses multi-head attention to capture long-range dependencies in the time-series data.
	•	The model consists of:
	•	Embedding Layer: Converts input features to an embedding space of a specified dimension (embed_dim).
	•	Transformer Encoder: A multi-layer transformer encoder with a defined number of attention heads (num_heads), feedforward network dimension (ff_dim), and layers (num_layers).
	•	Output Layer: A fully connected layer that outputs a single value for each timestep.
	2.	Hyperparameters:
	•	input_dim: The number of features in the input data (5).
	•	embed_dim: The size of the embedding dimension for transformer attention (64).
	•	num_heads: The number of attention heads in the multi-head attention mechanism (8).
	•	ff_dim: The size of the hidden layer in the transformer’s feedforward network (128).
	•	num_layers: The number of transformer encoder layers (4).
	•	output_dim: The number of output values per timestep (since it’s a regression task, it’s set to 1).
	•	seq_length: The length of the sequence (number of timesteps, set to 293).
	3.	Loss and Optimizer:
	•	Loss Function: MSELoss (Mean Squared Error) is used because this is a regression task, where the model is predicting continuous values.
	•	Optimizer: The Adam optimizer is used with a learning rate of 1e-3 to update the model’s parameters during training.
	4.	Dataset and DataLoader:
	•	The get_tensor_dataset() function from data.preprocess_get_data is used to preprocess the data and return a PyTorch DataLoader for batching during training.
	5.	Training Loop:
	•	The training loop runs for num_epochs (15 in this case).
	•	TQDM Progress Bar: The progress bar is initialized using tqdm to show the current batch’s progress, along with the loss for each batch.
	•	The model is trained by performing a forward pass, calculating the loss, performing a backward pass, and then optimizing the weights using the Adam optimizer.
	6.	Saving the Model:
	•	After training, the model’s parameters are saved to a file (model.pth) using torch.save().

4. test.py

	1.	Model Loading:
	•	The Transformer model is instantiated with specified hyperparameters (input_dim, embed_dim, num_heads, etc.) and loaded with pre-trained weights from saved_model/model.pth.
	•	transformer.load_state_dict(torch.load('saved_model/model.pth')) loads the saved model parameters.
	2.	Test Data Preparation:
	•	The test data is preprocessed using the test_data() function from data.preprocess_get_data, which returns the test input features (X_tensor_test) and the corresponding true values (y_tensor_test).
	3.	Model Inference:
	•	The model is set to evaluation mode using transformer.eval() to disable dropout layers and batch normalization.
	•	The model makes predictions on the test set using predicted_values_test = transformer(X_tensor_test). The predicted values are stored in predicted_values_test.
	4.	RMSE Calculation:
	•	The Root Mean Squared Error (RMSE) is computed using the mean_squared_error function from sklearn.metrics. RMSE measures the average magnitude of the prediction errors (the difference between the predicted and actual values).
	•	The getRMSE() function computes the RMSE for the first instance in the test set.

5. plots.py

	•	A loop iterates through each channel (representing different series or variables). The number of channels is defined by num_cities = 10.
	•	For each channel (in this case, num_cities = 10), the following steps are performed:
	•	Plot Ground Truth: The true values are plotted using plt.plot(y_tensor_test.numpy()[:][i]).
	•	Smooth the Predicted Values: The predictions are smoothed using a moving average (np.convolve() with a kernel of size window_size).
	•	Calculate RMSE: The RMSE is calculated between the true values (y_tensor_test[0,:]) and the predicted values (predicted_values_test[0,:]).
	•	Plot and Save: Both the ground truth and smoothed predicted values are plotted, and the figure is saved to the plots/ directory with a unique filename (my_plot_{i+1}.png).

# Steps to run it locally

6. Ipython Notebooks

    1. Datahandling.ipnyb - In this file we have worked on data preprocessing and created a subset of one city time series data set to use for the statistical analysis (document has required comments at each step)

    2. StatisticalAnalysis.ipnyb - This file has the analysis of the time series data set using ARMA, ARIMA, SARIMA and SARMAX (please refer the document, it has comments along side the code)

1. Create a virtual environment :

    python3 -m venv venv
    source venv/bin/activate  # For macOS/Linux
    venv\Scripts\activate  # For Windows

2. Once your virtual environment is activated (or if you’re not using one), run the following command in the terminal or command prompt to install the packages listed in the requirements.txt file:

    pip install -r requirements.txt

3. The model is saved and stored after training in the folder 'saved_model/model.pth'

4. the model.pth is used to predict on the test data set to perform the metrics.

5. the plots.py will save the plots to the "plots" folder for evaluation and visualization purposes.

6. use "python -m Vizualization.plots" to run and get the plots.

7. use  "python -m Model.test" to test the saved model.

8. use "python -m Model.train" to train the model.

9. The model is already trained, the user can change hyperparameters of the model and the data used for training purposes accordingly.