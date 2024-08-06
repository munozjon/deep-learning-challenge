# deep-learning-challenge

## Module 21 Challenge Report

### Overview of the Analysis

For this challenge, I worked on creating a neural network model for the nonprofit foundation Alphabet Soup. The reason for this model was to select applicants for funding with the best chance of success. The model is a binary classifier predicting if applicants will be successful if funded by Alphabet Soup. I created 4 models in total in effort to reach an accuracy score of at least 75%.

The columns in the dataset included EIN, NAME, APPLICATION_TYPE, AFFILIATION, CLASSIFICATION, USE_CASE, ORGANIZATION, STATUS, INCOME_AMT, SPECIAL CONSIDERATIONS, ASK_AMT, and IS_SUCCESSFUL. The target labels being tested are in the IS_SUCCESSFUL column, while the features are the rest of the columns apart from EIN and NAME. These features identify the borrower from Alphabet Soup previously, along with information regarding the use case and other contextual insights.

### Results

#### Data Preprocessing
I used a supervised model to fit, predict, and evaluate the data. I started by reading an online CSV file and dropping the ID columns EIN and NAME, as they will not be beneficial to the model. After, I determined the value counts of the APPLICATION_TYPE column and determined a cutoff value, where application types fewer than 200 were grouped together into 'Other'. I then did the same thing with the for CLASSIFICATION, cutting of values less than 1000. After, I converted the categorical data into numberical/binary data with pd.get_dummies() and joined it with the ASK_AMT column to get the data used for testing.

#### Compiling, Training, and Evaluating the Model
I split the preprocessed data into the features, X, and the target, y. I further split these arrays using the train_test_split, specifying a test size and ensuring the data would be shuffled. I then scaled the data using the StandardScaler() module and scaled the X_train and X_test arrays. After, I initialized the neural network model using TensorFlow keras sequential modeling and used Dense layering.

- On the first layer, I used 42 neurons, an input dimension of 42, and an activation function of 'relu'. This was because there was 42 columns in the X_train array. Relu was used as a standard function.
- On the second layer, I used 84 neurons and an activation function of 'relu'. I simply doubled the number of neurons from the first layer.
- On the output layer, I used the 'sigmoid' activation function because it fits a linear regression.

After compiling the model with the 'adam' optimizer, 'binary_crossentropy' loss, and 'accuracy' for metrics, I trained the model by fitting the X_train_scaled and y_train data. I used 100 epochs and a validation split of 0.2.

For this first model, I achieved an accuracy score of 72%, not the target model performance. In efforts to improve it, I created 3 optimized models.

- On the first model, I increased the cutoff values for APPLICATION_TYPE and CLASSIFICATION, changed how the data was split, added a third layer, increased/decreased the number of neurons at each layer, and changed the activation functions at the second and third layer. The result was a 73% accuracy.
- On the second model, I removed the ASK_AMT and SPECIAL_CONSIDERATIONS columns, as well as changing how the data was split and removing the third hidden layer. I also included a batch size to the training. This model returned a 72% accuracy.
- The third model also removed ASK_AMT, modified the test size, re-added the third hidden layer and adjusted the neurons and activation functions, and decreased the batch size. The result was also 72% accuracy.

### Summary
In conclusion, none of the models created reached the target model performance of 75%. This could be because the model is not linear, and so it is unable to properly fit to the sigmoid function on output. A future model could be done that tests other output activation functions, while also striving for more parity between the training and testing data to ensure each sample is emblematic of the overall dataset.