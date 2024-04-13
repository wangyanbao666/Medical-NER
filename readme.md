## BiLSTM For Named Entity Recognition In Medical Dataset

### Getting Started
NER.ipynb is the notebook we used to load data, build, train and evaluate our models. To run the script, please install all the libraries in the requirements.txt by the following command:
```
pip install -r requirements.txt
```
You might need to change the torch version according to your cuda version, please find the proper version here:
https://pytorch.org/get-started/previous-versions/

Due to the extremely large size of the checkpoints, we could not upload all the checkpoints and only transfer the final checkpoints of every model to the google drive. They are used as the final parameters to evaluate our models. Please find them in the following link: 
https://drive.google.com/drive/folders/1yWpeD4Si2F4Sfoo2pxOlLKw1t3tJqhRO?usp=sharing

We have also uploaded our training and testing dataset which is originally from n2c2 NLP Research Data Sets. Please find them in the google drive link as well:
https://drive.google.com/drive/folders/1tZ5eiPyt2IB89eeuciTS3zfymBAJxtXm?usp=sharing


Before running our script, please make sure that your file structure looks like:
```
input/ -> training and testing data
models/ -> the final model
checkpoints/ -> storing checkpoints
NER.ipynb
```
If you want to train the model by yourself and evaluate your own checkpoint, please rename the checkpoint file as best_model.pth and put them to the correct place under models according to the model type and hyperparameters.


### Result
The evaluation of all the models are stored in an excel sheet under the result folder. We used accuracy, precision, recall and f1 on an independent testing dataset to compare the performance of all the models with different hyperparameters.


### Prediction
If you want to play with our model, you can use the predict function at the last section which asks you to feed in a text and specify the model you are going to use. It will take out the named entity found in the text and print something like this:
```
[{'token': '[CLS]', 'label': 'I-DRUG'}, {'token': 'paracetamol', 'label': 'B-DRUG'}, {'token': '500mg', 'label': 'B-STRENGTH'}, {'token': 'tablet', 'label': 'B-FORM'}, {'token': 'tablet', 'label': 'B-FORM'}, {'token': 'po', 'label': 'B-ROUTE'}, {'token': 'q6h', 'label': 'B-FREQUENCY'}, {'token': 'prn', 'label': 'I-FREQUENCY'}]
```