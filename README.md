This project implements a parallel ensemble of neural networks to perform spam detection on email text data.
It uses MPI  to scale the training and inference/prompting phases across multiple processes, making the system capable of handling larger datasets and providing faster predictions.

Each process trains its own lightweight neural network model on a bootstrapped subset of the data.
During inference, the ensemble of models votes to classify emails as Spam or Not Spam.

This project was developed as part of a university course on Parallel Programming — which is why detailed comments are included throughout the code.

Emails are tokenized.
A vocabulary of unique words is built from the training set.
Each email is converted to a frequency vector based on the presence of vocabulary words.
The project uses the spam assasin data set https://www.kaggle.com/datasets/ganiyuolalekan/spam-assassin-email-classification-dataset

The Neural networks and training processes are extremly simplistic because the project is mostly focused about Parallel Programming. 
