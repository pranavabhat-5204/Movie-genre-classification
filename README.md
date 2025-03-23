# Movie-genre-classification
This is a SVC based ML model for Movie genre classification. The output of this code is:



<img width="383" alt="Screenshot 2025-03-23 181215" src="https://github.com/user-attachments/assets/38f7fe19-ce87-46b9-ac48-f19823720752" />

Now about the steps involved in this model:
1. I used the description as the input as given in the task assigned.
2. The file was a table with separator ':::'.
3. Here the training and texting data are in separate files, so there is no needed for spliting.
4. I used the TfidfVectorizer to convert the texts to some numeral format using their frequencies in the document.
5. The features needs conversion to numerical format while label does not as the label consists for fixed terms of genre type.
6. I used SVC model for this text classification as these are labelled data and SVC is generally good with such data.
7. It achieved an accuracy of about 0.7 or 70%. With kernel='linear' we get about 0.62 accuracy which increases to 0.7 when kernel='sigmoid'.
8. SVC is less scalable so naive bayes classifier is another option for such data.
