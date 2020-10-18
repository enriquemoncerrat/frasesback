
import string

import pandas as pd
from django.contrib.auth.models import User, Group
from rest_framework import serializers
from rest_framework.serializers import ModelSerializer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC


class UserSerializer( serializers.HyperlinkedModelSerializer ):
    class Meta:
        model = User
        fields = ['url', 'username', 'email', 'groups']


class GroupSerializer( serializers.HyperlinkedModelSerializer ):
    class Meta:
        model = Group
        fields = ['url', 'name']


class UserDetailSerializer( ModelSerializer ):



    def get_relationship_to_user(self,  format=None):
        request = 'request'
        if request.exists():
            return 'Existed'
        else:
            return 'Not Existed'

    def getFrase(request):

        # Code source: Jaques Grobler
        # License: BSD 3 clause

        # Load the diabetes dataset
        ##leo el csv de entrenenamient
        spam_or_ham = pd.read_csv( "sentimientos.csv", encoding='latin-1' )[
            ["Sentiment", "SentimentText"]]
        spam_or_ham.columns = ["label", "text"]
        ##   data = spam_or_ham.head()
        ##cuantas columnas positivoas y cuantas negativas tengo
        data = spam_or_ham["label"].value_counts()

        # tokenizar es separar palabra por palabra en un array de string. donde esta punctuation lo que se hace es indicar
        # que no tenga en cuenta los . (puntos)

        punctuation = set( string.punctuation )

        def tokenize(sentence):
            tokens = []
            for token in sentence.split():
                new_token = []
                for character in token:
                    if character not in punctuation:
                        new_token.append( character.lower() )
                if new_token:
                    tokens.append( "".join( new_token ) )
            return tokens

        demo_vectorizer = CountVectorizer(
            tokenizer=tokenize,
            binary=True
        )

        spam_or_ham.head()["text"].apply( tokenize )
        examples = [
            "Call FREEPHONE 0800 542 0578 now!",
            "Did you call me just now ah?",
            "Call FREEPHONE 0800 542 0578 now!"
        ]
        demo_vectorizer.fit( examples )
        vectors = demo_vectorizer.transform( examples ).toarray()
        headers = sorted( demo_vectorizer.vocabulary_.keys() )
        pd.DataFrame( vectors, columns=headers )

        train_text, test_text, train_labels, test_labels = train_test_split( spam_or_ham["text"],
                                                                             spam_or_ham["label"],
                                                                             stratify=spam_or_ham["label"] )
        print( f'Training examples: {len( train_text )}, testing examples {len( test_text )}' )
        real_vectorizer = CountVectorizer( tokenizer=tokenize, binary=True )

        train_X = real_vectorizer.fit_transform( train_text )
        test_X = real_vectorizer.transform( test_text )

        train_X.shape
        classifier = LinearSVC()
        classifier.fit( train_X, train_labels )

        predicciones = classifier.predict( test_X )

        accuracy = accuracy_score( test_labels, predicciones )

        print( f'Accuracy: {accuracy:.4%}' )

        spamm = "fuck"

        examples = [
            spamm
        ]

        examples_X = real_vectorizer.transform( examples )
        predicciones = classifier.predict( examples_X )

        for text, label in zip( examples, predicciones ):
            # print( f'{label:5} - {text}' )
            if (label == 1):
                respuesta = 'Esta frase tiene un sentimiento positivo!'
            else:
                respuesta = 'Esta frase tiene un sentimiento negativo!'
            print( respuesta )
        return request
