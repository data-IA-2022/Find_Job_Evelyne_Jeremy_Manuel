# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 18:17:56 2023

@author: Utilisateur
"""

import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
import plotly.express as px

# machine learning - scikit learn:
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, RobustScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn import set_config
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVC, SVR
from sklearn.model_selection import GridSearchCV
from PIL import Image

# choix des couleurs
couleurs =['#EF553B', '#636EFA', '#00CC96', '#FF6692', '#AB63FA', '#19D3F3', '#B6E880', '#FF97FF', '#FECB52']

# CSS to inject contained in a string
hide_table_row_index = """
            <style>
            thead tr th:first-child {display:none}
            tbody th {display:none}
            </style>
            """

# Inject CSS with Markdown
st.markdown(hide_table_row_index, unsafe_allow_html=True)



st.markdown('<style>body{background-color: Blue;}</style>',unsafe_allow_html=True)

@st.cache(suppress_st_warning=True)
def load():
    data=pd.read_csv('C:/Users/Utilisateur/workspace/IA_Projets/GRETA/Projet 2/df_travail_poste.csv')
    
    if data.columns[0] == "Unnamed: 0":
      data=data.rename({"Unnamed: 0":"Index"}, axis='columns')
      data=data.set_index('Index')
    
    if data.columns[0] == "Unnamed: 0.1":
      data=data.rename({"Unnamed: 0.1":"Index"}, axis='columns')
      data=data.set_index('Index')
   
    return data

@st.cache(suppress_st_warning=True)
def set_df_circular_barplot():
    df = pd.DataFrame(
            {
                'Name': ['item ' + str(i) for i in list(range(1, 51)) ],
                'Value': np.random.randint(low=10, high=100, size=50)
            })

    # Reorder the dataframe
    df = df.sort_values(by=['Value']);
    
    return df

def tauxDeRemplissage(df):
  return (df.notna().sum()*100/len(df)).sort_values()

def correlation(dataframe, method):
  liste = []
  corr = dataframe.corr(method = method)
  colonnesIndex = corr.index
  colonnesNoms = corr.columns
  for colonne in colonnesIndex:
    for colonne2 in colonnesNoms:
      if colonne != colonne2:
        trie = [colonne, colonne2]
        trie.sort()
        liste.append([trie[0], trie[1], corr.loc[colonne, colonne2]])
  return pd.DataFrame(liste, columns=["col1", "col2", "values"]).drop_duplicates(["col1", "col2"])


def exploded(data):
    df_exploded=pd.DataFrame()
    df_exploded[["comp1", "comp2", "comp3", "comp4", "comp5"]] = data["Compétences"].str.split(',', expand=True)
    df_exploded.fillna("", inplace=True)
    return df_exploded

def circular_barplot(df, df_values, df_labels):
    # initialize the figure
    plt.figure(figsize=(20,10))
    
    ax = plt.subplot(111, polar=True)
    plt.axis('off')
    
    # Constants = parameters controling the plot layout:
    upperLimit = 40
    lowerLimit = 2
    labelPadding = 9
    
    # Compute max and min in the dataset
    max = df_values.max()
    
    # Let's compute heights: they are a conversion of each item value in those new coordinates
    # In our example, 0 in the dataset will be converted to the lowerLimit (10)
    # The maximum will be converted to the upperLimit (100)
    slope = (max - lowerLimit) / max
    heights = slope * df.Value + lowerLimit
    heights*=.9
    
    # Compute the width of each bar. In total we have 2*Pi = 360°
    width = 2*np.pi / len(df.index)
    
    # Compute the angle each bar is centered on:
    indexes = list(range(1, len(df.index)+1))
    angles = [element * width for element in indexes]
    # angles
    
    # Draw bars
    bars = ax.bar(
        x=angles, 
        height=heights, 
        width=width, 
        bottom=lowerLimit,
        linewidth=2, 
        edgecolor="white",
        color="#61a4b2",
    )
    
    for bar, angle, height, label in zip(bars,angles, heights, df_labels):
        labelPadding-= 0.1
        # Labels are rotated. Rotation must be specified in degrees :(
        rotation = np.rad2deg(angle)
    
        # Flip some labels upside down
        alignment = ""
        if angle >= np.pi/2 and angle < 3*np.pi/2:
            alignment = "right"
            rotation = rotation + 180
        else: 
            alignment = "left"
    
        # Finally add the labels
        ax.text(
            x=angle, 
            y=lowerLimit + bar.get_height() + labelPadding, 
            s=label, 
            ha=alignment, 
            va='center', 
            rotation=rotation, 
            rotation_mode="anchor") 
    
    st.pyplot(plt)

csvExtract =  load()  
df = set_df_circular_barplot()  
    
st.sidebar.header("Menu")

menu = ["Accueil",
        "Exploration des données",
        "Analyse du marché" , 
        'Filtrage des offres & Prédiction']

selected_menu = st.sidebar.radio("", menu)




# st.beta_layout(st.shared.header_container(level=3, title=selected_menu))

if selected_menu == menu[0]:
    st.image(Image.open("C:/Users/Utilisateur/workspace/IA_Projets/GRETA/Projet 2/Logo.png"))

if selected_menu == menu[1]:
    
    lst= ['Informations sur la table',
          'Taux de remplissage de la table',
          'Répartition des salaires', 
          'Correlation' 
            ]
    
    app_mode1 = st.sidebar.selectbox('Exploration des données', lst)
   
    if app_mode1 == lst[0]:
        
        st.title(app_mode1)
        st.write(csvExtract.describe())
        
        
    if app_mode1 == lst[1]:
        fig = plt.figure(figsize=(10, 4))
        df = pd.DataFrame(tauxDeRemplissage(csvExtract))
        sns.barplot(x= df.index, y= df[0], data=df)
        plt.xticks(rotation=90)
        st.title(app_mode1)
        st.pyplot(fig)
        pass
    
    if app_mode1 == lst[2]:
        plt.tight_layout()
        fig = plt.figure(figsize=(10, 4))
        # Récupérer les données à partir du fichier csv
        salaireMin = csvExtract['Salaire minimum']
        salaireMax = csvExtract['Salaire maximum']
        
        # Créer le premier histogramme pour le salaire minimum
        sns.histplot(salaireMin, kde=False, bins=25, color='lightblue', edgecolor='black', linewidth=0.5)
        
        # Ajouter les lignes verticales pour la moyenne et la médiane du salaire minimum
        plt.axvline(salaireMin.mean(), color='red', linestyle='dashed', linewidth=2, label='moyenne salaire min')
        plt.axvline(salaireMin.median(), color='darkred', linestyle='dashed', linewidth=2, label='médiane salaire min')
        
        # Créer le deuxième histogramme pour le salaire maximum
        ax=sns.histplot(salaireMax, kde=False, bins=25, color='blue', edgecolor='black', linewidth=0.5)
        
        # Ajouter les lignes verticales pour la moyenne et la médiane du salaire maximum
        plt.axvline(salaireMax.mean(), color='green', linestyle='dashed', linewidth=2, label='moyenne salaire max')
        plt.axvline(salaireMax.median(), color='darkgreen', linestyle='dashed', linewidth=2, label='médiane salaire max')
        
        ax.set_title('Répartition des salaires')
        ax.set(xlabel='Salaire')
        
        # Ajouter une légende
        plt.legend(fontsize=7)
        sns.despine(bottom=True)
    
        # Afficher le graphique
        st.title(app_mode1)
        st.pyplot(plt)
        
        pass
    
    if app_mode1 == lst[3]:
        st.title(app_mode1)
        st.write(correlation(csvExtract, "kendall"))
        
# =============================================================================
# 
#Debut Evelyne
#     
# =============================================================================
if selected_menu == menu[2]:
    
    lst= ['Compétences les plus recherchées',
          'Entreprises qui recrutent le plus',
          'Les types de contrat',
          'Les Compétences les mieux payées',
          "Nombre d'annonces par lieu",
          "Localisation des entreprises qui recrutent",
          "Choisis l'entreprise !",          
            ]
    
    app_mode1 = st.sidebar.selectbox('Exploration des données', lst)

    
    if app_mode1 == lst[0]:  
        agree  = st.sidebar.checkbox("Avec compétences orephelines", True)
                    
        if agree :
            couleurs =['#fefefe'] + couleurs
            #Compétences les plus recherchées
            # from sklearn.feature_extraction.text import CountVectorizer
            vectorizer = CountVectorizer(strip_accents='unicode', tokenizer=lambda x: x.split(', '))
            matrix = vectorizer.fit_transform(csvExtract.dropna()['Compétences'])
            #Transformation de la matrice en un dataframe avec le nom des compétences
            counts = pd.DataFrame(matrix.toarray(), columns=vectorizer.get_feature_names())
            # Ajout d'une ligne 'total' au dataframe
            ctotal = counts.copy()
            ctotal.loc['total']= ctotal.sum()
            # Transposition du dataframe pour classer les compétences par la colomne 'total'
            pd.set_option('display.max_rows', None)
            competences = ctotal.transpose().sort_values(by=['total'], ascending=False)
            # Formatage du dataframe
            competences = competences.iloc[:,[-1]]
            competences = competences.reset_index()
            competences.columns = ['name', 'total']
            # Regroupement des compétences figurant dans une seule annonce
            compcat = competences.copy()
            compcat.loc[compcat['total'] < 2, 'name'] = 'compétences figurant dans une seule annonce'
            # Pie chart des principales compétences
            fig = px.pie(compcat, names='name', values='total', color_discrete_sequence=couleurs, width=1000, height=1000, hole=.3, title='Principales compétences en pourcentage')
            fig.update_traces(texttemplate='%{percent:.1%}')
    
            # Affichage
            st.title("Elargis ton champ de compétences !")
            st.plotly_chart(fig)
        else:
            
            # couleurs =['white'] + couleurs
            #Compétences les plus recherchées
            # from sklearn.feature_extraction.text import CountVectorizer
            vectorizer = CountVectorizer(strip_accents='unicode', tokenizer=lambda x: x.split(', '))
            matrix = vectorizer.fit_transform(csvExtract.dropna()['Compétences'])
            #Transformation de la matrice en un dataframe avec le nom des compétences
            counts = pd.DataFrame(matrix.toarray(), columns=vectorizer.get_feature_names())
            # Ajout d'une ligne 'total' au dataframe
            ctotal = counts.copy()
            ctotal.loc['total']= ctotal.sum()
            # Transposition du dataframe pour classer les compétences par la colomne 'total'
            pd.set_option('display.max_rows', None)
            competences = ctotal.transpose().sort_values(by=['total'], ascending=False)
            # Formatage du dataframe
            competences = competences.iloc[:,[-1]]
            competences = competences.reset_index()
            competences.columns = ['name', 'total']
            # Regroupement des compétences figurant dans une seule annonce
            compcat = competences.copy()
            compcat.loc[compcat['total'] < 2, 'name'] = 'compétences figurant dans une seule annonce'
            # Pie chart des principales compétences
            
            compcat=compcat[compcat['name'] != 'compétences figurant dans une seule annonce']
         
            fig = px.pie(compcat, names='name', values='total', color_discrete_sequence=couleurs, width=1000, height=1000, hole=.3, title='Principales compétences en pourcentage presente au moins 2 fois')
            fig.update_traces(texttemplate='%{percent:.1%}')
    
            # Affichage
            st.title("Compétences les plus recherchées")
            st.plotly_chart(fig)
            
            pass
        
    if app_mode1 == lst[2]: 
        
        # Les types de contrat
        contrats = csvExtract['Type de contrat'].value_counts()
        contrats = contrats.reset_index()
        contrats.columns = ['type', 'count']
        # Pie chart des types de contrats
        fig = px.pie(contrats, names='type', values='count', color_discrete_sequence=couleurs, width=1000, height=1000, hole=.4, title='Répartition en pourcentage')
        fig.update_traces(texttemplate='%{percent:.1%}')
        
        # Affichage
        st.title("Les types de contrat")
        st.plotly_chart(fig)
       
    if app_mode1 == lst[1]:
        # Entreprises qui recrutent le plus
        df = csvExtract['Nom de la société'].value_counts()
        # Formatage du dataframe
        df = df.rename_axis('Name').reset_index(name='Value')
        
        # Affichage
        st.title('Explore les entreprises qui recrutent !')
        
        # Reorder the dataframe
        df = df.sort_values(by=['Value']);
        
        circular_barplot(df, df['Value'], df["Name"])
        
    if app_mode1 == lst[3]:
        
        vectorizer = CountVectorizer(strip_accents='unicode', tokenizer=lambda x: x.split(', '))
        matrix = vectorizer.fit_transform(csvExtract.dropna()['Compétences'])
        #Transformation de la matrice en un dataframe avec le nom des compétences
        counts = pd.DataFrame(matrix.toarray(), columns=vectorizer.get_feature_names())
        # Les postes les mieux payés
        # Extraction des lignes où les salaires sont présents
        df = csvExtract[csvExtract['Salaire minimum'].notna()]
        df.reset_index(drop=True, inplace=True)
        # Création d'une columne pour le salaire moyen
        df['Salaire moyen'] = df[['Salaire minimum', 'Salaire maximum']].mean(axis=1)
        # Les Compétences les mieux payées
        # Multiplication du dataframe obtenu avec CountVectorizer par la colonne 'Salaire moyen'
        comp_salaire = counts.multiply(df['Salaire moyen'], axis=0)
        # Calcul de la moyenne pour chaque colomne compétence.
        # Pour ne pas prendre en compte les valeurs égales à 0 dans la moyenne, on les remplace par NaN :
        comp_salaire.replace(0, np.nan, inplace=True)
        comp_salaire.loc['moyenne']= comp_salaire.mean()
        salaire_moyen = comp_salaire.iloc[[-1],:]
        # Les compétences les mieux payées
        #Transposition du dataframe pour classer les compétences par la colomne 'moyenne'
        classement = salaire_moyen.transpose().sort_values(by=['moyenne'], ascending=False)
        classement = classement.reset_index()
        classement.columns =['Compétences', 'Salaire moyen']
        # Bar chart des compétences les mieux payées
        classcat = classement[classement['Salaire moyen']>60000]
        fig = px.bar(classcat, x='Compétences', y='Salaire moyen', color='Compétences', width=1200, height=900, title='Salaire moyen par compétence', text_auto='.2s')
        
        # Affichage
        st.title("Les Compétences les mieux payées")
        st.plotly_chart(fig)
        
    if app_mode1 == lst[4]:
        # Localisation des annonces
        # Retrait de la catégorie 'île-de-france' pour visualiser les données.
        noregion = csvExtract[csvExtract['Lieu'] != 'île-de-france' ]
        localisation = noregion['Lieu'].value_counts()
        localisation = localisation.reset_index()
        localisation.columns = ['nom', 'count']
        
        # Bar chart en fonction du lieu
        fig = px.funnel(localisation, x="count", y='nom', color_discrete_sequence=couleurs[1:], width=1200, height=1200, title="Nombre d'annonces par lieu")
        
        # Affichage
        st.title("Ici c'est Paris !")
        st.plotly_chart(fig)
        
    if app_mode1 == lst[5]: 
        #image à recupérer sur le drive
        st.title("Sors de ton ghetto !")
        st.image(Image.open("C:/Users/Utilisateur/workspace/IA_Projets/GRETA/Projet 2/annonces_departement.png"))
        
        pass
        
    if app_mode1 == lst[6]: 
        menu_ = ["Post",
                "Classes de poste",
                "Niveau" 
                ]

        menu_st = st.sidebar.radio("Choisis l'entreprise !", menu_)
        # st.write(menu_st)
        if menu_st == menu_[0]: 
            # Scatter plot des postes en fonction des entreprises
            fig = px.scatter(csvExtract, y="Nom de la société", x="Poste")
            fig.update_traces(marker_size=10, marker_color=couleurs[0])
            
            # Affichage
            st.title("Choisis l'entreprise qui te convient !")
            st.plotly_chart(fig)
            pass
        
        if menu_st == menu_[1]: 
            # Scatter plot des classes de postes en fonction des entreprises
            fig = px.scatter(csvExtract, y="Nom de la société", x="Poste_class")
            fig.update_traces(marker_size=10, marker_color=couleurs[1])

            # Affichage
            st.title("Pour ta carrière présente...")
            st.plotly_chart(fig)
            pass
        
        if menu_st == menu_[2]: 
            # Scatter plot des niveaux en fonction des entreprises
            fig = px.scatter(csvExtract, y="Nom de la société", x="Niveau")
            fig.update_traces(marker_size=10, marker_color=couleurs[2])
            
            # Affichage
            st.title("... et future !")
            st.plotly_chart(fig)
            pass
            
        
        
        pass
        
# =============================================================================
# Fin Evelyne
# =============================================================================
                
if selected_menu == menu[3]:
    
    
    lst= ['Offres',
          'Tableau des compétences',
          'Toutes les offres et leurs compétences',
          'Salaires minimum evalué par un RandomForestRegressor', 
          'Meilleur modele pour evalué le salaire minimum & maximum'
          ]
    
    app_mode1 = st.sidebar.selectbox('', lst)
    
    st.sidebar.write("_____________________________________________________")
    
    
    if app_mode1 == lst[0]:
        app_mode11 = st.sidebar.selectbox('Classe de poste',np.sort(csvExtract['Poste_class'].unique()))
        app_mode12 = st.sidebar.selectbox('Poste',np.sort((csvExtract['Poste'][csvExtract['Poste_class']==app_mode11]).unique()))
 
        try :              
            vectorizer = CountVectorizer(strip_accents='unicode', tokenizer=lambda x: x.split(', '))
            matrix2 = vectorizer.fit_transform(csvExtract.dropna()['Compétences'][csvExtract['Poste']==app_mode12])
                       
            counts = pd.DataFrame(matrix2.toarray(), columns=vectorizer.get_feature_names())
            words=vectorizer.get_feature_names_out()
            
            app_mode13 = st.sidebar.selectbox('Compétence',words)

            st.title(app_mode1)
            st.write(app_mode13)
                  
            csvExtract1 = csvExtract[['Intitulé du poste','Nom de la société','Compétences']][csvExtract['Poste']==app_mode12].index
          
            h=[]
            
            for i in csvExtract[['Intitulé du poste','Nom de la société','Compétences']][csvExtract['Poste']==app_mode12].index:
                lst = csvExtract['Compétences'][i]
                if app_mode13 in lst:
                    h.append(i)
            
            if len(csvExtract.axes[0])>0:
                st.write(csvExtract[['Intitulé du poste','Nom de la société','Compétences']].iloc[h])    
        except:
             st.title(app_mode1)
             pass
       
    if app_mode1 == lst[1]:
        st.title(app_mode1)
        st.write(exploded(csvExtract))
        
    if app_mode1 == lst[2]:
        st.title(app_mode1)
        df_pour_tri = csvExtract.join(exploded(csvExtract))
        st.write(df_pour_tri)

    if app_mode1 == lst[3]:
        #Définition de notre X(données) et y(target) pour salaire minimum
        csvExtract = csvExtract.dropna()
        y = csvExtract['Salaire minimum']
        y2 = csvExtract['Salaire maximum']
        X = csvExtract.drop(['Salaire minimum', 'Salaire maximum', 'Intitulé du poste'], axis=1)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)
        X_train, X_test, y2_train, y2_test = train_test_split(X, y2, test_size=0.3, random_state=10)
        
        
        #Création de l'étape preparation du pipeline
        #vectorizer = CountVectorizer(strip_accents='unicode', tokenizer=lambda x: x.split(', '))
        preparation = ColumnTransformer(
            transformers=[       
                ('data_cat',
                 OneHotEncoder(handle_unknown='ignore'),
                 ['Nom de la société', 'Poste', 'Poste_class', 'Niveau']), #étendre avec poste & classe de poste & niveau
                ('data_tex',
                 CountVectorizer(strip_accents='unicode', tokenizer=lambda x: x.split(', ')),
                 'Compétences')         
            ])
        
        set_config(display='diagram')
        
        #Création du pipeline
        model_lm = Pipeline([('scaler', preparation),
                                ('classifier',RandomForestRegressor())])
        
        #Entraînement du modèle
        model_lm.fit(X_train, y_train);
        y_pred = model_lm.predict(X_test)
        
        st.title(app_mode1)
        st.write(f"La moyenne : {y_pred.mean()}")
        st.write(f"La médiane : {np.median(y_pred)}")
        
        st.write(f"r2_score : {r2_score(y_test, y_pred)*100:.2f}%")
         
    if app_mode1 == lst[4]:
        csvExtract = csvExtract.dropna()
        y = csvExtract['Salaire minimum']
        y2 = csvExtract['Salaire maximum']
        X = csvExtract.drop(['Salaire minimum', 'Salaire maximum', 'Intitulé du poste'], axis=1)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)
        X_train, X_test, y2_train, y2_test = train_test_split(X, y2, test_size=0.3, random_state=10)
        
        #Création de l'étape preparation du pipeline
        #vectorizer = CountVectorizer(strip_accents='unicode', tokenizer=lambda x: x.split(', '))
        preparation = ColumnTransformer(
            transformers=[       
                ('data_cat',
                 OneHotEncoder(handle_unknown='ignore'),
                 ['Nom de la société', 'Poste', 'Poste_class', 'Niveau']), #étendre avec poste & classe de poste & niveau
                ('data_tex',
                 CountVectorizer(strip_accents='unicode', tokenizer=lambda x: x.split(', ')),
                 'Compétences')         
            ])
        
        set_config(display='diagram')
        
        #Création du pipeline
        model_lm = Pipeline([('scaler', preparation),
                                ('classifier',RandomForestRegressor())])

        #TEST DU MEILLEURE MODELE POUR SALAIRE MINIMUM & MAX
        # Définition des paramètres à ajuster
        param_grid = [
            {
                'classifier': [LogisticRegression()],
                'classifier__C': [0.1, 1, 10],
            },
            {
                'classifier': [RandomForestClassifier()],
                'classifier__n_estimators': [10, 50, 100],
            },
            # {
            #     'classifier': [SVC()],
            #     'classifier__kernel': ['linear', 'rbf'],
            #     'classifier__C': [0.1, 1, 10],
            # },
            # {
            #     'classifier': [GradientBoostingRegressor()],
            #     'classifier__n_estimators': [50, 100, 200],
            #     'classifier__learning_rate': [0.1, 0.5, 1.0],
            #     'classifier__max_depth': [1, 3, 5]
            # },
            {
                'classifier': [RandomForestRegressor()],
                'classifier__n_estimators': [50, 100, 200],
                'classifier__max_depth': [5, 10, 20],
            },
            # {
            #     'classifier': [SVR()],
            #     'classifier__kernel':['linear','rbf'],
            #     'classifier__C':[0.1,1,10]
            # },
            # {
            #     'classifier': [Lasso()],
            #     'classifier__alpha':[0.1, 1, 10]
            # },
            # {
            #     'classifier': [Ridge()],
            #     'classifier__alpha':[0.1, 1, 10]
            # },
            {
                'classifier': [LinearRegression()],
            }
        ]
        
        # Initialisation de GridSearchCV
        grid = GridSearchCV(model_lm, param_grid, cv=5)
        grid2 = GridSearchCV(model_lm, param_grid, cv=5)
        
        # Entraînement du modèle
        grid.fit(X_train, y_train)
        grid2.fit(X_train, y2_train)
        
        st.title(app_mode1)
        
        # Affichage des meilleurs paramètres salaire maximum
        st.write("\n-----------------------------------------------------------")
        st.write("Meilleurs salaire max paramètres : ", grid2.best_params_)
        
        predictions2 = grid2.best_estimator_.predict(X_test)
        
        # st.write(predictions2)
        st.write(f"La moyenne : {(predictions2.mean()//100)*100}")
        st.write(f"La médiane : {np.median(predictions2)}")
        
        # accuracy2 = grid2.best_estimator_.score(X_test, y2_test)
        # st.write("Accuracy: {:.2f}%".format(accuracy2*100))
        
        # Affichage des meilleurs paramètres salaire minimum
        st.write("\n-----------------------------------------------------------")
        st.write("Meilleurs salaire min paramètres : ", grid.best_params_)
        
        predictions = grid.best_estimator_.predict(X_test)
        
        # st.write(predictions)
        st.write(f"La moyenne : {(predictions.mean()//100)*100}")
        st.write(f"La médiane : {np.median(predictions)}")
        
        # accuracy = grid.best_estimator_.score(X_test, y_test)
        # st.write("Accuracy: {:.2f}%".format(accuracy*100))
        
    

