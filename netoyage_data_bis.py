# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 12:23:29 2023

@author: Utilisateur
"""
import numpy as np
import pandas as pd

def labelisations_colonne_intitule_poste(tab_origin):

    x=0
    
    df1=pd.DataFrame()
    df2=pd.DataFrame()
    
    for lng in tab_origin.index:
        
        #extraction de la colonne Intitulé du poste
        val = tab_origin.iloc[lng]['Intitulé du poste']
    
        #suppression de H/F...    
        # val = val.replace('(', '').replace(')', '').replace('H/F', '').replace('F/H', '').replace('h/f', '')
      
        #simplifier le traitement en passant tout en minuscule
        val = val.lower()
    
        job=""  
        class_job =""
        niveau =""
        
        #Ce sont les intitulés en rapport avec les annoces pour la data 
        if val.find('data')!=-1 or val.find('données')!=-1 or val.find('virtualisation')!=-1 or val.find('digital')!=-1:
            
            class_job="Data"
            
            if val.find('analyst')!=-1 or val.find('analytics')!=-1 or val.find('analyses')!=-1  or val.find('analyse')!=-1 :
                job = "Data analyst" 
                # print(str(lng) + "  "+val)
            
            elif val.find('scientist')!=-1 or val.find('datascience')!=-1:
                job = "Data scientist"
                # print(str(lng) + "  "+val)
            
            elif val.find('manager')!=-1 or val.find('management')!=-1:
                job = "Data manager"
                # print(str(lng) + "  "+val)
            
            elif val.find('contrôleur')!=-1:
                job = "Data controler"
                # print(str(lng) + "  "+val)
            
            elif val.find('big data')!=-1 or val.find('bigdata')!=-1: 
                job = "Big Data"
                #print(str(lng) + "  "+val)
            
            elif val.find('ingenieur')!=-1 or val.find('engineer')!=-1 or val.find('ingénieur')!=-1:
                job = "Data ingenier"
                # print(str(lng) + "  "+val)
            
            elif val.find('architecte')!=-1:
                job = "Data architecte"
                # print(str(lng) + "  "+val)
            
            elif val.find('consultant')!=-1:
                job = "Data consultant"
                # print(str(lng) + "  "+val)
            
            elif val.find('viz')!=-1 or val.find('visualisation')!=-1 or val.find('tableau de bord')!=-1:
                job = "Data viz"
                # print(str(lng) + "  "+val)
            
            elif val.find('personnelles')!=-1 or val.find('protection et digital')!=-1:
                job = "DPO"
                # print(str(lng) + "  "+val)
            
            elif val.find('chef de projet')!=-1:
                job = "Chef de projet data"
                # print(str(lng) + "  "+val)
            
            else:
                job = "Data other"  
                # print(str(lng) + "  "+val)
            
        #Ce sont les intitulés en rapport avec les annoces pour le BI     
        elif val.find('business')!=-1 or val.find('bi')!=-1 or val.find('msbi')!=-1 or val.find('sap')!=-1 or val.find('marketing')!=-1 or val.find('décisionnel')!=-1 or val.find('talend')!=-1 or  val.find('cognos')!=-1 or  val.find('anaplan')!=-1 or  val.find('customer')!=-1:
            
            class_job="Business"
            # print(str(lng) + "  "+val)
            
            if val.find('analyst')!=-1 :
                job = "Business analyst" 
                # print(str(lng) + "  "+val)
                
            elif val.find('intelligence')!=-1 or val.find('bi')!=-1 or val.find('msbi')!=-1 or val.find('cognos')!=-1 or val.find('anaplan')!=-1 or val.find('sap')!=-1 or val.find('talend')!=-1:
                job = "Business intelligence" 
                # print(str(lng) + "  "+val)
                
            elif val.find('marketing')!=-1:
                job = "Marketing" 
                # print(str(lng) + "  "+val)
                
            else:
                job = "Business other"  
                #print(str(lng) + "  "+val)
                

        #Ce sont les intitulés en rapport avec les annoces pour le cloud         
        elif val.find('cloud')!=-1:
            class_job="Net"
            
            if val.find('sécurité')!=-1:
                job = "Cyber cloud" 
                # print(str(lng) + "  "+val)
                
            elif val.find('cloud')!=-1:
                job = "Ingenieurie cloud" 
                # print(str(lng) + "  "+val)
                
            else:
                job = "Net other"  
                # print(str(lng) + "  "+val)
            
        #Ce sont les intitulés en rapport avec les annoces pour developpeur 
        elif val.find('developpeur')!=-1 or val.find('développeur')!=-1 or val.find('logiciel')!=-1 or val.find('windows')!=-1 or val.find('web')!=-1 or val.find('unix/linux')!=-1 or val.find('developer')!=-1 or val.find('devops')!=-1 or val.find('développement')!=-1 or val.find('test')!=-1 or val.find('application')!=-1 or val.find('informatique')!=-1 or val.find('applicatif')!=-1:
            
            class_job="Génie informatique"
            # print(str(lng) + "  "+val)
            
            if val.find('web')!=-1 or val.find('angular')!=-1 or  val.find('php')!=-1 or  val.find('j2ee')!=-1:
                 job = "Developpement web" 
                 # print(str(lng) + "  "+val)
                
            elif val.find('test')!=-1 :
                job = "Developpement test" 
                # print(str(lng) + "  "+val)
                
            elif val.find('application')!=-1 or val.find('ios')!=-1 or val.find('applicatif')!=-1:
                job = "Developpement application" 
                # print(str(lng) + "  "+val)
       
            elif val.find('logiciel')!=-1 or val.find('software')!=-1 or val.find('windows')!=-1 or val.find('linux')!=-1 or val.find('etude')!=-1 or val.find('vba')!=-1:
                job = "Developpement logiciel" 
                # print(str(lng) + "  "+val)
                
            elif val.find('agile')!=-1 or val.find('devops')!=-1:
                job = "Developpement agile" 
                # print(str(lng) + "  "+val)
            
            elif val.find('full')!=-1 :
                job = "Developpement full stack" 
                # print(str(lng) + "  "+val)
            
            elif val.find('backend')!=-1 :
                job = "Developpement backend" 
                # print(str(lng) + "  "+val)
                       
            elif val.find('développeur')!=-1 or val.find('developpeur')!=-1 :
                job = "Developpement logiciel" 
                # print(str(lng) + "  "+val)
                
            elif val.find('technicien')!=-1 :
                job = "Technicien informatique" 
                # print(str(lng) + "  "+val)
               
            else:
                job = "Business other"  
                # print(str(lng) + "  "+val)
            
            
        #Ce sont les intitulés en rapport avec les reseaux autres que le cloud     
        elif val.find('réseau')!=-1 or val.find('cloud')!=-1 or val.find('cyber')!=-1 or val.find('middleware')!=-1 or val.find('vmware')!=-1 or val.find('cybersoc')!=-1 or val.find('infrastructure')!=-1:
            
            class_job="Net"
            # print(str(lng) + "  "+val)
            
            if val.find('cyber')!=-1 or val.find('sécurité')!=-1:
                job = "Cyber" 
                # print(str(lng) + "  "+val)
                
            elif val.find('middleware')!=-1 :
                job = "Middleware"
                # print(str(lng) + "  "+val)
                
            elif val.find('infrastructure')!=-1 or val.find('cisco')!=-1  or val.find('vmware')!=-1 :
                job = "Infrastructures"
                # print(str(lng) + "  "+val)
                
            else:
                job = "Net other"  
                # print(str(lng) + "  "+val)
            
        #Ce sont les intitulés en rapport avec le support aux entreprises   
        elif val.find('quality')!=-1 or val.find('risque')!=-1 or val.find('risques')!=-1 or val.find('recette')!=-1 or val.find('grands comptes')!=-1 or val.find('avant-vente')!=-1 or val.find('production')!=-1  or val.find('support')!=-1  or val.find('support')!=-1 or val.find('projets')!=-1 or val.find('architecte')!=-1 or val.find('technique')!=-1 or val.find('investor')!=-1 or val.find('qa')!=-1 or val.find('salesforce')!=-1 or  val.find('supply chain')!=-1  or val.find('crm')!=-1:
            
            class_job="Support"
            # print(str(lng) + "  "+val)
            
            if val.find('projet')!=-1 or val.find('architecte')!=-1:
                job = "Chef de projets" 
                # print(str(lng) + "  "+val)
                
            elif val.find('qa')!=-1 :
                job = "Responsable qualité" 
                # print(str(lng) + "  "+val)
                
            elif val.find('risques')!=-1 or val.find('expert')!=-1:
                job = "Auditeur/Expert" 
                # print(str(lng) + "  "+val)
                
            elif val.find('avant-vente')!=-1 or val.find('salesforce')!=-1  or val.find('grands comptes')!=-1:
                job = "Commercial" 
                # print(str(lng) + "  "+val)
                
            elif val.find('maintenance')!=-1 or val.find('production')!=-1  or val.find('relations')!=-1:
                job = "Operationnel" 
                # print(str(lng) + "  "+val)
                
            else:
                job = "Support other"  
                #print(str(lng) + "  "+val)
            
        #Ce sont les rejets qui n'ont pas pu être labélisées   
        else:
            x+=1
            class_job="Other"
            job = "Other"  
            # print(str(lng) + "  "+val)
            pass
        
        
             
        if val.find('technicien')!=-1 or val.find('contrôleur')!=-1 or val.find('technical')!=-1:
            niveau='Technicien'
            #print(str(lng) + "  "+val)
            
        elif val.find('doctorant')!=-1:
            niveau = "Doctorant" 
            # print(str(lng) + "  "+val)
            
        elif val.find('ingénieur')!=-1 or val.find('ingenieur')!=-1 or val.find('concepteur')!=-1 or val.find('engineer')!=-1:
            niveau = "Ingénieur" 
            # print(str(lng) + "  "+val)
            
        elif val.find('expert')!=-1 or val.find('consultant')!=-1  or val.find('auditeur')!=-1 or val.find('spécialiste')!=-1 :
            niveau = "Expert" 
            # print(str(lng) + "  "+val)
           
        elif val.find('développeur')!=-1 or val.find('developpeur')!=-1 or val.find('developer')!=-1  :
            niveau = "Développeur" 
            # print(str(lng) + "  "+val)
            
        elif val.find('responsable')!=-1 or val.find('chef de projet')!=-1 or val.find('manager')!=-1 or val.find('référent')!=-1:
            niveau = "Managment" 
            # print(str(lng) + "  "+val)
                
        elif val.find('analyst')!=-1 :
            niveau = "Analyst" 
            # print(str(lng) + "  "+val)
            
        elif val.find('scientist')!=-1 :
            niveau = "Scientist" 
            # print(str(lng) + "  "+val)
            
        elif val.find('architecte')!=-1 :
            niveau = "Architecte" 
            # print(str(lng) + "  "+val)
            
        elif val.find('stage')!=-1 or  val.find('alternance')!=-1 or  val.find('apprenti')!=-1:
            niveau = "Stage/Alternance/Apprenti" 
            # print(str(lng) + "  "+val)
            
        else:
            niveau='Autre'
            print(str(lng) + "  "+val)
            pass
        
        
        df1=pd.DataFrame([pd.Series([job, class_job, niveau])])
        df2= pd.concat([df2,df1], ignore_index=True)
    
    df2.columns = ['Poste', 'Poste_class', 'Niveau']
    
    return df2

url = "df_travail.csv"
df_csv = pd.read_csv(url)


df=labelisations_colonne_intitule_poste(df_csv)
csvExtract=pd.concat([df_csv,df], ignore_index=True, axis=1)
csvExtract.columns=[*df_csv.columns,*df.columns]
csvExtract['Salaire minimum'].replace(0, np.nan, inplace=True)
csvExtract['Salaire maximum'].replace(0, np.nan, inplace=True)

idf_data = pd.read_csv('communes_ile-de-france.csv')
idf_data.head()

# Formatage des noms de villes.
idf_data['nomcom'] = idf_data['nomcom'].str.lower()
idf_data['nomcom'] = idf_data['nomcom'].str.replace(' ','-')
idf_data.head()

# Création d'une liste de villes à partir de la colonne 'nomcom'.
villes = idf_data['nomcom'].tolist()
# La Défense n'est pas vraiment une commune, mais peut être rajoutée manuellement.
villes.append('la-defense')
villes[:10]

# Formatage des noms de villes dans la colomne 'Lieu' de notre dataframe principal.
csvExtract['Lieu'] = csvExtract['Lieu'].str.lower()
csvExtract['Lieu'] = csvExtract['Lieu'].str.replace(' ','-')
csvExtract.head()

# Filtrage de la colonne 'Lieu' pour conserver soit le nom de la ville, soit le département.
departements = ['paris', 'seine-et-marne', 'yvelines', 'essonne', 'hauts-de-seine', 'seine-saint-denis', 'val-de-marne', "val-d'oise"]
def clean_lieu(lieu):
  for ville in villes:
    if ville in lieu:
      return ville
    else:
      for departement in departements:
        if departement in lieu:
          return departement
      
csvExtract['Lieu'] = csvExtract['Lieu'].apply(clean_lieu)
csvExtract['Lieu'].fillna('île-de-france', inplace=True) 

csvExtract['Nom de la société'] = csvExtract['Nom de la société'].str.lower()
csvExtract['Nom de la société'] = csvExtract['Nom de la société'].str.replace(' sas','')
csvExtract['Nom de la société'] = csvExtract['Nom de la société'].str.replace(' s.a.s.','')
csvExtract['Nom de la société'] = csvExtract['Nom de la société'].str.replace(' group','')
csvExtract['Nom de la société'] = csvExtract['Nom de la société'].str.replace('-group','')
csvExtract['Nom de la société'] = csvExtract['Nom de la société'].str.replace('groupe ','')
csvExtract['Nom de la société'] = csvExtract['Nom de la société'].str.replace('.com','')
csvExtract['Nom de la société'] = csvExtract['Nom de la société'].str.replace(' & ','&')


df.to_csv('df_poste.csv')
csvExtract.to_csv('df_travail_poste.csv')

