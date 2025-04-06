"""
Link features to the radiomics dataframe 
"""

import pandas as pd
import numpy as np 

RADIOMICS = 'Users/jeleke/radiomics_base.xlsx'
CLINICAL = 'Users/jeleke/clinical_features.xlsx' 
QUAL = 'Users/jeleke/qualitatuve_features.xlsx' 

ROOT = '/Users/jeleke/'

# Load variables 
df = pd.read_excel(RADIOMICS) 
clinical_features = pd.read_excel(CLINICAL) 
qualitative_features = pd.read_excel(QUAL) 

# Merge dataframes 
age_dic = clinical_features.set_index('ID')['Age'].to_dict()
sex_dic = clinical_features.set_index('ID')['Sex'].to_dict()

infra_dic = qualitative_features.set_index('ID')['Infratentorial Involvement'].to_dict() 
extranodal_dic = qualitative_features.set_index('ID')['Extranodal Metastasis'].to_dict() 
itss_dic = qualitative_features.set_index('ID')['Intratumoral Susceptibiltiy'].to_dict() 
cystic_dic = qualitative_features.set_index('ID')['Cystic Degeneration'].to_dict() 

# Map onto radiomics frame
df['age'] = df['id'].map(age_dic)
df['sex'] = df['id'].map(sex_dic)

df['infra'] = df['id'].map(infra_dic)
df['extranodal'] = df['id'].map(extranodal_dic)
df['itss'] = df['id'].map(itss_dic)
df['cystic'] = df['id'].map(cystic_dic)

# Save linked df 
df.to_csv(ROOT + 'linked_df.csv', index = True) 
