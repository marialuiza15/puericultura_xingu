import pandas as pd

# Função genérica para carregar e transformar dados com idade em dias
def carregar_percentis_com_idade(arquivo, coluna_idade):
    df = pd.read_excel(arquivo)
    df['Idade_meses'] = df[coluna_idade] / 30.4375
    df_long = df.melt(id_vars=[coluna_idade, 'Idade_meses'],
                      var_name='percentil',
                      value_name='valor')
    return df_long

# Função para carregar e transformar dados de peso por estatura (sem idade)
def carregar_percentis_sem_idade(arquivo, coluna_base):
    df = pd.read_excel(arquivo)
    df_long = df.melt(id_vars=[coluna_base],
                      var_name='percentil',
                      value_name='valor')
    return df_long

# Peso por idade 
wfa_percentiles_M = carregar_percentis_com_idade("apoio/wfa-boys-percentiles-expanded-tables.xlsx", "Age")
wfa_percentiles_F = carregar_percentis_com_idade("apoio/wfa-girls-percentiles-expanded-tables.xlsx", "Age")

# Altura por idade 
lhfa_percentiles_M = carregar_percentis_com_idade("apoio/lhfa-boys-percentiles-expanded-tables.xlsx", "Day")
lhfa_percentiles_F = carregar_percentis_com_idade("apoio/lhfa-girls-percentiles-expanded-tables.xlsx", "Day")

# Peso para altura 
bfa_percentiles_M = carregar_percentis_com_idade("apoio/bfa-boys-percentiles-expanded-tables.xlsx", "Age")
bfa_percentiles_F = carregar_percentis_com_idade("apoio/bfa-girls-percentiles-expanded-tables.xlsx", "Age")

# Perímetro cefálico por idade 
hcfa_percentiles_M = carregar_percentis_com_idade("apoio/hcfa-boys-percentiles-expanded-tables.xlsx", "Age")
hcfa_percentiles_F = carregar_percentis_com_idade("apoio/hcfa-girls-percentiles-expanded-tables.xlsx", "Age")

# Peso por estatura 
wfl_percentiles_M = carregar_percentis_sem_idade("apoio/wfl-boys-percentiles-expanded-tables.xlsx", "Length")
wfl_percentiles_F = carregar_percentis_sem_idade("apoio/wfl-girls-percentiles-expanded-tables.xlsx", "Length")
