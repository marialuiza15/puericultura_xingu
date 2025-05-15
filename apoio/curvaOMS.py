import pandas as pd

# Função auxiliar para carregar e transformar os dados
def carregar_transformar_percentis(caminho_arquivo, coluna_idade, nome_coluna_idade_meses):
    df = pd.read_excel(caminho_arquivo)
    df[nome_coluna_idade_meses] = df[coluna_idade] / 30.4375
    df_long = df.melt(id_vars=[coluna_idade, nome_coluna_idade_meses], 
                      var_name='percentil', 
                      value_name='valor')
    return df_long

# Peso por idade
wfa_percentiles_M = carregar_transformar_percentis("apoio/wfa-boys-percentiles-expanded-tables.xlsx", "Age", "Idade_meses")
wfa_percentiles_F = carregar_transformar_percentis("apoio/wfa-girls-percentiles-expanded-tables.xlsx", "Age", "Idade_meses")

# Altura por idade
lhfa_percentiles_M = carregar_transformar_percentis("apoio/lhfa-boys-percentiles-expanded-tables.xlsx", "Day", "Idade_meses")
lhfa_percentiles_F = carregar_transformar_percentis("apoio/lhfa-girls-percentiles-expanded-tables.xlsx", "Day", "Idade_meses")

# Peso para altura
bfa_percentiles_M = carregar_transformar_percentis("apoio/bfa-boys-percentiles-expanded-tables.xlsx", "Age", "Idade_meses")
bfa_percentiles_F = carregar_transformar_percentis("apoio/bfa-girls-percentiles-expanded-tables.xlsx", "Age", "Idade_meses")

# Perímetro cefálico por idade
hcfa_percentiles_M = carregar_transformar_percentis("apoio/hcfa-boys-percentiles-expanded-tables.xlsx", "Age", "Idade_meses")
hcfa_percentiles_F = carregar_transformar_percentis("apoio/hcfa-girls-percentiles-expanded-tables.xlsx", "Age", "Idade_meses")

# Peso por estatura
def carregar_transformar_peso_por_estatura(caminho_arquivo):
    df = pd.read_excel(caminho_arquivo)
    df_long = df.melt(id_vars=["Length"], 
                      var_name='percentil', 
                      value_name='valor')
    return df_long

wfl_percentiles_M = carregar_transformar_peso_por_estatura("apoio/wfl-boys-percentiles-expanded-tables.xlsx")
wfl_percentiles_F = carregar_transformar_peso_por_estatura("apoio/wfl-girls-percentiles-expanded-tables.xlsx")
