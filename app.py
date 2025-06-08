import dash
from dash import dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import numpy as np
import io
import openpyxl
from dateutil.relativedelta import relativedelta
import base64
from apoio.curvaOMS import *
import os

# wfa_percentiles_M = pd.read_csv('apoio/wfa_percentiles_M.csv')
# wfa_percentiles_F = pd.read_csv('apoio/wfa_percentiles_F.csv')
# lhfa_percentiles_M = pd.read_csv('apoio/lhfa_percentiles_M.csv')
# lhfa_percentiles_F = pd.read_csv('apoio/lhfa_percentiles_F.csv')
# bfa_percentiles_M = pd.read_csv('apoio/bfa_percentiles_M.csv')
# bfa_percentiles_F = pd.read_csv('apoio/bfa_percentiles_F.csv')
# hcfa_percentiles_M = pd.read_csv('apoio/hcfa_percentiles_M.csv')
# wfl_percentiles_M = pd.read_csv('apoio/wfl_percentiles_M.csv')

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container([
    html.Div([
        html.H1('Análise de dados antropométricos: Puericultura Xingu', 
                style={'color': '#3474A7'})
    ]),
    
    dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Arraste ou ',
            html.A('Selecione o arquivo .xlsx')
        ]),
        style={
            'borderStyle': 'dashed',
            'textAlign': 'center',
            'margin': '10px'
        }
    ),
    
    dcc.Tabs([
        dcc.Tab(label='Descritivas', children=[
            html.H3('Análise Descritiva', style={'color': '#3474A7'}),
            html.Div(id='descritiva-table')
        ]),
        
        dcc.Tab(label='Curvas', children=[
            html.H2('Curvas com medidas antropométricas - com Escores z', 
                    style={'color': '#3474A7'}),
            
            dbc.Row([
                dbc.Col([
                    html.H4('Curva Peso vs Idade - Meninos', style={'color': '#3474A4'}),
                    dcc.Graph(id='curva-peso-id-m'),
                    html.H4('Curva Estatura vs Idade - Meninos', style={'color': '#3474A4'}),
                    dcc.Graph(id='curva-estatura-id-m'),
                    html.H4('Curva IMC vs Idade - Meninos', style={'color': '#3474A4'}),
                    dcc.Graph(id='curva-imc-id-m'),
                    html.H4('Perímetro Encefálico vs Idade - Meninos', style={'color': '#3474A4'}),
                    dcc.Graph(id='curva-pc-id-m'),
                    html.H4('Peso vs Estatura - Meninos', style={'color': '#3474A4'}),
                    dcc.Graph(id='curva-peso-est-m')
                ], width=5),
                
                dbc.Col([
                    html.H4('Curva Peso vs Idade - Meninas', style={'color': '#3474A4'}),
                    dcc.Graph(id='curva-peso-id-f'),
                    html.H4('Curva Estatura vs Idade - Meninas', style={'color': '#3474A4'}),
                    dcc.Graph(id='curva-estatura-id-f'),
                    html.H4('Curva IMC vs Idade - Meninas', style={'color': '#3474A4'}),
                    dcc.Graph(id='curva-imc-id-f'),
                    html.H4('Perímetro Encefálico vs Idade - Meninas', style={'color': '#3474A4'}),
                    dcc.Graph(id='curva-pc-id-f'),
                    html.H4('Peso vs Estatura - Meninas', style={'color': '#3474A4'}),
                    dcc.Graph(id='curva-peso-est-f')
                ], width=5)
            ])
        ]),
        
        dcc.Tab(label='Controle', children=[
            html.Div(id='file-list')
        ])
    ]),
    
    html.Footer(
        children=[
            "© 2023 Autores: Julia Jallad, Lucas Calixto. Orientadores: Igor Peres, Leonardo Bastos e Silvio Hamacher; ",
            "Colaboradores: DEI PUC-Rio e Projeto Conexão Saúde Alto Xingu- Fundação Oswaldo Cruz; "
            "Última atualização: 2025 - Maria Luiza Lima Bastos",
        ],
        style={'margin-top': '20px', 'font-size': '0.8em'}
    )
])

def parse_data(contents):
    """Lê o arquivo enviado, processa e retorna o DataFrame pronto para análise."""

    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    df = pd.read_excel(io.BytesIO(decoded), sheet_name='PUERICULTURA', skiprows=3)
    df.columns = df.columns.str.strip()

    required_cols = ['DN', 'DATA AVALIAÇÃO', 'PESO', 'ESTATURA', 'SEXO', 'CONDUTA']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Coluna obrigatória '{col}' não encontrada. Colunas disponíveis: {df.columns.tolist()}")

    df['DN'] = pd.to_datetime(df['DN'], errors='coerce')
    df['DATA AVALIAÇÃO'] = pd.to_datetime(df['DATA AVALIAÇÃO'], errors='coerce')

    df['Meses_ajus'] = (df['DATA AVALIAÇÃO'] - df['DN']).dt.days // 30
    if df['Meses_ajus'].isnull().all():
        raise ValueError("Não foi possível calcular 'Meses_ajus'. Verifique as datas no arquivo.")

    df['Conduta_ajus'] = np.where(
        df['CONDUTA'] == 'ACOMPANHAMENTO', 'Acompanhamento',
        np.where(df['CONDUTA'] == 'TTO MEDICAMENTOSO', 'Em tratamento medicamentoso', df['CONDUTA'])
    )

    df['Peso_ajus'] = np.where(df['PESO'] > 1000, df['PESO']/1000, df['PESO'])
    df['IMC'] = df['Peso_ajus'] / ((df['ESTATURA']/100)**2)

    conditions = [
        df['Meses_ajus'] <= 12,
        (df['Meses_ajus'] > 12) & (df['Meses_ajus'] <= 24),
        (df['Meses_ajus'] > 24) & (df['Meses_ajus'] <= 48),
        df['Meses_ajus'] > 48
    ]
    choices = ['Lactente', 'Lactente 2', 'Pré-escola', '4 anos ou mais']
    df['Idade_categ'] = np.select(conditions, choices, default='Desconhecido')
    df['Idade_categ'] = pd.Categorical(df['Idade_categ'], categories=choices + ['Desconhecido'], ordered=True)

    if 'PT ' in df.columns and 'PT' not in df.columns:
        df = df.rename(columns={'PT ': 'PT'})

    return df

def create_summary_table(df):
    """Gera a tabela descritiva com estatísticas e frequências para a aba Descritivas."""

    idade_categ = pd.crosstab(df['Idade_categ'], df['SEXO'], margins=True, margins_name='Overall')
    idade_categ_percent = idade_categ.div(idade_categ.loc['Overall'], axis=1).fillna(0) * 100
    idade_categ_fmt = idade_categ.astype(str) + ' (' + idade_categ_percent.round(1).astype(str) + '%)'
    idade_categ_fmt = idade_categ_fmt.drop('Overall')

    def mean_std(series):
        return f"{series.mean():.2f} ± {series.std():.2f}" if not series.isnull().all() else "-"

    def stat_row(var):
        return [
            mean_std(df[var]),
            mean_std(df[df['SEXO']=='F'][var]),
            mean_std(df[df['SEXO']=='M'][var])
        ]

    def cat_count(var):
        tab = pd.crosstab(df[var], df['SEXO'], margins=True, margins_name='Overall')
        tab_percent = tab.div(tab.loc['Overall'], axis=1).fillna(0) * 100
        tab_fmt = tab.astype(str) + ' (' + tab_percent.round(1).astype(str) + '%)'
        tab_fmt = tab_fmt.drop('Overall')
        return tab_fmt

    rows = []
    rows.append(['Characteristic', 'Overall', 'F', 'M'])

    for idx, row in idade_categ_fmt.iterrows():
        rows.append([f'Idade: {idx}', row.get('Overall',''), row.get('F',''), row.get('M','')])

    rows.append(['Peso', *stat_row('Peso_ajus')])

    rows.append(['Estatura', *stat_row('ESTATURA')])

    pc_tab = cat_count('PC')
    for idx, row in pc_tab.iterrows():
        rows.append([f'PC: {idx}', row.get('Overall',''), row.get('F',''), row.get('M','')])

    if 'PA' in df.columns:
        rows.append(['Perímetro Abdominal', *stat_row('PA')])

    if 'PT' in df.columns:
        pt_tab = cat_count('PT')
        for idx, row in pt_tab.iterrows():
            rows.append([f'PT: {idx}', row.get('Overall',''), row.get('F',''), row.get('M','')])

    rows.append(['IMC', *stat_row('IMC')])

    return dash.dash_table.DataTable(
        columns=[{"name": i, "id": i} for i in rows[0]],
        data=[dict(zip(rows[0], r)) for r in rows[1:]],
        style_cell={'textAlign': 'center'},
        style_header={'fontWeight': 'bold'},
        style_table={'overflowX': 'auto'}
    )

@app.callback(
    Output('descritiva-table', 'children'),
    Input('upload-data', 'contents')
)
def update_summary(contents):
    """Callback: atualiza a tabela descritiva ao receber novo arquivo."""

    if contents is None:
        return []
    df = parse_data(contents)
    return create_summary_table(df)

def create_plot(percentile_df, data_df, x_col, y_col, title, x_title, y_title, gender):
    """Cria o gráfico de percentis e dados individuais para cada curva antropométrica."""
    fig = go.Figure()

    is_peso_est = x_col == 'ESTATURA'
    x_percentil = 'Length' if is_peso_est else 'Idade_meses'

    percentil_cores = {
        'P3': 'red',
        'P15': 'orange',
        'P50': 'blue',
        'P85': 'green',
        'P97': 'purple'
    }
    for percentile in ['P3', 'P15', 'P50', 'P85', 'P97']:
        df = percentile_df[percentile_df['percentil'] == percentile]
        fig.add_trace(go.Scatter(
            x=df[x_percentil], y=df['valor'],
            mode='lines', name=percentile,
            line=dict(color=percentil_cores.get(percentile, 'grey'), width=2)
        ))

    if gender == 'M':
        filtered_df = data_df[data_df['SEXO'] == 'M']
    else:
        filtered_df = data_df[data_df['SEXO'] == 'F']
    filtered_df = filtered_df.dropna(subset=[x_col, y_col])
    if filtered_df.empty:
        fig.add_annotation(text=f"Sem dados válidos para {y_title}",
                           xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False,
                           font=dict(size=16, color="red"))
        return fig
    fig.add_trace(go.Scatter(
        x=filtered_df[x_col], y=filtered_df[y_col],
        mode='markers', 
        marker=dict(color='black', size=8, opacity=0.7),
        hovertext=filtered_df.apply(lambda row: f"""
            Idade (meses): {row['Meses_ajus']}<br>
            {y_title}: {row[y_col]}<br>
            Aldeia: {row['ALDEIA']}<br>
            Nome: {row['NOME']}""", axis=1),
        hoverinfo='text',
        name='Dados Paciente'
    ))
    fig.update_layout(
        title=title,
        xaxis_title=x_title,
        yaxis_title=y_title,
        legend_title='Percentis',
        legend=dict(itemsizing='constant')
    )
    return fig

#Callbacks para cada gráfico
@app.callback(
    Output('curva-peso-id-m', 'figure'),
    Input('upload-data', 'contents')
)
def update_peso_id_m(contents):
    """Callback: atualiza o gráfico Peso vs Idade - Meninos."""

    if contents is None:
        return go.Figure()
    try:
        df = parse_data(contents)
        if 'Meses_ajus' not in df.columns:
            raise ValueError("Coluna 'Meses_ajus' não encontrada no DataFrame final.")
        return create_plot(wfa_percentiles_M, df, 'Meses_ajus', 'Peso_ajus',
                          'Peso vs Idade - Meninos', 'Idade (meses)', 'Peso (kg)', 'M')
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(text=f"Erro: {str(e)}",
                           xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False,
                           font=dict(size=16, color="red"))
        return fig

@app.callback(
    Output('curva-peso-id-f', 'figure'),
    Input('upload-data', 'contents')
)
def update_peso_id_f(contents):
    """Callback: atualiza o gráfico Peso vs Idade - Meninas."""

    if contents is None:
        return go.Figure()
    try:
        df = parse_data(contents)
        return create_plot(wfa_percentiles_F, df, 'Meses_ajus', 'Peso_ajus',
                          'Peso vs Idade - Meninas', 'Idade (meses)', 'Peso (kg)', 'F')
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(text=f"Erro: {str(e)}",
                           xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False,
                           font=dict(size=16, color="red"))
        return fig

@app.callback(
    Output('curva-estatura-id-m', 'figure'),
    Input('upload-data', 'contents')
)
def update_estatura_id_m(contents):
    """Callback: atualiza o gráfico Estatura vs Idade - Meninos."""

    if contents is None:
        return go.Figure()
    try:
        df = parse_data(contents)
        return create_plot(lhfa_percentiles_M, df, 'Meses_ajus', 'ESTATURA',
                          'Estatura vs Idade - Meninos', 'Idade (meses)', 'Estatura (cm)', 'M')
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(text=f"Erro: {str(e)}",
                           xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False,
                           font=dict(size=16, color="red"))
        return fig

@app.callback(
    Output('curva-estatura-id-f', 'figure'),
    Input('upload-data', 'contents')
)
def update_estatura_id_f(contents):
    """Callback: atualiza o gráfico Estatura vs Idade - Meninas."""

    if contents is None:
        return go.Figure()
    try:
        df = parse_data(contents)
        return create_plot(lhfa_percentiles_F, df, 'Meses_ajus', 'ESTATURA',
                          'Estatura vs Idade - Meninas', 'Idade (meses)', 'Estatura (cm)', 'F')
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(text=f"Erro: {str(e)}",
                           xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False,
                           font=dict(size=16, color="red"))
        return fig

@app.callback(
    Output('curva-imc-id-m', 'figure'),
    Input('upload-data', 'contents')
)
def update_imc_id_m(contents):
    """Callback: atualiza o gráfico IMC vs Idade - Meninos."""

    if contents is None:
        return go.Figure()
    try:
        df = parse_data(contents)
        return create_plot(bfa_percentiles_M, df, 'Meses_ajus', 'IMC',
                          'IMC vs Idade - Meninos', 'Idade (meses)', 'IMC (kg/m²)', 'M')
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(text=f"Erro: {str(e)}",
                           xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False,
                           font=dict(size=16, color="red"))
        return fig

@app.callback(
    Output('curva-imc-id-f', 'figure'),
    Input('upload-data', 'contents')
)
def update_imc_id_f(contents):
    """Callback: atualiza o gráfico IMC vs Idade - Meninas."""

    if contents is None:
        return go.Figure()
    try:
        df = parse_data(contents)
        return create_plot(bfa_percentiles_F, df, 'Meses_ajus', 'IMC',
                          'IMC vs Idade - Meninas', 'Idade (meses)', 'IMC (kg/m²)', 'F')
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(text=f"Erro: {str(e)}",
                           xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False,
                           font=dict(size=16, color="red"))
        return fig

@app.callback(
    Output('curva-pc-id-m', 'figure'),
    Input('upload-data', 'contents')
)
def update_pc_id_m(contents):
    """Callback: atualiza o gráfico Perímetro Cefálico vs Idade - Meninos."""

    if contents is None:
        return go.Figure()
    try:
        df = parse_data(contents)
        return create_plot(hcfa_percentiles_M, df, 'Meses_ajus', 'PC',
                          'Perímetro Cefálico vs Idade - Meninos', 'Idade (meses)', 'PC (cm)', 'M')
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(text=f"Erro: {str(e)}",
                           xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False,
                           font=dict(size=16, color="red"))
        return fig

@app.callback(
    Output('curva-pc-id-f', 'figure'),
    Input('upload-data', 'contents')
)
def update_pc_id_f(contents):
    """Callback: atualiza o gráfico Perímetro Cefálico vs Idade - Meninas."""

    if contents is None:
        return go.Figure()
    try:
        df = parse_data(contents)
        return create_plot(hcfa_percentiles_F, df, 'Meses_ajus', 'PC',
                          'Perímetro Cefálico vs Idade - Meninas', 'Idade (meses)', 'PC (cm)', 'F')
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(text=f"Erro: {str(e)}",
                           xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False,
                           font=dict(size=16, color="red"))
        return fig

@app.callback(
    Output('curva-peso-est-m', 'figure'),
    Input('upload-data', 'contents')
)
def update_peso_est_m(contents):
    """Callback: atualiza o gráfico Peso vs Estatura - Meninos."""

    if contents is None:
        return go.Figure()
    try:
        df = parse_data(contents)
        # Para peso vs estatura, x = 'ESTATURA', y = 'Peso_ajus'
        return create_plot(wfl_percentiles_M, df, 'ESTATURA', 'Peso_ajus',
                          'Peso vs Estatura - Meninos', 'Estatura (cm)', 'Peso (kg)', 'M')
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(text=f"Erro: {str(e)}",
                           xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False,
                           font=dict(size=16, color="red"))
        return fig

@app.callback(
    Output('curva-peso-est-f', 'figure'),
    Input('upload-data', 'contents')
)
def update_peso_est_f(contents):
    """Callback: atualiza o gráfico Peso vs Estatura - Meninas."""

    if contents is None:
        return go.Figure()
    try:
        df = parse_data(contents)
        return create_plot(wfl_percentiles_F, df, 'ESTATURA', 'Peso_ajus',
                          'Peso vs Estatura - Meninas', 'Estatura (cm)', 'Peso (kg)', 'F')
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(text=f"Erro: {str(e)}",
                           xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False,
                           font=dict(size=16, color="red"))
        return fig

@app.callback(
    Output('file-list', 'children'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def show_uploaded_file_info(contents, filename):
    """Callback: exibe nome e tamanho do arquivo recebido na aba Controle."""
    
    if contents is None or filename is None:
        return "Nenhum arquivo carregado."
    try:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        file_size_kb = len(decoded) / 1024
        return html.Div([
            html.P(f"Nome do arquivo: {filename}"),
            html.P(f"Tamanho: {file_size_kb:.2f} KB")
        ])
    except Exception as e:
        return f"Erro ao exibir informações do arquivo: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)