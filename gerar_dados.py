#  ARQUIVO 1 — gerar_dados.py
# Objetivo: gerar dados simulados de sensores agrícolas e armazená-los
# em um arquivo CSV e em um banco de dados SQLite.



# Importação das bibliotecas necessárias para manipulação de dados,
# geração de valores aleatórios e armazenamento das informações.

import pandas as pd          # Manipulação de dados tabulares
import numpy as np           # Geração de números aleatórios e operações numéricas
import sqlite3               # Criação e acesso ao banco de dados SQLite
import os                    # Manipulação de diretórios e arquivos

# CONFIGURAÇÃO INICIAL
# Define uma semente para o gerador de números aleatórios,
# permitindo a reprodução dos mesmos resultados em diferentes execuções.
np.random.seed(42)

# Quantidade de registros simulados que serão gerados.
# Cada registro representa um dia de leitura dos sensores.
# 200 = 200 dias de leituras dos sensores
NUM_AMOSTRAS = 200

print("Gerando dados simulados dos sensores...")
print(f"Serão criadas {NUM_AMOSTRAS} amostras (dias de leitura).\n")



#  BLOCO 1 — CRIAR OS DADOS
#  Colunas da tabela
#  Cada coluna representa uma medição diferente do campo.


#  COLUNA 1: umidade_solo 
# Umidade do solo em porcentagem (0% = totalmente seco, 100% = encharcado).
# np.random.uniform(a, b, N) cria N números aleatórios entre a e b.
# Gera valores de umidade do solo entre 20% e 80%.
umidade_solo = np.random.uniform(low=20.0, high=80.0, size=NUM_AMOSTRAS)

#  COLUNA 2: ph_solo 
# Gera valores de pH em uma faixa compatível com condições agrícolas. pH do solo (escala de 0 a 14).
# pH ideal para a maioria das culturas: entre 5.5 e 7.5.
# Valores fora disso prejudicam a absorção de nutrientes.
ph_solo = np.random.uniform(low=5.0, high=8.0, size=NUM_AMOSTRAS)

# COLUNA 3: temperatura
# Temperatura do solo em graus Celsius.
# Variação típica: 15°C a 35°C.
temperatura = np.random.uniform(low=15.0, high=35.0, size=NUM_AMOSTRAS)

# COLUNA 4: volume_irrigacao 
# Quantidade de água aplicada em litros por metro quadrado (L/m²).
# Pode variar de 0 (dia sem irrigação) a 15 litros.
volume_irrigacao = np.random.uniform(low=0.0, high=15.0, size=NUM_AMOSTRAS)

# COLUNA 5: fertilizante_aplicado 
# 1 = fertilizante foi aplicado naquele dia | 0 = não foi aplicado
# np.random.randint(0, 2, N) cria N números inteiros: 0 ou 1.
fertilizante_aplicado = np.random.randint(low=0, high=2, size=NUM_AMOSTRAS)

#  COLUNA 6: produtividade (VARIÁVEL ALVO) 
# Variável alvo do conjunto de dados, representando a produtividade agrícola.
# Produtividade em kg por hectare (kg/ha).
#
# A fórmula simula como a produtividade se comporta no campo:
#   - Umidade alta → mais produtividade (peso 1.5)
#   - pH próximo de 6.5 → melhor absorção → mais produtividade (peso 20.0)
#   - Temperatura alta demais → prejudica (peso -2.0, ou seja, subtrai)
#   - Mais irrigação → mais produtividade (peso 10.0)
#   - Fertilizante aplicado → mais produtividade (peso 80.0 fixos)
#   - np.random.normal(0, 30, N) adiciona um "ruído" aleatório
#     (porque na vida real os dados nunca são perfeitos)
#
# A construção dessa variável permite avaliar posteriormente
# o desempenho de modelos preditivos aplicados ao conjunto de dados.

produtividade = (
1.5 * umidade_solo         +   # efeito positivo da umidade
20.0 * ph_solo            +   # influência do pH do solo
-2.0 * temperatura        +   # efeito negativo de temperaturas elevadas
10.0 * volume_irrigacao   +   # contribuição da irrigação
80.0 * fertilizante_aplicado + # incremento associado à aplicação de fertilizante
    np.random.normal(0, 30, NUM_AMOSTRAS)  # ruído aleatório (realismo)
)

# Garante que a produtividade nunca seja negativa
# (não faz sentido ter produção negativa no mundo real)
produtividade = np.maximum(produtividade, 0)



#  BLOCO 2 — MONTAR A TABELA (DataFrame)
#  Um DataFrame, uma planilha do Excel dentro do Python.
#  cada chave do dicionário vira um título de coluna.


df = pd.DataFrame({
    'umidade_solo':          np.round(umidade_solo, 2),         # arredonda para 2 casas decimais
    'ph_solo':               np.round(ph_solo, 2),
    'temperatura':           np.round(temperatura, 2),
    'volume_irrigacao':      np.round(volume_irrigacao, 2),
    'fertilizante_aplicado': fertilizante_aplicado,
    'produtividade':         np.round(produtividade, 2)
})

# Mostra as 5 primeiras linhas para conferir se ficou certo
print("As primeiras 5 linhas da tabela criada:")
print(df.head())
print(f"\nTotal de linhas: {len(df)}")
print(f"Total de colunas: {len(df.columns)}")
print(f"\nNomes das colunas: {list(df.columns)}")


#  BLOCO 3 — SALVAR EM ARQUIVO CSV
# Exporta os dados para um arquivo CSV.
# O parâmetro index=False impede a gravação do índice do DataFrame.


os.makedirs('data', exist_ok=True)   # cria a pasta "data" se ela não existir
caminho_csv = 'data/dados_sensores.csv'
df.to_csv(caminho_csv, index=False)
print(f"\nArquivo CSV salvo em: {caminho_csv}")



#  BLOCO 4 — SQLite
# Criação de um banco de dados SQLite para armazenamento persistente
# das leituras simuladas.
#
#  TABELA PRINCIPAL: leituras_sensores
#  Estrutura:
#    id                     INTEGER PRIMARY KEY AUTOINCREMENT  → número único de cada linha
#    umidade_solo         REAL    → número decimal (umidade %)
#    ph_solo                REAL    → número decimal (pH)
#    temperatura          REAL    → número decimal (°C)
#    volume_irrigacao      REAL    → número decimal (L/m²)
#    fertilizante_aplicado INTEGER → inteiro 0 ou 1
#    produtividade        REAL    → número decimal (kg/ha)
#    data_registro         TEXT    → texto com data/hora


caminho_banco = 'data/farmtech.db'

# sqlite3.connect() abre (ou cria) o arquivo de banco de dados
conn = sqlite3.connect(caminho_banco)

print(f"\nConectado ao banco de dados: {caminho_banco}")

# Cria a tabela com a estrutura definida
# IF NOT EXISTS = cria se a tabela não existir
conn.execute("""
    CREATE TABLE IF NOT EXISTS leituras_sensores (
        id                    INTEGER PRIMARY KEY AUTOINCREMENT,
        umidade_solo          REAL    NOT NULL,
        ph_solo               REAL    NOT NULL,
        temperatura           REAL    NOT NULL,
        volume_irrigacao      REAL    NOT NULL,
        fertilizante_aplicado INTEGER NOT NULL,
        produtividade         REAL    NOT NULL,
        data_registro         TEXT    DEFAULT (datetime('now'))
    )
""")

# Apaga dados antigos para não duplicar se rodar o script de novo
conn.execute("DELETE FROM leituras_sensores")
print("Tabela limpa para nova inserção.")

# Insere todos os dados do DataFrame na tabela de uma vez
# if_exists='append' = adiciona os dados sem apagar a tabela
df.to_sql('leituras_sensores', conn, if_exists='append', index=False)

# Confirma (salva) as alterações no banco
conn.commit()

# Verifica se os dados foram inseridos corretamente
cursor = conn.execute("SELECT COUNT(*) FROM leituras_sensores")
total = cursor.fetchone()[0]
print(f"Total de registros inseridos no banco: {total}")

# Mostra as 3 primeiras linhas diretamente do banco para confirmar
print("\nPrimeiros 3 registros no banco de dados:")
cursor = conn.execute("SELECT * FROM leituras_sensores LIMIT 3")
for linha in cursor.fetchall():
    print(linha)

# Encerra a conexão com o banco de dados.
conn.close()
print("\nConexão com o banco de dados encerrada.")
print("\n✅ gerar_dados.py concluído com sucesso!")
print("   Arquivos criados:")
print(f"   → {caminho_csv}")
print(f"   → {caminho_banco}")
