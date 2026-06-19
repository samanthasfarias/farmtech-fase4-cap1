#  ARQUIVO 2 — treinar_modelo.py
#  O script pega os dados que geramos antes e ensina um modelo de Inteligência Artificial a prever a produtividade
#  agrícola com base nas leituras dos sensores.
#  A ideia é simples mostrar 200 dias de dados para o computador, ele aprende o padrão, e depois consegue prever sozinho
#  o que vai acontecer com novos valores que ele nunca viu.


import pandas as pd
import numpy as np
import joblib
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model    import LinearRegression
from sklearn.preprocessing   import StandardScaler
from sklearn.metrics         import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# BLOCO 1 — CARREGAR OS DADOS
# Primeiro passo: ler o arquivo CSV que foi gerado pelo gerar_dados.py.
# O pandas transforma esse arquivo numa tabela que o Python consegue manipular.

print("Carregando dados dos sensores...")
df = pd.read_csv('data/dados_sensores.csv')

print(f"Dados carregados: {df.shape[0]} linhas × {df.shape[1]} colunas")
print("\nEstatísticas gerais dos dados:")
print(df.describe().round(2))


# BLOCO 2 — SEPARAR O QUE O MODELO VAI USAR (X) DO QUE ELE VAI PREVER (y)
#
# X são as "pistas" que o modelo recebe para fazer a previsão:
# umidade, pH, temperatura, irrigação e fertilizante.
#
# y é o resultado que queremos prever: a produtividade em kg/ha.

FEATURES = ['umidade_solo', 'ph_solo', 'temperatura',
            'volume_irrigacao', 'fertilizante_aplicado']
TARGET   = 'produtividade'

X = df[FEATURES]
y = df[TARGET]

print(f"\nVariáveis preditoras (X): {FEATURES}")
print(f"Variável alvo (y): {TARGET}")


# BLOCO 3 — DIVIDIR OS DADOS EM TREINO E TESTE
#
# Não posso usar os mesmos dados para treinar e testar o modelo.
# Por isso dividi assim:
#   80% dos dados → o modelo estuda e aprende com eles (treino)
#   20% dos dados → ficam guardados para testar se aprendeu de verdade (teste)

X_treino, X_teste, y_treino, y_teste = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

print(f"\nDados de treino: {len(X_treino)} amostras")
print(f"Dados de teste:  {len(X_teste)} amostras")


# BLOCO 4 — NORMALIZAÇÃO DOS DADOS
# Cada variável tem uma escala completamente diferente.
# A umidade vai de 20 a 80, enquanto o pH vai de 5 a 8.
# Se deixar assim, o modelo pode achar que umidade é mais importante
# só porque os números são maiores, o que seria erro.
# O StandardScaler resolve isso transformando tudo para a mesma escala.
# Depois da normalização, todas as variáveis ficam comparáveis entre si.
# Importante: aprender a escala só com os dados de treino (fit_transform),
# e aplicar essa mesma escala nos dados de teste (só transform).
# Isso evita que informações do teste "vazem" para o treino.

scaler = StandardScaler()
X_treino_norm = scaler.fit_transform(X_treino)
X_teste_norm  = scaler.transform(X_teste)

print("\nNormalização aplicada com StandardScaler.")


# BLOCO 5 — TREINAR O MODELO DE REGRESSÃO LINEAR
# cada variável recebe um peso (coeficiente) que mostra exatamente quanto ela contribui para aumentar ou diminuir a produtividade.
# O model.fit() é o  "aprendizado" —
# o modelo analisa os 160 exemplos de treino e ajusta os pesos para minimizar os erros de previsão.

print("\nTreinando modelo de Regressão Linear...")
model = LinearRegression()
model.fit(X_treino_norm, y_treino)
print("Modelo treinado com sucesso!")

# Aqui o modelo aprendeu sobre cada variável.
# Peso positivo = essa variável ajuda a produtividade.
# Peso negativo = essa variável prejudica a produtividade.
coefs = pd.Series(model.coef_, index=FEATURES).sort_values(ascending=False)
print("\nCoeficientes do modelo (influência de cada variável):")
for var, coef in coefs.items():
    sinal = "↑ aumenta" if coef > 0 else "↓ diminui"
    print(f"  {var:25s}: {coef:+.2f}  ({sinal} a produtividade)")
print(f"\n  Intercepto (valor base): {model.intercept_:.2f}")


# BLOCO 6 — AVALIAR O MODELO COM MÉTRICAS
# Agora testa o modelo com os 40 dados que ele nunca viu.
#  quatro métricas para medir o quanto ele acertou:
# MAE - erro médio em kg/ha. Fácil de entender: se o MAE for 20, o modelo erra em média 20 kg/ha.
# MSE - parecido com MAE, mas pune mais os erros grandes.
# RMSE - raiz do MSE, fica na mesma unidade que a produtividade. É o mais usado na prática para comparar modelos.
# R² Vai de 0 a 1. R²=0.86 significa que o modelo explica 86% da variação  na produtividade. Quanto mais perto de 1, melhor.

y_previsto = model.predict(X_teste_norm)

mae  = mean_absolute_error(y_teste, y_previsto)
mse  = mean_squared_error(y_teste, y_previsto)
rmse = np.sqrt(mse)
r2   = r2_score(y_teste, y_previsto)

print("\n" + "="*50)
print("       MÉTRICAS DE DESEMPENHO DO MODELO")
print("="*50)
print(f"  MAE  (Erro Médio Absoluto):        {mae:.2f} kg/ha")
print(f"  MSE  (Erro Quadrático Médio):      {mse:.2f}")
print(f"  RMSE (Raiz do Erro Quadrático):    {rmse:.2f} kg/ha")
print(f"  R²   (Coeficiente Determinação):   {r2:.4f}  ({r2*100:.1f}% explicado)")
print("="*50)

if r2 >= 0.85:
    print("  ✅ Excelente! O modelo aprendeu muito bem os padrões dos dados.")
elif r2 >= 0.70:
    print("  ✅ Bom! O modelo capturou bem os padrões principais.")
elif r2 >= 0.50:
    print("  ⚠️  Razoável. O modelo aprendeu algo, mas pode melhorar.")
else:
    print("  ❌ O modelo precisa de ajustes — R² muito baixo.")


# BLOCO 7 — GERAR OS GRÁFICOS DE AVALIAÇÃO
# Três gráficos para visualizar o desempenho do modelo:
# 1. Real vs Previsto — pontos perto da diagonal = modelo bom
# 2. Importância das variáveis — quais fatores mais influenciam
# 3. Distribuição dos resíduos — erros centrados em zero = modelo bem calibrado
# 4. Matriz de correlação — como as variáveis se relacionam entre si

os.makedirs('data', exist_ok=True)

# Gráfico 1: Real vs Previsto + Importância das Variáveis
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('FarmTech Solutions — Avaliação do Modelo de Regressão',
             fontsize=14, fontweight='bold')

ax1 = axes[0]
ax1.scatter(y_teste, y_previsto, alpha=0.6, color='steelblue',
            edgecolors='white', linewidths=0.5)
min_val = min(y_teste.min(), y_previsto.min())
max_val = max(y_teste.max(), y_previsto.max())
ax1.plot([min_val, max_val], [min_val, max_val], 'r--',
         linewidth=2, label='Previsão perfeita')
ax1.set_xlabel('Produtividade Real (kg/ha)')
ax1.set_ylabel('Produtividade Prevista (kg/ha)')
ax1.set_title(f'Real vs Previsto  (R² = {r2:.3f})')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2 = axes[1]
cores = ['green' if c > 0 else 'red' for c in coefs.values]
bars = ax2.barh(coefs.index, coefs.values, color=cores, edgecolor='white')
ax2.axvline(x=0, color='black', linewidth=0.8)
ax2.set_xlabel('Coeficiente (impacto na produtividade)')
ax2.set_title('Importância de Cada Variável')
ax2.grid(True, alpha=0.3, axis='x')
for bar, val in zip(bars, coefs.values):
    ax2.text(val + (2 if val >= 0 else -2),
             bar.get_y() + bar.get_height()/2,
             f'{val:.1f}', va='center',
             ha='left' if val >= 0 else 'right', fontsize=9)

plt.tight_layout()
plt.savefig('data/grafico_avaliacao_modelo.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nGráfico salvo: data/grafico_avaliacao_modelo.png")

# Gráfico 2: Matriz de Correlação
fig, ax = plt.subplots(figsize=(8, 6))
correlacao = df.corr()
sns.heatmap(correlacao, annot=True, fmt='.2f', cmap='RdYlGn',
            center=0, square=True, ax=ax, linewidths=0.5)
ax.set_title('Matriz de Correlação entre Variáveis',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('data/grafico_correlacao.png', dpi=150, bbox_inches='tight')
plt.close()
print("Gráfico salvo: data/grafico_correlacao.png")

# Gráfico 3: Distribuição dos Resíduos
residuos = y_teste.values - y_previsto
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(residuos, bins=20, color='steelblue', edgecolor='white', alpha=0.8)
ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero (ideal)')
ax.set_xlabel('Resíduo (Real − Previsto) em kg/ha')
ax.set_ylabel('Frequência')
ax.set_title('Distribuição dos Resíduos', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('data/grafico_residuos.png', dpi=150, bbox_inches='tight')
plt.close()
print("Gráfico salvo: data/grafico_residuos.png")


# BLOCO 8 — SALVAR O MODELO TREINADO
# Salvo o modelo e o scaler em arquivos .pkl usando joblib.
# Assim não preciso treinar de novo toda vez que abrir o dashboard —
# o Streamlit carrega esses arquivos prontos.

os.makedirs('models', exist_ok=True)
joblib.dump(model,  'models/modelo_regressao.pkl')
joblib.dump(scaler, 'models/scaler.pkl')

print("\nModelo salvo: models/modelo_regressao.pkl")
print("Scaler salvo: models/scaler.pkl")
print("\n✅ treinar_modelo.py concluído com sucesso!")
print("   Arquivos gerados:")
print("   → models/modelo_regressao.pkl  (modelo treinado)")
print("   → models/scaler.pkl            (normalizador)")
print("   → data/grafico_avaliacao_modelo.png")
print("   → data/grafico_correlacao.png")
print("   → data/grafico_residuos.png")
