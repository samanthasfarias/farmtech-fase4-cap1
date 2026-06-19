# 
#  ARQUIVO - app.py  (DASHBOARD — STREAMLIT)
#
#  O QUE FAZ: Cria um site interativo.
#  Nele o gestor consegue:
#   Ver os dados dos sensores em tabelas e gráficos
#   Ver as métricas do modelo de Machine Learning
#   Digitar novos valores e receber uma previsão de produtividade
#   Receber recomendações automáticas de irrigação e fertilização
#
#  COMO RODAR: No terminal, dentro da pasta farmtech, digitar:
#    streamlit run app.py
#  O Streamlit abre automaticamente no navegador.
#

# IMPORTAÇÕES 
import streamlit as st          # streamlit  = cria o site/dashboard
import pandas as pd             # pandas     = manipula tabelas
import numpy as np              # numpy      = cálculos matemáticos
import joblib                   # joblib     = carrega modelo salvo
import sqlite3                  # sqlite3    = lê banco de dados
import matplotlib.pyplot as plt # matplotlib = gráficos
import seaborn as sns           # seaborn    = gráficos mais bonitos
import os
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import StandardScaler


# 
#  CONFIGURAÇÃO DA PÁGINA
#  Isso define o título que aparece na aba do navegador,
#  o ícone e o layout (wide = usa a largura total da tela).
# 

st.set_page_config(
    page_title="Dashboard FarmTech Gestão ",
    page_icon="📊🍃🚜",
    layout="wide",
    initial_sidebar_state="expanded"
)


# 
#  ESTILO VISUAL (CSS)
#  Personaliza a aparência do dashboard com cores e fontes.
# 

st.markdown("""
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #2e7d32;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #555;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #f1f8e9;
        border-left: 5px solid #4caf50;
        padding: 14px 18px;
        border-radius: 8px;
        margin-bottom: 10px;
    }
    .metric-card h3 { margin: 0; color: #1b5e20; font-size: 1rem; }
    .metric-card p  { margin: 4px 0 0 0; font-size: 1.5rem; font-weight: bold; color: #2e7d32; }
    .rec-card {
        background: #fff8e1;
        border-left: 5px solid #ffa000;
        padding: 12px 16px;
        border-radius: 8px;
        margin-bottom: 8px;
    }
    .alert-green  { background:#e8f5e9; border-left:5px solid #43a047; padding:10px 14px; border-radius:6px; }
    .alert-yellow { background:#fff9c4; border-left:5px solid #f9a825; padding:10px 14px; border-radius:6px; }
    .alert-red    { background:#ffebee; border-left:5px solid #e53935; padding:10px 14px; border-radius:6px; }
    .section-header {
        font-size: 1.3rem;
        font-weight: 700;
        color: #2e7d32;
        border-bottom: 2px solid #a5d6a7;
        padding-bottom: 6px;
        margin: 24px 0 14px 0;
    }
    </style>
""", unsafe_allow_html=True)


# 
#  FUNÇÕES AUXILIARES - blocos de código reutilizáveis.
#  Definimos  e usamos várias vezes ao longo do código.
# 

@st.cache_data   # cache = o Streamlit memoriza o resultado para não recarregar toda hora
def carregar_dados():
    """Lê os dados do arquivo CSV. Se não existir, gera automaticamente."""
    if not os.path.exists('data/dados_sensores.csv'):
        # Roda o script de geração de dados automaticamente
        import subprocess
        subprocess.run(['python', 'gerar_dados.py'], check=True)
    return pd.read_csv('data/dados_sensores.csv')


@st.cache_resource   # cache_resource = carrega o modelo uma só vez (ele é pesado)
def carregar_modelo():
    """Carrega o modelo e o scaler. Se não existirem, treina automaticamente."""
    if not os.path.exists('models/modelo_regressao.pkl'):
        import subprocess
        subprocess.run(['python', 'treinar_modelo.py'], check=True)
    model  = joblib.load('models/modelo_regressao.pkl')
    scaler = joblib.load('models/scaler.pkl')
    return model, scaler


def calcular_metricas(df, model, scaler):
    """Recalcula as métricas do modelo para exibir no dashboard."""
    FEATURES = ['umidade_solo', 'ph_solo', 'temperatura',
                'volume_irrigacao', 'fertilizante_aplicado']
    X = df[FEATURES]
    y = df['produtividade']
    X_tr, X_ts, y_tr, y_ts = train_test_split(X, y, test_size=0.2, random_state=42)
    X_ts_norm = scaler.transform(X_ts)
    y_prev    = model.predict(X_ts_norm)
    mae  = mean_absolute_error(y_ts, y_prev)
    mse  = mean_squared_error(y_ts, y_prev)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_ts, y_prev)
    return mae, mse, rmse, r2, y_ts, y_prev


def gerar_recomendacoes(umidade, ph, temperatura, volume_irrig, fertilizante, producao_prevista):
    """
    Gera recomendações automáticas de manejo agrícola
    com base nos valores digitados pelo usuário.
    Retorna uma lista de dicionários com tipo e mensagem.
    """
    recs = []

    # --- Recomendações de Umidade ---
    if umidade < 30:
        recs.append({
            'tipo': 'danger',
            'icone': '🚨',
            'titulo': 'Solo muito seco!',
            'texto': f'Umidade atual: {umidade:.1f}%. Recomenda-se irrigação imediata '
                     f'com pelo menos 10–12 L/m². Risco alto de estresse hídrico na cultura.'
        })
    elif umidade < 45:
        recs.append({
            'tipo': 'warning',
            'icone': '⚠️',
            'titulo': 'Umidade abaixo do ideal',
            'texto': f'Umidade atual: {umidade:.1f}%. Programar irrigação nas próximas '
                     f'24–48 horas (6–8 L/m²). Monitorar a cultura de perto.'
        })
    elif umidade > 75:
        recs.append({
            'tipo': 'warning',
            'icone': '💧',
            'titulo': 'Solo com excesso de umidade',
            'texto': f'Umidade atual: {umidade:.1f}%. Suspender irrigação por 3–5 dias. '
                     f'Excesso de água pode causar apodrecimento de raízes.'
        })
    else:
        recs.append({
            'tipo': 'success',
            'icone': '✅',
            'titulo': 'Umidade do solo adequada',
            'texto': f'Umidade atual: {umidade:.1f}%. Dentro da faixa ideal (45%–75%). '
                     f'Manter o volume de irrigação atual.'
        })

    # --- Recomendações de pH ---
    if ph < 5.5:
        recs.append({
            'tipo': 'danger',
            'icone': '🧪',
            'titulo': 'Solo muito ácido — pH crítico!',
            'texto': f'pH atual: {ph:.1f}. Aplicar calcário (calagem) para elevar o pH. '
                     f'pH abaixo de 5.5 bloqueia a absorção de fósforo e cálcio pela planta.'
        })
    elif ph < 6.0:
        recs.append({
            'tipo': 'warning',
            'icone': '🧪',
            'titulo': 'Solo levemente ácido',
            'texto': f'pH atual: {ph:.1f}. Considerar aplicação leve de calcário dolomítico. '
                     f'pH ideal para a maioria das culturas: 6.0–7.0.'
        })
    elif ph > 7.5:
        recs.append({
            'tipo': 'warning',
            'icone': '🧪',
            'titulo': 'Solo alcalino — pH elevado',
            'texto': f'pH atual: {ph:.1f}. Aplicar enxofre elementar ou matéria orgânica '
                     f'para reduzir o pH. Solos muito alcalinos bloqueiam ferro e manganês.'
        })
    else:
        recs.append({
            'tipo': 'success',
            'icone': '✅',
            'titulo': 'pH do solo ideal',
            'texto': f'pH atual: {ph:.1f}. Dentro da faixa ótima (6.0–7.5). '
                     f'Não é necessária correção no momento.'
        })

    # --- Recomendações de Temperatura ---
    if temperatura > 32:
        recs.append({
            'tipo': 'warning',
            'icone': '🌡️',
            'titulo': 'Temperatura elevada',
            'texto': f'Temperatura: {temperatura:.1f}°C. Aumentar a frequência de irrigação '
                     f'para compensar a evapotranspiração elevada. Evitar irrigação no horário '
                     f'de pico solar (11h–15h).'
        })
    elif temperatura < 18:
        recs.append({
            'tipo': 'warning',
            'icone': '🌡️',
            'titulo': 'Temperatura baixa',
            'texto': f'Temperatura: {temperatura:.1f}°C. Reduzir volume de irrigação — '
                     f'a evapotranspiração é baixa e o solo demora mais para secar. '
                     f'Atentar para risco de geada se temperatura cair abaixo de 5°C.'
        })
    else:
        recs.append({
            'tipo': 'success',
            'icone': '✅',
            'titulo': 'Temperatura favorável',
            'texto': f'Temperatura: {temperatura:.1f}°C. Condições térmicas adequadas '
                     f'para o desenvolvimento vegetativo.'
        })

    # --- Recomendações de Fertilizante ---
    if fertilizante == 0:
        recs.append({
            'tipo': 'warning',
            'icone': '🌿',
            'titulo': 'Fertilização não aplicada',
            'texto': 'Considerar aplicação de fertilizante NPK conforme análise do solo. '
                     'A fertilização pode aumentar a produtividade prevista em até 80 kg/ha.'
        })
    else:
        recs.append({
            'tipo': 'success',
            'icone': '✅',
            'titulo': 'Fertilização em dia',
            'texto': 'Fertilizante aplicado neste ciclo. Monitorar resposta da cultura '
                     'nas próximas semanas.'
        })

    # --- Avaliação da produtividade prevista ---
    if producao_prevista < 300:
        recs.append({
            'tipo': 'danger',
            'icone': '📉',
            'titulo': 'Produtividade prevista muito baixa!',
            'texto': f'Estimativa: {producao_prevista:.0f} kg/ha. Revise urgentemente todos '
                     f'os parâmetros acima. Considerar análise de solo completa.'
        })
    elif producao_prevista < 500:
        recs.append({
            'tipo': 'warning',
            'icone': '📊',
            'titulo': 'Produtividade abaixo do potencial',
            'texto': f'Estimativa: {producao_prevista:.0f} kg/ha. Há margem para melhoria. '
                     f'Aplicar as recomendações acima para aumentar o rendimento.'
        })
    else:
        recs.append({
            'tipo': 'success',
            'icone': '🏆',
            'titulo': 'Boa produtividade estimada!',
            'texto': f'Estimativa: {producao_prevista:.0f} kg/ha. As condições estão favoráveis. '
                     f'Manter o manejo atual e monitorar os sensores regularmente.'
        })

    return recs


# 
#  SIDEBAR (MENU LATERAL)
#  A sidebar é o painel à esquerda do dashboard.
#  st.sidebar.xxx coloca elementos na barra lateral.
# 

with st.sidebar:
    st.image("https://img.icons8.com/color/96/plant-under-sun.png", width=80)
    st.markdown("## 📊 FarmTech ")
    st.markdown("**Assistente Inteligente**")
    st.markdown("---")

    pagina = st.radio(
        "📂 Navegação",
        options=[
            "🌎 Geral",
            "📊 Análise dos Dados",
            "🛠️ Modelo de Machine Learning",
            "⏳ Previsão Interativa",
            "ℹ️ Banco de Dados"
        ]
    )

    st.markdown("---")
    st.markdown("#### 🛠️ O Projeto")
    st.markdown("""
    Dashboard desenvolvido para o capitulo 1.

    **Tecnologias usadas:**
    - 🐍 Python
    - 📦 Scikit-learn
    - 📊 Streamlit
    - 🗃️ SQLite
    """)


# 
#  CARREGAMENTO DOS DADOS E DO MODELO - Executado quando o app inicia.
# 

df            = carregar_dados()
model, scaler = carregar_modelo()
mae, mse, rmse, r2, y_ts, y_prev = calcular_metricas(df, model, scaler)

FEATURES = ['umidade_solo', 'ph_solo', 'temperatura',
            'volume_irrigacao', 'fertilizante_aplicado']


# 
# ---------------------------------------------------------------------
#  PÁGINA 1 —  GERAL
# ---------------------------------------------------------------------
# 

if pagina == "🌎 Geral":

    st.markdown('<div class="main-title">🌿🌿 FarmTech Solutions</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Dashboard Agrícola Inteligente — Fase 4 | Grupo: Heleno Madeira, Matheus Fantini, Maykon Souza e Samanta Farias</div>', unsafe_allow_html=True)

    # --- KPIs (Indicadores-chave) no topo ---
    # st.columns(N) divide a linha em N colunas lado a lado
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>📋 Total de Amostras</h3>
            <p>{len(df)}</p>
        </div>""", unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>💧 Umidade Média</h3>
            <p>{df['umidade_solo'].mean():.1f}%</p>
        </div>""", unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>🌾 Produtiv. Média</h3>
            <p>{df['produtividade'].mean():.0f} kg/ha</p>
        </div>""", unsafe_allow_html=True)

    with c4:
        st.markdown(f"""
        <div class="metric-card">
            <h3>🤖 R² do Modelo</h3>
            <p>{r2:.3f}</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # --- Gráfico de linha: evolução das variáveis ---
    st.markdown('<div class="section-header">📈 Evolução das Leituras dos Sensores</div>', unsafe_allow_html=True)

    var_escolhida = st.selectbox(
        "Selecione a variável para visualizar ao longo do tempo:",
        options=['umidade_solo', 'ph_solo', 'temperatura', 'volume_irrigacao', 'produtividade'],
        format_func=lambda x: {
            'umidade_solo':     'Umidade do Solo (%)',
            'ph_solo':          'pH do Solo',
            'temperatura':      'Temperatura (°C)',
            'volume_irrigacao': 'Volume de Irrigação (L/m²)',
            'produtividade':    'Produtividade (kg/ha)'
        }[x]
    )

    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.plot(df.index, df[var_escolhida], color='#2e7d32', linewidth=1.2, alpha=0.8)
    ax.fill_between(df.index, df[var_escolhida], alpha=0.15, color='#4caf50')
    ax.set_xlabel('Registro (dia)')
    ax.set_ylabel(var_escolhida.replace('_', ' ').title())
    ax.set_title(f'Variação de {var_escolhida.replace("_", " ").title()} ao longo do tempo')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # --- Estatísticas descritivas ---
    st.markdown('<div class="section-header">📋 Resumo Estatístico dos Dados</div>', unsafe_allow_html=True)
    st.markdown("Esta tabela mostra para cada variável: quantidade de dados, média, desvio padrão, mínimo, máximo e quartis.")
    st.dataframe(df.describe().round(2), use_container_width=True)


# 
# ----------------------------------------------------------------------
#  PÁGINA 2 — ANÁLISE DOS DADOS
# -------------------------------------------------------------------------------
# 
elif pagina == "📊 Análise dos Dados":

    st.markdown('<div class="main-title">📊 Análise Exploratória dos Dados</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Visualize correlações, distribuições e padrões nos dados dos sensores</div>', unsafe_allow_html=True)

    # --- Matriz de Correlação ---
    st.markdown('<div class="section-header">🔗 Matriz de Correlação</div>', unsafe_allow_html=True)
    st.markdown("""
    **Como ler:** Cada célula mostra o quanto duas variáveis estão relacionadas.
    - **+1.0** (verde escuro) = quando uma sobe, a outra também sobe.
    - **-1.0** (vermelho escuro) = quando uma sobe, a outra desce.
    - **0.0** (amarelo) = não há relação entre as duas.
    """)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='RdYlGn',
                center=0, square=True, ax=ax, linewidths=0.5, annot_kws={'size': 10})
    ax.set_title('Correlação entre Variáveis dos Sensores', fontsize=13, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")

    # --- Histogramas de distribuição ---
    st.markdown('<div class="section-header">📊 Distribuição das Variáveis</div>', unsafe_allow_html=True)
    st.markdown("Cada gráfico abaixo mostra quantas vezes cada valor aparece nos dados (frequência).")

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle('Distribuição de Frequência das Variáveis', fontsize=14, fontweight='bold')
    axes = axes.flatten()   # transforma a grade 2×3 em lista de 6 eixos

    cores = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0', '#F44336', '#009688']
    for i, col in enumerate(df.columns):
        axes[i].hist(df[col], bins=20, color=cores[i], edgecolor='white', alpha=0.85)
        axes[i].set_title(col.replace('_', ' ').title(), fontsize=11)
        axes[i].set_xlabel('Valor')
        axes[i].set_ylabel('Frequência')
        axes[i].grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")

    # --- Scatter plot: variável vs produtividade ---
    st.markdown('<div class="section-header">🔍 Relação com a Produtividade</div>', unsafe_allow_html=True)
    st.markdown("Selecione uma variável para ver como ela se relaciona com a produtividade:")

    var_scatter = st.selectbox(
        "Variável para comparar com Produtividade:",
        ['umidade_solo', 'ph_solo', 'temperatura', 'volume_irrigacao']
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    scatter = ax.scatter(df[var_scatter], df['produtividade'],
                         alpha=0.5, c=df['produtividade'],
                         cmap='YlGn', edgecolors='none', s=40)
    plt.colorbar(scatter, ax=ax, label='Produtividade (kg/ha)')
    # Linha de tendência
    z = np.polyfit(df[var_scatter], df['produtividade'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df[var_scatter].min(), df[var_scatter].max(), 100)
    ax.plot(x_line, p(x_line), 'r--', linewidth=2, label='Tendência linear')
    ax.set_xlabel(var_scatter.replace('_', ' ').title())
    ax.set_ylabel('Produtividade (kg/ha)')
    ax.set_title(f'{var_scatter.replace("_", " ").title()} × Produtividade')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


# 
# ----------------------------------------------------------------------------------------------
#  PÁGINA 3 — MODELO DE MACHINE LEARNING
# ----------------------------------------------------------------------------------------------
# 

elif pagina == "🛠️ Modelo de Machine Learning":

    st.markdown('<div class="main-title">🤖 Modelo de Machine Learning</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Regressão Linear treinada com Scikit-learn para prever produtividade agrícola</div>', unsafe_allow_html=True)

    # --- Métricas em destaque ---
    st.markdown('<div class="section-header">📏 Métricas de Desempenho</div>', unsafe_allow_html=True)

    st.markdown("""
    As métricas abaixo indicam o quão bem o modelo aprendeu a prever a produtividade.
    Foram calculadas nos **dados de teste** (20% dos dados que o modelo nunca viu durante o treino).
    """)

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric("MAE (kg/ha)", f"{mae:.2f}",
                  help="Erro Médio Absoluto: em média, o modelo erra esse valor para cima ou para baixo.")
    with c2:
        st.metric("MSE", f"{mse:.2f}",
                  help="Erro Quadrático Médio: penaliza erros grandes com mais força.")
    with c3:
        st.metric("RMSE (kg/ha)", f"{rmse:.2f}",
                  help="Raiz do Erro Quadrático Médio: mesma unidade que a produtividade.")
    with c4:
        st.metric("R²", f"{r2:.4f}",
                  help="Quanto mais próximo de 1.0, melhor. Indica a % da variação explicada pelo modelo.")

    # Barra de progresso visual do R²
    st.markdown("**Qualidade do modelo (R²):**")
    st.progress(float(r2))
    if r2 >= 0.85:
        st.success(f"✅ Excelente! O modelo explica {r2*100:.1f}% da variação na produtividade.")
    elif r2 >= 0.70:
        st.success(f"✅ Bom! O modelo explica {r2*100:.1f}% da variação na produtividade.")
    else:
        st.warning(f"⚠️ O modelo explica {r2*100:.1f}% da variação — pode melhorar com mais dados.")

    st.markdown("---")

    # --- Gráfico: Real vs Previsto ---
    st.markdown('<div class="section-header">🎯 Valores Reais × Previstos</div>', unsafe_allow_html=True)
    st.markdown("Num modelo perfeito, todos os pontos estariam sobre a linha vermelha diagonal. Quanto mais próximos, melhor.")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(y_ts, y_prev, alpha=0.55, color='steelblue', edgecolors='white',
               linewidths=0.4, s=50)
    mn = min(y_ts.min(), y_prev.min())
    mx = max(y_ts.max(), y_prev.max())
    ax.plot([mn, mx], [mn, mx], 'r--', linewidth=2, label='Previsão perfeita')
    ax.set_xlabel('Produtividade Real (kg/ha)')
    ax.set_ylabel('Produtividade Prevista (kg/ha)')
    ax.set_title(f'Real vs Previsto  —  R² = {r2:.4f}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")

    # --- Importância das variáveis ---
    st.markdown('<div class="section-header">⚖️ Importância de Cada Variável</div>', unsafe_allow_html=True)
    st.markdown("""
    O gráfico abaixo mostra o coeficiente de cada variável no modelo.
    - **Barra verde (positivo):** quando essa variável aumenta, a produtividade tende a subir.
    - **Barra vermelha (negativo):** quando essa variável aumenta, a produtividade tende a cair.
    """)

    coefs = pd.Series(model.coef_, index=FEATURES).sort_values()
    fig, ax = plt.subplots(figsize=(8, 4))
    cores_bar = ['#e53935' if c < 0 else '#43a047' for c in coefs.values]
    bars = ax.barh(coefs.index, coefs.values, color=cores_bar, edgecolor='white')
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlabel('Coeficiente (impacto na produtividade kg/ha)')
    ax.set_title('Importância das Variáveis no Modelo de Regressão')
    ax.grid(True, alpha=0.3, axis='x')
    for bar, val in zip(bars, coefs.values):
        ax.text(val + (1 if val >= 0 else -1), bar.get_y() + bar.get_height()/2,
                f'{val:.1f}', va='center', ha='left' if val >= 0 else 'right', fontsize=9)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")

    # --- Distribuição dos Resíduos ---
    st.markdown('<div class="section-header">📉 Distribuição dos Resíduos</div>', unsafe_allow_html=True)
    st.markdown("""
    **Resíduo** = valor real − valor previsto.
    Um bom modelo tem resíduos centrados em zero e distribuídos simetricamente.
    Se o histograma parecer um sino centrado no zero, o modelo está bem calibrado.
    """)

    residuos = y_ts.values - y_prev
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(residuos, bins=20, color='steelblue', edgecolor='white', alpha=0.85)
    ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero ideal')
    ax.axvline(residuos.mean(), color='orange', linestyle='-', linewidth=1.5,
               label=f'Média ({residuos.mean():.1f})')
    ax.set_xlabel('Resíduo (kg/ha)')
    ax.set_ylabel('Frequência')
    ax.set_title('Distribuição dos Resíduos do Modelo')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


# 
# --------------------------------------------------------------------------------------------
#  PÁGINA 4 — PREVISÃO INTERATIVA
# --------------------------------------------------------------------------------------------
# 
elif pagina == "⏳ Previsão Interativa":

    st.markdown('<div class="main-title">⏳ Previsão Interativa de Produtividade</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Digite os valores dos sensores e receba uma previsão e recomendações automáticas</div>', unsafe_allow_html=True)

    st.markdown("### 🎛️ Insira os valores atuais do campo:")
    st.markdown("Use os controles abaixo para informar as condições do solo. O modelo calculará a produtividade estimada em tempo real.")

    col1, col2 = st.columns(2)

    with col1:
        umidade = st.slider(
            "💧 Umidade do Solo (%)",
            min_value=0.0, max_value=100.0, value=50.0, step=0.5,
            help="Porcentagem de água no solo. Ideal: 45%–75%."
        )
        ph = st.slider(
            "🧪 pH do Solo",
            min_value=4.0, max_value=9.0, value=6.5, step=0.1,
            help="Acidez/basicidade do solo. Ideal para maioria das culturas: 6.0–7.0."
        )
        temperatura = st.slider(
            "🌡️ Temperatura (°C)",
            min_value=5.0, max_value=45.0, value=25.0, step=0.5,
            help="Temperatura do solo em graus Celsius."
        )

    with col2:
        volume_irrig = st.slider(
            "🚿 Volume de Irrigação (L/m²)",
            min_value=0.0, max_value=20.0, value=7.0, step=0.5,
            help="Quantidade de água aplicada por metro quadrado."
        )
        fertilizante = st.radio(
            "🌿 Fertilizante Aplicado?",
            options=[0, 1],
            format_func=lambda x: "✅ Sim" if x == 1 else "❌ Não",
            horizontal=True,
            help="Indica se fertilizante foi aplicado neste ciclo."
        )

    # --- FAZER A PREVISÃO ---
    # Monta um array com os valores inseridos pelo usuário
    entrada = np.array([[umidade, ph, temperatura, volume_irrig, fertilizante]])
    # Normaliza com o mesmo scaler usado no treino
    entrada_norm = scaler.transform(entrada)
    # Faz a previsão
    producao_prevista = model.predict(entrada_norm)[0]
    producao_prevista = max(0, producao_prevista)  # garante não negativo

    # --- EXIBIR O RESULTADO ---
    st.markdown("---")
    st.markdown("### 🌾 Resultado da Previsão")

    col_res1, col_res2, col_res3 = st.columns(3)
    col_res1.metric("🌾 Produtividade Estimada", f"{producao_prevista:.0f} kg/ha")
    col_res2.metric("📊 Média Histórica", f"{df['produtividade'].mean():.0f} kg/ha")
    diferenca = producao_prevista - df['produtividade'].mean()
    col_res3.metric("📈 Diferença da Média", f"{diferenca:+.0f} kg/ha",
                    delta=f"{diferenca/df['produtividade'].mean()*100:+.1f}%")

    # Gauge visual da produtividade
    max_prod = df['produtividade'].max()
    pct = min(producao_prevista / max_prod, 1.0)
    cor_gauge = '#43a047' if pct > 0.6 else '#ffa000' if pct > 0.3 else '#e53935'

    st.markdown(f"""
    <div style="background:#f5f5f5;border-radius:10px;padding:6px 10px;margin:10px 0;">
        <div style="height:22px;width:{pct*100:.1f}%;background:{cor_gauge};
                    border-radius:8px;transition:width 0.5s;">
        </div>
    </div>
    <p style="text-align:center;color:#555;font-size:0.9rem;">
        {pct*100:.1f}% do máximo histórico ({max_prod:.0f} kg/ha)
    </p>
    """, unsafe_allow_html=True)

    # --- RECOMENDAÇÕES ---
    st.markdown("---")
    st.markdown("### 💡 Recomendações de Manejo Agrícola")
    st.markdown("Com base nos valores inseridos, o sistema gerou as seguintes recomendações:")

    recs = gerar_recomendacoes(umidade, ph, temperatura, volume_irrig, fertilizante, producao_prevista)

    mapa_css = {'success': 'alert-green', 'warning': 'alert-yellow', 'danger': 'alert-red'}
    for r in recs:
        css = mapa_css.get(r['tipo'], 'alert-green')
        st.markdown(f"""
        <div class="{css}" style="margin-bottom:10px;">
            <strong>{r['icone']} {r['titulo']}</strong><br>
            <span style="font-size:0.93rem;">{r['texto']}</span>
        </div>
        """, unsafe_allow_html=True)


# 
# ----------------------------------------------------------------------------------
#  PÁGINA 5 — BANCO DE DADOS
# ----------------------------------------------------------------------------------
# 

elif pagina == "ℹ️ Banco de Dados":

    st.markdown('<div class="main-title">🗄️ Banco de Dados SQLite</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Visualização dos dados armazenados no banco relacional da FarmTech Solutions</div>', unsafe_allow_html=True)

    st.markdown("""
    O banco de dados **farmtech.db** foi criado com **SQLite**, um sistema de banco de dados
    relacional leve que não precisa de instalação separada. Todos os dados dos sensores
    são armazenados na tabela **leituras_sensores**.
    """)

    # --- Estrutura da tabela ---
    st.markdown('<div class="section-header">🏗️ Estrutura da Tabela</div>', unsafe_allow_html=True)

    estrutura = pd.DataFrame({
        'Coluna':      ['id', 'umidade_solo', 'ph_solo', 'temperatura',
                        'volume_irrigacao', 'fertilizante_aplicado', 'produtividade', 'data_registro'],
        'Tipo SQL':    ['INTEGER PK', 'REAL', 'REAL', 'REAL',
                        'REAL', 'INTEGER', 'REAL', 'TEXT'],
        'Descrição':   [
            'Identificador único, gerado automaticamente',
            'Umidade do solo em porcentagem (%)',
            'Potencial hidrogeniônico do solo (0–14)',
            'Temperatura do solo em graus Celsius (°C)',
            'Volume de irrigação aplicado (L/m²)',
            '1 = fertilizante aplicado | 0 = não aplicado',
            'Produtividade estimada em kg por hectare (kg/ha)',
            'Data e hora do registro (gerado automaticamente)'
        ]
    })
    st.dataframe(estrutura, use_container_width=True, hide_index=True)

    st.markdown("---")

    # --- Dados do banco ---
    st.markdown('<div class="section-header">📋 Dados Armazenados</div>', unsafe_allow_html=True)

    n_linhas = st.slider("Quantas linhas deseja visualizar?", 5, 200, 20)

    try:
        conn = sqlite3.connect('data/farmtech.db')
        df_banco = pd.read_sql(
            f"SELECT * FROM leituras_sensores LIMIT {n_linhas}", conn
        )
        conn.close()
        st.dataframe(df_banco, use_container_width=True)

        # Contagem total
        conn = sqlite3.connect('data/farmtech.db')
        total = pd.read_sql("SELECT COUNT(*) as total FROM leituras_sensores", conn).iloc[0,0]
        conn.close()
        st.success(f"✅ Total de registros no banco de dados: **{total}**")

    except Exception as e:
        st.error(f"Erro ao conectar ao banco: {e}. Execute gerar_dados.py primeiro.")

    # --- DDL (código SQL que cria a tabela) ---
    st.markdown("---")
    st.markdown('<div class="section-header">📝 DDL — Código SQL de Criação da Tabela</div>', unsafe_allow_html=True)
    st.markdown("DDL significa *Data Definition Language* — é o código SQL usado para CRIAR a estrutura do banco.")

    st.code("""
CREATE TABLE IF NOT EXISTS leituras_sensores (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    umidade_solo          REAL    NOT NULL,
    ph_solo               REAL    NOT NULL,
    temperatura           REAL    NOT NULL,
    volume_irrigacao      REAL    NOT NULL,
    fertilizante_aplicado INTEGER NOT NULL,
    produtividade         REAL    NOT NULL,
    data_registro         TEXT    DEFAULT (datetime('now'))
);
    """, language='sql')
