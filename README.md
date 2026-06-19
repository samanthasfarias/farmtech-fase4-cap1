# FIAP - Faculdade de Informática e Administração Paulista

<p align="center">
<a href= "https://www.fiap.com.br/"><img src="docs/images/logo-fiap.png" alt="FIAP - Faculdade de Informática e Admnistração Paulista" border="0" width=40% height=40%></a>
</p>

<br>

# Fase 4: Previsão Inteligente na Agricultura


## FIAP — FarmTech Solutions

### 👨‍🎓 Integrantes
* Heleno Madeira RM570302
* Matheus Fantini RM574078
* Maykon Souza RM574011
* Samanta Silva RM574120


## 📜 Descrição

Esse projeto é a continuação de tudo que construímos nas fases anteriores da FarmTech Solutions. Chegamos na parte que achamos mais interessante do curso: usar Inteligência Artificial de verdade para prever o que vai acontecer no campo antes mesmo de plantar.
A ideia principal é: e se a gente pudesse ensinar o computador a entender os dados dos sensores do campo e com isso prever a produtividade da lavoura?  Coletamos (de forma simulada) dados de umidade do solo, pH, temperatura, irrigação e fertilizante, e usamos esses dados para treinar um modelo de Machine Learning.
Para isso, usamos a biblioteca Scikit-learn do Python, que tem algoritmos de aprendizado de máquina prontos para usar. O modelo que escolhemos foi a Regressão Linear: cada variável recebe um peso que mostra exatamente o quanto ela influencia a produtividade. Por exemplo, o modelo aprendeu que aplicar fertilizante tem um impacto de +41 kg/ha na produtividade, enquanto temperatura alta tem impacto de -12 kg/ha. Isso faz sentido com o que acontece de verdade no campo.
Para guardar os dados, criamos um banco de dados SQLite — uma forma de organizar as informações dos sensores de um jeito que o sistema consiga acessar e atualizar facilmente. Essa parte corresponde ao requisito "Ir Além" da atividade.
E para apresentar tudo de um jeito que qualquer pessoa consegue entender, desenvolvemos um dashboard interativo com Streamlit. Nele, um gestor agrícola consegue ver os gráficos dos dados, as métricas do modelo e até testar previsões em tempo real — é só mover os controles deslizantes com os valores do campo e o sistema já mostra a produtividade estimada e o que fazer para melhorar.
No final, o modelo alcançou um R² de 0.8651, o que significa que ele consegue explicar 86,5% da variação na produtividade. Para nós, que estamos aprendendo isso agora, achamos um resultado bastante satisfatório.


## 📁 Estrutura de pastas

- **data/** —  pasta gerada automaticamente ao rodar os scripts, contendo o arquivo CSV com os dados dos sensores (`dados_sensores.csv`), o banco de dados SQLite (`farmtech.db`) e os gráficos de avaliação do modelo.

- **models** —  pasta gerada automaticamente ao treinar o modelo, contendo o modelo treinado (`modelo_regressao.pkl`) e o normalizador (`scaler.pkl`).
- **gerar_dados.py** — script Python que simula 200 registros de leituras de sensores agrícolas e os armazena em CSV e banco de dados SQLite.
- **treinar_modelo.py** — script Python que treina o modelo de Regressão Linear, calcula as métricas de desempenho (MAE, MSE, RMSE, R²) e gera os gráficos de avaliação.
- **app.py** — dashboard interativo desenvolvido com Streamlit, com 5 abas: visão geral, análise dos dados, modelo de ML, previsão interativa e banco de dados.


## 🔧 Como executar

Pré-requisitos: Python 3.10 ou superior instalado.

## Passo a passo

# Passo 1 — Clone o repositório
```bash
git clone https://github.com/samanthasfarias/farmtech-fase4-cap1.git
cd farmtech-fase4-cap1
```
# Passo 2 — Instale as dependências
```bash
pip install pandas scikit-learn streamlit matplotlib seaborn joblib
```
# Passo 3 — Gere os dados simulados dos sensores
```bash
py gerar_dados.py
```
Cria a pasta `data/` com 200 registros em CSV e banco SQLite.
# Passo 4 — Treine o modelo de Machine Learning
```bash
py treinar_modelo.py
```
# Treina a Regressão Linear, exibe as métricas no terminal e salva o modelo na pasta `models/`.

# Passo 5 — Abra o dashboard interativo
```bash
py -m streamlit run app.py
```
O navegador abrirá automaticamente em `http://localhost:8501`.

## 🗃 Resultados obtidos

# MAE	
20.46 kg/ha	- Erro médio de 20kg por hectare
# MSE
684.69 - Penaliza erros grandes
# RMSE 
26.17 kg/ha - Erro típico na mesma unidade da produtividade
# R² - 0.8651 - O modelo explica 86.5% da variação nos dados ✅



## 📎 Links

- **Repositório:** [Acesse o link do repositório](https://github.com/samanthasfarias/farmtech-fase4-cap1.git)
- **Vídeo demonstrativo (YouTube):** [Acesse o link da apresentação - Youtube] XXXXX


## 📋 Observações

Dados e Estáticas apenas para a demonstração.
