# Simulação de Controle de Temperatura com Q-Learning

## 🚀 Sobre o Projeto

Este projeto utiliza **Q-Learning**, uma técnica de aprendizado por reforço, para controlar a temperatura de um ambiente com base na quantidade de pessoas presentes. O objetivo é manter a temperatura entre 20°C e 23°C, proporcionando um ambiente confortável e produtivo, como uma sala de aula.

A simulação leva em consideração a ação de *aquecimento*, *resfriamento* ou *manter a temperatura*, com a recompensa sendo dada conforme a temperatura ideal é mantida. O número de pessoas no ambiente varia ao longo da simulação, o que influencia o controle de temperatura.

---

## 📈 Funcionalidades

- **Controle de Temperatura:** Mantém a temperatura entre 20°C e 23°C para maximizar o desempenho do ambiente.
- **Simulação de Q-Learning:** O agente aprende a tomar decisões (aquecimento, resfriamento ou manter a temperatura) com base nas recompensas obtidas.
- **Variação de Pessoas:** O número de pessoas no ambiente é aleatório e pode ser alterado para simular diferentes cenários.
- **Análise de Desempenho:** Visualização de recompensas e temperatura média ao longo dos episódios de aprendizado.

---

## 🧑‍💻 Como Executar

### Pré-requisitos

- Python 3.x
- Bibliotecas:
  - `numpy`
  - `matplotlib`
  - `random`

### Instalação

1. Clone o repositório:
   ```bash
   git clone https://github.com/seuusuario/controle-temperatura-ql.git

2. Navegue até o diretório do projeto:
    ```bash
    cd controle-temperatura-ql

3. Instale as dependências:
    ```bash
    pip install -r requirements.txt

4. Execute o script de simulação:
    ```bash
    python main.py

## 📊 Resultados Esperados

Ao executar a simulação, o agente aprenderá a controlar a temperatura do ambiente, buscando sempre manter a faixa ideal. Os resultados serão visualizados em gráficos, incluindo:

   - Recompensa Total ao Longo dos Episódios
   - Temperatura Média
   - Número Médio de Pessoas

## ⚙️ Como Funciona

O agente toma decisões de acordo com a técnica de Q-Learning, utilizando a tabela Q para atualizar e aprender a melhor ação com base nas recompensas que recebe. O modelo é treinado por meio de episódios, em que o ambiente simula variações no número de pessoas e na temperatura.
### Funções Principais:

  - **criar_estado(temp, pessoas):** Cria o estado do ambiente com a temperatura e número de pessoas.
  - **executar_acao(temp, acao, pessoas):** Aplica a ação escolhida (aquecimento, resfriamento ou manter) e retorna a nova temperatura.
  - **calcular_recompensa(temp):** Calcula a recompensa dependendo da temperatura (penalidades para temperaturas fora da faixa ideal).
  - **q_learning_simulacao(agent):** Executa a simulação do aprendizado, coletando dados para análise.
