import numpy as np
import matplotlib.pyplot as plt
import random
import agent

# Definindo os parâmetros do ambiente
TEMPERATURA_INICIAL = 31  # Temperatura inicial
TEMP_IDEAL = 20           # Temperatura ideal
AÇÕES = ['heating', 'cooling', 'idle']  # Ações possíveis
EPSILON = 0.1             # Taxa de exploração
ALPHA = 0.1               # Taxa de aprendizado
GAMMA = 0.9               # Fator de desconto

# Inicializa o agente
agent = agent.TemperatureAgent(target_temp=TEMP_IDEAL, alpha=ALPHA, gamma=GAMMA, epsilon=EPSILON)

# Inicializa tabela Q (via agente)
q_table = agent.q_table

# Função para criar um estado
def criar_estado(temp, pessoas):
    return (round(temp), pessoas)

# Função para inicializar a tabela Q
def inicializar_estado(estado):
    if estado not in q_table:
        q_table[estado] = {a: 0 for a in AÇÕES}

# Função para calcular recompensa
def calcular_recompensa(temp):
    if 20 <= temp <= 23:  # Se estiver dentro da faixa ideal
        return 10  # Recompensa por manter a temperatura dentro da faixa
    elif temp > 23:  # Temperatura acima da faixa
        return -5  # Penalidade por temperatura acima do ideal
    elif temp < 20:  # Temperatura abaixo da faixa
        return -5  # Penalidade por temperatura abaixo do ideal
    return 0  # Sem recompensa/penalidade

# Função para executar a ação de controle da temperatura com variação gradual e limites
def executar_acao(temp, acao, pessoas):
    # Variação máxima de 1°C por ação, podendo ser negativa (diminuindo) ou positiva (aumentando)
    if acao == "heating":
        nova_temp = temp + 1 if temp < 23 else temp  # Limita a temperatura a 23°C para evitar excesso
    elif acao == "cooling":
        nova_temp = temp - 1 if temp > 20 else temp  # Limita a temperatura a 20°C para evitar frio extremo
    else:
        nova_temp = temp  # Se a ação for "idle", a temperatura permanece a mesma
    return nova_temp

# Função para simulação de aprendizado com variação gradual da temperatura
def q_learning_simulacao(agent, episodios=1000):
    historico = []
    temperaturas = []
    pessoas_totais = []
    for episodio in range(episodios):
        temp = TEMPERATURA_INICIAL
        pessoas = random.randint(12, 40)
        estado = criar_estado(temp, pessoas)
        inicializar_estado(estado)
        recompensa_total = 0

        for _ in range(100):  # Máximo de iterações por episódio
            acao = agent.decide_action(temp, pessoas)  # Decisão do agente
            nova_temp = executar_acao(temp, acao, pessoas)  # Temperatura agora varia mais lentamente
            nova_pessoas = random.randint(1, 20)  # Simula variação de pessoas
            novo_estado = criar_estado(nova_temp, nova_pessoas)
            inicializar_estado(novo_estado)  # Garante que o estado exista na tabela Q
            
            # Usando a função get_reward para calcular a recompensa
            recompensa = calcular_recompensa(nova_temp)
            recompensa_total += recompensa
            
            # Atualiza a tabela Q usando o método do agente
            agent.update_q_table(estado, acao, recompensa, novo_estado)

            # Atualiza estado
            estado = novo_estado
            temp = nova_temp
            pessoas = nova_pessoas

        # Coleta os dados de cada episódio para análise
        historico.append(recompensa_total)
        temperaturas.append(temp)
        pessoas_totais.append(pessoas)

        # Exibe os detalhes de cada episódio
        print(f"Episódio {episodio+1}/{episodios} | Temperatura: {temp}°C | Pessoas: {pessoas} | Recompensa: {recompensa_total}")

    return historico, temperaturas, pessoas_totais

# Rodar simulação
resultados, temperaturas, pessoas_totais = q_learning_simulacao(agent)

# Visualizar aprendizado
fig, ax1 = plt.subplots()

ax1.set_xlabel('Episódios')
ax1.set_ylabel('Recompensa Total', color='tab:blue')
ax1.plot(resultados, color='tab:blue', label='Recompensa Total')

ax2 = ax1.twinx()
ax2.set_ylabel('Temperatura Média', color='tab:red')
ax2.plot(np.mean(temperaturas), color='tab:red', label='Temperatura Média', linestyle='--')

ax3 = ax1.twinx()
ax3.spines['right'].set_position(('outward', 60))  # Ajustar posição do terceiro eixo Y
ax3.set_ylabel('Pessoas Médias', color='tab:green')
ax3.plot(np.mean(pessoas_totais), color='tab:green', label='Pessoas Médias', linestyle='-.')

fig.tight_layout()  # Ajusta o layout
plt.title("Simulação de Q-Learning para Controle de Temperatura")
plt.show()
