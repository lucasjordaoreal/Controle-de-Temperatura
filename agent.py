import numpy as np
import random

class TemperatureAgent:
    def __init__(self, target_temp=22, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.target_temp = target_temp  # Temperatura desejada
        self.current_temp = np.random.uniform(15, 35)  # Inicial aleatório
        self.alpha = alpha  # Taxa de aprendizado
        self.gamma = gamma  # Fator de desconto
        self.epsilon = epsilon  # Taxa de exploração
        self.states = [(temp, people) for temp in range(10, 40) for people in range(12, 32)]
        self.actions = ["cooling", "heating", "idle"]
        # Inicializa a Q-Table
        self.q_table = {state: {action: 0.0 for action in self.actions} for state in self.states}

    def sense_environment(self):
        """
        Simula a leitura de sensores para retornar temperatura e número de pessoas.
        """
        temp = self.current_temp + np.random.uniform(-0.5, 0.5)  # Pequenas variações
        people = np.random.randint(0, 11)  # Número de pessoas
        return round(temp), people

    def choose_action(self, state):
        """
        Escolhe uma ação usando o método epsilon-greedy.
        """
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(self.actions)  # Exploração
        # Escolha a ação com o maior valor Q
        return max(self.q_table[state], key=self.q_table[state].get)
    def update_q_table(self, state, action, reward, next_state):
        """
        Atualiza a tabela Q com base na equação de Q-Learning.
        """
        # Inicializa os estados na tabela Q, se necessário
        if state not in self.q_table:
            self.q_table[state] = {a: 0 for a in self.actions}
        if next_state not in self.q_table:
            self.q_table[next_state] = {a: 0 for a in self.actions}

        # Calcula o valor máximo do próximo estado
        max_q_next = max(self.q_table[next_state].values())

        # Atualiza a tabela Q usando a equação de Q-Learning
        current_q = self.q_table[state][action]
        self.q_table[state][action] = current_q + self.alpha * (reward + self.gamma * max_q_next - current_q)


    def decide_action(self, temp, people):
        """
        Mapeia o estado para a escolha de uma ação baseada em Q-Learning.
        """
        state = (temp, people)
        if state not in self.q_table:  # Estado fora do esperado
            return "idle"
        return self.choose_action(state)

    def act(self, action):
        """
        Executa a ação no ambiente, ajustando a temperatura.
        """
        if action == "cooling":
            self.current_temp -= 1
        elif action == "heating":
            self.current_temp += 1

    def get_reward(self, temp):
        """
        Define a recompensa baseada na proximidade da temperatura ao alvo.
        """
        if abs(temp - self.target_temp) <= 1:
            return 10  # Recompensa alta para atingir o alvo
        elif abs(temp - self.target_temp) <= 3:
            return 5  # Recompensa moderada
        else:
            return -1  # Penalidade para estados muito distantes

    def run_episode(self):
        """
        Executa um ciclo do agente, incluindo aprendizado.
        """
        state = self.sense_environment()
        action = self.decide_action(*state)
        self.act(action)
        new_state = self.sense_environment()
        reward = self.get_reward(new_state[0])
        # Atualiza a Q-Table
        self.update_q_table(state, action, reward, new_state)
        return state, action, reward, new_state
