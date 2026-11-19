# my_team.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from contest.capture_agents import CaptureAgent
from contest.game import Actions, Directions


def create_team(first_index, second_index, is_red,
                first='OffensiveReflexAgent', second='DefensiveReflexAgent',
                num_training=0):
    agents = {
        'OffensiveReflexAgent': OffensiveReflexAgent,
        'DefensiveReflexAgent': DefensiveReflexAgent,
    }
    return [agents[first](first_index), agents[second](second_index)]


class ReflexCaptureAgent(CaptureAgent):
    def register_initial_state(self, game_state):
        super().register_initial_state(game_state)
        self.start = game_state.get_agent_position(self.index)
        walls = game_state.get_walls()
        home_x = walls.width // 2 - int(self.red)
        enemy_x = home_x + (1 if self.red else -1)
        self.crossings = [
            (home_x, y) for y in range(1, walls.height - 1)
            if not walls[home_x][y] and not walls[enemy_x][y]
        ]
        self.home_targets = self.crossings or [self.start]

    def visible_enemies(self, game_state):
        enemies = [
            game_state.get_agent_state(index)
            for index in self.get_opponents(game_state)
        ]
        return [enemy for enemy in enemies if enemy.get_position() is not None]

    def distance_to(self, position, targets):
        return min((self.get_maze_distance(position, target) for target in targets),
                   default=0)

    def best_action(self, game_state, score):
        reverse = Directions.REVERSE[
            game_state.get_agent_state(self.index).configuration.direction
        ]

        def rank(action):
            successor = game_state.generate_successor(self.index, action)
            return score(successor, action), action != Directions.STOP, action != reverse

        return max(game_state.get_legal_actions(self.index), key=rank)


class OffensiveReflexAgent(ReflexCaptureAgent):
    def choose_action(self, game_state):
        own = game_state.get_agent_state(self.index)
        position = own.get_position()
        food = self.get_food(game_state).as_list()
        capsules = self.get_capsules(game_state)
        ghosts = [enemy for enemy in self.visible_enemies(game_state)
                  if not enemy.is_pacman and enemy.scared_timer <= 2]
        danger = min((self.get_maze_distance(position, ghost.get_position())
                      for ghost in ghosts), default=float('inf'))
        home_distance = self.distance_to(position, self.home_targets)
        carrying = own.num_carrying
        moves_left = game_state.data.timeleft // game_state.get_num_agents()
        return_home = own.is_pacman and (
            not food
            or (carrying > 0 and (
                carrying >= 4
                or len(food) <= 2
                or danger <= 4
                or moves_left <= home_distance + 3
            ))
        )

        def score(successor, action):
            next_state = successor.get_agent_state(self.index)
            next_position = next_state.get_position()
            dx, dy = Actions.direction_to_vector(action)
            intended_position = (position[0] + dx, position[1] + dy)
            if next_position != intended_position:
                return -1000

            deposited = next_state.num_returned - own.num_returned
            eaten = next_state.num_carrying + deposited - carrying
            took_capsule = len(capsules) > len(self.get_capsules(successor))
            active_ghosts = [
                enemy.get_position() for enemy in self.visible_enemies(successor)
                if not enemy.is_pacman and enemy.scared_timer <= 2
            ]
            risk = 0
            if next_state.is_pacman:
                ghost_distance = min(
                    (self.get_maze_distance(next_position, ghost) for ghost in active_ghosts),
                    default=float('inf'),
                )
                risk = 4 * max(0, 4 - ghost_distance) ** 2

            if return_home:
                target_distance = self.distance_to(next_position, self.home_targets)
            else:
                target_distance = self.distance_to(next_position, food)
                if capsules and danger <= 5:
                    target_distance = min(
                        target_distance, self.distance_to(next_position, capsules) - 2
                    )

            return 24 * deposited + 8 * eaten + 14 * took_capsule - target_distance - risk

        return self.best_action(game_state, score)


class DefensiveReflexAgent(ReflexCaptureAgent):
    def register_initial_state(self, game_state):
        super().register_initial_state(game_state)
        self.defended_food = set(self.get_food_you_are_defending(game_state).as_list())
        self.defended_capsules = set(self.get_capsules_you_are_defending(game_state))
        self.last_intrusion = None
        self.patrol_index = 0
        gates = self.home_targets
        self.patrol = list(dict.fromkeys(gates[index] for index in (
            len(gates) // 2, len(gates) // 4, 3 * len(gates) // 4,
        )))

    def choose_action(self, game_state):
        own = game_state.get_agent_state(self.index)
        position = own.get_position()
        invaders = [enemy.get_position() for enemy in self.visible_enemies(game_state)
                    if enemy.is_pacman]
        food = set(self.get_food_you_are_defending(game_state).as_list())
        capsules = set(self.get_capsules_you_are_defending(game_state))
        lost = (self.defended_food - food) | (self.defended_capsules - capsules)
        self.defended_food = food
        self.defended_capsules = capsules
        if invaders or lost:
            self.last_intrusion = min(
                invaders or sorted(lost),
                key=lambda target: self.get_maze_distance(position, target),
            )
        elif position == self.last_intrusion:
            self.last_intrusion = None

        if position == self.patrol[self.patrol_index]:
            self.patrol_index = (self.patrol_index + 1) % len(self.patrol)
        targets = invaders or [self.last_intrusion or self.patrol[self.patrol_index]]

        def score(successor, action):
            next_state = successor.get_agent_state(self.index)
            next_position = next_state.get_position()
            if next_state.is_pacman:
                return -1000 - self.distance_to(next_position, self.home_targets)
            distance = self.distance_to(next_position, targets)
            if own.scared_timer > 1 and invaders:
                dx, dy = Actions.direction_to_vector(action)
                if next_position != (position[0] + dx, position[1] + dy):
                    return -1000
                return -abs(distance - 3) - 8 * max(0, 3 - distance)

            remaining = sum(enemy.is_pacman for enemy in self.visible_enemies(successor))
            captured = len(invaders) - remaining
            return 30 * captured - distance

        return self.best_action(game_state, score)
