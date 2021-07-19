from random import choice
import time
import sys


class Tictactoe():
    """
    A simple Tic tac toe implementation with minimax.

    """

    def __init__(self):
        """
        The board is initialized.

        """
        while True:
            self.__player_plays_first = input("Wanna play first or second \n\t- (1)\n\t- (2)\n")
            if self.__player_plays_first in ['1', 1, 'first', '1st', 'f']:
                self.__turn = 0
                self.__player_plays_first = True
                break
            elif self.__player_plays_first in ['2', 2, 'second', '2nd', 's']:
                self.__turn = 1
                self.__player_plays_first = False
                break
            else:
                continue

        self.__current_board_state = {
            (0, 0): ' ',
            (0, 1): ' ',
            (0, 2): ' ',
            (1, 0): ' ',
            (1, 1): ' ',
            (1, 2): ' ',
            (2, 0): ' ',
            (2, 1): ' ',
            (2, 2): ' ',
        }

        self.__players_tick = "X"
        self.__ais_tick = "O"

    def start_playing(self):
        """
        Method called to start playing the game.

        """
        for state_number in range(0, 9):
            # if self.gam
            if state_number % 2 == self.__turn:
                # Request input
                tuple_of_coordinates = Tictactoe.player_input([x for x in self.__current_board_state.keys() if self.__current_board_state[x] == " "])
                self.__current_board_state[tuple_of_coordinates] = self.__players_tick
                # print(self.__current_board_state)
                Tictactoe.Node.print_board(self.__current_board_state, state_number + 1)
            else:
                # In case the Ai plays first, the first move is hardcoded in (1,1) cell
                if state_number == 0:
                    time.sleep(1)
                    tuple_of_coordinates = (1, 1)
                    self.__current_board_state[tuple_of_coordinates] = self.__ais_tick
                    Tictactoe.Node.print_board(self.__current_board_state, state_number + 1)
                    continue
                tuple_of_coordinates = self.grow_graph(initial_board_state=self.__current_board_state, player_plays_first=self.__player_plays_first)
                self.__current_board_state[tuple_of_coordinates] = self.__ais_tick
                Tictactoe.Node.print_board(self.__current_board_state, state_number + 1)
            game_over, winner = Tictactoe.Node.check_if_the_game_is_over(self.__current_board_state, state_number + 1)

            if game_over:
                print(f'-Game Over-')
                if winner:
                    print(f'The winner is {winner}!')
                else:
                    print('Nobody won...')
                time.sleep(5)
                break

    def grow_graph(cls, initial_board_state=None, player_plays_first=True):
        """
        For each state of the game this method is called in order to grow a tree-graph (acyclic) starting from a root node, up to
        game-over states of the game (leaf nodes). After reaching the leaf nodes, the minimax values are assigned and a backtrack procedure
        takes place in order to assign minimax values to the parent nodes (bottom-up) till the root node is reached. This way the optimal move
        is found in a "minimax way"

        Parameters
        ----------
        initial_board_state : Expects a dictionary of the form shown in __init__method.
        player_plays_first : (True/False) Depending on whether the player plays first or not.

        Returns
        -------
        A tuple of the optimal move given the current state of the board, that can be fed as an input to TicTacToe.Node class.

        """
        root_node = cls.Node(new_input=None, parent_node=None, initial_board_state=initial_board_state)
        nodes_per_level = {}
        queue = [root_node]
        initial_depth = 9 - sum([x == " " for x in initial_board_state.values()])

        while queue:
            current_node = queue.pop(0)
            try:
                # print(f"Current level: {current_node.depth}")
                nodes_per_level[current_node.depth].append(current_node)
            except:
                # print(f"Current level: {current_node.depth}")
                nodes_per_level[current_node.depth] = [current_node]
            game_over, _ = current_node.check_if_the_game_is_over(current_node.board_state, current_node.depth)
            if game_over:
                current_node.assign_minimax_value()
                # print(current_node.minimax_value)
                # assign_minimax_value internally calls self.check_if_game_is_over()
                continue
            moves = current_node.available_moves
            nodes_per_level[current_node.depth + 1] = []
            for move in moves:
                child_node = cls.Node(new_input=move, parent_node=current_node)
                # nodes_per_level[current_node.depth+1].append(child_node)
                queue.append(child_node)

        # Now that the minimax values of the leaf nodes have been found, gotta backtrack assigning values to parent nodes
        current_depth = 9
        while current_depth > initial_depth:
            for node in nodes_per_level[current_depth]:
                if node.parent_node is None:  # Reached root_node
                    continue
                if node.parent_node.current_player == "X":
                    if node.minimax_value < node.parent_node.minimax_value:
                        node.parent_node.minimax_value = node.minimax_value
                elif node.parent_node.current_player == "O":
                    if node.minimax_value > node.parent_node.minimax_value:
                        node.parent_node.minimax_value = node.minimax_value
            current_depth -= 1

        nodes_list = nodes_per_level[initial_depth + 1]
        optimal_nodes_list = []

        optimal_minimax_value = max([val.minimax_value for val in nodes_list]) if player_plays_first == True else min([val.minimax_value for val in nodes_list])
        [optimal_nodes_list.append(node) for node in nodes_list if node.minimax_value == optimal_minimax_value]
        # To make it non-deterministic, gonna choose among the optimal moves with random.choice
        optimal_node = choice(optimal_nodes_list)

        # print(f'Chosen from a total of {len(optimal_nodes_list)} optimal move(s)')
        # print(f'Chosen optimal move coordinates: {optimal_node.move}')
        return optimal_node.move

    @staticmethod
    def player_input(available_moves):
        """
        Static method that just takes care of the player's input

        Parameters
        ----------
        available_moves : A list of the available moves for the current board state.

        Returns
        -------
        A tuple of the input coordinates, formatted such that it can be fed to TicTacToe.Node class.

        """
        while True:
            try:
                input_ = input("Input move: ")
                pos_1, pos_2 = input_.replace(" ", "").split(sep=",")
                position = (int(pos_1), int(pos_2))
                print(pos_1, pos_2)
                # Make sure that the input is in the right range
                if int(pos_1) not in [0, 1, 2] or int(pos_2) not in [0, 1, 2]:
                    print('Both inputs should be between 0 and 2. Please retry')
                    continue
                elif (int(pos_1), int(pos_2)) not in available_moves:
                    print('Cell already taken. Please retry')
                    continue
            except:
                print("Input should be of the form 'x,y'")
                continue
            else:
                # Also gotta make sure that the input is available

                if tuple([int(pos_1), int(pos_2)]) not in available_moves:
                    continue
                # if the following line of code is reached, it means that the input is valid, so gotta break the loop
                break
        return position

    class Node():
        """
        A node class used to wrap together the information of each possible state of the game, while growing the tree.

        """

        def __init__(self, new_input=None, parent_node=None, initial_board_state=None):
            self.__new_input = new_input

            self.parent_node = parent_node
            if parent_node is not None:
                self.depth = parent_node.depth + 1
            else:
                self.depth = 9 - sum([x == " " for x in initial_board_state.values()])
            if parent_node is not None:
                self.board_state = parent_node.board_state.copy()
                self.board_state[self.__new_input] = self.current_player
            else:
                self.board_state = initial_board_state

            if self.current_player == 'X':
                self.__minimax_value = 10
            else:
                self.__minimax_value = -10
            # self.print_board(self.board_state, self.depth)

        @property
        def move(self):
            return self.__new_input

        @property
        def current_player(self):
            """
            Returns
            -------
            The current player's tick (str): 'X' / 'O'

            """
            x_count = sum('X' == val for val in self.board_state.values())
            o_count = sum('O' == val for val in self.board_state.values())
            if x_count == o_count:
                return 'X'
            else:
                return 'O'

        @property
        def available_moves(self):
            """

            Returns
            -------
            A list of the available moves for the current node state.

            """
            available_list = []
            for cell in self.board_state.keys():
                if self.board_state[cell] == " ":
                    available_list.append(cell)
            return available_list

        @classmethod
        def print_board(cls, board_state, depth):
            """
            Class-method used to output a user-friendly visualization of the board.

            Parameters
            ----------
            board_state : Dictionary
            depth : The current state's depth

            """
            b00, b01, b02, b10, b11, b12, b20, b21, b22 = [vals for vals in board_state.values()]
            row_0 = '          |         |          \n' + '     ' + b00 + '    |    ' + b01 + '    |    ' + b02 + '     \n' + '          |         |         \n'
            row_1 = '          |         |          \n' + '     ' + b10 + '    |    ' + b11 + '    |    ' + b12 + '     \n' + '          |         |         \n'
            row_2 = '          |         |          \n' + '     ' + b20 + '    |    ' + b21 + '    |    ' + b22 + '     \n' + '          |         |         \n'
            horizontal_lines = '  __ _ __ + __ _ __ + __ _ __  \n'
            print(f"\nDepth: {depth}\n")
            print(row_0,
                  horizontal_lines,
                  row_1,
                  horizontal_lines,
                  row_2, sep="")
            print(" ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~")

        @staticmethod
        def check_if_the_game_is_over(current_board_state, current_board_depth):
            """
            Static-method that finds a winner if there is one. Returns a tuple of True or False, depending on whether the game is over or not and
            the winner tick 'X' or 'O'.

            Returns
            -------
            Tuple : (True/False , 'X'/'O')

            """
            # look for horizontal winner
            for i in range(3):
                if current_board_state[(i, 0)] == current_board_state[i, 1] and current_board_state[i, 0] == current_board_state[i, 2] and current_board_state[
                    (i, 0)] != " ":
                    return True, current_board_state[(i, 0)]
            # look for vertical winner
            for j in range(3):
                if current_board_state[(0, j)] == current_board_state[1, j] and current_board_state[0, j] == current_board_state[2, j] and current_board_state[
                    (0, j)] != " ":
                    return True, current_board_state[(0, j)]
            # diagonal winner
            if current_board_state[(0, 0)] == current_board_state[1, 1] and current_board_state[0, 0] == current_board_state[2, 2] and current_board_state[
                (0, 0)] != " ":
                return True, current_board_state[(0, 0)]
            if current_board_state[(0, 2)] == current_board_state[1, 1] and current_board_state[0, 2] == current_board_state[2, 0] and current_board_state[
                (0, 2)] != " ":
                return True, current_board_state[(0, 2)]
            # Game is over with no winner
            if current_board_depth >= 9:
                return True, None
            return False, None

        def assign_minimax_value(self):
            """
            Method used when a leaf node is reached. Assigns a minimax_value (one from -1/0/1) to the the node and returns (True/False).

            """
            game_over, winner = self.check_if_the_game_is_over(self.board_state, self.depth)
            if game_over:
                # Assuming that AI is the 'O' player
                if winner == 'O':
                    self.__minimax_value = 1
                elif winner == 'X':
                    self.__minimax_value = -1
                else:
                    self.__minimax_value = 0
                return True
            return False

        @property
        def minimax_value(self):
            """
            Returns
            -------
            The minimax value of the current node.

            """
            return self.__minimax_value

        @minimax_value.setter
        def minimax_value(self, value):
            """
            Used to override the minimax value while backtracking.

            """
            self.__minimax_value = value


if __name__ == "__main__":
    game = Tictactoe()
    game.start_playing()
    sys.exit()
