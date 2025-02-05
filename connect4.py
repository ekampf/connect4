from abc import ABC, abstractmethod
import random
import time
from typing import List, Tuple, Optional
from dataclasses import dataclass
import numpy as np
import math

import signal
from contextlib import contextmanager


class TimeoutException(Exception):
    pass


@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")

    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


@dataclass
class GameState:
    """Represents the current state of the game"""

    board: List[List[str]]
    current_player: str
    last_move: Optional[Tuple[int, int]] = None  # (row, col)


class Player(ABC):
    """Abstract base class that competitors must implement"""

    def __init__(self, symbol: str):
        self.symbol = symbol  # 'X' or 'O'

    @abstractmethod
    def get_move(self, state: GameState) -> int:
        """
        Given the current game state, return the column (0-6) to play in
        Must return a valid move (column that isn't full)
        """
        pass


class ConnectFour:
    def __init__(
        self,
        player1: Player,
        player2: Player,
        columns: int = 14,
        rows: int = 12,
        delay: float = 1.0,
        animate: bool = True,
    ):
        self.columns = columns
        self.rows = rows
        self.board = [[" " for _ in range(columns)] for _ in range(rows)]
        self.player1 = player1  # X
        self.player2 = player2  # O
        self.current_player = player1
        self.delay = delay  # Delay between moves for visualization
        self.animate = animate

    def is_valid_move(self, col: int) -> bool:
        """Check if a move is valid"""
        return 0 <= col < self.columns and self.board[0][col] == " "

    def get_next_row(self, col: int) -> Optional[int]:
        """Get the next available row in the given column"""
        for row in range(self.rows - 1, -1, -1):
            if self.board[row][col] == " ":
                return row
        return None

    def make_move(self, col: int) -> Tuple[int, int]:
        """Make a move in the given column"""
        row = self.get_next_row(col)
        if row is None:
            raise ValueError("Invalid move")

        if self.animate:
            # Animate the piece falling
            for r in range(row + 1):
                if r > 0:
                    self.board[r - 1][col] = " "
                self.board[r][col] = self.current_player.symbol
                self.display_board()
                time.sleep(0.1)
                if r < row:
                    print("\033[F" * 8)  # Move cursor up 8 lines
        else:
            self.board[row][col] = self.current_player.symbol
            self.display_board()

        return row, col

    def check_winner(self, row: int, col: int) -> bool:
        """Check if the last move created a winning condition"""
        directions = [
            (0, 1),  # horizontal
            (1, 0),  # vertical
            (1, 1),  # diagonal right
            (1, -1),  # diagonal left
        ]

        symbol = self.board[row][col]

        for dr, dc in directions:
            count = 1

            # Check in positive direction
            r, c = row + dr, col + dc
            while (
                0 <= r < self.rows
                and 0 <= c < self.columns
                and self.board[r][c] == symbol
            ):
                count += 1
                r, c = r + dr, c + dc

            # Check in negative direction
            r, c = row - dr, col - dc
            while (
                0 <= r < self.rows
                and 0 <= c < self.columns
                and self.board[r][c] == symbol
            ):
                count += 1
                r, c = r - dr, c - dc

            if count >= 4:
                return True

        return False

    def is_board_full(self) -> bool:
        """Check if the board is full (tie game)"""
        return all(cell != " " for cell in self.board[0])

    def display_board(self):
        """Display the current board state with colors"""
        # ANSI color codes
        RED = "\033[91m"
        YELLOW = "\033[93m"
        RESET = "\033[0m"
        BLUE = "\033[94m"

        # Print column numbers
        print("\n " + BLUE + " ".join((str(i) for i in range(self.columns))) + RESET)
        print(BLUE + "--" * self.columns + RESET)

        # Print board with colored pieces
        for row in self.board:
            print(BLUE + "|" + RESET, end="")
            for cell in row:
                if cell == "X":
                    print(RED + "X" + RESET, end="")
                elif cell == "O":
                    print(YELLOW + "O" + RESET, end="")
                else:
                    print(" ", end="")
                print(BLUE + "|" + RESET, end="")
            print()
        print(BLUE + "--" * self.columns + RESET)

    def play_game(self) -> Optional[Player]:
        """Play a full game, returns the winner or None if tie"""
        while True:
            # Display current state
            if not self.animate:
                print(
                    f"\nPlayer {self.current_player.symbol}'s ({type(self.current_player).__name__}) turn"
                )
                self.display_board()

            # Get player's move
            try:
                state = GameState(
                    board=[row[:] for row in self.board],
                    current_player=self.current_player.symbol,
                )

                # TODO: Add timeout for player moves
                with time_limit(1):
                    col = self.current_player.get_move(state)

                if not self.is_valid_move(col):
                    print(f"Invalid move by player {self.current_player.symbol}")
                    return (
                        self.player1
                        if self.current_player == self.player2
                        else self.player2
                    )

                row, col = self.make_move(col)

            except Exception as e:
                print(f"Error from player {self.current_player.symbol}: {e}")
                return (
                    self.player1
                    if self.current_player == self.player2
                    else self.player2
                )

            # Check for winner
            if self.check_winner(row, col):
                if not self.animate:
                    self.display_board()
                print(
                    f"\nPlayer {self.current_player.symbol} ({type(self.current_player).__name__}) wins!"
                )
                return self.current_player

            # Check for tie
            if self.is_board_full():
                if not self.animate:
                    self.display_board()
                print("\nTie game!")
                return None

            # Switch players
            self.current_player = (
                self.player2 if self.current_player == self.player1 else self.player1
            )

            # Add delay for visualization
            if self.delay:
                time.sleep(self.delay)

class MCTSNode:
    def __init__(self, state: GameState, parent=None, move=None):
        self.state = state
        self.parent = parent
        self.move = move  # Move that led to this state
        self.children = {}
        self.wins = 0
        self.visits = 0
        self.untried_moves = self._get_valid_moves()
        
    def _get_valid_moves(self) -> List[int]:
        """Get list of valid moves from current state"""
        return [col for col in range(len(self.state.board[0])) 
                if self.state.board[0][col] == ' ']
    
    def ucb1(self, exploration: float = 1.41) -> float:
        """Calculate UCB1 value for this node"""
        if self.visits == 0:
            return float('inf')
        return (self.wins / self.visits + 
                exploration * math.sqrt(math.log(self.parent.visits) / self.visits))

# The MCTSPlayer uses the Monte Carlo Tree Search algorithm, which is a probabilistic search algorithm that finds optimal decisions by building a search tree through random sampling of the decision space. The algorithm consists of four main phases that repeat until a time limit is reached:
#
# 1. **Selection**
#    - Starting from the root node (current game state)
#    - Uses UCB1 (Upper Confidence Bound 1) formula to balance exploration vs exploitation
#    - Traverses down the tree, selecting nodes with highest UCB1 scores
#    - Continues until reaching a node that hasn't been fully explored
#
# 2. **Expansion**
#    - When reaching a node that hasn't been fully explored
#    - Creates a new child node for one of the unexplored moves
#    - Adds this new node to the tree
#
# 3. **Simulation**
#    - From the new node position
#    - Plays out a complete game using random moves
#    - This is called a "rollout" or "playout"
#
# 4. **Backpropagation**
#    - Takes the result of the simulation
#    - Updates statistics (visits and wins) for all nodes in the path from the simulated node back to the root


class MCTSPlayer(Player):
    """Monte Carlo Tree Search player implementation"""
    def __init__(self, symbol: str, simulation_time: float = 0.9):
        super().__init__(symbol)
        self.simulation_time = simulation_time
    
    def get_move(self, state: GameState) -> int:
        root = MCTSNode(state)
        end_time = time.time() + self.simulation_time
        
        while time.time() < end_time:
            # Selection
            node = root
            temp_board = [row[:] for row in state.board]
            current_player = self.symbol
            
            # Select a path through the tree
            while node.untried_moves == [] and node.children:
                node = max(node.children.values(), key=lambda n: n.ucb1())
                if node.move is not None:
                    row = self._get_next_row(temp_board, node.move)
                    temp_board[row][node.move] = current_player
                    current_player = 'O' if current_player == 'X' else 'X'
            
            # Expansion
            if node.untried_moves:
                col = random.choice(node.untried_moves)
                row = self._get_next_row(temp_board, col)
                temp_board[row][col] = current_player
                current_player = 'O' if current_player == 'X' else 'X'
                
                new_state = GameState(
                    board=[row[:] for row in temp_board],
                    current_player=current_player
                )
                node.untried_moves.remove(col)
                node.children[col] = MCTSNode(new_state, parent=node, move=col)
                node = node.children[col]
            
            # Simulation
            winner = self._simulate_game(temp_board, current_player)
            
            # Backpropagation
            while node:
                node.visits += 1
                if winner:
                    if type(self).__name__ == winner:
                        node.wins += 1
                node = node.parent
        
        # Choose best move
        best_move = max(root.children.items(), 
                       key=lambda x: x[1].visits)[0]
        return best_move
    
    def _get_next_row(self, board: List[List[str]], col: int) -> int:
        """Get the next available row in the given column"""
        for row in range(len(board)-1, -1, -1):
            if board[row][col] == ' ':
                return row
        return -1
    
    def _simulate_game(self, board: List[List[str]], current_player: str) -> Optional[str]:
        """Simulate a random game from the current position"""
        temp_board = [row[:] for row in board]
        player = current_player
        
        while True:
            # Get valid moves
            valid_moves = []
            for col in range(len(temp_board[0])):
                if temp_board[0][col] == ' ':
                    valid_moves.append(col)
            
            if not valid_moves:
                return None  # Tie game
            
            # Make random move
            col = random.choice(valid_moves)
            row = self._get_next_row(temp_board, col)
            temp_board[row][col] = player
            
            # Check for winner
            if self._check_winner(temp_board, row, col):
                return 'MCTSPlayer' if player == self.symbol else 'Opponent'
            
            player = 'O' if player == 'X' else 'X'
    
    def _check_winner(self, board: List[List[str]], row: int, col: int) -> bool:
        """Check if the last move created a winning condition"""
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        symbol = board[row][col]
        
        for dr, dc in directions:
            count = 1
            
            # Check positive direction
            r, c = row + dr, col + dc
            while (0 <= r < len(board) and 0 <= c < len(board[0]) and 
                   board[r][c] == symbol):
                count += 1
                r, c = r + dr, c + dc
            
            # Check negative direction
            r, c = row - dr, col - dc
            while (0 <= r < len(board) and 0 <= c < len(board[0]) and 
                   board[r][c] == symbol):
                count += 1
                r, c = r - dr, c - dc
            
            if count >= 4:
                return True
        return False


# Example player implementations
class RandomPlayer(Player):
    """Makes random valid moves"""

    def get_move(self, state: GameState) -> int:
        columns = len(state.board[0])
        valid_moves = [col for col in range(columns) if state.board[0][col] == " "]
        return random.choice(valid_moves)


class SimplePlayer(Player):
    """Looks for winning moves and blocking moves"""

    def get_move(self, state: GameState) -> int:
        columns = len(state.board[0])
        rows = len(state.board)

        # First check for winning moves
        for col in range(columns):
            if state.board[0][col] != " ":
                continue

            # Find row where piece would land
            row = rows - 1
            while row >= 0 and state.board[row][col] != " ":
                row -= 1

            # Try move
            test_board = [row[:] for row in state.board]
            test_board[row][col] = self.symbol

            # Check if this move wins
            if self.check_winner(test_board, row, col):
                return col

        # Then check for blocking moves
        opponent = "O" if self.symbol == "X" else "X"
        for col in range(columns):
            if state.board[0][col] != " ":
                continue

            row = rows - 1
            while row >= 0 and state.board[row][col] != " ":
                row -= 1

            test_board = [row[:] for row in state.board]
            test_board[row][col] = opponent

            if self.check_winner(test_board, row, col):
                return col

        # Otherwise, pick random valid move
        valid_moves = [col for col in range(columns) if state.board[0][col] == " "]
        return random.choice(valid_moves)

    def check_winner(self, board: List[List[str]], row: int, col: int) -> bool:
        """Check if there's a winner on the test board"""
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        symbol = board[row][col]
        columns = len(board[0])
        rows = len(board)

        for dr, dc in directions:
            count = 1

            # Check positive direction
            r, c = row + dr, col + dc
            while 0 <= r < rows and 0 <= c < columns and board[r][c] == symbol:
                count += 1
                r, c = r + dr, c + dc

            # Check negative direction
            r, c = row - dr, col - dc
            while 0 <= r < rows and 0 <= c < columns and board[r][c] == symbol:
                count += 1
                r, c = r - dr, c - dc

            if count >= 4:
                return True
        return False


class LousyPlayer(Player):
    """A terrible Connect Four player that makes many strategic mistakes"""

    def __init__(self, symbol: str):
        super().__init__(symbol)
        self.last_col = None  # Remember last move to make repeated mistakes

    def get_move(self, state: GameState) -> int:
        # List of bad strategic decisions
        columns = len(state.board[0])
        rows = len(state.board)

        # 1. If we played before, try to play in the same column
        # (This is terrible because it makes our moves predictable and
        # often leads to playing in full columns)
        if self.last_col is not None and state.board[0][self.last_col] == " ":
            self.last_col = self.last_col
            return self.last_col

        # 2. Avoid center columns at all costs
        # (This is terrible because center control is crucial in Connect Four)
        edge_columns = [0, 1, columns - 2, columns - 1]
        for col in edge_columns:
            if state.board[0][col] == " ":
                self.last_col = col
                return col

        # 3. Prefer playing in higher positions
        # (This is terrible because it leaves gaps that opponents can use)
        for row in range(rows):
            for col in range(columns):
                if (
                    row > 0
                    and state.board[row][col] == " "
                    and state.board[row - 1][col] == " "
                ):
                    if state.board[0][col] == " ":  # Make sure column isn't full
                        self.last_col = col
                        return col

        # 4. Ignore obvious winning moves
        # (This is terrible because we'll miss easy wins)
        for col in range(columns):
            if state.board[0][col] != " ":
                continue

            # Find where piece would land
            row = rows - 1
            while row >= 0 and state.board[row][col] != " ":
                row -= 1

            # If this would be a winning move, avoid it!
            test_board = [r[:] for r in state.board]
            test_board[row][col] = self.symbol
            if self._would_win(test_board, row, col):
                continue

            self.last_col = col
            return col

        # 5. If all else fails, pick the first available column
        # (This is terrible because it's predictable and ignores strategy)
        for col in range(columns):
            if state.board[0][col] == " ":
                self.last_col = col
                return col

        return 0  # Should never reach here if game isn't over

    def _would_win(self, board: List[List[str]], row: int, col: int) -> bool:
        """Check if a move would win (so we can avoid it!)"""
        symbol = board[row][col]
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        columns = len(board[0])
        rows = len(board)

        for dr, dc in directions:
            count = 1

            # Check positive direction
            r, c = row + dr, col + dc
            while 0 <= r < rows and 0 <= c < columns and board[r][c] == symbol:
                count += 1
                r, c = r + dr, c + dc

            # Check negative direction
            r, c = row - dr, col - dc
            while 0 <= r < rows and 0 <= c < columns and board[r][c] == symbol:
                count += 1
                r, c = r - dr, c - dc

            if count >= 4:
                return True
        return False

class FlossyPlayer(Player):
    """Looks for winning moves and blocking moves"""

    def get_move(self, state: GameState) -> int:
        columns = len(state.board[0])
        rows = len(state.board)

        # First check for winning moves
        for col in range(columns):
            if state.board[0][col] != ' ':
                continue

            # Find row where piece would land
            row = rows - 1
            while row >= 0 and state.board[row][col] != ' ':
                row -= 1

            # Try move
            test_board = [row[:] for row in state.board]
            test_board[row][col] = self.symbol

            # Check if this move wins
            if self.check_winner(test_board, row, col):
                return col

            test_board[row][col] = ' '

        # Then check for blocking moves
        opponent = 'O' if self.symbol == 'X' else 'X'
        for col in range(columns):
            if state.board[0][col] != ' ':
                continue

            row = rows - 1
            while row >= 0 and state.board[row][col] != ' ':
                row -= 1

            test_board = [row[:] for row in state.board]
            test_board[row][col] = opponent

            if self.check_winner(test_board, row, col):
                return col

            test_board[row][col] = ' '

        # Then check for move that will force winning
        for col in range(columns):
            if state.board[0][col] != ' ':
                continue

            # Find row where piece would land
            row = rows - 1
            while row >= 0 and state.board[row][col] != ' ':
                row -= 1

            # Try move
            test_board = [row[:] for row in state.board]
            test_board[row][col] = self.symbol

            # Check if this move wins
            if self.check_future_winner(test_board, row, col):
                return col

            test_board[row][col] = ' '

        # Then check for move that will force losing
        for col in range(columns):
            if state.board[0][col] != ' ':
                continue

            row = rows - 1
            while row >= 0 and state.board[row][col] != ' ':
                row -= 1

            test_board = [row[:] for row in state.board]
            test_board[row][col] = opponent

            if self.check_future_winner(test_board, row, col):
                return col

            test_board[row][col] = ' '

        # Otherwise, pick random valid move
        valid_moves = [
            col for col in range(columns)
            if state.board[0][col] == ' '
        ]
        return random.choice(valid_moves)

    def check_winner(self, board: List[List[str]], row: int, col: int) -> bool:
        """Check if there's a winner on the test board"""
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        symbol = board[row][col]
        columns = len(board[0])
        rows = len(board)

        for dr, dc in directions:
            count = 1

            # Check positive direction
            r, c = row + dr, col + dc
            while (0 <= r < rows and 0 <= c < columns and
                   board[r][c] == symbol):
                count += 1
                r, c = r + dr, c + dc

            # Check negative direction
            r, c = row - dr, col - dc
            while (0 <= r < rows and 0 <= c < columns and
                   board[r][c] == symbol):
                count += 1
                r, c = r - dr, c - dc

            if count >= 4:
                return True
        return False

    def check_future_winner(self, board: List[List[str]], row, col) -> bool:
        """Check if there's a move that will lead to an unavoidable winner"""
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        symbol = board[row][col]
        columns = len(board[0])
        rows = len(board)
        winning_moves_number = 0

        for dr, dc in directions:
            count = 1

            # Check positive direction
            r, c = row + dr, col + dc
            while (0 <= r < rows and 0 <= c < columns and
                   board[r][c] == symbol):
                count += 1
                r, c = r + dr, c + dc

            # Check negative direction
            r, c = row - dr, col - dc
            while (0 <= r < rows and 0 <= c < columns and
                   board[r][c] == symbol):
                count += 1
                r, c = r - dr, c - dc

            if count >= 3:
                winning_moves_number += len([i for i in (board[r + dr][c + dc], board[r - dr][c - dc]) if i == ' '])
                if winning_moves_number >= 1:
                    return True

        return False


import math

import random
import math
from typing import List

import random
import math
import time
from typing import List

import random
import math
import time
from typing import List


class Bob(Player):
    """Improved AI: Uses minimax with alpha-beta pruning, advanced heuristics, and dynamic depth."""

    def __init__(self, symbol, depth=4):
        super().__init__(symbol)
        self.depth = depth  # Search depth for minimax

    def get_move(self, state: GameState) -> int:
        board = np.array(state.board)  # Convert to NumPy array for efficiency

        # 1. Check for immediate winning move
        for col in self.get_valid_moves(board):
            row = self.get_drop_row(board, col)
            if row is None:
                continue

            board[row, col] = self.symbol
            if self.check_winner(board, row, col):
                return col  # Win immediately
            board[row, col] = ' '  # Undo test move

        # 2. Check for opponent's winning move and block it
        opponent = 'O' if self.symbol == 'X' else 'X'
        for col in self.get_valid_moves(board):
            row = self.get_drop_row(board, col)
            if row is None:
                continue

            board[row, col] = opponent
            if self.check_winner(board, row, col):
                return col  # Block the opponent
            board[row, col] = ' '  # Undo test move

        # 3. Use minimax with alpha-beta pruning to find the best move
        best_move, _ = self.minimax(board, self.depth, -float('inf'), float('inf'), True)
        return best_move if best_move is not None else random.choice(self.get_valid_moves(board))

    def minimax(self, board, depth, alpha, beta, maximizing_player):
        """Minimax algorithm with alpha-beta pruning."""
        valid_moves = self.get_valid_moves(board)

        # Base cases: terminal state or depth limit reached
        if depth == 0 or not valid_moves:
            return None, self.evaluate_board(board)

        if maximizing_player:
            max_eval = -float('inf')
            best_move = None

            for col in valid_moves:
                row = self.get_drop_row(board, col)
                if row is None:
                    continue

                board[row, col] = self.symbol
                _, eval = self.minimax(board, depth - 1, alpha, beta, False)
                board[row, col] = ' '  # Undo test move

                if eval > max_eval:
                    max_eval = eval
                    best_move = col

                alpha = max(alpha, eval)
                if beta <= alpha:
                    break  # Beta cutoff

            return best_move, max_eval
        else:
            min_eval = float('inf')
            best_move = None

            for col in valid_moves:
                row = self.get_drop_row(board, col)
                if row is None:
                    continue

                board[row, col] = 'O' if self.symbol == 'X' else 'X'
                _, eval = self.minimax(board, depth - 1, alpha, beta, True)
                board[row, col] = ' '  # Undo test move

                if eval < min_eval:
                    min_eval = eval
                    best_move = col

                beta = min(beta, eval)
                if beta <= alpha:
                    break  # Alpha cutoff

            return best_move, min_eval

    def evaluate_board(self, board):
        """Evaluate the board position for the AI player."""
        score = 0

        # Favor center control
        center_col = board.shape[1] // 2
        for col in range(board.shape[1]):
            if board[0, col] == ' ':
                score += 3 - abs(center_col - col)

        # Evaluate connections
        for row in range(board.shape[0]):
            for col in range(board.shape[1]):
                if board[row, col] == self.symbol:
                    score += self.count_connections(board, row, col)
                elif board[row, col] != ' ':
                    score -= self.count_connections(board, row, col)

        return score

    def count_connections(self, board, row, col):
        """Counts connected pieces for a given position."""
        symbol = board[row, col]
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        total = 0

        for dr, dc in directions:
            count = 1
            for i in range(1, 4):
                r, c = row + dr * i, col + dc * i
                if 0 <= r < board.shape[0] and 0 <= c < board.shape[1] and board[r, c] == symbol:
                    count += 1
                else:
                    break
            total += count ** 2  # Favor longer streaks

        return total

    def get_valid_moves(self, board):
        """Returns a list of valid (non-full) columns."""
        return [col for col in range(board.shape[1]) if board[0, col] == ' ']

    def get_drop_row(self, board, col):
        """Find the lowest available row for a given column."""
        for row in range(board.shape[0] - 1, -1, -1):
            if board[row, col] == ' ':
                return row
        return None

    def check_winner(self, board: List[List[str]], row: int, col: int) -> bool:
        """Check if there's a winner on the test board."""
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        symbol = board[row][col]

        for dr, dc in directions:
            count = 1

            # Check positive direction
            r, c = row + dr, col + dc
            while 0 <= r < board.shape[0] and 0 <= c < board.shape[1] and board[r][c] == symbol:
                count += 1
                r, c = r + dr, c + dc

            # Check negative direction
            r, c = row - dr, col - dc
            while 0 <= r < board.shape[0] and 0 <= c < board.shape[1] and board[r][c] == symbol:
                count += 1
                r, c = r - dr, c - dc

            if count >= 4:
                return True
        return False

from copy import deepcopy


class NeedMoreGluePlayer(Player):
    def __init__(self, symbol: str, depth: int = 4):
        super().__init__(symbol)
        self.depth = depth  # How many moves ahead the AI should look

    def get_move(self, state: GameState) -> int:
        _, best_col = self.minimax(state, self.depth, -math.inf, math.inf, True)
        return best_col

    def minimax(
        self, state: GameState, depth: int, alpha: float, beta: float, maximizing: bool
    ):
        valid_moves = self.get_valid_moves(state.board)
        if depth == 0 or not valid_moves or self.is_terminal(state):
            return self.evaluate(state), None

        if maximizing:
            max_eval = -math.inf
            best_col = random.choice(valid_moves)
            for col in valid_moves:
                new_state = self.simulate_move(state, col, self.symbol)
                eval, _ = self.minimax(new_state, depth - 1, alpha, beta, False)
                if eval > max_eval:
                    max_eval = eval
                    best_col = col

                alpha = max(alpha, eval)
                if beta <= alpha:
                    break  # Alpha-beta pruning

            return max_eval, best_col
        else:
            min_eval = math.inf
            best_col = random.choice(valid_moves)
            opponent_symbol = "X" if self.symbol == "O" else "O"
            for col in valid_moves:
                new_state = self.simulate_move(state, col, opponent_symbol)
                eval, _ = self.minimax(new_state, depth - 1, alpha, beta, True)
                if eval < min_eval:
                    min_eval = eval
                    best_col = col
                beta = min(beta, eval)
                if beta <= alpha:
                    break  # Alpha-beta pruning

            return min_eval, best_col

    def get_valid_moves(self, board):
        return [col for col in range(len(board[0])) if board[0][col] == " "]

    def simulate_move(self, state: GameState, col: int, symbol: str) -> GameState:
        new_board = deepcopy(state.board)
        for row in range(len(new_board) - 1, -1, -1):
            if new_board[row][col] == " ":
                new_board[row][col] = symbol
                break
        return GameState(new_board, symbol, (row, col))

    def is_terminal(self, state: GameState) -> bool:
        return self.check_winner(state.board, state.last_move) or all(
            cell != " " for cell in state.board[0]
        )

    def evaluate(self, state: GameState) -> int:
        if self.check_winner(state.board, state.last_move):
            return 1000 if state.current_player == self.symbol else -1000

        return self.score_position(state.board, self.symbol) - self.score_position(
            state.board, "X" if self.symbol == "O" else "O"
        )

    def score_position(self, board, symbol):
        score = 0
        for row in range(len(board)):
            for col in range(len(board[0])):
                if board[row][col] == symbol:
                    score += self.evaluate_position(board, row, col, symbol)

        return score

    def evaluate_position(self, board, row, col, symbol):
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        score = 0
        for dr, dc in directions:
            count = 1
            for i in range(1, 4):
                r, c = row + dr * i, col + dc * i
                if (
                    0 <= r < len(board)
                    and 0 <= c < len(board[0])
                    and board[r][c] == symbol
                ):
                    count += 1
                else:
                    break
            score += 10 ** (count - 1)  # Exponential scoring
        return score

    def check_winner(self, board, last_move):
        if not last_move:
            return False
        row, col = last_move
        symbol = board[row][col]
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for dr, dc in directions:
            count = 1
            for i in range(1, 4):
                r, c = row + dr * i, col + dc * i
                if (
                    0 <= r < len(board)
                    and 0 <= c < len(board[0])
                    and board[r][c] == symbol
                ):
                    count += 1
                else:
                    break
            for i in range(1, 4):
                r, c = row - dr * i, col - dc * i
                if (
                    0 <= r < len(board)
                    and 0 <= c < len(board[0])
                    and board[r][c] == symbol
                ):
                    count += 1
                else:
                    break
            if count >= 4:
                return True
        return False


class WinningPlayer(Player):
    """A Connect 4 player that prioritizes winning moves, blocks opponents, and picks strong positions."""

    def get_move(self, state: GameState) -> int:
        columns = len(state.board[0])
        valid_moves = [col for col in range(columns) if state.board[0][col] == " "]

        # Check for a winning move
        for col in valid_moves:
            if self.is_winning_move(state, col, self.symbol):
                return col

        # Block opponent's winning move
        opponent_symbol = "X" if self.symbol == "O" else "O"
        for col in valid_moves:
            if self.is_winning_move(state, col, opponent_symbol):
                return col

        # Prevent opponent from stacking vertically
        for col in valid_moves:
            if self.is_dangerous_stack(state, col, opponent_symbol):
                return col

        # Block opponent from forming three in a row, column, or diagonal
        for col in valid_moves:
            if self.is_threatening_three(state, col, opponent_symbol):
                return col

        # Play in the center if possible (strongest position)
        center_col = columns // 2
        if center_col in valid_moves:
            return center_col

        # Play in columns closest to the center
        sorted_moves = sorted(valid_moves, key=lambda x: abs(x - center_col))
        return sorted_moves[0]

    def is_winning_move(self, state: GameState, col: int, symbol: str) -> bool:
        """Check if playing in a given column results in a win."""
        row = self.get_next_row(state.board, col)
        if row is None:
            return False

        temp_board = [r[:] for r in state.board]
        temp_board[row][col] = symbol
        return self.check_winner(temp_board, row, col)

    def is_dangerous_stack(self, state: GameState, col: int, symbol: str) -> bool:
        """Check if the opponent is about to stack four vertically."""
        row = self.get_next_row(state.board, col)
        if row is None or row < 3:
            return False

        count = 0
        for r in range(row, max(row - 4, -1), -1):
            if state.board[r][col] == symbol:
                count += 1
            else:
                break

        return count >= 3

    def is_threatening_three(self, state: GameState, col: int, symbol: str) -> bool:
        """Check if the opponent is about to form three in a row, column, or diagonal."""
        row = self.get_next_row(state.board, col)
        if row is None:
            return False

        temp_board = [r[:] for r in state.board]
        temp_board[row][col] = symbol
        return self.check_threat(temp_board, row, col)

    def check_threat(self, board: List[List[str]], row: int, col: int) -> bool:
        """Check if placing a piece at (row, col) sets up a three-in-a-row threat."""
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        symbol = board[row][col]

        for dr, dc in directions:
            count = 1

            # Check positive direction
            r, c = row + dr, col + dc
            while (
                0 <= r < len(board) and 0 <= c < len(board[0]) and board[r][c] == symbol
            ):
                count += 1
                r, c = r + dr, c + dc

            # Check negative direction
            r, c = row - dr, col - dc
            while (
                0 <= r < len(board) and 0 <= c < len(board[0]) and board[r][c] == symbol
            ):
                count += 1
                r, c = r - dr, c - dc

            if count >= 3:
                return True

        return False

    def get_next_row(self, board: List[List[str]], col: int) -> Optional[int]:
        """Find the next available row in the given column."""
        for row in range(len(board) - 1, -1, -1):
            if board[row][col] == " ":
                return row
        return None

    def check_winner(self, board: List[List[str]], row: int, col: int) -> bool:
        """Check if placing a piece at (row, col) creates a winning condition."""
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        symbol = board[row][col]

        for dr, dc in directions:
            count = 1

            # Check positive direction
            r, c = row + dr, col + dc
            while (
                0 <= r < len(board) and 0 <= c < len(board[0]) and board[r][c] == symbol
            ):
                count += 1
                r, c = r + dr, c + dc

            # Check negative direction
            r, c = row - dr, col - dc
            while (
                0 <= r < len(board) and 0 <= c < len(board[0]) and board[r][c] == symbol
            ):
                count += 1
                r, c = r - dr, c - dc

            if count >= 4:
                return True

        return False


class Tournament:
    """Runs a tournament between multiple Connect Four strategies"""

    def __init__(self, strategy_classes: List[type], games_per_match: int = 10):
        self.strategy_classes = strategy_classes
        self.games_per_match = games_per_match
        self.results = {
            name: {
                "wins": 0,
                "losses": 0,
                "ties": 0,
                "points": 0,  # 3 for win, 1 for tie
                "avg_moves": 0,
                "total_games": 0,
            }
            for name in [cls.__name__ for cls in strategy_classes]
        }

    def run_tournament(self, display_games: bool = False):
        """Run a full tournament where each strategy plays against all others"""
        total_matches = (
            len(self.strategy_classes)
            * (len(self.strategy_classes) - 1)
            * self.games_per_match
            // 2
        )
        matches_played = 0

        print(f"\nStarting tournament with {len(self.strategy_classes)} strategies")
        print(f"Each matchup will play {self.games_per_match} games")
        print(f"Total matches to be played: {total_matches}\n")

        # Each strategy plays against every other strategy
        for i, strat1 in enumerate(self.strategy_classes):
            for j, strat2 in enumerate(self.strategy_classes[i + 1 :], i + 1):
                print(f"\nMatch: {strat1.__name__} vs {strat2.__name__}")

                # Play multiple games per matchup, alternating who goes first
                for game_num in range(self.games_per_match):
                    matches_played += 1
                    print(
                        f"\rProgress: {matches_played}/{total_matches} matches", end=""
                    )

                    # Alternate who goes first
                    if game_num % 2 == 0:
                        player1, player2 = strat1("X"), strat2("O")
                        name1, name2 = strat1.__name__, strat2.__name__
                    else:
                        player1, player2 = strat2("X"), strat1("O")
                        name1, name2 = strat2.__name__, strat1.__name__

                    # Play game
                    game = ConnectFour(
                        player1,
                        player2,
                        delay=0.5 if display_games else 0,
                        animate=display_games,
                    )
                    winner = game.play_game()

                    # Update statistics
                    if winner is None:  # Tie
                        self.results[name1]["ties"] += 1
                        self.results[name2]["ties"] += 1
                        self.results[name1]["points"] += 1
                        self.results[name2]["points"] += 1
                    else:
                        winner_name = type(winner).__name__
                        loser_name = name1 if winner_name == name2 else name2
                        self.results[winner_name]["wins"] += 1
                        self.results[winner_name]["points"] += 3
                        self.results[loser_name]["losses"] += 1

                    self.results[name1]["total_games"] += 1
                    self.results[name2]["total_games"] += 1

    def display_results(self):
        """Display tournament results in a formatted table"""
        print("\n\nTournament Results:")
        print("-" * 80)
        print(
            f"{'Strategy':<20} {'Games':<8} {'Wins':<8} {'Losses':<8} {'Ties':<8} {'Points':<8} {'Win %':<8}"
        )
        print("-" * 80)

        # Sort by points
        sorted_results = sorted(
            self.results.items(),
            key=lambda x: (x[1]["points"], x[1]["wins"]),
            reverse=True,
        )

        for name, stats in sorted_results:
            total_games = stats["total_games"]
            win_percentage = (
                (stats["wins"] / total_games * 100) if total_games > 0 else 0
            )
            print(
                f"{name:<20} {total_games:<8} {stats['wins']:<8} {stats['losses']:<8} "
                f"{stats['ties']:<8} {stats['points']:<8} {win_percentage:>6.1f}%"
            )
        print("-" * 80)


if __name__ == "__main__":
    # Create tournament with list of strategies
    strategies = [RandomPlayer, SimplePlayer, LousyPlayer, MCTSPlayer, Bob, FlossyPlayer, NeedMoreGluePlayer, WinningPlayer]
    tournament = Tournament(strategies, games_per_match=10)

    # Run tournament
    tournament.run_tournament(display_games=False)  # Set True to watch games

    # Show results
    tournament.display_results()
