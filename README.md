# Connect Four Game

A Python implementation of Connect Four featuring multiple AI players with different strategies, from basic to advanced, and a tournament system to evaluate their performance.

## Features

- Customizable board size (default: 14x12)
- Multiple AI player implementations
- Colorized console output
- Move animation support
- Tournament system for AI evaluation
- Timeout protection for player moves
- Alpha-beta pruning and MCTS implementations

## AI Players

The game includes several AI player implementations with varying levels of sophistication:

- `RandomPlayer`: Makes random valid moves
- `SimplePlayer`: Looks for winning moves and blocking moves
- `LousyPlayer`: Intentionally makes suboptimal moves for testing


## Getting Started

### Prerequisites

```bash
pip install -r requirements.txt
```

### Basic Usage

```python
from connect_four import ConnectFour, RandomPlayer, MCTSPlayer

# Create players
player1 = RandomPlayer("X")
player2 = MCTSPlayer("O")

# Initialize game
game = ConnectFour(player1, player2, columns=14, rows=12, delay=1.0, animate=True)

# Play a single game
winner = game.play_game()
```

### Running a Tournament

```python
from connect_four import Tournament

# Define strategies to compete
strategies = [RandomPlayer, SimplePlayer, MCTSPlayer, MinMaxPlayer]

# Create and run tournament
tournament = Tournament(strategies, games_per_match=10)
tournament.run_tournament(display_games=False)
tournament.display_results()
```

## Game Class Structure

### Main Classes

- `GameState`: Represents the current state of the game
- `Player`: Abstract base class for implementing new players
- `ConnectFour`: Main game engine
- `Tournament`: Handles multiple games between different players

### Player Implementation

To create a new AI player, extend the `Player` class:

```python
class MyPlayer(Player):
    def __init__(self, symbol: str):
        super().__init__(symbol)

    def get_move(self, state: GameState) -> int:
        # Implement your move logic here
        # Must return a valid column number (0-6)
        pass
```

## Advanced Features

### Move Animation

The game supports animated piece dropping with configurable delay:

```python
game = ConnectFour(player1, player2, delay=0.5, animate=True)
```

### Timeout Protection

Players have a 1-second timeout for making moves to prevent infinite loops:

```python
with time_limit(1):
    move = player.get_move(state)
```

### Board Evaluation

Advanced players use sophisticated board evaluation techniques:

- Center control preference
- Threat detection
- Pattern recognition
- Positional scoring

## Tournament System

The tournament system provides:

- Round-robin matches between all players
- Configurable games per match
- Win/loss/tie tracking
- Points system (3 for win, 1 for tie)
- Performance statistics
- Formatted results table

## Contributing

To add a new AI player:

1. Create a new class extending `Player`
2. Implement the `get_move` method
3. Add your player to the tournament in `__main__`

## License

This project is open source and available under the MIT License.
