import chess
import torch
import torch.nn as nn
import numpy as np

# ============ НЕЙРОСЕТЬ ============
class ChessNeuralNetwork(nn.Module):
    def __init__(self, num_moves):
        super(ChessNeuralNetwork, self).__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(12, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_moves)
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

# ============ КОНВЕРТАЦИЯ ДОСКИ ============
def board_to_tensor(board):
    tensor = np.zeros((12, 8, 8), dtype=np.float32)
    
    piece_to_idx = {
        chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
        chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5
    }
    
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            row = 7 - (square // 8)
            col = square % 8
            
            channel = piece_to_idx[piece.piece_type]
            if piece.color == chess.WHITE:
                tensor[channel][row][col] = 1.0
            else:
                tensor[channel + 6][row][col] = 1.0
    
    return tensor

# ============ ЗАГРУЗКА МОДЕЛИ ============
print("Загрузка модели...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

checkpoint = torch.load('chess_engine_trained.pth', map_location=device)
idx_to_move = checkpoint['idx_to_move']
num_moves = len(idx_to_move)

model = ChessNeuralNetwork(num_moves).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print("Модель загружена!\n")

# ============ ФУНКЦИЯ ПОЛУЧЕНИЯ ХОДА ОТ ДВИЖКА ============
def get_engine_move(board):
    """Возвращает лучший ход от движка"""
    
    position_tensor = board_to_tensor(board)
    position_tensor = torch.FloatTensor(position_tensor).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(position_tensor)
        probabilities = torch.softmax(outputs, dim=1)
    
    best_idx = torch.argmax(probabilities, dim=1).item()
    best_move_str = idx_to_move[best_idx]
    
    try:
        best_move = chess.Move.from_uci(best_move_str)
        if best_move in board.legal_moves:
            return best_move
    except:
        pass
    
    sorted_indices = torch.argsort(probabilities[0], descending=True)
    for idx in sorted_indices:
        move_str = idx_to_move[idx.item()]
        try:
            move = chess.Move.from_uci(move_str)
            if move in board.legal_moves:
                return move
        except:
            continue
    
    return list(board.legal_moves)[0]

# ============ ИГРА ============

board = chess.Board()

print("=" * 50)
print("ИГРА ПРОТИВ ДВИЖКА")
print("=" * 50)
print("Вводите ходы в формате UCI (например: e2e4, g1f3)")
print("'exit' - выход")
print("'board' - показать доску")
print("=" * 50)
print()

# Выбираем цвет
color_choice = input("Вы играете за (w)hite или (b)lack? ").lower()
human_is_white = color_choice.startswith('w')
engine_is_white = not human_is_white

print(f"\nВы играете за {'белых' if human_is_white else 'черных'}")
print(f"Движок играет за {'белых' if engine_is_white else 'черных'}")
print()

# Если движок начинает первым (человек за черных)
if not human_is_white:
    print("Движок думает...")
    engine_move = get_engine_move(board)
    board.push(engine_move)
    print(f"Движок делает ход: {engine_move}")
    print()

# Игровой цикл
while not board.is_game_over():
    # Показываем доску
    print(board)
    print()
    
    # Ход человека
    if (board.turn == chess.WHITE and human_is_white) or (board.turn == chess.BLACK and not human_is_white):
        while True:
            user_input = input("Ваш ход (например e2e4): ").strip()
            
            if user_input == 'exit':
                print("Игра завершена")
                exit()
            elif user_input == 'board':
                print(board)
                continue
            
            try:
                move = chess.Move.from_uci(user_input)
                if move in board.legal_moves:
                    board.push(move)
                    break
                else:
                    print("Невозможный ход! Попробуйте снова.")
            except:
                print("Неверный формат! Используйте UCI (например: e2e4)")
    
    # Ход движка
    else:
        print("Движок думает...")
        engine_move = get_engine_move(board)
        board.push(engine_move)
        print(f"Движок делает ход: {engine_move}")
        print()

# Конец игры
print()
print("=" * 50)
print("ИГРА ОКОНЧЕНА")
print("=" * 50)
print(board)
print()

if board.is_checkmate():
    winner = "белые" if board.turn == chess.BLACK else "черные"
    if (winner == "белые" and human_is_white) or (winner == "черные" and not human_is_white):
        print("🎉 ВЫ ПОБЕДИЛИ! 🎉")
    else:
        print("🤖 ДВИЖОК ПОБЕДИЛ! 🤖")
elif board.is_stalemate():
    print("🤝 ПАТ - НИЧЬЯ! 🤝")
else:
    print("Игра завершена!")