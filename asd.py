import chess
import chess.pgn
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import os
from tqdm import tqdm

# ============ НАСТРОЙКИ ============
# Укажите путь к вашему PGN файлу (замените на свой путь)
PGN_FILE_PATH = "C:/Users/asdzxc/Desktop/wtf/data/lichess.pgn"
# Или для Linux/Mac: "/home/user/Downloads/lichess_elite_2024-01.pgn"

NUM_GAMES_TO_USE = 500      # Количество партий для обучения
BATCH_SIZE = 64               # Размер батча для обучения
EPOCHS = 10                   # Количество эпох обучения
LEARNING_RATE = 0.001         # Скорость обучения

# ============ ШАГ 1: КОНВЕРТАЦИЯ ПОЗИЦИИ В ТЕНЗОР ============

def board_to_tensor(board):
    """
    Превращает шахматную позицию в тензор 8x8x12
    12 каналов: 6 типов фигур * 2 цвета
    """
    tensor = np.zeros((12, 8, 8), dtype=np.float32)
    
    # Соответствие фигур индексам
    piece_to_idx = {
        chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
        chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5
    }
    
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            row = 7 - (square // 8)  # инвертируем, чтобы белые были снизу
            col = square % 8
            
            # Канал для белых (0-5) или чёрных (6-11)
            channel = piece_to_idx[piece.piece_type]
            if piece.color == chess.WHITE:
                tensor[channel][row][col] = 1.0
            else:
                tensor[channel + 6][row][col] = 1.0
    
    return tensor

# ============ ШАГ 2: ИЗВЛЕЧЕНИЕ ДАННЫХ ИЗ PGN ============

def extract_positions_and_moves(pgn_file_path, max_games=None):
    """
    Извлекает позиции и ходы из PGN файла
    Возвращает: список позиций (тензоры), список ходов (строки)
    """
    positions = []
    moves = []
    move_counter = 0
    
    print(f"Открываем файл: {pgn_file_path}")
    
    with open(pgn_file_path, 'r', encoding='utf-8') as pgn_file:
        games_processed = 0
        
        while True:
            # Читаем следующую партию
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                break
            
            games_processed += 1
            board = game.board()
            
            # Проходим по всем ходам в партии
            for move in game.mainline_moves():
                # Сохраняем позицию перед ходом
                positions.append(board_to_tensor(board))
                moves.append(move.uci())
                move_counter += 1
                
                # Делаем ход на доске
                board.push(move)
            
            # Выводим прогресс каждые 100 партий
            if games_processed % 100 == 0:
                print(f"Обработано партий: {games_processed}, собрано ходов: {move_counter}")
            
            # Ограничиваем количество партий
            if max_games and games_processed >= max_games:
                break
    
    print(f"\nГотово! Обработано {games_processed} партий")
    print(f"Собрано {len(positions)} позиций для обучения")
    
    return positions, moves

# ============ ШАГ 3: СОЗДАНИЕ СЛОВАРЯ ХОДОВ ============

def create_move_dictionary(moves_list):
    """
    Создаёт словарь для преобразования ходов в индексы
    Используем только те ходы, которые встречаются в данных
    """
    move_counts = Counter(moves_list)
    
    # Берём топ-2000 самых частых ходов (можно увеличить)
    top_moves = [move for move, count in move_counts.most_common(2000)]
    
    move_to_idx = {move: idx for idx, move in enumerate(top_moves)}
    idx_to_move = {idx: move for move, idx in move_to_idx.items()}
    
    print(f"Всего уникальных ходов в данных: {len(move_counts)}")
    print(f"Используем топ-{len(top_moves)} самых частых ходов")
    
    return move_to_idx, idx_to_move

# ============ ШАГ 4: КЛАСС DATASET ДЛЯ PYTORCH ============
class ChessDataset(Dataset):
    def __init__(self, positions, moves, move_to_idx):
        self.positions = torch.FloatTensor(positions)
        self.move_indices = torch.LongTensor([move_to_idx[m] for m in moves])
        
    def __len__(self):
        return len(self.positions)
    
    def __getitem__(self, idx):
        return self.positions[idx], self.move_indices[idx]

# ============ ШАГ 5: НЕЙРОСЕТЬ ============

class ChessNeuralNetwork(nn.Module):
    def __init__(self, num_moves):
        super(ChessNeuralNetwork, self).__init__()
        
        # Свёрточная часть (анализирует доску)
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
        
        # Полносвязная часть (выбирает ход)
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

# ============ ШАГ 6: ОБУЧЕНИЕ ============

def train_model(model, train_loader, device, epochs=10):
    """
    Обучает нейросеть
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        
        progress_bar = tqdm(train_loader, desc=f"Эпоха {epoch+1}/{epochs}")
        
        for batch_positions, batch_moves in progress_bar:
            batch_positions = batch_positions.to(device)
            batch_moves = batch_moves.to(device)
            
            # Forward pass
            outputs = model(batch_positions)
            loss = criterion(outputs, batch_moves)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Статистика
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == batch_moves).sum().item()
            total_predictions += batch_moves.size(0)
            
            # Обновляем прогресс-бар
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100 * correct_predictions / total_predictions:.2f}%'
            })
        
        epoch_loss = total_loss / len(train_loader)
        epoch_acc = 100 * correct_predictions / total_predictions
        
        print(f"\nЭпоха {epoch+1} завершена:")
        print(f"  Средняя потеря: {epoch_loss:.4f}")
        print(f"  Точность: {epoch_acc:.2f}%")
        print("-" * 50)

# ============ ШАГ 7: ИСПОЛЬЗОВАНИЕ ОБУЧЕННОГО ДВИЖКА ============

class TrainedChessEngine:
    def __init__(self, model, idx_to_move, device):
        self.model = model
        self.idx_to_move = idx_to_move
        self.device = device
        self.model.eval()
    
    def get_best_move(self, board):
        """
        Возвращает лучший ход для текущей позиции
        """
        # Конвертируем позицию в тензор
        position_tensor = board_to_tensor(board)
        position_tensor = torch.FloatTensor(position_tensor).unsqueeze(0).to(self.device)
        
        # Получаем предсказания сети
        with torch.no_grad():
            outputs = self.model(position_tensor)
            probabilities = torch.softmax(outputs, dim=1)
        
        # Выбираем ход с максимальной вероятностью

        best_idx = torch.argmax(probabilities, dim=1).item()
        best_move_str = self.idx_to_move[best_idx]
        
        # Проверяем, что ход легален
        best_move = chess.Move.from_uci(best_move_str)
        if best_move in board.legal_moves:
            return best_move
        
        # Если ход нелегальный (редко для обученной сети), ищем следующий лучший
        sorted_indices = torch.argsort(probabilities[0], descending=True)
        for idx in sorted_indices:
            move_str = self.idx_to_move[idx.item()]
            move = chess.Move.from_uci(move_str)
            if move in board.legal_moves:
                return move
        
        # Если ничего не нашли (защита от ошибок)
        return list(board.legal_moves)[0]

# ============ ГЛАВНАЯ ФУНКЦИЯ ============

def main():
    print("=" * 60)
    print("ШАХМАТНЫЙ ДВИЖОК С ОБУЧЕНИЕМ ПО PGN ПАРТИЯМ")
    print("=" * 60)
    
    # Проверяем существование файла
    if not os.path.exists(PGN_FILE_PATH):
        print(f"\nОШИБКА: Файл не найден по пути:")
        print(f"  {PGN_FILE_PATH}")
        print(f"\nПожалуйста, укажите правильный путь к PGN файлу.")
        print(f"Пример: C:/Users/YourName/Downloads/games.pgn")
        return
    
    # Проверяем устройство (GPU или CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nИспользуем устройство: {device}")
    
    # ШАГ 1: Извлекаем данные из PGN
    print(f"\n[1/5] Извлекаем данные из PGN...")
    positions, moves = extract_positions_and_moves(PGN_FILE_PATH, NUM_GAMES_TO_USE)
    
    if len(positions) == 0:
        print("ОШИБКА: Не удалось извлечь позиции из PGN файла")
        return
    
    # ШАГ 2: Создаём словарь ходов
    print(f"\n[2/5] Создаём словарь ходов...")
    move_to_idx, idx_to_move = create_move_dictionary(moves)
    
    # ШАГ 3: Создаём датасет и загрузчик
    print(f"\n[3/5] Создаём датасет...")
    dataset = ChessDataset(positions, moves, move_to_idx)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # ШАГ 4: Создаём нейросеть
    print(f"\n[4/5] Создаём нейросеть...")
    num_moves = len(move_to_idx)
    model = ChessNeuralNetwork(num_moves).to(device)
    
    # Подсчитываем параметры сети
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Всего параметров в сети: {total_params:,}")
    
    # ШАГ 5: Обучаем
    print(f"\n[5/5] Начинаем обучение...\n")
    train_model(model, train_loader, device, epochs=EPOCHS)
    
    # Сохраняем обученную модель
    model_path = "chess_engine_trained.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'idx_to_move': idx_to_move,
        'move_to_idx': move_to_idx
    }, model_path)
    print(f"\nМодель сохранена в файл: {model_path}")
    
    # Демонстрация работы
    print("\n" + "=" * 60)
    print("ДЕМОНСТРАЦИЯ РАБОТЫ ДВИЖКА")
    print("=" * 60)
    
    # Создаём экземпляр движка
    engine = TrainedChessEngine(model, idx_to_move, device)
    
    # Начальная позиция
    board = chess.Board()
    print("\nНачальная позиция:")
    print(board)
    
    # Делаем несколько ходов движком
    print("\nДвижок выбирает ходы:")
    for i in range(5):
        move = engine.get_best_move(board)
        print(f"Ход {i+1}: {move}")
        board.push(move)
        print(f"После хода:\n{board}\n")
    
    print("Обучение завершено! Движок готов к использованию.")

# ============ ЗАПУСК ============
if __name__ == "__main__":
    main()