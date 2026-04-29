import sys
import chess
import torch
import torch.nn as nn
import numpy as np
from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
from PyQt6.QtGui import *
import os


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


# ============ ШАХМАТНЫЙ ДВИЖОК ============
class ChessEngine(QThread):
    move_ready = pyqtSignal(object)

    def __init__(self, model_path='chess_engine_trained.pth'):
        super().__init__()
        self.board = None
        self.model = None
        self.idx_to_move = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.load_model(model_path)

    def load_model(self, model_path):
        """Загрузка модели нейросети"""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.idx_to_move = checkpoint['idx_to_move']
            num_moves = len(self.idx_to_move)

            self.model = ChessNeuralNetwork(num_moves).to(self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            print("Модель успешно загружена")
            return True
        except Exception as e:
            print(f"Ошибка загрузки модели: {e}")
            return False

    def get_best_move(self, board):
        """Возвращает лучший ход от движка"""
        if self.model is None:
            moves = list(board.legal_moves)
            return moves[0] if moves else None

        position_tensor = board_to_tensor(board)
        position_tensor = torch.FloatTensor(position_tensor).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(position_tensor)
            probabilities = torch.softmax(outputs, dim=1)

        best_idx = torch.argmax(probabilities, dim=1).item()
        best_move_str = self.idx_to_move[best_idx]

        try:
            best_move = chess.Move.from_uci(best_move_str)
            if best_move in board.legal_moves:
                return best_move
        except:
            pass

        sorted_indices = torch.argsort(probabilities[0], descending=True)
        for idx in sorted_indices:
            move_str = self.idx_to_move[idx.item()]
            try:
                move = chess.Move.from_uci(move_str)
                if move in board.legal_moves:
                    return move
            except:
                continue

        moves = list(board.legal_moves)
        return moves[0] if moves else None

    def run(self):
        """Поток для хода движка"""
        if self.board:
            move = self.get_best_move(self.board)
            self.move_ready.emit(move)


# ============ ШАХМАТНАЯ ДОСКА (ВИДЖЕТ) ============
class ChessBoardWidget(QWidget):
    square_clicked = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(500, 500)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self.board = chess.Board()
        self.selected_square = None
        self.legal_moves = []
        self.last_move = None
        self.square_size = 0

        # Загрузка изображений фигур из папки spraut
        self.piece_images = {}
        self.load_piece_images()

        # Цвета клеток
        self.light_color = QColor(240, 217, 181)
        self.dark_color = QColor(181, 136, 99)
        self.highlight_color = QColor(255, 255, 0, 100)
        self.last_move_color = QColor(0, 255, 0, 80)
        self.check_color = QColor(255, 0, 0, 120)

    def load_piece_images(self):
        """Загрузка изображений фигур из папки spraut"""
        # Соответствие символов фигур именам файлов
        piece_files = {
            'r': 'br', 'n': 'bn', 'b': 'bb', 'q': 'bq', 'k': 'bk', 'p': 'bp',
            'R': 'wr', 'N': 'wn', 'B': 'wb', 'Q': 'wq', 'K': 'wk', 'P': 'wp'
        }

        # Проверяем наличие папки spraut
        spraut_folder = "spraut"
        if not os.path.exists(spraut_folder):
            print(f"Папка '{spraut_folder}' не найдена, ищу в текущей директории")
            spraut_folder = "."

        # Загружаем изображения
        loaded_count = 0
        for symbol, filename in piece_files.items():
            # Проверяем разные возможные расширения
            for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.PNG', '.JPG']:
                filepath = os.path.join(spraut_folder, filename + ext)
                if os.path.exists(filepath):
                    try:
                        pixmap = QPixmap(filepath)
                        if not pixmap.isNull():
                            self.piece_images[symbol] = pixmap
                            loaded_count += 1
                            print(f"Загружена фигура: {filename}{ext}")
                            break
                    except Exception as e:
                        print(f"Ошибка загрузки {filepath}: {e}")

        print(f"Загружено {loaded_count} из 12 изображений фигур")

        # Если изображения не найдены, используем символы Юникода
        if loaded_count == 0:
            print("Изображения фигур не найдены, используются текстовые символы")

    def set_board(self, board):
        """Установка новой позиции"""
        self.board = board
        self.selected_square = None
        self.legal_moves = []
        self.update()

    def get_square_from_pos(self, pos):
        """Получение клетки по координатам мыши"""
        x, y = pos.x(), pos.y()
        if 0 <= x < self.width() and 0 <= y < self.height():
            col = x // self.square_size
            row = y // self.square_size
            if 0 <= col < 8 and 0 <= row < 8:
                return (7 - row) * 8 + col
        return None

    def mousePressEvent(self, event):
        """Обработка клика мыши"""
        if event.button() == Qt.MouseButton.LeftButton:
            square = self.get_square_from_pos(event.pos())
            if square is not None:
                self.square_clicked.emit(square)

    def paintEvent(self, event):
        """Отрисовка доски"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Вычисляем размер клетки
        self.square_size = min(self.width(), self.height()) // 8

        # Отрисовка клеток
        for row in range(8):
            for col in range(8):
                x = col * self.square_size
                y = row * self.square_size

                # Цвет клетки
                if (row + col) % 2 == 0:
                    color = self.light_color
                else:
                    color = self.dark_color

                painter.fillRect(x, y, self.square_size, self.square_size, color)

                # Подсветка последнего хода
                if self.last_move:
                    from_row = 7 - (self.last_move.from_square // 8)
                    from_col = self.last_move.from_square % 8
                    to_row = 7 - (self.last_move.to_square // 8)
                    to_col = self.last_move.to_square % 8

                    if (row, col) == (from_row, from_col) or (row, col) == (to_row, to_col):
                        painter.fillRect(x, y, self.square_size, self.square_size, self.last_move_color)

                # Подсветка выбранной клетки
                if self.selected_square is not None:
                    selected_row = 7 - (self.selected_square // 8)
                    selected_col = self.selected_square % 8
                    if (row, col) == (selected_row, selected_col):
                        painter.fillRect(x, y, self.square_size, self.square_size, self.highlight_color)

                # Подсветка возможных ходов
                for move in self.legal_moves:
                    to_row = 7 - (move.to_square // 8)
                    to_col = move.to_square % 8
                    if (row, col) == (to_row, to_col):
                        painter.setBrush(QBrush(self.highlight_color))
                        painter.setPen(Qt.PenStyle.NoPen)
                        painter.drawEllipse(QPoint(x + self.square_size // 2, y + self.square_size // 2),
                                            self.square_size // 4, self.square_size // 4)

                # Подсветка короля под шахом
                if self.board.is_check():
                    king_square = self.board.king(self.board.turn)
                    if king_square:
                        king_row = 7 - (king_square // 8)
                        king_col = king_square % 8
                        if (row, col) == (king_row, king_col):
                            painter.fillRect(x, y, self.square_size, self.square_size, self.check_color)

        # Отрисовка фигур
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                row = 7 - (square // 8)
                col = square % 8
                x = col * self.square_size
                y = row * self.square_size

                piece_symbol = piece.symbol()

                # Рисуем изображение или текст
                if piece_symbol in self.piece_images:
                    # Масштабируем изображение под размер клетки
                    pixmap = self.piece_images[piece_symbol]
                    scaled_pixmap = pixmap.scaled(self.square_size - 10, self.square_size - 10,
                                                  Qt.AspectRatioMode.KeepAspectRatio,
                                                  Qt.TransformationMode.SmoothTransformation)
                    img_x = x + (self.square_size - scaled_pixmap.width()) // 2
                    img_y = y + (self.square_size - scaled_pixmap.height()) // 2
                    painter.drawPixmap(img_x, img_y, scaled_pixmap)
                else:
                    # Рисуем текстовый символ как запасной вариант
                    self.draw_piece_text(painter, piece, x, y)

    def draw_piece_text(self, painter, piece, x, y):
        """Отрисовка фигуры текстом"""
        piece_symbols = {
            'r': '♜', 'n': '♞', 'b': '♝', 'q': '♛', 'k': '♚', 'p': '♟',
            'R': '♖', 'N': '♘', 'B': '♗', 'Q': '♕', 'K': '♔', 'P': '♙'
        }

        symbol = piece_symbols.get(piece.symbol(), piece.symbol())

        font = QFont("Arial", self.square_size // 2, QFont.Weight.Bold)
        painter.setFont(font)

        # Цвет для текста
        if piece.color == chess.WHITE:
            painter.setPen(QColor(255, 255, 255))
        else:
            painter.setPen(QColor(0, 0, 0))

        # Центрируем текст
        fm = QFontMetrics(font)
        text_width = fm.horizontalAdvance(symbol)
        text_height = fm.height()

        text_x = x + (self.square_size - text_width) // 2
        text_y = y + (self.square_size + text_height) // 2 - fm.descent()

        painter.drawText(text_x, text_y, symbol)

    def resizeEvent(self, event):
        """Обработка изменения размера"""
        self.update()


# ============ ГЛАВНОЕ ОКНО ПРИЛОЖЕНИЯ ============
class ChessMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Шахматы против ИИ")
        self.setMinimumSize(900, 600)

        # Инициализация движка
        self.engine = ChessEngine()
        self.engine.move_ready.connect(self.on_engine_move)

        # Игровые переменные
        self.board = chess.Board()
        self.human_plays_white = True
        self.waiting_for_engine = False
        self.game_over = False
        self.move_history = []
        self.waiting_for_promotion = False  # Флаг ожидания выбора превращения

        # Создание UI
        self.setup_ui()

        # Выбор цвета
        self.show_color_choice()

    def setup_ui(self):
        """Настройка интерфейса"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout(central_widget)

        # Левая панель - шахматная доска
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        self.board_widget = ChessBoardWidget()
        self.board_widget.square_clicked.connect(self.on_square_clicked)
        left_layout.addWidget(self.board_widget)

        main_layout.addWidget(left_panel, stretch=2)

        # Правая панель - информация и управление
        right_panel = QWidget()
        right_panel.setMaximumWidth(300)
        right_layout = QVBoxLayout(right_panel)

        # Заголовок
        title_label = QLabel("ШАХМАТЫ")
        title_font = QFont("Arial", 20, QFont.Weight.Bold)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        right_layout.addWidget(title_label)

        right_layout.addSpacing(20)

        # Информационная группа
        info_group = QGroupBox("Информация")
        info_layout = QVBoxLayout(info_group)

        self.turn_label = QLabel("Ход: Белые")
        self.turn_label.setFont(QFont("Arial", 12))
        info_layout.addWidget(self.turn_label)

        self.status_label = QLabel("Статус: Игра активна")
        self.status_label.setFont(QFont("Arial", 12))
        info_layout.addWidget(self.status_label)

        right_layout.addWidget(info_group)

        right_layout.addSpacing(20)

        # Кнопки управления
        buttons_group = QGroupBox("Управление")
        buttons_layout = QVBoxLayout(buttons_group)

        self.new_game_btn = QPushButton("Новая игра")
        self.new_game_btn.clicked.connect(self.new_game)
        buttons_layout.addWidget(self.new_game_btn)

        self.resign_btn = QPushButton("Сдаться")
        self.resign_btn.clicked.connect(self.resign)
        buttons_layout.addWidget(self.resign_btn)

        self.undo_btn = QPushButton("Отменить ход")
        self.undo_btn.clicked.connect(self.undo_move)
        buttons_layout.addWidget(self.undo_btn)

        right_layout.addWidget(buttons_group)

        right_layout.addSpacing(20)

        # История ходов
        history_group = QGroupBox("История ходов")
        history_layout = QVBoxLayout(history_group)

        self.history_list = QListWidget()
        history_layout.addWidget(self.history_list)

        right_layout.addWidget(history_group)

        right_layout.addStretch()

        main_layout.addWidget(right_panel)

    def show_color_choice(self):
        """Диалог выбора цвета"""
        dialog = QDialog(self)
        dialog.setWindowTitle("Выбор цвета")
        dialog.setModal(True)
        dialog.setFixedSize(300, 200)

        layout = QVBoxLayout(dialog)

        label = QLabel("Выберите цвет фигур:")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setFont(QFont("Arial", 14))
        layout.addWidget(label)

        layout.addSpacing(20)

        buttons_layout = QHBoxLayout()

        white_btn = QPushButton("Белые")
        white_btn.clicked.connect(lambda: self.set_color(True, dialog))
        buttons_layout.addWidget(white_btn)

        black_btn = QPushButton("Черные")
        black_btn.clicked.connect(lambda: self.set_color(False, dialog))
        buttons_layout.addWidget(black_btn)

        layout.addLayout(buttons_layout)

        dialog.exec()

    def set_color(self, human_white, dialog):
        """Установка цвета игрока"""
        self.human_plays_white = human_white
        dialog.accept()

        # Если ИИ начинает первым
        if not self.human_plays_white and self.board.turn == chess.WHITE:
            self.make_engine_move()

    def update_ui(self):
        """Обновление интерфейса"""
        # Обновляем доску
        self.board_widget.set_board(self.board)

        # Обновляем информацию о ходе
        turn_text = "Белых" if self.board.turn == chess.WHITE else "Черных"
        is_human_turn = (self.board.turn == chess.WHITE and self.human_plays_white) or \
                        (self.board.turn == chess.BLACK and not self.human_plays_white)

        if self.game_over:
            self.turn_label.setText("Игра окончена")
        else:
            self.turn_label.setText(f"Ход: {turn_text} ({'Вы' if is_human_turn else 'ИИ'})")

        # Обновляем историю ходов
        if len(self.move_history) > self.history_list.count():
            last_move = self.move_history[-1]
            move_number = len(self.move_history)
            move_text = f"{move_number}. {last_move}"
            self.history_list.addItem(move_text)
            self.history_list.scrollToBottom()

    def on_square_clicked(self, square):
        """Обработка клика по клетке - ПОЛНОСТЬЮ ИСПРАВЛЕННАЯ ВЕРСИЯ"""
        # Если игра окончена или ждем ход движка - игнорируем
        if self.game_over or self.waiting_for_engine or self.waiting_for_promotion:
            return

        # Проверяем, чей сейчас ход
        is_human_turn = (self.board.turn == chess.WHITE and self.human_plays_white) or \
                        (self.board.turn == chess.BLACK and not self.human_plays_white)

        if not is_human_turn:
            QMessageBox.information(self, "Не ваш ход", "Сейчас ходит компьютер!")
            return

        # Если клетка не выбрана
        if self.board_widget.selected_square is None:
            # Выбираем клетку с фигурой нужного цвета
            piece = self.board.piece_at(square)
            if piece and piece.color == self.board.turn:
                self.board_widget.selected_square = square
                self.board_widget.legal_moves = [move for move in self.board.legal_moves
                                                 if move.from_square == square]
                self.board_widget.update()
                print(f"Выбрана клетка: {chess.square_name(square)}")
            else:
                # Если кликнули на пустую клетку или чужую фигуру
                if piece:
                    QMessageBox.warning(self, "Невозможно выбрать", "Это не ваша фигура!")
                else:
                    QMessageBox.warning(self, "Невозможно выбрать", "Выберите свою фигуру!")
        else:
            # Уже есть выбранная клетка - пытаемся сделать ход
            from_square = self.board_widget.selected_square
            to_square = square

            # Создаем ход
            move = chess.Move(from_square, to_square)

            # Проверка на превращение пешки
            piece = self.board.piece_at(from_square)
            if piece and piece.piece_type == chess.PAWN:
                to_row = 7 - (to_square // 8)
                if (piece.color == chess.WHITE and to_row == 0) or \
                        (piece.color == chess.BLACK and to_row == 7):
                    # Нужно превращение
                    self.waiting_for_promotion = True
                    promotion = self.show_promotion_dialog()
                    self.waiting_for_promotion = False

                    if promotion:
                        move = chess.Move(from_square, to_square, promotion=promotion)
                    else:
                        # Пользователь отменил - сбрасываем выделение
                        self.board_widget.selected_square = None
                        self.board_widget.legal_moves = []
                        self.board_widget.update()
                        return

            # Проверяем и делаем ход
            if move in self.board.legal_moves:
                self.make_move(move)
            else:
                # Если ход неверный, очищаем выделение
                self.board_widget.selected_square = None
                self.board_widget.legal_moves = []
                self.board_widget.update()
                """
                QMessageBox.warning(self, "Невозможный ход", "Этот ход недопустим!")"""

    def show_promotion_dialog(self):
        """Диалог выбора фигуры для превращения - ИСПРАВЛЕННАЯ ВЕРСИЯ"""
        dialog = QDialog(self)
        dialog.setWindowTitle("Превращение пешки")
        dialog.setModal(True)
        dialog.setFixedSize(400, 150)

        layout = QVBoxLayout(dialog)

        label = QLabel("Выберите фигуру для превращения:")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(label)

        buttons_layout = QHBoxLayout()

        # Создаем кнопки с фигурами
        pieces = [
            (chess.QUEEN, "Ферзь ♕"),
            (chess.ROOK, "Ладья ♖"),
            (chess.BISHOP, "Слон ♗"),
            (chess.KNIGHT, "Конь ♘")
        ]

        result = [None]

        for piece, name in pieces:
            btn = QPushButton(name)
            btn.clicked.connect(lambda checked, p=piece: self.on_promotion_selected(p, result, dialog))
            buttons_layout.addWidget(btn)

        layout.addLayout(buttons_layout)

        # Добавляем кнопку Cancel
        cancel_btn = QPushButton("Отмена")
        cancel_btn.clicked.connect(dialog.reject)
        layout.addWidget(cancel_btn)

        if dialog.exec() == QDialog.DialogCode.Accepted:
            return result[0] if result[0] is not None else chess.QUEEN
        else:
            return None

    def on_promotion_selected(self, piece, result, dialog):
        """Обработчик выбора фигуры для превращения"""
        result[0] = piece
        dialog.accept()

    def make_move(self, move):
        """Совершить ход"""
        # Сохраняем ход
        self.board.push(move)
        self.move_history.append(move)
        self.board_widget.last_move = move
        self.board_widget.selected_square = None
        self.board_widget.legal_moves = []

        # Обновляем UI
        self.update_ui()

        # Проверка окончания игры
        if self.check_game_over():
            return

        # Ход ИИ
        if not self.game_over:
            self.make_engine_move()

    def make_engine_move(self):
        """Запуск хода движка"""
        if self.game_over:
            return

        is_engine_turn = (self.board.turn == chess.WHITE and not self.human_plays_white) or \
                         (self.board.turn == chess.BLACK and self.human_plays_white)

        if is_engine_turn:
            self.waiting_for_engine = True
            self.update_ui()

            # Запускаем движок в отдельном потоке
            self.engine.board = self.board.copy()  # Отправляем копию доски
            self.engine.start()

    def on_engine_move(self, move):
        """Получение хода от движка"""
        self.waiting_for_engine = False

        if move and move in self.board.legal_moves:
            self.board.push(move)
            self.move_history.append(move)
            self.board_widget.last_move = move
            self.update_ui()

            self.check_game_over()
        else:
            # Если модель не смогла найти ход, делаем случайный
            moves = list(self.board.legal_moves)
            if moves:
                self.board.push(moves[0])
                self.move_history.append(moves[0])
                self.board_widget.last_move = moves[0]
                self.update_ui()
                self.check_game_over()

    def check_game_over(self):
        """Проверка окончания игры"""
        if self.board.is_game_over():
            self.game_over = True
            message = ""

            if self.board.is_checkmate():
                # Определяем победителя
                if self.board.turn == chess.WHITE:
                    winner = "Черные"
                else:
                    winner = "Белые"

                if (winner == "Белые" and self.human_plays_white) or \
                        (winner == "Черные" and not self.human_plays_white):
                    message = f"Поздравляем! Вы победили!"
                else:
                    message = f"ИИ победил!"

            elif self.board.is_stalemate():
                message = "Пат! Ничья!"
            elif self.board.is_insufficient_material():
                message = "Недостаточно фигур для мата! Ничья!"
            else:
                message = "Игра окончена!"

            self.status_label.setText(f"Статус: {message}")
            self.turn_label.setText("Игра окончена")

            QMessageBox.information(self, "Игра окончена", message)
            return True

        self.status_label.setText("Статус: Игра активна")
        return False

    def new_game(self):
        """Начало новой игры"""
        self.board = chess.Board()
        self.move_history = []
        self.game_over = False
        self.waiting_for_engine = False
        self.waiting_for_promotion = False
        self.history_list.clear()

        self.board_widget.set_board(self.board)
        self.board_widget.last_move = None
        self.board_widget.selected_square = None
        self.board_widget.legal_moves = []

        self.update_ui()

        # Если ИИ играет первым
        if not self.human_plays_white and self.board.turn == chess.WHITE:
            self.make_engine_move()

    def resign(self):
        """Сдаться"""
        if not self.game_over:
            reply = QMessageBox.question(self, "Сдаться",
                                         "Вы уверены, что хотите сдаться?",
                                         QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)

            if reply == QMessageBox.StandardButton.Yes:
                self.game_over = True
                if self.board.turn == chess.WHITE:
                    winner = "Черные"
                else:
                    winner = "Белые"

                message = f"Вы сдались. Победил ИИ ({winner})"
                self.status_label.setText(f"Статус: {message}")
                self.turn_label.setText("Игра окончена")
                QMessageBox.information(self, "Игра окончена", message)

    def undo_move(self):
        """Отмена последнего хода"""
        if self.game_over:
            QMessageBox.information(self, "Отмена", "Игра уже окончена!")
            return

        if len(self.move_history) >= 2:
            # Отменяем два хода (последний ход человека и ход ИИ)
            self.board.pop()
            self.board.pop()
            self.move_history.pop()
            self.move_history.pop()

            # Обновляем историю в списке
            self.history_list.clear()
            for i, move in enumerate(self.move_history, 1):
                self.history_list.addItem(f"{i}. {move}")

            self.board_widget.set_board(self.board)
            self.board_widget.last_move = None
            self.board_widget.selected_square = None
            self.board_widget.legal_moves = []
            self.game_over = False
            self.waiting_for_engine = False

            self.update_ui()
        else:
            QMessageBox.information(self, "Отмена", "Нет ходов для отмены!")


# ============ ЗАПУСК ПРИЛОЖЕНИЯ ============
def main():
    app = QApplication(sys.argv)
    window = ChessMainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()