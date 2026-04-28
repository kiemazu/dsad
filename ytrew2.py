import chess
import torch
import torch.nn as nn
import numpy as np
import pygame
import sys
from pygame.locals import *


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
class ChessEngine:
    def __init__(self, model_path='chess_engine_trained.pth'):
        print("Загрузка модели...")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.idx_to_move = checkpoint['idx_to_move']
            num_moves = len(self.idx_to_move)

            self.model = ChessNeuralNetwork(num_moves).to(self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            print("Модель успешно загружена!")
        except FileNotFoundError:
            print(f"Файл модели {model_path} не найден!")
            print("Создаю заглушку...")
            self.idx_to_move = {}
            self.model = None

    def get_best_move(self, board):
        """Возвращает лучший ход от движка"""
        if self.model is None:
            # Заглушка: случайный ход
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


# ============ ГРАФИЧЕСКИЙ ИНТЕРФЕЙС ============
class ChessGUI:
    def __init__(self, engine):
        pygame.init()

        # Константы
        self.SQUARE_SIZE = 80
        self.BOARD_SIZE = self.SQUARE_SIZE * 8
        self.INFO_PANEL_WIDTH = 300

        # Окно
        self.screen = pygame.display.set_mode((self.BOARD_SIZE + self.INFO_PANEL_WIDTH, self.BOARD_SIZE))
        pygame.display.set_caption("Шахматы против ИИ")

        # Цвета
        self.WHITE_SQUARE = (240, 217, 181)
        self.BLACK_SQUARE = (181, 136, 99)
        self.HIGHLIGHT_COLOR = (255, 255, 0, 128)
        self.LAST_MOVE_COLOR = (0, 255, 0, 100)
        self.CHECK_COLOR = (255, 0, 0, 150)

        # Загрузка изображений фигур
        self.pieces_images = self.load_pieces()

        # Состояние игры
        self.board = chess.Board()
        self.engine = engine
        self.selected_square = None
        self.legal_moves = []
        self.human_plays_white = True
        self.game_over = False
        self.winner = None
        self.last_move = None
        self.waiting_for_engine = False

        # Шрифты
        self.font = pygame.font.Font(None, 24)
        self.big_font = pygame.font.Font(None, 36)

        # Кнопки
        self.setup_buttons()

    def load_pieces(self):
        """Загрузка изображений фигур"""
        pieces = {}
        piece_names = {
            'p': 'bp', 'n': 'bn', 'b': 'bb', 'r': 'br', 'q': 'bq', 'k': 'bk',
            'P': 'wp', 'N': 'wn', 'B': 'wb', 'R': 'wr', 'Q': 'wq', 'K': 'wk'
        }

        # Пытаемся загрузить изображения, если их нет - используем текстовые символы
        try:
            for symbol, name in piece_names.items():
                img = pygame.image.load(f'pieces/{name}.png')
                img = pygame.transform.scale(img, (self.SQUARE_SIZE - 10, self.SQUARE_SIZE - 10))
                pieces[symbol] = img
        except:
            # Если нет изображений, используем текстовые символы
            for symbol in piece_names.keys():
                pieces[symbol] = None

        return pieces

    def setup_buttons(self):
        """Создание кнопок"""
        button_y = 50
        button_height = 40
        button_margin = 10

        self.buttons = {
            'new_game': pygame.Rect(self.BOARD_SIZE + 20, button_y, 260, button_height),
            'flip_board': pygame.Rect(self.BOARD_SIZE + 20, button_y + button_height + button_margin, 260,
                                      button_height),
            'resign': pygame.Rect(self.BOARD_SIZE + 20, button_y + 2 * (button_height + button_margin), 260,
                                  button_height)
        }

    def draw_board(self):
        """Отрисовка шахматной доски"""
        for row in range(8):
            for col in range(8):
                x = col * self.SQUARE_SIZE
                y = row * self.SQUARE_SIZE

                if (row + col) % 2 == 0:
                    color = self.WHITE_SQUARE
                else:
                    color = self.BLACK_SQUARE

                pygame.draw.rect(self.screen, color, (x, y, self.SQUARE_SIZE, self.SQUARE_SIZE))

                # Подсветка последнего хода
                if self.last_move:
                    from_row = 7 - (self.last_move.from_square // 8)
                    from_col = self.last_move.from_square % 8
                    to_row = 7 - (self.last_move.to_square // 8)
                    to_col = self.last_move.to_square % 8

                    if (row, col) == (from_row, from_col) or (row, col) == (to_row, to_col):
                        s = pygame.Surface((self.SQUARE_SIZE, self.SQUARE_SIZE))
                        s.set_alpha(128)
                        s.fill((0, 255, 0))
                        self.screen.blit(s, (x, y))

                # Подсветка выбранной клетки
                if self.selected_square:
                    selected_row = 7 - (self.selected_square // 8)
                    selected_col = self.selected_square % 8
                    if (row, col) == (selected_row, selected_col):
                        s = pygame.Surface((self.SQUARE_SIZE, self.SQUARE_SIZE))
                        s.set_alpha(128)
                        s.fill((255, 255, 0))
                        self.screen.blit(s, (x, y))

                # Подсветка возможных ходов
                for move in self.legal_moves:
                    to_row = 7 - (move.to_square // 8)
                    to_col = move.to_square % 8
                    if (row, col) == (to_row, to_col):
                        # Рисуем кружок для возможного хода
                        center_x = x + self.SQUARE_SIZE // 2
                        center_y = y + self.SQUARE_SIZE // 2
                        pygame.draw.circle(self.screen, (0, 255, 0, 100), (center_x, center_y),
                                           self.SQUARE_SIZE // 4)

                # Подсветка короля под шахом
                if self.board.is_check():
                    king_square = self.board.king(self.board.turn)
                    if king_square:
                        king_row = 7 - (king_square // 8)
                        king_col = king_square % 8
                        if (row, col) == (king_row, king_col):
                            s = pygame.Surface((self.SQUARE_SIZE, self.SQUARE_SIZE))
                            s.set_alpha(100)
                            s.fill((255, 0, 0))
                            self.screen.blit(s, (x, y))

    def draw_pieces(self):
        """Отрисовка фигур"""
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                row = 7 - (square // 8)
                col = square % 8
                x = col * self.SQUARE_SIZE
                y = row * self.SQUARE_SIZE

                piece_symbol = piece.symbol()

                if self.pieces_images[piece_symbol] is not None:
                    img = self.pieces_images[piece_symbol]
                    img_x = x + (self.SQUARE_SIZE - img.get_width()) // 2
                    img_y = y + (self.SQUARE_SIZE - img.get_height()) // 2
                    self.screen.blit(img, (img_x, img_y))
                else:
                    # Рисуем текстовый символ как запасной вариант
                    text = self.font.render(piece_symbol, True, (0, 0, 0))
                    text_x = x + (self.SQUARE_SIZE - text.get_width()) // 2
                    text_y = y + (self.SQUARE_SIZE - text.get_height()) // 2
                    self.screen.blit(text, (text_x, text_y))

    def draw_info_panel(self):
        """Отрисовка информационной панели"""
        panel_x = self.BOARD_SIZE
        panel_width = self.INFO_PANEL_WIDTH

        # Фон панели
        pygame.draw.rect(self.screen, (50, 50, 50), (panel_x, 0, panel_width, self.BOARD_SIZE))

        # Заголовок
        title = self.big_font.render("ШАХМАТЫ", True, (255, 255, 255))
        title_x = panel_x + (panel_width - title.get_width()) // 2
        self.screen.blit(title, (title_x, 20))

        # Информация о ходе
        turn_text = "Ход: " + ("Белых" if self.board.turn == chess.WHITE else "Черных")
        if (self.board.turn == chess.WHITE and self.human_plays_white) or \
                (self.board.turn == chess.BLACK and not self.human_plays_white):
            turn_text += " (Вы)"
        else:
            turn_text += " (ИИ)"

        turn_surface = self.font.render(turn_text, True, (255, 255, 255))
        self.screen.blit(turn_surface, (panel_x + 20, 100))

        # Статус игры
        if self.game_over:
            if self.winner:
                status_text = f"Победили: {self.winner}"
            else:
                status_text = "Ничья!"
            status_surface = self.big_font.render(status_text, True, (255, 255, 0))
            self.screen.blit(status_surface, (panel_x + 20, 150))

        # Счет (можно добавить)
        # Рисуем кнопки
        for button_name, button_rect in self.buttons.items():
            # Цвет кнопки
            mouse_pos = pygame.mouse.get_pos()
            if button_rect.collidepoint(mouse_pos):
                button_color = (70, 70, 70)
            else:
                button_color = (100, 100, 100)

            pygame.draw.rect(self.screen, button_color, button_rect)
            pygame.draw.rect(self.screen, (150, 150, 150), button_rect, 2)

            # Текст кнопки
            if button_name == 'new_game':
                button_text = "Новая игра"
            elif button_name == 'flip_board':
                button_text = "Повернуть доску"
            else:
                button_text = "Сдаться"

            text = self.font.render(button_text, True, (255, 255, 255))
            text_x = button_rect.x + (button_rect.width - text.get_width()) // 2
            text_y = button_rect.y + (button_rect.height - text.get_height()) // 2
            self.screen.blit(text, (text_x, text_y))

    def get_square_from_mouse(self, pos):
        """Получение клетки по координатам мыши"""
        x, y = pos
        if x < self.BOARD_SIZE and y < self.BOARD_SIZE:
            col = x // self.SQUARE_SIZE
            row = y // self.SQUARE_SIZE
            square = (7 - row) * 8 + col
            return square
        return None

    def handle_click(self, pos):
        """Обработка клика мыши"""
        # Проверка клика по кнопкам
        for button_name, button_rect in self.buttons.items():
            if button_rect.collidepoint(pos):
                if button_name == 'new_game':
                    self.new_game()
                elif button_name == 'flip_board':
                    # Можно реализовать поворот доски
                    pass
                elif button_name == 'resign':
                    if not self.game_over:
                        self.game_over = True
                        if self.board.turn == chess.WHITE:
                            self.winner = "Черные"
                        else:
                            self.winner = "Белые"
                return

        if self.game_over or self.waiting_for_engine:
            return

        square = self.get_square_from_mouse(pos)
        if square is None:
            return

        # Проверяем, чей сейчас ход
        is_human_turn = (self.board.turn == chess.WHITE and self.human_plays_white) or \
                        (self.board.turn == chess.BLACK and not self.human_plays_white)

        if not is_human_turn:
            return

        # Если клетка не выбрана
        if self.selected_square is None:
            # Выбираем клетку с фигурой нужного цвета
            piece = self.board.piece_at(square)
            if piece and piece.color == self.board.turn:
                self.selected_square = square
                self.legal_moves = [move for move in self.board.legal_moves
                                    if move.from_square == square]
        else:
            # Пытаемся сделать ход
            move = chess.Move(self.selected_square, square)

            # Проверка на превращение пешки
            piece = self.board.piece_at(self.selected_square)
            if piece and piece.piece_type == chess.PAWN:
                to_row = 7 - (square // 8)
                if (piece.color == chess.WHITE and to_row == 0) or \
                        (piece.color == chess.BLACK and to_row == 7):
                    # Предлагаем выбрать фигуру для превращения
                    promotion = self.show_promotion_dialog()
                    if promotion:
                        move = chess.Move(self.selected_square, square, promotion=promotion)

            if move in self.board.legal_moves:
                self.board.push(move)
                self.last_move = move
                self.selected_square = None
                self.legal_moves = []

                # Проверка окончания игры
                self.check_game_over()

                if not self.game_over:
                    # Ход ИИ
                    self.waiting_for_engine = True
            else:
                # Если ход неверный, сбрасываем выделение
                self.selected_square = None
                self.legal_moves = []

    def show_promotion_dialog(self):
        """Диалог выбора фигуры для превращения пешки"""
        # Простое окно выбора
        choices = [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]
        choice_names = ['Ферзь', 'Ладья', 'Слон', 'Конь']

        # Создаем временное окно
        dialog_width = 300
        dialog_height = 200
        dialog_x = (self.BOARD_SIZE + self.INFO_PANEL_WIDTH - dialog_width) // 2
        dialog_y = (self.BOARD_SIZE - dialog_height) // 2

        # Рисуем диалог (упрощенная версия - возвращаем ферзя)
        # В реальной версии нужно реализовать полноценный диалог
        return chess.QUEEN

    def check_game_over(self):
        """Проверка окончания игры"""
        if self.board.is_game_over():
            self.game_over = True
            if self.board.is_checkmate():
                # Определяем победителя
                if self.board.turn == chess.WHITE:
                    self.winner = "Черные"
                else:
                    self.winner = "Белые"
            elif self.board.is_stalemate():
                self.winner = None
            elif self.board.is_insufficient_material():
                self.winner = None

    def engine_move(self):
        """Ход движка"""
        if not self.game_over and not self.waiting_for_engine:
            return

        move = self.engine.get_best_move(self.board)
        if move:
            self.board.push(move)
            self.last_move = move
            self.check_game_over()

        self.waiting_for_engine = False

    def new_game(self):
        """Начало новой игры"""
        self.board = chess.Board()
        self.selected_square = None
        self.legal_moves = []
        self.game_over = False
        self.winner = None
        self.last_move = None
        self.waiting_for_engine = False

        # Если ИИ играет первым (человек за черных)
        if not self.human_plays_white and self.board.turn == chess.WHITE:
            self.waiting_for_engine = True

    def run(self):
        """Главный игровой цикл"""
        clock = pygame.time.Clock()
        running = True

        # Выбор цвета
        self.show_color_choice()

        # Если ИИ начинает первым
        if not self.human_plays_white and self.board.turn == chess.WHITE:
            self.waiting_for_engine = True

        while running:
            for event in pygame.event.get():
                if event.type == QUIT:
                    running = False
                elif event.type == MOUSEBUTTONDOWN:
                    self.handle_click(event.pos)
                elif event.type == KEYDOWN:
                    if event.key == K_ESCAPE:
                        running = False

            # Ход ИИ
            if self.waiting_for_engine and not self.game_over:
                self.engine_move()
                pygame.time.wait(500)  # Небольшая задержка для имитации "мышления"

            # Отрисовка
            self.draw_board()
            self.draw_pieces()
            self.draw_info_panel()

            pygame.display.flip()
            clock.tick(60)

        pygame.quit()
        sys.exit()

    def show_color_choice(self):
        """Диалог выбора цвета"""
        dialog_width = 400
        dialog_height = 300
        dialog_x = (self.BOARD_SIZE + self.INFO_PANEL_WIDTH - dialog_width) // 2
        dialog_y = (self.BOARD_SIZE - dialog_height) // 2

        # Простой диалог выбора
        buttons = {
            'white': pygame.Rect(dialog_x + 50, dialog_y + 150, 120, 50),
            'black': pygame.Rect(dialog_x + dialog_width - 170, dialog_y + 150, 120, 50)
        }

        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == MOUSEBUTTONDOWN:
                    if buttons['white'].collidepoint(event.pos):
                        self.human_plays_white = True
                        waiting = False
                    elif buttons['black'].collidepoint(event.pos):
                        self.human_plays_white = False
                        waiting = False

            # Рисуем фон
            self.draw_board()
            self.draw_pieces()
            self.draw_info_panel()

            # Рисуем диалог
            s = pygame.Surface((dialog_width, dialog_height))
            s.set_alpha(200)
            s.fill((0, 0, 0))
            self.screen.blit(s, (dialog_x, dialog_y))

            pygame.draw.rect(self.screen, (100, 100, 100), (dialog_x, dialog_y, dialog_width, dialog_height), 2)

            text = self.big_font.render("Выберите цвет", True, (255, 255, 255))
            text_x = dialog_x + (dialog_width - text.get_width()) // 2
            self.screen.blit(text, (text_x, dialog_y + 50))

            # Кнопки
            for color, rect in buttons.items():
                pygame.draw.rect(self.screen, (70, 70, 70), rect)
                pygame.draw.rect(self.screen, (150, 150, 150), rect, 2)

                text = self.font.render("Белые" if color == 'white' else "Черные",
                                        True, (255, 255, 255))
                text_x = rect.x + (rect.width - text.get_width()) // 2
                text_y = rect.y + (rect.height - text.get_height()) // 2
                self.screen.blit(text, (text_x, text_y))

            pygame.display.flip()
            pygame.time.wait(50)


# ============ ЗАПУСК ИГРЫ ============
if __name__ == "__main__":
    engine = ChessEngine('chess_engine_trained.pth')
    game = ChessGUI(engine)
    game.run()