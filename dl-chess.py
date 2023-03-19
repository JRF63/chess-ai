import math
import time

import numpy as np

import torch
import torch.nn as nn
import torch.utils.data

import chess
import chess.engine
import chess.pgn


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NNUE_FILENAME = "nnue.pt"

NUM_SQ = 64
NUM_PT = 10
NUM_PLANES = (NUM_SQ * NUM_PT + 1)

NUM_FEATURES = NUM_PLANES * NUM_SQ
M = 256
N = 32
K = 1


def get_halfkp_features(board: chess.Board) -> tuple[torch.Tensor, torch.Tensor]:
    """Extracts the HalfKP features for the current board."""

    def orient(is_white_pov: bool, sq: int):
        return (63 * (not is_white_pov)) ^ sq

    def halfkp_idx(is_white_pov: bool, king_sq: int, sq: int, p: chess.Piece):
        p_idx = (p.piece_type - 1) * 2 + (p.color != is_white_pov)
        return 1 + orient(is_white_pov, sq) + p_idx * NUM_SQ + king_sq * NUM_PLANES

    def piece_features(turn: bool):
        indices = []
        for sq, p in board.piece_map().items():
            if p.piece_type == chess.KING:
                continue
            indices.append(
                halfkp_idx(turn, orient(turn, board.king(turn)), sq, p)
            )

        tensor = torch.sparse_coo_tensor(
            [[0 for _ in range(len(indices))], indices],
            torch.ones(len(indices)),
            (1, NUM_FEATURES),
            device=device
        )
        return tensor

    return (piece_features(chess.WHITE), piece_features(chess.BLACK))


class ChessNN(nn.Module):

    def __init__(self):
        super(ChessNN, self).__init__()

        self.l0 = nn.Linear(NUM_FEATURES, M)
        self.l1 = nn.Linear(2 * M, N)
        self.l2 = nn.Linear(N, N)
        self.l3 = nn.Linear(N, K)

    def forward(self, white_features, black_features, stm):
        w = self.l0(white_features)
        b = self.l0(black_features)

        accumulator = (
            stm * torch.cat([w, b], dim=1) +
            (1 - stm) * torch.cat([b, w], dim=1)
        )

        l1_x = torch.clamp(accumulator, 0.0, 1.0)
        l2_x = torch.clamp(self.l1(l1_x), 0.0, 1.0)
        l3_x = torch.clamp(self.l2(l2_x), 0.0, 1.0)
        return self.l3(l3_x)

    def play(self, board: chess.Board) -> chess.Move:
        return self.find_best_move(board, 4, board.turn)

    def find_best_move(self, board: chess.Board, depth: int, color: chess.Color) -> chess.Move:
        best_move = None
        max_eval = -math.inf
        alpha = -math.inf
        beta = math.inf
        for move in board.legal_moves:
            board.push(move)
            eval = self.alpha_beta_pruning(
                board, depth - 1, alpha, beta, False, color)
            board.pop()
            if best_move is None or eval > max_eval:
                max_eval = eval
                best_move = move
            alpha = max(alpha, eval)
        return best_move

    def evaluate(self, board: chess.Board, color: chess.Color) -> float:
        w, b = get_halfkp_features(board)

        if board.turn == chess.WHITE:
            stm = torch.ones(1, 2 * M, device=device)
        else:
            stm = torch.zeros(1, 2 * M, device=device)

        with torch.no_grad():
            output = self.forward(w, b, stm)
            if board.turn != color:
                output = -output
            return output

    def alpha_beta_pruning(
        self,
        board: chess.Board,
        depth: int,
        alpha: float,
        beta: float,
        maximizing_player: bool,
        color: chess.Color
    ) -> float:
        if depth == 0:
            return self.evaluate(board, color)

        if maximizing_player:
            max_eval = -math.inf
            for move in board.legal_moves:
                board.push(move)
                eval = self.alpha_beta_pruning(
                    board, depth - 1, alpha, beta, False, color)
                board.pop()
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = math.inf
            for move in board.legal_moves:
                board.push(move)
                eval = self.alpha_beta_pruning(
                    board, depth - 1, alpha, beta, True, color)
                board.pop()
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return min_eval


def load_raw_nnue(filename: str) -> None:
    """Convert a .nnue file into a PyTorch model."""

    np.set_printoptions(threshold=5)

    with open(filename, 'rb') as f:
        # Header section
        version = int.from_bytes(f.read(4), 'little')
        hash_value = int.from_bytes(f.read(4), 'little')
        size = int.from_bytes(f.read(4), 'little')
        architecture = f.read(size).decode("utf-8")

        print(f'Version: {hex(version)}')  # 0x7af32f16
        print(f'Hash value: {hex(hash_value)}')  # 0x3e5aa6ee
        print(f'Architecture: {architecture}')
        print('')

        # Feature Transformer
        ft_input_dim = 41024
        ft_output_dim = 256
        ft_input_type_size = 2
        ft_output_type_size = 2
        ft_hash = int.from_bytes(f.read(4), 'little')
        ft_biases = np.frombuffer(f.read(
            ft_output_dim * ft_output_type_size), dtype=np.dtype('<i2')).astype('float32')
        ft_weights = np.frombuffer(f.read(
            ft_output_dim * ft_input_dim * ft_input_type_size), dtype=np.dtype('<i2')).astype('float32')
        print(f'Feature Transformer hash: {hex(ft_hash)}')  # 0x5d69d7b8
        print(ft_biases)
        print(ft_weights)
        print('')

        network_hash = int.from_bytes(f.read(4), 'little')
        print(f'Network hash: {hex(network_hash)}')  # 0x63337156

        # Hidden Layer 1
        l1_input_dim = 512  # Same with/without padding
        l1_output_dim = 32
        l1_input_type_size = 1
        l1_output_type_size = 4
        l1_biases = np.frombuffer(f.read(
            l1_output_dim * l1_output_type_size), dtype=np.dtype('<i4')).astype('float32')
        l1_weights = np.frombuffer(f.read(
            l1_output_dim * l1_input_dim * l1_input_type_size), dtype=np.dtype('<i1')).astype('float32')
        print('    Hidden Layer #1')
        print('   ', l1_biases)
        print('   ', l1_weights)
        print('')

        # Hidden Layer 2
        l2_input_dim = 32  # Same with/without padding
        l2_output_dim = 32
        l2_input_type_size = 1
        l2_output_type_size = 4
        l2_biases = np.frombuffer(f.read(
            l2_output_dim * l2_output_type_size), dtype=np.dtype('<i4')).astype('float32')
        l2_weights = np.frombuffer(f.read(
            l2_output_dim * l2_input_dim * l2_input_type_size), dtype=np.dtype('<i1')).astype('float32')
        print('    Hidden Layer #2')
        print('   ', l2_biases)
        print('   ', l2_weights)
        print('')

        # Hidden Layer 3
        l3_input_dim = 32  # Same with/without padding
        l3_output_dim = 1
        l3_input_type_size = 1
        l3_output_type_size = 4
        l3_biases = np.frombuffer(f.read(
            l3_output_dim * l3_output_type_size), dtype=np.dtype('<i4')).astype('float32')
        l3_weights = np.frombuffer(f.read(
            l3_output_dim * l3_input_dim * l3_input_type_size), dtype=np.dtype('<i1')).astype('float32')
        print('    Hidden Layer #3')
        print('   ', l3_biases)
        print('   ', l3_weights)
        print('')

        assert len(f.read()) == 0

        chess_nn = ChessNN()

        with torch.no_grad():
            ft_weights /= 127
            ft_biases /= 127
            ft_weights.shape = (NUM_FEATURES, M)

            chess_nn.l0.weight.copy_(torch.from_numpy(ft_weights).t())
            chess_nn.l0.bias.copy_(torch.from_numpy(ft_biases))

            l1_weights /= 64
            l1_biases /= 127 * 64
            l1_weights.shape = (N, 2 * M)

            chess_nn.l1.weight.copy_(torch.from_numpy(l1_weights))
            chess_nn.l1.bias.copy_(torch.from_numpy(l1_biases))

            l2_weights /= 64
            l2_biases /= 127 * 64
            l2_weights.shape = (N, N)

            chess_nn.l2.weight.copy_(torch.from_numpy(l2_weights))
            chess_nn.l2.bias.copy_(torch.from_numpy(l2_biases))

            l3_weights *= 127
            l3_weights /= 600 * 16
            l3_biases /= 600 * 16
            l3_weights.shape = (K, N)

            chess_nn.l3.weight.copy_(torch.from_numpy(l3_weights))
            chess_nn.l3.bias.copy_(torch.from_numpy(l3_biases))

            print(f'Converting {filename} to {NNUE_FILENAME}')
            torch.save(chess_nn.state_dict(), NNUE_FILENAME)


def verify() -> None:
    """Verify the output of the PyTorch model against the output of the quantized NNUE in Stockfish
    commit 84f3e867"""

    chess_nn = ChessNN()

    chess_nn.load_state_dict(torch.load(NNUE_FILENAME))
    chess_nn.eval()

    # List of (position, quantized NNUE score)
    positions = [
        ("rnbqrbnk/p4pp1/1p2p2p/2pp4/3P1P2/6PP/PPP1P3/KNBRQBNR w - - 0 15", -186),
        ("rnbqrbnk/pp4p1/2p1pp1p/3p4/5P2/PP1P2P1/2P1P2P/KNBRQBNR w - - 0 15", -80),
        ("knbrqbnr/2p2p1p/pp1p2p1/4p3/4P3/1P3P1P/P1PP2P1/RNBQRBNK w - - 0 15", 38),
        ("r7/5pk1/3p2p1/pPpPp3/B1P1P2b/6R1/5PK1/8 w - - 0 98", -327),
        ("5k2/4r1pp/5n2/8/3NBP1P/R3P3/5K2/8 b - - 0 43", -1845),
        ("8/2p3k1/1p1p2p1/1P1PbP1p/2P5/4BK1P/r4P2/2R5 b - - 0 36", -97),
        ("rnb1kbnr/pp1p1ppp/4p3/1Bp3q1/8/2P1P3/PP1P1PPP/RNBQK1NR w KQkq -", 18),
        ("rnb1kbnr/ppq2ppp/2p1p3/3p4/8/1P1P1N2/PBP1PPPP/RN1QKB1R w KQkq -", 88),
        ("rnbqkbnr/1p1p1p2/p3p1pp/2p5/P1P5/3PBN2/1P2PPPP/RN1QKB1R w KQkq -", -5),
        ("knbrqbnr/pp5p/3pppp1/2p5/4P3/2PP2P1/PP3P1P/RNBQRBNK w - c6", 72)
    ]

    # The kPonanzaConstant
    stockfish_scaling_factor = 600

    for pos, score in positions:
        board = chess.Board(pos)

        w, b = get_halfkp_features(board)

        if board.turn == chess.WHITE:
            stm = torch.ones(1, 2 * M)
        else:
            stm = torch.zeros(1, 2 * M)

        with torch.no_grad():
            output = chess_nn.forward(w, b, stm)
            scaled = float(output * stockfish_scaling_factor)
            percent_diff = float(abs((scaled - score) / score)) * 100.0
            print(
                f"PyTorch: {scaled:8.2f}  |  NNUE: {score:5}  |  % diff.: {percent_diff:5.2f}")


def play_vs_stockfish(elo: int, rounds: int) -> None:
    """Test the neural net against an Elo-limited Stockfish"""

    chess_nn = ChessNN().to(device)

    chess_nn.load_state_dict(torch.load(NNUE_FILENAME))
    chess_nn.eval()

    opponent = chess.engine.SimpleEngine.popen_uci("stockfish.exe")
    opponent.configure({"UCI_Elo": elo, "UCI_LimitStrength": True})

    # Recent Stockfish's `UCI_Elo` is calibrated against time control 120.0s+1.0s, old versions are
    # using 60.0s+0.6s.
    clocks = [120.0, 120.0]  # [White's, Black's]
    clock_inc = 1.0

    score_table = [0, 0]

    chess_nn_color = chess.BLACK

    for game_num in range(rounds):
        chess_nn_color = not chess_nn_color
        board = chess.Board()

        outcome = board.outcome()
        while outcome is None:
            print('.', end='', flush=True)
            if board.turn == chess_nn_color:
                move = chess_nn.play(board)
            else:
                clock_idx = chess_nn_color == chess.WHITE

                # Increment Stockfish's clock
                clocks[clock_idx] += clock_inc

                # limit = chess.engine.Limit(
                #     white_clock=clocks[0],
                #     black_clock=clocks[1],
                #     white_inc=clock_inc,
                #     black_inc=clock_inc
                # )
                limit = chess.engine.Limit(time=0.001)

                start = time.time()
                result = opponent.play(board, limit, game=game_num)
                end = time.time()
                elapsed = end - start

                # Subtract the time it took Stockfish to make its move
                clocks[clock_idx] -= elapsed

                # Make it such that Stockfish cannot lose by timeout - it will instead be limited
                # to making 1 millisecond moves if it consumes all its time budget.
                if clocks[clock_idx] < 0:
                    clocks[clock_idx] = 0.001 - clock_inc

                move = result.move

            board.push(move)
            outcome = board.outcome()

        if outcome.winner is None:
            score_table[0] += 0.5
            score_table[1] += 0.5
            game_result = "1/2-1/2"
        else:
            if outcome.winner == chess.WHITE:
                game_result = "1-0"
            else:
                game_result = "0-1"

            if outcome.winner == chess_nn_color:
                score_table[0] += 1
            else:
                score_table[1] += 1

        if chess_nn_color == chess.WHITE:
            white_name = "ChessNN"
            black_name = f"Stockfish-Elo{elo}"
        else:
            white_name = f"Stockfish-Elo{elo}"
            black_name = "ChessNN"

        game = chess.pgn.Game({
            "White": white_name,
            "Black": black_name,
            "Result": game_result,
            "Round": game_num + 1
        })

        moves = iter(board.move_stack)
        node = game.add_variation(next(moves))

        try:
            while True:
                node = node.add_variation(next(moves))
        except:
            pass

        print(game)
        print('')

    print('Final score:')
    print(f'ChessNN: {score_table[0]}, Stockfish-Elo{elo}: {score_table[1]}')

    opponent.quit()


if __name__ == '__main__':
    # load_raw_nnue('nn-97f742aaefcd.nnue')
    # verify()
    play_vs_stockfish(1500, 1)
