"""
poker_metrics.py
~~~~~~~~~~~~~~~~
Одно-файловый скрипт, который считает покерные метрики **T, HS, PP, EHS, S**
для заданной ситуации.

• Работает «из коробки» — только стандартная библиотека Python 3.  
• Поддерживает произвольное число оппонентов.  
• Принимает позиции карт в привычном формате (`"Ah"`, `"Tc"`, …).  
• HS и EHS оцениваются Монте-Карло-симуляцией.

Запуск из терминала:

    python poker_metrics.py \
        --opps 2 \
        --hero "4s 4h" \
        --flop "Th Js 5h" \
        --turn "Td" \
        --river "8d" \
        --iters 50000
"""

import argparse
import math
import random
import collections
from itertools import combinations
import time
import multiprocessing
import os

try:
    import openai
except ImportError:
    openai = None

# ---------- вспомогательные таблицы ---------- #
RANKS = '23456789TJQKA'
VALUES = {r: i for i, r in enumerate(RANKS, start=2)}
DECK   = [r + s for r in RANKS for s in 'shdc']


# ---------- служебные функции работы с картами ---------- #
def rank(card):  return VALUES[card[0]]
def suit(card):  return card[1]


# ---------- оценка лучшей 5-картовой руки ---------- #
def rank_five(cards):
    """Возвращает (категория, тiebreaker-tuple)  – чем больше, тем сильнее."""
    ranks = sorted((rank(c) for c in cards), reverse=True)
    suits = [suit(c) for c in cards]
    counts = collections.Counter(ranks)

    # flush / straight
    is_flush = len(set(suits)) == 1
    unique   = sorted(set(ranks), reverse=True)
    straight_hi = None
    if len(unique) == 5 and unique[0] - unique[-1] == 4:
        straight_hi = unique[0]
    if set([14, 5, 4, 3, 2]).issubset(unique):     # колёсика A-5
        straight_hi = 5

    # hand categories
    if is_flush and straight_hi:
        return 8, (straight_hi,)
    if 4 in counts.values():
        quad = max(k for k, v in counts.items() if v == 4)
        kicker = max(k for k in ranks if k != quad)
        return 7, (quad, kicker)
    if 3 in counts.values() and 2 in counts.values():
        trips = max(k for k, v in counts.items() if v == 3)
        pair  = max(k for k, v in counts.items() if v == 2)
        return 6, (trips, pair)
    if is_flush:
        return 5, tuple(ranks)
    if straight_hi:
        return 4, (straight_hi,)
    if 3 in counts.values():
        trips = max(k for k, v in counts.items() if v == 3)
        kick  = tuple(k for k in ranks if k != trips)[:2]
        return 3, (trips,) + kick
    pairs = [k for k, v in counts.items() if v == 2]
    if len(pairs) == 2:
        hi, lo = sorted(pairs, reverse=True)
        kicker = max(k for k in ranks if k not in pairs)
        return 2, (hi, lo, kicker)
    if 2 in counts.values():
        pair = max(k for k, v in counts.items() if v == 2)
        kick = tuple(k for k in ranks if k != pair)[:3]
        return 1, (pair,) + kick
    return 0, tuple(ranks)


def best_of_seven(seven):
    if len(seven) < 5:
        return -1, ()
    return max(rank_five(combo) for combo in combinations(seven, 5))


# ---------- вычисление Board-texture T ---------- #
def board_texture_T(board):
    if not board:
        return 0.0
    ranks = [rank(c) for c in board]
    suits = [suit(c) for c in board]

    # straight-factor
    s = max(0, 4 - (max(ranks) - min(ranks))) / 4 if len(ranks) >= 2 else 0

    # flush-factor
    suit_counts = collections.Counter(suits)
    max_suit = max(suit_counts.values())
    f = 0.7 if max_suit >= 3 else (1.0 if max_suit == 2 else 0.0)
    r = 1 - f

    # pair-factor
    rc = collections.Counter(ranks)
    p = 1.0 if 3 in rc.values() or 4 in rc.values() else (0.5 if 2 in rc.values() else 0.0)

    v = 0.6 * f + 0.8 * s - 0.9 * p - 0.5 * r
    return math.tanh(v)


# ---------- Monte-Carlo (параллельная версия) ---------- #
def worker(args):
    """Рабочая функция для пула процессов, выполняет часть симуляций."""
    hero, board, unused, n_opps, iters_chunk, left_cards = args
    wins, ties = 0, 0
    for _ in range(iters_chunk):
        deck = unused.copy()
        random.shuffle(deck)

        opps = []
        try:
            for _ in range(n_opps):
                opps.append([deck.pop(), deck.pop()])
            
            if left_cards > 0:
                future = [deck.pop() for _ in range(left_cards)]
                full_board = board + future
            else:
                full_board = board
        except IndexError:
            # В редких случаях карт в колоде может не хватить на всех при большом
            # числе оппонентов. Пропускаем такую невозможную итерацию.
            continue

        h_rank = best_of_seven(hero + full_board)
        results = [best_of_seven(o + full_board) for o in opps]

        if all(h_rank > r for r in results):
            wins += 1
        elif all(h_rank >= r for r in results):
            ties += 1
    return wins, ties


def monte_carlo(hero, board, n_opps=2, iters=20000):
    hero, board = list(hero), list(board)
    unused = [c for c in DECK if c not in hero + board]

    try:
        n_cpu = min(multiprocessing.cpu_count(), iters)
    except NotImplementedError:
        n_cpu = 1
    
    if n_cpu == 1:
        # Если доступно только одно ядро, нет смысла в параллелизме.
        # Оставляем простой цикл для избежания накладных расходов.
        return monte_carlo_single_thread(hero, board, n_opps, iters)

    iters_per_cpu = iters // n_cpu
    actual_iters = iters_per_cpu * n_cpu
    
    with multiprocessing.Pool(processes=n_cpu) as pool:
        # HS (Hand Strength)
        hs_args = [(hero, board, unused, n_opps, iters_per_cpu, 0) for _ in range(n_cpu)]
        hs_results = pool.map(worker, hs_args)
        total_wins_hs = sum(r[0] for r in hs_results)
        total_ties_hs = sum(r[1] for r in hs_results)
        HS = (total_wins_hs + 0.5 * total_ties_hs) / actual_iters

        # EHS (Effective Hand Strength)
        if len(board) < 5:
            left = 5 - len(board)
            ehs_args = [(hero, board, unused, n_opps, iters_per_cpu, left) for _ in range(n_cpu)]
            ehs_results = pool.map(worker, ehs_args)
            total_wins_ehs = sum(r[0] for r in ehs_results)
            total_ties_ehs = sum(r[1] for r in ehs_results)
            EHS = (total_wins_ehs + 0.5 * total_ties_ehs) / actual_iters
        else:
            EHS = HS

    PP = (EHS - HS) / (1 - HS) if HS < 1 else 0.0
    NP = (HS - EHS) / HS if HS > 0 else 0.0
    S  = 2 * EHS - 1
    return HS, PP, NP, EHS, S


def monte_carlo_single_thread(hero, board, n_opps=2, iters=20000):
    """Оригинальная однопоточная реализация на случай, если параллелизм недоступен."""
    hero, board = list(hero), list(board)
    unused = [c for c in DECK if c not in hero + board]

    def showdown(full_board):
        wins, ties = 0, 0
        for _ in range(iters):
            deck = unused.copy()
            random.shuffle(deck)
            opps = []
            for _ in range(n_opps):
                opps.append([deck.pop(), deck.pop()])

            h_rank = best_of_seven(hero + full_board)
            results = [best_of_seven(o + full_board) for o in opps]

            if all(h_rank > r for r in results): wins += 1
            elif all(h_rank >= r for r in results): ties += 1
        return (wins + 0.5 * ties) / iters

    HS = showdown(board)

    if len(board) < 5:
        left = 5 - len(board)
        wins, ties = 0, 0
        for _ in range(iters):
            deck = unused.copy()
            random.shuffle(deck)
            opps = []
            for _ in range(n_opps):
                opps.append([deck.pop(), deck.pop()])
            future = random.sample(deck, left)
            full_board = board + future
            h_rank = best_of_seven(hero + full_board)
            results = [best_of_seven(o + full_board) for o in opps]

            if all(h_rank > r for r in results): wins += 1
            elif all(h_rank >= r for r in results): ties += 1
        EHS = (wins + 0.5 * ties) / iters
    else:
        EHS = HS

    PP = (EHS - HS) / (1 - HS) if HS < 1 else 0.0
    NP = (HS - EHS) / HS if HS > 0 else 0.0
    S  = 2 * EHS - 1
    return HS, PP, NP, EHS, S


# ---------- Анализ через OpenAI ---------- #
def get_openai_analysis(street, hero, board, n_opps, metrics, api_key):
    """Формирует промпт, отправляет в OpenAI и возвращает анализ."""
    if not openai:
        return "ОШИБКА: Библиотека OpenAI не найдена. Установите ее командой: pip install openai"

    if not api_key:
        return "ОШИБКА: Ключ API OpenAI не был предоставлен. Анализ невозможен."
    
    client = openai.OpenAI(api_key=api_key)

    T, HS, PP, NP, EHS, S = metrics
    hero_hand = ' '.join(hero)
    board_cards = ' '.join(board) or "(пусто)"
    street_name_map = {
        "Flop": "Флоп",
        "Turn": "Тёрн",
        "River": "Ривер"
    }
    street_name = street_name_map.get(street, street)


    system_prompt = "Ты — элитный покерный тренер. Твой анализ должен быть кратким, по существу и на русском языке. Анализируй ситуацию для 'Hero'."
    user_prompt = f"""**Ситуация:**
*   **Рука Hero:** {hero_hand}
*   **Оппоненты:** {n_opps}
*   **Улица:** {street_name}
*   **Борд:** {board_cards}
*   **Метрики:** T={T:.3f}, HS={HS:.3f}, EHS={EHS:.3f}, PP={PP:.3f}, NP={NP:.3f}, S={S:.3f}

**Формат ответа:**
**Анализ:** 1-2 предложения. Ключевое изменение на этой улице и как оно повлияло на силу руки.
**Рекомендация:** Конкретное действие (ставка, чек, фолд) и краткое обоснование (1 предложение).
**Итог:** Однозначный вывод: CHECK, FOLD, CALL, VALUE BET, BLUFF.
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.5,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"ОШИБКА при обращении к OpenAI API: {e}"


# ---------- точка входа CLI ---------- #
def parse_cards(s):
    return [c.strip() for c in s.split()] if s else []

def main():
    ap = argparse.ArgumentParser(description="Poker metric calculator")
    ap.add_argument('--opps',  type=int, default=1, help='кол-во оппонентов')
    ap.add_argument('--hero',  required=True, help='карты героя, напр. "Ah Kd"')
    ap.add_argument('--flop',  default='', help='флоп, напр. "7c 8c 9d"')
    ap.add_argument('--turn',  default='', help='терн, одна карта')
    ap.add_argument('--river', default='', help='ривер, одна карта')
    ap.add_argument('--iters', type=int, default=20000, help='число симуляций')
    ap.add_argument('--seed',  type=int, default=None,  help='зерно для генератора случайных чисел')
    ap.add_argument('--analyze', action='store_true', help='включить анализ результатов через OpenAI')
    args = ap.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    hero  = parse_cards(args.hero)
    flop  = parse_cards(args.flop)
    turn  = parse_cards(args.turn)
    river = parse_cards(args.river)

    if len(hero) != 2:
        raise ValueError("у героя должно быть ровно 2 карты")
    if len(flop) not in (0, 3):
        raise ValueError("флоп должен содержать 0 или 3 карты")
    if len(turn) not in (0, 1):
        raise ValueError("терн должен содержать 0 или 1 карту")
    if len(river) not in (0, 1):
        raise ValueError("ривер должен содержать 0 или 1 карту")
    if turn and not flop:
        raise ValueError("нельзя указать терн без флопа")
    if river and not turn:
        raise ValueError("нельзя указать ривер без терна")

    all_cards = hero + flop + turn + river
    if len(all_cards) != len(set(all_cards)):
        raise ValueError("обнаружены дубликаты карт")

    print(f"Hero: {' '.join(hero)}")
    print(f"Opps: {args.opps}, Iters: {args.iters}")
    print("-" * 92)
    print(f"{'Street':<8} | {'Board':<20} | {'T':>7} | {'HS':>7} | {'PP':>7} | {'NP':>7} | {'EHS':>7} | {'S':>7} | {'Time (s)':>9}")
    print("-" * 92)

    board = []
    if flop:
        board.extend(flop)
        start_time = time.time()
        T = board_texture_T(board)
        HS, PP, NP, EHS, S = monte_carlo(hero, board, args.opps, args.iters)
        duration = time.time() - start_time
        board_str = ' '.join(board)
        print(f"{'Flop':<8} | {board_str:<20} | {T:>7.3f} | {HS:>7.3f} | {PP:>7.3f} | {NP:>7.3f} | {EHS:>7.3f} | {S:>7.3f} | {duration:>9.2f}")
        if args.analyze:
            print("\n--- Анализ OpenAI ---")
            metrics = (T, HS, PP, NP, EHS, S)
            api_key = os.getenv("OPENAI_API_KEY")
            analysis = get_openai_analysis("Flop", hero, board, args.opps, metrics, api_key)
            print(analysis)
            print("---------------------\n")

    if turn:
        board.extend(turn)
        start_time = time.time()
        T = board_texture_T(board)
        HS, PP, NP, EHS, S = monte_carlo(hero, board, args.opps, args.iters)
        duration = time.time() - start_time
        board_str = ' '.join(board)
        print(f"{'Turn':<8} | {board_str:<20} | {T:>7.3f} | {HS:>7.3f} | {PP:>7.3f} | {NP:>7.3f} | {EHS:>7.3f} | {S:>7.3f} | {duration:>9.2f}")
        if args.analyze:
            print("\n--- Анализ OpenAI ---")
            metrics = (T, HS, PP, NP, EHS, S)
            api_key = os.getenv("OPENAI_API_KEY")
            analysis = get_openai_analysis("Turn", hero, board, args.opps, metrics, api_key)
            print(analysis)
            print("---------------------\n")

    if river:
        board.extend(river)
        start_time = time.time()
        T = board_texture_T(board)
        HS, PP, NP, EHS, S = monte_carlo(hero, board, args.opps, args.iters)
        duration = time.time() - start_time
        board_str = ' '.join(board)
        print(f"{'River':<8} | {board_str:<20} | {T:>7.3f} | {HS:>7.3f} | {PP:>7.3f} | {NP:>7.3f} | {EHS:>7.3f} | {S:>7.3f} | {duration:>9.2f}")
        if args.analyze:
            print("\n--- Анализ OpenAI ---")
            metrics = (T, HS, PP, NP, EHS, S)
            api_key = os.getenv("OPENAI_API_KEY")
            analysis = get_openai_analysis("River", hero, board, args.opps, metrics, api_key)
            print(analysis)
            print("---------------------\n")


if __name__ == "__main__":
    main()