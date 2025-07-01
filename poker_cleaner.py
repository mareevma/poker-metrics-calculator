# -*- coding: utf-8 -*-
"""poker_cleaner.py – v0.7  (2025‑07‑01)

* Потоковая очистка большого массива раздач 6‑max NLHE.
* Исправляет `raise` → `raise_to`, корректирует суммы `call`, пересчитывает банк.
* Добавлено вычисление покерных метрик T, HS, PP, EHS, S для каждой раздачи.
* Статистика:
    - наличие/корректность строки пота;
    - совпадение типа действия в `chosen` / `rejected`;
    - распределение действий в `chosen` и `rejected` (количество и суммы);
    - средние суммы на каждый тип действия.

Работает без внешних зависимостей; при наличии **orjson** читает ~3× быстрее.
"""
from __future__ import annotations

import json
import re
import sys
import math
import random
import collections
import multiprocessing
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Tuple
from itertools import combinations

try:
    import orjson as fastjson  # type: ignore
except ImportError:  # fallback на stdlib
    fastjson = None

# ───────────────────────── regex ────────────────────────── #
RE_POT_LINE = re.compile(r"current pot size is ([\d.]+) chips", re.I)
RE_IN_HAND  = re.compile(r"In this hand,", re.I)
RE_NOW_TURN = re.compile(r"Now it is your turn", re.I)
RE_COMES_FIX = re.compile(r"comes(UTG|HJ|CO|BTN|SB|BB)")
RE_STREET = re.compile(r"\n(The flop|The turn|The river) comes", re.I)
RE_ACTION = re.compile(r"(UTG|HJ|CO|BTN|SB|BB)\s+(bet|raise_to|raise|call|check|fold)\s*(\d+\.\d+)?", re.I)
RE_PREFLOP = re.compile(r"(Before the flop,)([\s\S]*?)(?=\nThe flop|\nThe turn|\nThe river|$)", re.I)
# Регулярное выражение для парсинга действий с суммами из chosen/rejected
RE_PARSE_ACTION = re.compile(r"^(bet|raise_to|raise|call|check|fold)(?:\s+([\d.]+))?", re.I)

# Regex для извлечения карт из промпта
RE_HERO_CARDS = re.compile(r"your holding is \[([^]]+)\]", re.I)
RE_FLOP_CARDS = re.compile(r"The flop comes ([^,]+), ([^,]+), ([^,.\n]+)", re.I)
RE_TURN_CARD = re.compile(r"The turn comes ([^,.\n]+)", re.I)
RE_RIVER_CARD = re.compile(r"The river comes ([^,.\n]+)", re.I)

POSITIONS = ("UTG", "HJ", "CO", "BTN", "SB", "BB")
ACTION_SET = {"call", "raise", "bet", "check", "fold"}

# ───────────────────────── poker constants ────────────────────────── #
RANKS = '23456789TJQKA'
VALUES = {r: i for i, r in enumerate(RANKS, start=2)}
DECK = [r + s for r in RANKS for s in 'shdc']

# Конвертация длинных названий карт в короткие
RANK_MAP = {
    'Two': '2', 'Three': '3', 'Four': '4', 'Five': '5', 'Six': '6', 'Seven': '7', 'Eight': '8', 'Nine': '9',
    'Ten': 'T', 'Jack': 'J', 'Queen': 'Q', 'King': 'K', 'Ace': 'A'
}

SUIT_MAP = {
    'Spade': 's', 'Heart': 'h', 'Diamond': 'd', 'Club': 'c'
}

def convert_card_name(card_text: str) -> str:
    """Конвертирует 'Two of Diamond' в '2d'."""
    card_text = card_text.strip()
    for rank_long, rank_short in RANK_MAP.items():
        for suit_long, suit_short in SUIT_MAP.items():
            if f"{rank_long} of {suit_long}" in card_text:
                return rank_short + suit_short
    return ""

def parse_hero_cards(hero_text: str) -> List[str]:
    """Парсит '[Two of Diamond and Two of Heart]' -> ['2d', '2h']."""
    cards = []
    # Убираем скобки и разделяем по "and"
    clean_text = hero_text.strip('[]')
    parts = clean_text.split(' and ')
    
    for part in parts:
        card = convert_card_name(part.strip())
        if card:
            cards.append(card)
    
    return cards

# ─────────────────── poker helpers ──────────────────────── #

def rank(card): 
    return VALUES[card[0]]

def suit(card): 
    return card[1]

def rank_five(cards):
    """Возвращает (категория, tiebreaker-tuple) – чем больше, тем сильнее."""
    ranks = sorted((rank(c) for c in cards), reverse=True)
    suits = [suit(c) for c in cards]
    counts = collections.Counter(ranks)

    # flush / straight
    is_flush = len(set(suits)) == 1
    unique = sorted(set(ranks), reverse=True)
    straight_hi = None
    if len(unique) == 5 and unique[0] - unique[-1] == 4:
        straight_hi = unique[0]
    if set([14, 5, 4, 3, 2]).issubset(unique):  # колёсика A-5
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
        pair = max(k for k, v in counts.items() if v == 2)
        return 6, (trips, pair)
    if is_flush:
        return 5, tuple(ranks)
    if straight_hi:
        return 4, (straight_hi,)
    if 3 in counts.values():
        trips = max(k for k, v in counts.items() if v == 3)
        kick = tuple(k for k in ranks if k != trips)[:2]
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

def worker_mc(args):
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

def monte_carlo_parallel(hero, board, n_opps=2, iters=20000):
    """Параллельная версия Монте-Карло с использованием всех CPU ядер."""
    hero, board = list(hero), list(board)
    unused = [c for c in DECK if c not in hero + board]

    try:
        n_cpu = multiprocessing.cpu_count()
    except NotImplementedError:
        n_cpu = 1
    
    if n_cpu == 1 or iters < 1000:
        # Если доступно только одно ядро или мало итераций, используем простую версию
        return monte_carlo_simple(hero, board, n_opps, iters)

    iters_per_cpu = iters // n_cpu
    actual_iters = iters_per_cpu * n_cpu
    
    with multiprocessing.Pool(processes=n_cpu) as pool:
        # HS (Hand Strength)
        hs_args = [(hero, board, unused, n_opps, iters_per_cpu, 0) for _ in range(n_cpu)]
        hs_results = pool.map(worker_mc, hs_args)
        total_wins_hs = sum(r[0] for r in hs_results)
        total_ties_hs = sum(r[1] for r in hs_results)
        HS = (total_wins_hs + 0.5 * total_ties_hs) / actual_iters

        # EHS (Effective Hand Strength)
        if len(board) < 5:
            left = 5 - len(board)
            ehs_args = [(hero, board, unused, n_opps, iters_per_cpu, left) for _ in range(n_cpu)]
            ehs_results = pool.map(worker_mc, ehs_args)
            total_wins_ehs = sum(r[0] for r in ehs_results)
            total_ties_ehs = sum(r[1] for r in ehs_results)
            EHS = (total_wins_ehs + 0.5 * total_ties_ehs) / actual_iters
        else:
            EHS = HS

    PP = (EHS - HS) / (1 - HS) if HS < 1 else 0.0
    NP = (HS - EHS) / HS if HS > 0 else 0.0
    S = 2 * EHS - 1
    return HS, PP, NP, EHS, S

def monte_carlo_simple(hero, board, n_opps=2, iters=5000):
    """Упрощенная однопоточная версия Монте-Карло для быстрого расчета."""
    hero, board = list(hero), list(board)
    unused = [c for c in DECK if c not in hero + board]

    def showdown(full_board):
        wins, ties = 0, 0
        for _ in range(iters):
            deck = unused.copy()
            random.shuffle(deck)
            opps = []
            try:
                for _ in range(n_opps):
                    opps.append([deck.pop(), deck.pop()])
            except IndexError:
                continue

            h_rank = best_of_seven(hero + full_board)
            results = [best_of_seven(o + full_board) for o in opps]

            if all(h_rank > r for r in results):
                wins += 1
            elif all(h_rank >= r for r in results):
                ties += 1
        return (wins + 0.5 * ties) / iters

    HS = showdown(board)

    if len(board) < 5:
        left = 5 - len(board)
        wins, ties = 0, 0
        for _ in range(iters):
            deck = unused.copy()
            random.shuffle(deck)
            opps = []
            try:
                for _ in range(n_opps):
                    opps.append([deck.pop(), deck.pop()])
                future = random.sample(deck, left)
                full_board = board + future
            except (IndexError, ValueError):
                continue

            h_rank = best_of_seven(hero + full_board)
            results = [best_of_seven(o + full_board) for o in opps]

            if all(h_rank > r for r in results):
                wins += 1
            elif all(h_rank >= r for r in results):
                ties += 1
        EHS = (wins + 0.5 * ties) / iters
    else:
        EHS = HS

    PP = (EHS - HS) / (1 - HS) if HS < 1 else 0.0
    NP = (HS - EHS) / HS if HS > 0 else 0.0
    S = 2 * EHS - 1
    return HS, PP, NP, EHS, S

def extract_cards_from_prompt(prompt: str):
    """Извлекает карты героя и борд из промпта."""
    hero_cards = []
    flop_cards = []
    turn_card = None
    river_card = None
    
    # Извлечение карт героя
    hero_match = RE_HERO_CARDS.search(prompt)
    if hero_match:
        hero_cards = parse_hero_cards(hero_match.group(1))
    
    # Извлечение флопа
    flop_match = RE_FLOP_CARDS.search(prompt)
    if flop_match:
        flop1 = convert_card_name(flop_match.group(1))
        flop2 = convert_card_name(flop_match.group(2))
        flop3 = convert_card_name(flop_match.group(3))
        if flop1 and flop2 and flop3:
            flop_cards = [flop1, flop2, flop3]
    
    # Извлечение терна
    turn_match = RE_TURN_CARD.search(prompt)
    if turn_match:
        turn_card = convert_card_name(turn_match.group(1))
    
    # Извлечение ривера
    river_match = RE_RIVER_CARD.search(prompt)
    if river_match:
        river_card = convert_card_name(river_match.group(1))
    
    return hero_cards, flop_cards, turn_card, river_card

def calculate_metrics(hero_cards, flop_cards, turn_card, river_card, n_opps=2):
    """Рассчитывает метрики для доступных этапов раздачи."""
    if len(hero_cards) != 2:
        return {}
    
    metrics = {}
    
    try:
        # Флоп метрики (20000 итераций по умолчанию)
        if len(flop_cards) == 3:
            board = flop_cards.copy()
            T = board_texture_T(board)
            HS, PP, NP, EHS, S = monte_carlo_parallel(hero_cards, board, n_opps)  # Используем дефолт 20000
            metrics["flop"] = {
                "T": round(T, 4),
                "HS": round(HS, 4),
                "PP": round(PP, 4),
                "NP": round(NP, 4),
                "EHS": round(EHS, 4),
                "S": round(S, 4)
            }
            
            # Терн метрики
            if turn_card:
                board = flop_cards + [turn_card]
                T = board_texture_T(board)
                HS, PP, NP, EHS, S = monte_carlo_parallel(hero_cards, board, n_opps)
                metrics["turn"] = {
                    "T": round(T, 4),
                    "HS": round(HS, 4),
                    "PP": round(PP, 4),
                    "NP": round(NP, 4),
                    "EHS": round(EHS, 4),
                    "S": round(S, 4)
                }
                
                # Ривер метрики
                if river_card:
                    board = flop_cards + [turn_card, river_card]
                    T = board_texture_T(board)
                    HS, PP, NP, EHS, S = monte_carlo_parallel(hero_cards, board, n_opps)
                    metrics["river"] = {
                        "T": round(T, 4),
                        "HS": round(HS, 4),
                        "PP": round(PP, 4),
                        "NP": round(NP, 4),
                        "EHS": round(EHS, 4),
                        "S": round(S, 4)
                    }
    except Exception as e:
        return {}
    
    return metrics

# ─────────────────── JSON helpers ───────────────────────── #

def stream_load(path: Path) -> Iterator[dict]:
    """Yield dicts from a large JSON array without loading it fully."""
    if fastjson:
        for obj in fastjson.loads(path.read_bytes()):
            yield obj
        return
    buf, depth = "", 0
    with path.open("r", encoding="utf8") as fh:
        for ch in fh.read():
            if ch == '{':
                if depth == 0:
                    buf = '{'
                else:
                    buf += ch
                depth += 1
            elif ch == '}':
                depth -= 1
                buf += ch
                if depth == 0:
                    yield json.loads(buf)
            else:
                if depth:
                    buf += ch


def stream_dump(path: Path, objs: Iterable[dict]) -> None:
    with path.open("w", encoding="utf8") as fh:
        fh.write("[\n")
        first = True
        for obj in objs:
            if not first:
                fh.write(",\n")
            else:
                first = False
            fh.write(json.dumps(obj, ensure_ascii=False))
        fh.write("\n]\n")

# ─────────────────── poker helpers ──────────────────────── #

def parse_action_string(action_str: str) -> Tuple[str, float]:
    """Парсит строку действия и возвращает (действие, сумма)."""
    if not action_str:
        return "", 0.0
    
    action_str = action_str.strip()
    match = RE_PARSE_ACTION.match(action_str)
    
    if not match:
        return "", 0.0
    
    action = match.group(1).lower()
    amount_str = match.group(2)
    amount = float(amount_str) if amount_str else 0.0
    
    # Нормализация действий
    if action == "raise":
        action = "raise_to"
    
    return action, amount

def calc_street(actions_raw: str, invested: Dict[str, float], current_bet: float) -> Tuple[str, float, float]:
    out: List[str] = []
    pot_add = 0.0
    for pos, act, amt_s in RE_ACTION.findall(actions_raw):
        pos, act = pos.upper(), act.lower()
        amt = float(amt_s) if amt_s else None
        if act == "raise":
            act = "raise_to"
        if act == "raise_to" and amt is not None:
            add = amt - invested[pos]
            invested[pos] = amt
            current_bet = amt
            out.append(f"{pos} raise_to {amt:.2f}")
        elif act == "bet" and amt is not None:
            add = amt
            invested[pos] += add
            current_bet = amt
            out.append(f"{pos} bet {amt:.2f}")
        elif act == "call":
            add = max(0.0, current_bet - invested[pos])
            invested[pos] += add
            out.append(f"{pos} call {add:.2f}")
        elif act in ("check", "fold"):
            add = 0.0
            out.append(f"{pos} {act}")
        else:
            add = 0.0
            out.append(f"{pos} {act} {amt_s or ''}")
        pot_add += add
    return ", ".join(out), pot_add, current_bet


def fix_preflop(text: str, invested: Dict[str, float], current_bet: float) -> Tuple[str, float]:
    m = RE_PREFLOP.search(text)
    if not m:
        return text, 0.0
    prefix, actions = m.groups()
    fixed_actions, add, _ = calc_street(actions, invested, current_bet)
    rest = text[m.end(2):]
    return RE_PREFLOP.sub(f"{prefix} {fixed_actions}{rest}", text, count=1), add


def recalc_prompt(prompt: str) -> Tuple[str, bool, bool]:
    prompt = RE_COMES_FIX.sub(r"comes \1", prompt).replace("..", ".")

    # dynamic slice containing everything from "In this hand," to pot‑line/turn marker
    start = RE_IN_HAND.search(prompt)
    if not start:
        dynamic = prompt
    else:
        s = start.start()
        pot_m = RE_POT_LINE.search(prompt, s)
        now_m = RE_NOW_TURN.search(prompt, s)
        end = pot_m.end() if pot_m else (now_m.end() if now_m else len(prompt))
        dynamic = prompt[s:end]

    # blinds
    sb, bb = 0.5, 1.0
    pot = sb + bb
    invested = {p: 0.0 for p in POSITIONS}
    invested["SB"], invested["BB"] = sb, bb
    current_bet = bb

    # pre‑flop
    dynamic, add = fix_preflop(dynamic, invested, current_bet)
    pot += add

    invested = {p: 0.0 for p in POSITIONS}  # reset each street
    current_bet = 0.0

    parts = RE_STREET.split(dynamic)
    rebuilt = parts[0]
    for i in range(1, len(parts), 2):
        header, body = parts[i], parts[i + 1]
        fixed_body, add, current_bet = calc_street(body, invested, current_bet)
        pot += add
        rebuilt += f"\n{header} comes {fixed_body}"
        invested = {p: 0.0 for p in POSITIONS}
        current_bet = 0.0

    pot_line = f"current pot size is {pot:.1f} chips"
    original_pot = RE_POT_LINE.search(prompt)
    pot_present = bool(original_pot)
    pot_ok = False
    if pot_present:
        pot_ok = abs(float(original_pot.group(1)) - pot) < 0.1
        rebuilt = RE_POT_LINE.sub(pot_line, rebuilt)
    else:
        rebuilt += f"\n\nTo remind you, the {pot_line}."

    return rebuilt, pot_ok, pot_present

# ─────────────────── pipeline ─────────────────────────────── #

def process_file(inp: Path, out: Path) -> None:
    total = pot_present = pot_match = pot_mismatch = same_act = 0
    chosen_counter: Counter[str] = Counter()
    rejected_counter: Counter[str] = Counter()
    # Новые счетчики для сумм действий
    chosen_sums: Counter[str] = Counter()
    rejected_sums: Counter[str] = Counter()
    # Счетчики для метрик
    metrics_calculated = 0
    metrics_failed = 0

    def generator() -> Iterator[dict]:
        nonlocal total, pot_present, pot_match, pot_mismatch, same_act, metrics_calculated, metrics_failed
        for obj in stream_load(inp):
            total += 1
            
            # Сначала извлекаем карты из оригинального промпта
            original_prompt = obj["prompt"]
            try:
                hero_cards, flop_cards, turn_card, river_card = extract_cards_from_prompt(original_prompt)
                metrics = calculate_metrics(hero_cards, flop_cards, turn_card, river_card)
                if metrics:
                    obj["metrics"] = metrics
                    metrics_calculated += 1
                else:
                    metrics_failed += 1
            except Exception as e:
                metrics_failed += 1
            
            # Затем обрабатываем промпт для исправления покерной логики
            fixed, pot_ok, present = recalc_prompt(original_prompt)
            obj["prompt"] = fixed

            if present:
                pot_present += 1
                if pot_ok:
                    pot_match += 1
                else:
                    pot_mismatch += 1

            # Парсинг chosen и rejected с извлечением действий и сумм
            chosen_str = obj.get("chosen", "")
            rejected_str = obj.get("rejected", "")
            
            ch_action, ch_amount = parse_action_string(chosen_str)
            rj_action, rj_amount = parse_action_string(rejected_str)
            
            # Старая логика для совместимости
            ch = chosen_str.split()[0].lower() if chosen_str else ""
            rj = rejected_str.split()[0].lower() if rejected_str else ""
            
            if ch == rj and ch in ACTION_SET:
                same_act += 1
            
            # Счетчики количества действий
            if ch in ACTION_SET:
                chosen_counter[ch] += 1
            if rj in ACTION_SET:
                rejected_counter[rj] += 1
            
            # Счетчики сумм действий
            if ch_action in ACTION_SET:
                chosen_sums[ch_action] += ch_amount
            if rj_action in ACTION_SET:
                rejected_sums[rj_action] += rj_amount
            
            yield obj

    stream_dump(out, generator())

    # ── report ──
    miss = total - pot_present
    try:
        cpu_count = multiprocessing.cpu_count()
    except NotImplementedError:
        cpu_count = 1
    
    print(f"Processed                      : {total:,}")
    print(f"CPU cores used                 : {cpu_count}")
    print(f"Pot lines present              : {pot_present:,} ({pot_present/total:.1%})")
    if pot_present:
        print(f"  ├─ matched                   : {pot_match:,} ({pot_match/pot_present:.1%})")
        print(f"  └─ mismatched                : {pot_mismatch:,} ({pot_mismatch/pot_present:.1%})")
    print(f"Pot lines missing              : {miss:,} ({miss/total:.1%})")
    print(f"Same action chosen ≡ rejected  : {same_act:,} ({same_act/total:.1%})")
    print(f"Metrics calculated             : {metrics_calculated:,} ({metrics_calculated/total:.1%})")
    print(f"Metrics failed                 : {metrics_failed:,} ({metrics_failed/total:.1%})")
    print()

    def show_counts(counter: Counter[str], title: str) -> None:
        """Pretty-print action distribution in fixed order."""
        order = ["call", "bet", "raise_to", "check", "fold"]
        total_cnt  = sum(counter.values()) or 1
        print(title)
        for act in order:
            cnt = counter.get(act, 0)
            print(f"  {act:<8}: {cnt:>7,} ({cnt/total_cnt:5.1%})")
        print()

    def show_sums(counter: Counter[str], title: str) -> None:
        """Pretty-print action sums in fixed order."""
        order = ["call", "bet", "raise_to", "check", "fold"]
        total_sum = sum(counter.values())
        print(title)
        for act in order:
            amount = counter.get(act, 0.0)
            pct = (amount / total_sum * 100) if total_sum > 0 else 0
            print(f"  {act:<8}: {amount:>12,.2f} ({pct:5.1f}%)")
        print(f"  {'Total':<8}: {total_sum:>12,.2f}")
        print()

    def show_averages(sums: Counter[str], counts: Counter[str], title: str) -> None:
        """Pretty-print average amounts per action."""
        order = ["call", "bet", "raise_to", "check", "fold"]
        print(title)
        for act in order:
            total_sum = sums.get(act, 0.0)
            count = counts.get(act, 0)
            avg = total_sum / count if count > 0 else 0.0
            print(f"  {act:<8}: {avg:>12,.2f} (avg per action)")
        print()

    show_counts(chosen_counter, "Chosen actions (counts):")
    show_counts(rejected_counter, "Rejected actions (counts):")
    print()
    show_sums(chosen_sums, "Chosen actions (sums):")
    show_sums(rejected_sums, "Rejected actions (sums):")
    show_averages(chosen_sums, chosen_counter, "Chosen actions (averages):")
    show_averages(rejected_sums, rejected_counter, "Rejected actions (averages):")
    
    print("="*60)
    print("SUMMARY:")
    print(f"Total chosen actions   : {sum(chosen_counter.values()):,}")
    print(f"Total rejected actions : {sum(rejected_counter.values()):,}")
    print(f"Total chosen sum       : {sum(chosen_sums.values()):,.2f}")
    print(f"Total rejected sum     : {sum(rejected_sums.values()):,.2f}")
    
    # Общие средние
    total_chosen = sum(chosen_counter.values())
    total_rejected = sum(rejected_counter.values())
    avg_chosen = sum(chosen_sums.values()) / total_chosen if total_chosen > 0 else 0
    avg_rejected = sum(rejected_sums.values()) / total_rejected if total_rejected > 0 else 0
    
    print(f"Average chosen amount  : {avg_chosen:,.2f}")
    print(f"Average rejected amount: {avg_rejected:,.2f}")
    print(f"Metrics success rate   : {metrics_calculated/total:.1%}")

# ─────────────────────── main ─────────────────────────────── #
if __name__ == "__main__":
    # Защита для multiprocessing на Windows и macOS
    multiprocessing.set_start_method('spawn', force=True)
    
    if len(sys.argv) != 3:
        sys.exit("Usage: python poker_cleaner.py input.json output.json")
    process_file(Path(sys.argv[1]), Path(sys.argv[2]))
