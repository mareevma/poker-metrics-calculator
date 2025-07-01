# 🃏 Poker Metrics Calculator

Высокопроизводительная система для расчета покерных метрик (T, HS, PP, NP, EHS, S) с использованием многопоточности и Monte-Carlo симуляций.

## 🚀 Быстрый старт

### 1. Проверка системы
```bash
python check_system.py
```

### 2. Обработка данных
```bash
python poker_cleaner.py input.json output.json
```

## 📊 Что делает система

- **Извлекает карты** из текста раздач
- **Рассчитывает 6 метрик** для каждой улицы:
  - **T** - Board Texture (текстура борда)
  - **HS** - Hand Strength (сила руки)
  - **PP** - Positive Potential (потенциал улучшения) 
  - **NP** - Negative Potential (риск ухудшения)
  - **EHS** - Effective Hand Strength (эффективная сила)
  - **S** - Strength Index (индекс силы)
- **Поддерживает** флоп, терн и ривер
- **Использует** все CPU ядра (20,000 итераций Монте-Карло)

## 📁 Включенные данные

- **`clean.json`** (9.2MB) - Обработанные покерные раздачи без метрик
- **`clean_with_metrics.json`** (15MB) - Полные данные с рассчитанными метриками
- **`raw.json`** (28MB) - Исходные данные (не включены в репозиторий)

## 📈 Производительность

| Файл | Раздач | Время (8 ядер) |
|------|--------|-----------------|
| 10 MB | ~1,000 | 2-5 минут |
| 100 MB | ~10,000 | 20-40 минут |
| 1 GB | ~100,000 | 3-6 часов |

## 🛠️ Установка на сервере

### Ubuntu/Debian:
```bash
sudo apt update && sudo apt install python3 python3-pip -y
mkdir ~/poker && cd ~/poker
# Скопируйте файлы: poker_cleaner.py, check_system.py
python check_system.py
```

### Запуск в фоне:
```bash
nohup python poker_cleaner.py large_file.json output.json > process.log 2>&1 &
tail -f process.log
```

### Использование screen:
```bash
screen -S poker
python poker_cleaner.py huge_dataset.json output.json
# Ctrl+A, D для отключения
# screen -r poker для подключения
```

## 🎯 Рекомендуемые серверы

### AWS EC2:
- **c6i.2xlarge** (8 vCPU, 16 GB) - средние датасеты
- **c6i.4xlarge** (16 vCPU, 32 GB) - большие датасеты

### Google Cloud:
- **c2-standard-8** (8 vCPU, 32 GB)
- **c2-standard-16** (16 vCPU, 64 GB)

## 📝 Формат входных данных

```json
[
  {
    "prompt": "In this hand, your position is BB, and your holding is [Ace of Spade and King of Clubs].\nThe flop comes Queen of Heart, Jack of Spade, Ten of Diamond...",
    "chosen": "bet 12.00",
    "rejected": "check"
  }
]
```

## 📤 Формат выходных данных

```json
[
  {
    "prompt": "...",
    "chosen": "bet 12.00", 
    "rejected": "check",
    "metrics": {
      "flop": {"T": 0.537, "HS": 0.842, "PP": -1.546, "NP": 0.294, "EHS": 0.593, "S": 0.186},
      "turn": {"T": 0.537, "HS": 0.620, "PP": -0.166, "NP": 0.101, "EHS": 0.558, "S": 0.115},
      "river": {"T": 0.264, "HS": 0.505, "PP": 0.0, "NP": 0.0, "EHS": 0.505, "S": 0.009}
    }
  }
]
```

## 💻 Системные требования

- **CPU**: 4+ ядра (рекомендуется 8-16)
- **RAM**: 8+ GB (рекомендуется 16-32 GB)
- **Python**: 3.8+
- **Диск**: SSD предпочтительно

## 📚 Дополнительные файлы

- `SERVER_SETUP.md` - Подробное руководство по установке
- `check_system.py` - Проверка производительности системы
- `poker_metrics.py` - Автономный калькулятор метрик

---

**🎲 Готово к обработке миллионов покерных раздач!** 