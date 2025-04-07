# data_generation/filter_minigrid.py

import numpy as np
import argparse
import os

def filter_dataset_by_return(
    input_path,
    output_path,
    min_return_threshold=0.0,
    ):

    print(f"Загрузка исходного датасета из: {input_path}")
    try:
        data = np.load(input_path, allow_pickle=True)
        states_all = data['states']
        actions_all = data['actions']
        returns_all = data['returns']
        # Загружаем метаданные, если есть, чтобы передать дальше
        metadata_in = data['metadata'].item() if 'metadata' in data else {}
        print(f"Загружено {len(states_all)} эпизодов.")
    except Exception as e:
        print(f"Ошибка загрузки датасета {input_path}: {e}")
        return

    filtered_indices = []
    for i in range(len(returns_all)):
        if len(returns_all[i]) > 0 and returns_all[i][0] >= min_return_threshold:
            filtered_indices.append(i)

    num_filtered = len(filtered_indices)
    print(f"Найдено {num_filtered} эпизодов с RTG[0] >= {min_return_threshold}")

    if num_filtered == 0:
        print("Нет эпизодов, удовлетворяющих критерию. Выход.")
        return

    states_filtered = states_all[filtered_indices]
    actions_filtered = actions_all[filtered_indices]
    returns_filtered = returns_all[filtered_indices]

    episode_returns = [ep[0] for ep in returns_filtered if len(ep) > 0]
    max_return_filt = max(episode_returns) if episode_returns else 0
    avg_return_filt = sum(episode_returns) / len(episode_returns) if episode_returns else 0
    total_steps_filt = sum(len(ep) for ep in states_filtered)
    avg_steps_filt = total_steps_filt / num_filtered if num_filtered else 0

    print(f"\nСтатистика отфильтрованного датасета:")
    print(f"  Количество эпизодов: {num_filtered}")
    print(f"  Всего шагов: {total_steps_filt}")
    print(f"  Средняя длина эпизода: {avg_steps_filt:.2f}")
    print(f"  Макс. суммарная награда: {max_return_filt:.4f}")
    print(f"  Сред. суммарная награда: {avg_return_filt:.4f}")

    metadata_out = metadata_in.copy()
    metadata_out.update({
        'data_type': f"medium_filtered_return_ge_{min_return_threshold}",
        'num_episodes': num_filtered,
        'max_return': max_return_filt,
        'avg_return': avg_return_filt,
        'source_file': os.path.basename(input_path)
    })

    print(f"\nСохранение отфильтрованного датасета в {output_path}...")
    np.savez_compressed(
        output_path,
        states=states_filtered,
        actions=actions_filtered,
        returns=returns_filtered,
        metadata=metadata_out
    )
    print("Отфильтрованный датасет сохранен.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Путь к исходному .npz датасету (например, случайному)")
    parser.add_argument("--output", type=str, required=True, help="Путь для сохранения отфильтрованного .npz датасета")
    parser.add_argument("--min_return", type=float, default=0.01, help="Минимальный RTG[0] (суммарная награда) для включения эпизода")
    args = parser.parse_args()
    
    output_dir = os.path.dirname(args.output)
    if output_dir:
         os.makedirs(output_dir, exist_ok=True)

    filter_dataset_by_return(
        input_path=args.input,
        output_path=args.output,
        min_return_threshold=args.min_return
    )

if __name__ == "__main__":
    main()