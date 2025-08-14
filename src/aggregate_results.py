import pandas as pd
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re # ★ 正規表現モジュールをインポート

def aggregate_and_visualize(base_dir: Path, eval_intervals: list, models: list):
    """
    指定されたディレクトリ内の全 result_*.csv を集約し、
    サマリーテキストと性能推移プロットを生成する。
    """
    all_dfs = []
    csv_files = list(base_dir.rglob("result_*.csv"))
    
    if not csv_files:
        print(f"No result CSV files found in {base_dir}")
        return

    print(f"Found {len(csv_files)} result files. Aggregating...")
    
    # === ★ 変更点: 正規表現による堅牢なファイル名パース ===
    # パターン: result_(モデル名)_(データセット名)_split_(スプリットID).csv
    # モデル名とデータセット名にはアンダースコアが含まれる可能性がある
    pattern = re.compile(r"result_([A-Z\-]+)_(.+)_split_(\d+)\.csv")

    for f in csv_files:
        try:
            # 正規表現でファイル名をマッチング
            match = pattern.search(f.name)
            if not match:
                print(f"Warning: Skipping file with unexpected name format: {f.name}")
                continue

            model_name, dataset_name, split_id_str = match.groups()
            split_id = int(split_id_str)
            
            df = pd.read_csv(f)
            if not df.empty:
                df['model'] = model_name
                df['dataset'] = dataset_name
                df['split_id'] = split_id
                all_dfs.append(df)
        except (pd.errors.EmptyDataError, IndexError, ValueError) as e:
            print(f"Warning: Skipping malformed or empty file {f} due to {e}")
            continue
    # ========================================================
            
    if not all_dfs:
        print("No valid data to aggregate.")
        return

    full_df = pd.concat(all_dfs, ignore_index=True)

    # # --- 1. サマリーテキストの生成 (以降のロジックは変更なし) ---
    # summary_text_path = base_dir / "performance_summary.txt"
    # with open(summary_text_path, 'w') as f:
    #     # ... (以前の回答と同じ) ...
    #     for epoch in eval_intervals:
    #         f.write(f"epoch={epoch}\n")
    #         epoch_df = full_df[full_df['iteration'] == epoch]
    #         if epoch_df.empty:
    #             f.write("No data for this epoch.\n\n")
    #             continue
            
    #         summary = epoch_df.groupby(['model', 'dataset']).agg(
    #             avg_rmse=('rmse', 'mean'),
    #             std_rmse=('rmse', 'std'),
    #         ).reset_index()

    #         time_summary = full_df.loc[full_df.groupby(['model', 'dataset', 'split_id'])['iteration'].idxmax()]
    #         time_summary = time_summary.groupby(['model', 'dataset'])['total_time_s'].mean().reset_index()
            
    #         pivot_rmse = summary.pivot_table(index='dataset', columns='model', values=['avg_rmse', 'std_rmse'])
            
    #         header = "dataset," + ",".join([f"{model}_rmse,{model}_time(s)" for model in models])
    #         f.write(header + "\n")

    #         for dataset in sorted(pivot_rmse.index):
    #             row_items = [dataset]
    #             for model in models:
    #                 try:
    #                     avg = pivot_rmse.loc[dataset, ('avg_rmse', model)]
    #                     std = pivot_rmse.loc[dataset, ('std_rmse', model)]
    #                     rmse_str = f"{avg:.4f} ± {std:.4f}" if pd.notna(avg) else "N/A"
    #                 except KeyError:
    #                     rmse_str = "N/A"
    #                 row_items.append(rmse_str)
                    
    #                 try:
    #                     time_val = time_summary[(time_summary['model'] == model) & (time_summary['dataset'] == dataset)]['total_time_s'].values
    #                     time_str = f"{time_val[0]:.2f}" if len(time_val) > 0 else "N/A"
    #                 except (KeyError, IndexError):
    #                     time_str = "N/A"
    #                 row_items.append(time_str)
                
    #             f.write(",".join(row_items) + "\n")
    #         f.write("\n")
            
    # print(f"Performance summary saved to {summary_text_path}")
    
    # --- 2. 可視化プロットの生成 (以降のロジックは変更なし) ---
    plot_dir = base_dir / "plots"
    plot_dir.mkdir(exist_ok=True)
    
    datasets = sorted(full_df['dataset'].unique())
    
    for dataset in datasets:
        plt.figure(figsize=(12, 8))
        dataset_df = full_df[full_df['dataset'] == dataset]
        
        sns.lineplot(data=dataset_df, x='iteration', y='rmse', hue='model', style='model',
                     markers=True, dashes=False, errorbar=('ci', 68), err_style='band')
        
        for model in models:
            model_df = dataset_df[dataset_df['model'] == model]
            if model_df.empty: continue
            
            max_iters_per_split = model_df.groupby('split_id')['iteration'].max()
            incomplete_splits = max_iters_per_split[max_iters_per_split < max(eval_intervals)]
            
            for split_id, last_iter in incomplete_splits.items():
                last_rmse_series = model_df[(model_df['split_id'] == split_id) & (model_df['iteration'] == last_iter)]['rmse']
                if not last_rmse_series.empty:
                    last_rmse = last_rmse_series.iloc[0]
                    plt.scatter([last_iter], [last_rmse], marker='x', color='red', s=150, zorder=10,
                                label=f'Timeout ({model})' if f'Timeout ({model})' not in plt.gca().get_legend_handles_labels()[1] else "")
        
        plt.title(f'RMSE Convergence on {dataset} Dataset', fontsize=16)
        plt.xlabel('Iterations', fontsize=12)
        plt.ylabel('Test RMSE', fontsize=12)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.legend(title='Model')
        plt.tight_layout()
        
        plot_path = plot_dir / f"rmse_convergence_{dataset}.png"
        plt.savefig(plot_path)
        plt.close()
        
    print(f"Convergence plots saved to {plot_dir}")

# ... (main関数は変更なし) ...
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate and visualize TPRT evaluation results.")
    parser.add_argument('base_dir', type=str, help='The base directory containing the result subdirectories.')
    args = parser.parse_args()
    
    # --- 実験設定に合わせて変更 ---
    EVAL_MODELS = ['TPRT-VEM', 'TPRT-LA'] 
    EVAL_INTERVALS = [i for i in range(10, 1001, 10)] 
    # ---------------------------

    aggregate_and_visualize(Path(args.base_dir), EVAL_INTERVALS, EVAL_MODELS)