import pandas as pd
from pathlib import Path
import argparse

def aggregate(base_dir: Path):
    """
    指定されたディレクトリ内の全 result_*.csv を集約し、
    モデルとデータセットごとに平均と標準偏差を計算する。
    """
    all_dfs = []
    # サブディレクトリを含めて全てのCSVファイルを再帰的に検索
    csv_files = list(base_dir.rglob("result_*.csv"))
    
    if not csv_files:
        print(f"No result CSV files found in {base_dir}")
        return

    print(f"Found {len(csv_files)} result files. Aggregating...")

    for f in csv_files:
        try:
            all_dfs.append(pd.read_csv(f))
        except pd.errors.EmptyDataError:
            print(f"Warning: Skipping empty file {f}")
            continue
            
    if not all_dfs:
        print("No valid data to aggregate.")
        return

    # 全てのDataFrameを結合
    full_df = pd.concat(all_dfs, ignore_index=True)

    # モデルとデータセットでグループ化し、統計量を計算
    summary = full_df.groupby(['model', 'dataset']).agg(
        avg_rmse=('rmse', 'mean'),
        std_rmse=('rmse', 'std'),
        avg_time_s=('time_s', 'mean')
    ).reset_index()

    # 整形して最終的なテーブルを作成
    final_table = summary.pivot(index='dataset', columns='model')
    # 列名を整形 (例: ('avg_rmse', 'TPRT-LA') -> 'TPRT-LA_RMSE')
    final_table.columns = [f"{col[1]}_{col[0].replace('avg_', '').replace('_s', '(s)')}" for col in final_table.columns]
    
    # RMSEの列に標準偏差を追加
    for model in summary['model'].unique():
        rmse_col = f'{model}_rmse'
        std_col = f'{model}_std_rmse'
        if rmse_col in final_table.columns:
            # 対応するstd_rmseを取得
            std_series = summary.set_index(['model', 'dataset']).loc[model]['std_rmse']
            # 文字列として結合
            final_table[rmse_col] = final_table[rmse_col].round(4).astype(str) + " ± " + std_series.round(4).astype(str)
    
    # 不要になったstd_rmse列を削除
    std_cols_to_drop = [f'{model}_std_rmse' for model in summary['model'].unique()]
    final_table = final_table.drop(columns=std_cols_to_drop, errors='ignore').reset_index()


    # 結果を保存
    summary_file_path = base_dir / "final_summary.csv"
    final_table.to_csv(summary_file_path, index=False)
    
    print("\n\n--- 📊 Final Performance Summary 📊 ---")
    print(final_table.to_string(index=False))
    print(f"\nSummary saved to: {summary_file_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate TPRT evaluation results.")
    parser.add_argument('base_dir', type=str, help='The base directory containing the result subdirectories.')
    args = parser.parse_args()
    
    aggregate(Path(args.base_dir))