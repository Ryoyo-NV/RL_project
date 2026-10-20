# SO-101 RL Sim6

SO-101ロボットの到達動作を対象とした、Isaac Labの強化学習プロジェクトです。学習環境・設定、ロボットモデル、学習結果、ROS 2による実機向け推論コードをまとめています。

## 構成

| 配置先 | 内容 |
| --- | --- |
| `source/so101_rl_sim6/` | タスク登録、観測・制御・報酬・成功条件の実装、環境設定、PPO設定 |
| `scripts/rsl_rl/` | 学習、シミュレーション再生、TorchScript・ONNX形式へのモデル出力 |
| `scripts/ros2/real_policy_runner.py` | 実機の関節状態とTFから観測を作り、推論結果を関節目標として配信するROS 2ノード |
| `scripts/list_envs.py`、`zero_agent.py`、`random_agent.py` | 環境の登録・動作確認用スクリプト |
| `so101_3.usd` | SO-101の形状・関節・物理設定を含むロボットモデル |
| `logs/` | 保存済み1回分の学習ログ、チェックポイント、学習時の設定・Git情報、出力済み推論モデル |
| `.vscode/`、各種設定ファイル | エディタ、コード整形、Gitなどの開発用設定 |
| `LICENSE`、`LICENSES/` | 本プロジェクトと、取り込んだモデル・コードのライセンス本文 |

## ベースと主な変更

Isaac Labの外部プロジェクト用テンプレート（Direct方式・単一エージェント・RSL-RL PPO）を基にしています。比較元は[v3.0.0-beta2.patch1 / ffff603](https://github.com/isaac-sim/IsaacLab/tree/ffff603eafc6b74264a5261cc0183d6a65390d78)です。

標準環境をSO-101の到達タスクへ変更し、観測・行動・報酬・成功条件とPPO設定を調整しています。RSL-RL 5.0.1への対応、ロボットUSD、実機向け推論コードを追加しています。

`so101_3.usd`は、[MuammerBay/isaac_so_arm101のSO-101用URDF](https://github.com/MuammerBay/isaac_so_arm101/blob/main/src/isaac_so_arm101/robots/trs_so101/urdf/so_arm101.urdf)を改変し、Isaac SimのURDFインポート機能でUSDへ変換したモデルです。

## ライセンス

本リポジトリの独自追加・修正部分と学習成果物は、個別に別の指定がある場合を除き、[BSD-3-Clause](LICENSE)で提供します。取り込んだコード・モデルの既存の著作権・帰属表示を保持しています。

- Isaac Lab由来のコードには、元のBSD-3-Clauseが適用されます。著作権表示と条件・免責条項は[LICENSE](LICENSE)に保持しています。
- `so101_3.usd`には、元モデルの[Apache License 2.0](LICENSES/SO101-Apache-2.0.txt)の条件が適用されます。出典とUSDへの変換・改変については上記に記載しています。
- 出力済みの`policy.pt`に含まれる第三者コードには、それぞれのライセンスが適用されます。[RSL-RL 5.0.1](LICENSES/RSL-RL-BSD-3-Clause.txt)、[PyTorch 2.10.0](LICENSES/PyTorch-LICENSE.txt)の本文と著作権表示を同梱しています。
