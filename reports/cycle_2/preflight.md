# Preflight Check — Cycle 2 (Phase 2: Electricity Data Pipeline)

## 1. データ境界表

| 項目 | 値 |
|---|---|
| データ取得終了日 | 2026-03-27 (今日 2026-03-30 以前) |
| Train期間 | 2024-03-28 〜 2025-07-28 |
| Validation期間 | 2025-07-28 〜 2025-11-28 |
| Test期間 | 2025-11-28 〜 2026-03-27 |
| 重複なし確認 | Yes |
| 未来日付なし確認 | Yes |

**注意**: UCI Electricityデータセットのダウンロードに失敗したため（zipファイル破損）、ARF Data APIの
hourly OHLCVデータ（18銘柄）を代替として使用。各銘柄を「顧客」として扱い、close価格を「電力消費量」の
アナロジーとして使用する。この制約はdocs/open_questions.mdに記録。

## 2. Feature timestamp 契約

- 全ての特徴量は時刻 t の予測に t-1 以前のデータのみを使用しているか？ → **Yes**
  - lookbackウィンドウは過去のデータのみ使用
  - known future inputs (hour_of_day, day_of_week, month) はカレンダー情報のみ（将来既知）
- Scaler / Imputer は train データのみで fit しているか？ → **Yes**
  - StandardScalerはtrain splitでfitし、val/testにtransform
- Centered rolling window を使用していないか？ → **Yes** (使用していない)

## 3. Paper spec 差分表

| パラメータ | 論文の値 | 現在の実装 | 一致? |
|---|---|---|---|
| データセット | Electricity (370顧客, 15min間隔) | ARF OHLCV (18銘柄, 1h間隔) | No (代替データ) |
| Lookback期間 | 168 (7日 × 24h) | 168 | Yes |
| 予測Horizon | 24 (24時間先) | 24 | Yes |
| 静的特徴 | 顧客ID | 銘柄ID (entity_id) | Yes (アナロジー) |
| 時間特徴 | hour, day_of_week, month, holiday | hour, day_of_week, month, day_of_month | 概ね一致 |
| 正規化 | エンティティ別正規化 | エンティティ別StandardScaler | Yes |
| 量子化損失 | P10, P50, P90 | P10, P50, P90 | Yes |
