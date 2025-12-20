# 未実装モデルリスト（deaRパッケージとの比較）

## 概要

deaRパッケージ（v1.5.2）と既存の実装を比較し、未実装のモデルをリストアップしました。

## 未実装モデル

### 1. Non-Radial and Goal-Oriented Models

#### model_nonradial
- **説明**: Non-radial DEA model
- **特徴**: 各入力/出力の非比例的な削減/増加を許容
- **実装状況**: 未実装

#### model_lgo (Linear Goal-Oriented)
- **説明**: Linear Goal-Oriented DEA model
- **特徴**: 線形目標指向DEAモデル
- **実装状況**: 未実装

#### model_qgo (Quadratic Goal-Oriented)
- **説明**: Quadratic Goal-Oriented DEA model
- **特徴**: 二次目標指向DEAモデル
- **実装状況**: 未実装

#### model_rdm (Range Directional Model)
- **説明**: Range Directional Model
- **特徴**: 範囲方向性モデル
- **実装状況**: 未実装

### 2. Additive Model Variants

#### model_addmin
- **説明**: Additive min model
- **特徴**: 加法的最小モデル
- **実装状況**: 未実装
- **注記**: 既に`AdditiveModel`は実装済みだが、これは異なるバリアント

#### model_addsupereff
- **説明**: Additive super-efficiency model
- **特徴**: Du, Liang and Zhu (2010)による加法的スーパー効率モデル
- **実装状況**: 未実装
- **注記**: SBMスーパー効率の加法的DEAへの拡張

### 3. Other Models

#### model_basic
- **説明**: Basic DEA model
- **特徴**: 基本的なDEAモデル
- **実装状況**: 未実装（CCR/BCCと重複する可能性あり）
- **注記**: 確認が必要

#### model_deaps
- **説明**: DEA-PS model
- **特徴**: DEA-PSモデル
- **実装状況**: 未実装

### 4. Fuzzy DEA Models

#### modelfuzzy_guotanaka
- **説明**: Guo-Tanaka fuzzy DEA model
- **特徴**: Guo-Tanaka手法によるファジィDEA
- **実装状況**: 未実装

#### modelfuzzy_kaoliu
- **説明**: Kao-Liu fuzzy DEA model
- **特徴**: Kao-Liu手法によるファジィDEA
- **実装状況**: 未実装

#### modelfuzzy_possibilistic
- **説明**: Possibilistic fuzzy DEA model
- **特徴**: 可能性測度によるファジィDEA
- **実装状況**: 未実装

### 5. Additional Functions

#### cross_efficiency
- **説明**: Cross-efficiency analysis
- **特徴**: 
  - Arbitrary, benevolent, aggressive formulations
  - Maverick index calculation
  - Doyle and Green (1994)の手法
- **実装状況**: 未実装

#### undesirable_basic
- **説明**: Undesirable inputs and outputs handling
- **特徴**: 
  - Seiford and Zhu (2002)の手法
  - 望ましくない入出力の変換
- **実装状況**: 未実装

## 実装済みモデル（参考）

以下のモデルは既に実装済みです：

- ✓ model_additive → `AdditiveModel`
- ✓ model_fdh → `FDHModel`, `FDHPlusModel`
- ✓ model_multiplier → `CCRModel.solve_multiplier`, `BCCModel.solve_multiplier`
- ✓ model_supereff → `APModel`, `MAJModel`
- ✓ model_sbmeff → `SBMModel`
- ✓ model_sbmsupereff → `SBMModel`
- ✓ model_dir → `DirectionalEfficiencyModel`
- ✓ model_profit → `ProfitEfficiencyModel`
- ✓ bootstrap_basic → `BootstrapDEAModel`

## 統計

- **未実装モデル数**: 11モデル + 2機能 = **合計13項目**
- **実装済みモデル数**: 9モデル（マッピング済み）

## 推奨実装順序

1. **Non-radial models** (model_nonradial, model_rdm)
2. **Goal-oriented models** (model_lgo, model_qgo)
3. **Additive variants** (model_addmin, model_addsupereff)
4. **Cross-efficiency** (cross_efficiency)
5. **Undesirable I/O** (undesirable_basic)
6. **Fuzzy DEA** (modelfuzzy_*)

