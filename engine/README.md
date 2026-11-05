# LianLianKan (連連看) Ultra-Fast Visual Solver Engine

高效能、低延遲的連連看視覺自動解題核心引擎（純 Rust 實作）。
採用 **DXGI Desktop Duplication API** 達成毫秒級畫面擷取，結合 **NCC (正規化互相關) 模板比對**、**3-Segment (最多兩次轉彎) 掃描線尋路**、與 **Win32 SendInput 滑鼠點擊注入**。

---

## 模組架構說明 (Engine Architecture)

原始碼位於 `engine/src/`：

1. **`board.rs` (棋盤資料結構)**
   - 網格狀態定義：`0 = 空格 (EMPTY)`、`-1 = 障礙物 (OBSTACLE)`、`1..N = 可配對方塊 ID`。
   - 提供外圍 1 圈虛擬邊界的 `PaddedBoard`，使穿過外圍邊界的外繞路徑可與內部路徑統一處理。
2. **`solver.rs` (3-Segment 尋路與啟發式核心)**
   - **0-turn（直線）**：水平或垂直無阻擋直連。
   - **1-turn（L 型）**：由單一空格拐角連接。
   - **2-turn（Z 型 / U 型 / 外環繞）**：雙向掃描線尋路，自動支援外圍邊界繞行。
   - **Heuristic 優先權評分**：優先消除長距離與邊緣方塊，避免死局並儘早開闢外圍路徑。
3. **`matcher.rs` (NCC 模板比對)**
   - 快速正規化互相關 (Normalized Cross-Correlation) 與方差分析。
   - 自動檢測背景空格（超低方差）與方塊 ID。
4. **`capture.rs` (超低延遲畫面擷取)**
   - 封裝 Windows DXGI Desktop Duplication API（GPU VRAM 直接複製，延遲 < 3ms）。
   - 具備自動降級 GDI BitBlt 備援機制，確保在任何環境皆可穩定運作。
5. **`controller.rs` (Win32 滑鼠注入與驗證)**
   - 透過 `SendInput` 精準點擊方塊中心。
   - 具備消除後驗證迴圈（Click A -> Click B -> 等待消除動畫 -> 驗證兩格是否變為 0）。
6. **`profile.rs` (設定檔模型)**
   - 載入與解析 `game_profile.json`（ROI 座標、網格行列數、信心門檻值、動畫等待延遲等）。
7. **`main.rs` (引擎排程主迴圈)**
   - 整合擷取、矩陣化、尋路、點擊與統計資訊。
   - 支援全域熱鍵：`[F8]` 暫停/繼續、`[ESC]` 安全退出。

---

## 編譯與測試指引 (Build & Test Guide)

### 1. 前置需求 (在具有 Rust 環境的電腦)
- 作業系統：Windows 10 / 11 (x86_64)
- Rust Toolchain：`stable-x86_64-pc-windows-msvc` (Rust 1.75+)
- Visual Studio C++ Build Tools (MSVC 連結器)

### 2. 靜態 CRT 連結 (Static CRT Linking)
本專案已在 `.cargo/config.toml` 配置：
```toml
[target.x86_64-pc-windows-msvc]
rustflags = ["-C", "target-feature=+crt-static"]
```
透過此設定，編譯出之 `engine.exe` 會將 C Runtime (MSVCRT) 靜態打包，形成**單一獨立執行檔**，可直接複製到未安裝 VC++ 執行庫的目標電腦上獨立執行。

### 3. 執行單元與整合測試
```bash
cargo test
```
包含直線、L 型、Z 型、U 型外圍繞行、障礙物阻擋、死局檢測、啟發式權重及 600 方塊大型棋盤效能測試。

### 4. 編譯發布版本 (Release Build)
```bash
cargo build --release
```
編譯完成後，執行檔將位於：
```
target/release/engine.exe
```

---

## 執行與使用方式 (Usage)

### 基本執行
```bash
# 自動讀取同目錄下的 game_profile.json
engine.exe

# 指定設定檔與模板目錄
engine.exe --profile "D:\Git\20261105\game_profile.json" --templates "D:\Git\20261105\templates"

# 模擬模式（Dry-run：僅比對尋路，不送出實際滑鼠點擊）
engine.exe --dry-run

# 限制步數與調整每次消除間隔
engine.exe --max-moves 50 --delay 80
```

### 控制熱鍵
- **`[F8]`**：暫停 / 恢復自動解題（當需要手動操作或重置遊戲時使用）。
- **`[ESC]`**：立即終止並退出引擎。

---

## `game_profile.json` 格式範例
```json
{
  "name": "LianLianKan Standard",
  "board": {
    "rows": 10,
    "cols": 14,
    "roi": {
      "x": 200,
      "y": 150,
      "width": 700,
      "height": 500
    }
  },
  "matcher": {
    "threshold": 0.82,
    "empty_variance_threshold": 15.0,
    "template_dir": "templates"
  },
  "timing": {
    "click_interval_ms": 30,
    "animation_delay_ms": 100,
    "verify_timeout_ms": 500
  }
}
```
