# LianLianKan Visual Solver (連連看通用視覺解題器)

基於「分離式架構（Studio-Engine Pattern）」設計的通用型連連看自動解題套件：
- **Studio (Rust)**：離線建模工具，負責滑鼠框選、自適應網格訊號分析、特徵分類、無監督分群與樣本庫建立，並可追加新關卡的新物件（單一執行檔約 2 MB，不依賴 Python／OpenCV）。
- **Engine (Rust)**：在線執行引擎，基於 DirectX DXGI Desktop Duplication 硬體截圖（<3ms 延遲）、三段線尋路核心、Win32 `SendInput` 點擊注入與狀態驗證閉環（靜態 CRT 編譯，免安裝運行庫）。

---

## 系統架構

```
                       [ 遊戲畫面 ]
                            │
            ┌───────────────┴───────────────┐
            ▼                               ▼
    【Studio 建模工具】              【Engine 求解引擎】
        (Rust)                      (Rust / 靜態 CRT)
┌───────────────────────────┐   ┌───────────────────────────┐
│ 1. 使用者選定/框選 ROI    │   │ 1. DXGI 截圖 (<3ms)       │
│ 2. 自適應網格週期偵測     │   │ 2. NCC 模板快速矩陣化     │
│ 3. 特徵抽取與奇偶性分群   │   │ 3. 0/1/2 轉彎三段線尋路   │
│ 4. 輸出 Profile 與範本庫  │   │ 4. SendInput 點擊與驗證   │
└─────────────┬─────────────┘   └─────────────▲─────────────┘
              │                               │
              └──────► game_profile.json ─────┘
                       templates/*.png
```

---

## GitHub Actions 自動編譯與發布

本專案配置了完整 CI/CD 工作流 (`.github/workflows/build.yml`)：
- **Rust Engine**：在 Windows runner 上透過 `cargo build --release` 編譯為獨立 `engine.exe`。
- **Rust Studio**：透過 `cargo build --release` 編譯為單一獨立 `studio.exe`。
- **Release Bundle**：自動整合打包為 `LianLianKan-Solver-windows-x64.zip`。

### 觸發機制
- 推送至 `main` 分支時自動執行編譯測試。
- 支援 GitHub 頁面「Run workflow」手動觸發。
- 推送版號標籤（如 `git tag v1.0.0 && git push --tags`）時自動在 GitHub Releases 發布下載包。

---

## 本地開發指引

### 1. Studio (Rust)
```bash
cd studio
cargo build --release
./target/release/studio.exe     # 不帶參數會出現選單；詳見 studio/README.md
```

### 2. Engine (Rust)
```bash
cd engine
cargo test
cargo build --release
```

---

## 快捷鍵說明 (Engine 運行時)
- **`[F8]`**：暫停 / 恢復解題迴圈。
- **`[ESC]`**：緊急終止程式。
