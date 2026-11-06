# Studio（建模工具，Rust）

離線建模工具：自適應網格訊號分析、特徵分類（磚塊／空格／障礙物）、無監督分群與樣本庫建立，
輸出 `game_profile.json`、`templates/` 與可手動編輯的 `preview.html`，供 `engine` 使用。
單一執行檔，不需要 Python、OpenCV 或 numpy。

## 編譯

```bash
cd studio
cargo build --release        # 產出 target/release/studio.exe
```

## 使用方式

不帶參數執行 `studio.exe` 會出現選單；把截圖檔直接拖到 exe 上則會開啟該圖進行框選。

```
studio --snip [--monitor N]               螢幕即時擷取 + 滑鼠框選（多螢幕、高 DPI）
studio --image board.png --select         對截圖檔滑鼠框選
studio --add --snip                       追加新物件（新關卡）：從螢幕框選
studio --add --image level2.png --select  追加新物件：從截圖檔框選
studio --reroi [--monitor N]              調整既有 profile 的框選區域（移動／縮放）
studio --image board.png                  不開視窗：整張圖當作框選範圍（腳本／測試用）
studio --image board.png --rough-roi X Y W H   指定大略範圍並對齊磚塊格線
studio --image board.png --roi X Y W H         直接指定棋盤範圍，不對齊
        [--templates templates] [--profile game_profile.json] [--preview preview.html]
        [--screenshot screenshot_board.png] [--dump out.json] [--open | --no-browser]
```

### 框選視窗
左鍵拖曳框選；框好後可拖曳框內移動、拖曳邊或角的方塊縮放（方向鍵微調，Shift = 10 px，Ctrl+方向鍵調右／下邊），框外拖曳則重畫。**Enter／Space** 確認、**Esc** 取消。框選範圍超出棋盤外緣請勿大於 1 個磚塊，
系統會自動對齊磚塊格線。`--snip` 的 `game_profile.json` 內是虛擬桌面的實體像素座標，engine 可直接使用。

### 重新對準既有 profile（`--reroi`，選單 [4]）
遊戲視窗移動或**縮放**後使用。會保留 profile 的**列數與欄數**，其餘自動處理：
1. 擷取目前畫面，舊 ROI 預先畫成紅黃框；把它大略拖到新的棋盤上（可移動、縮放，可比棋盤大不超過 1 個磚塊）。
2. 依原本的列／欄數，在附近用畫面上的格線擬合出外框與新的格子大小（會顯示可信度），
   再以「磚塊與既有模板的比對結果」微調位置與大小（外框偏幾個像素就會切到磚塊，比對分數會明顯下降）。
3. 把 `templates/` 的所有參照圖依「新格子大小 ÷ 舊格子大小」縮放，再把每個已知物件的參照圖換成
   目前畫面上的實際樣本（同縮放、同抗鋸齒，engine 比對才不會因遊戲重繪而失準）；
   同時顯示「磚塊中有幾格比對到既有物件」，低於 80% 會警告。
4. 只改寫 `board.roi`／`cell_size`／`offset`，並同步 `preview.html` 內的 ROI；規則、合併與障礙物設定不動。
貼合後的盤面存為 `screenshot_board.png`，可開啟確認外框是否正確。

### 追加截圖（`--add`）
後面的關卡常出現新物件。`--add` 會把新截圖的棋盤與現有 `game_profile.json`／`templates/` 全部模板比對，
已知物件沿用原編號，只把認不出的物件新增為新類別（編號 = 目前最大編號 + 1…），不會改動舊編號。
完成後 `preview.html` 會以追加模式顯示這張截圖與所有編號，新物件若被誤分成多類可在網頁中合併。

### 編輯頁面（`preview.html`）
單檔 HTML，可點選格子改成空格／障礙物／指定類別、將類別設為障礙物或合併到另一類、
切換「允許連線繞行棋盤外圍」，並下載更新後的 `game_profile.json`。

## 原始碼結構

| 檔案 | 內容 |
|---|---|
| `src/calibrate.rs` | 週期估計、格線相位、框選範圍對齊、切格 |
| `src/features.rs` | 磚塊／空格／障礙物分類（與顏色無關） |
| `src/cluster.rs` | 磚塊相似度、奇偶限制的階層式分群 |
| `src/augment.rs` | 追加截圖：比對現有模板、只新增新物件 |
| `src/export.rs` | 輸出模板、profile、可編輯 HTML（`src/editor.html`） |
| `src/capture.rs` / `src/select.rs` | 螢幕擷取（GDI）／滑鼠框選視窗（`minifb`） |
| `src/imgproc.rs` | 影像處理基礎（灰階、Sobel、縮放、HSV、Lab） |
