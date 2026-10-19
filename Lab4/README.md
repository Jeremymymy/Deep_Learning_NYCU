### Lab 4: Conditional VAE for Video Prediction

> 本文件是這個專題的**快速理解參考**：從任務目標、資料形狀、模型架構、訓練機制到核心技術一次說清楚。
> 想看數學推導與直覺，往下到「核心技術詳解」。

---

## 📌 任務目標

用 **Conditional VAE (CVAE)** 做影片序列預測：
- **輸入**：一張起始影格（第 0 幀真實 RGB 圖）+ 一整段人體骨架序列（pose skeleton，每幀一張）
- **輸出**：接下來完整的舞蹈影片（第 1～N 幀）

骨架是「劇本」（規定每幀手腳擺哪），外觀細節（人是誰、衣服、光影）靠模型自己維持。

**核心挑戰**：訓練影片固定 **16 幀**，但測試需連續預測 **630 幀**——巨大的泛化落差，是整個 Lab 的設計主軸。

**為什麼選 CVAE？**
| 做法 | 為什麼不選 / 選 |
|---|---|
| AutoEncoder | latent 是散亂的點、空間有空洞，無法隨機抽樣生成 |
| 純 VAE | 能生成但**無法用骨架控制姿勢** |
| **CVAE** ✅ | 能隨機抽樣生成 **+** 用條件（骨架）控制 → 跟著骨架跳舞 |
| GAN | 訓練不穩、無 encoder、不貼合 PSNR 評分（細節見下文） |

---

## 📊 資料集與 DataLoader（資料形狀）

`Dataset_Dance`（`dataloader.py`）載入人體跳舞資料集。**骨架 label 本身也是一張 3 通道 PNG 圖**（不是座標數字），所以 RGB 影格與骨架都走 CNN 編碼。

**目錄結構**
```
root/
├─ train/train_img/   frame0.png, frame1.png, ...   ← 真實影格
├─ train/train_label/ frame0.png, frame1.png, ...   ← 對應骨架圖（同檔名）
├─ val/val_img/
└─ val/val_label/
```
骨架路徑由影格路徑推導：把倒數第二層資料夾名 `*_img` 換成 `*_label`（`dataloader.py:47-51`）。

**切片邏輯**（`__getitem__`，`dataloader.py:41-55`）
- 影格依檔名數字排序（`get_key` 抽出 `frameXXX` 的整數）。
- 第 `index` 個 clip = 連續且不重疊的 `video_len` 幀：`frame[index*video_len : index*video_len + video_len]`。
- clip 數量 `__len__ = 總幀數 // video_len`。
- `video_len`：訓練用 `train_vi_len=16`，驗證用 `val_vi_len=630`。

**形狀流轉**（transform = `Resize((32,64))` + `ToTensor()`，像素值 → [0,1]）
```
單幀                     (3, 32, 64)              C, H, W
__getitem__ 回傳一個 clip (video_len, 3, 32, 64)   imgs 與 labels 各一份
DataLoader 疊成 batch     (B, video_len, 3, 32, 64)
訓練/驗證內 permute        (video_len, B, 3, 32, 64) 把時間軸換到最前 → img[i] = 整批的第 i 幀
```
> 訓練 `batch_size=2`、`shuffle=False`、`drop_last=True`；驗證 `batch_size=1`（一段 630 幀的長 clip）。

---

## 🧩 模型架構（五大模組）

預設超參：`F_dim=128`（影像特徵）、`L_dim=32`（骨架特徵）、`N_dim=12`（latent z）、`D_out_dim=192`。空間尺寸全程維持 **32×64**（所有 conv 為 stride=1 + padding，不做下採樣）。

```
 img[i]（影格） ─→[RGB_Encoder 3→128]──────────────┐
                                                  ├─→[Gaussian_Predictor 160→12]─→ z, μ, logvar
 label[i]（骨架）─→[Label_Encoder 3→32]────────────┘            （posterior，偷看第 i 幀）
                          │                                                  │ z (12)
 前一幀 ─→[RGB_Encoder]──→ feat_prev (128) ──┐                                │
 label[i]─→[Label_Encoder]→ label_feat (32) ─┼───────────────────────────────┴─→[Decoder_Fusion 172→192]
                                             │                                          │
                                             └──────────────────────────────────[Generator 192→3]─→ 預測影格 x̂
```

| 模組 | 輸入(通道) | 輸出(通道) | 職責 | 程式碼 |
|---|---|---|---|---|
| `RGB_Encoder` | 3 | 128 | 影格 → 視覺特徵 | `modules.py:33` |
| `Label_Encoder` | 3 | 32 | 骨架 → 條件特徵（7×7 大核抓全域姿態） | `modules.py:52` |
| `Gaussian_Predictor` | 160 = 128+32 | 12 | 出 μ、logvar 並 reparameterize 出 z（posterior） | `modules.py:66` |
| `Decoder_Fusion` | 172 = 128+32+12 | 192 | 融合 [前一幀, 骨架, z] | `modules.py:94` |
| `Generator` | 192 | 3 | 融合特徵 → RGB 影格 | `modules.py:15` |

**兩個值得注意的設計**
- **z 是空間張量** `(12, 32, 64)`，不是教科書 VAE 的扁平向量。理由：影片預測是密集像素任務，空間 latent 與輸出天然對齊。
- **效率型卷積塊**：`ResidualBlock`（標準殘差）、`DepthConv`（深度可分離卷積，省算力）、`ConvFFN`（門控 FFN，`x1 * ReLU(x2)`，類似 Transformer FFN）。詳見下方「🏎️ 訓練效能優化技巧」。

---

## ⚙️ 訓練機制

**逐幀 autoregressive 迴圈**（`Trainer.py:training_one_step`）。訓練 clip 16 幀，第 0 幀給定，迴圈生成第 1~15 幀、逐幀累積 loss：

```
out₀ = img[0]                                   （給定起始幀）
for i = 1..15:
    label_feat = Label_Encoder(label[i])
    前一幀特徵 = RGB_Encoder( img[i-1] )  若 TF 開      ← 真實前一幀
              = RGB_Encoder( out )      若 TF 關      ← 自己上一輪的輸出（自迴歸）
    z,μ,logvar = Gaussian_Predictor( RGB_Encoder(img[i]), label_feat )   ← posterior 偷看答案 img[i]
    out = Generator( Decoder_Fusion(前一幀特徵, label_feat, z) )
    loss += MSE(out, img[i]) + β · KL(μ, logvar)
backward()  ← 整段序列的 loss 累加後「一次」反向傳播
```

**關鍵概念**
- **BPTT（時間反向傳播）**：16 幀展開成一張大計算圖、一次 backward。TF 關時 `out` 鏈相連 → 梯度穿越每幀 → 學長期穩定（但易梯度爆炸，故用 **gradient clipping** 範數=1.0）。TF 開時鏈被打斷 → 好訓練但學不到長期。
- **資訊不對稱（VAE 精髓）**：**posterior（Gaussian_Predictor）偷看第 i 幀答案**學出有意義的 z；**decoder（Fusion+Generator）看不到答案**，只靠 [前一幀+骨架+z] 重建。
- **Loss**：`MSE 重建 + β·KL`，由 ELBO 推導而來（見下文）。

**訓練設施**：`Adam(lr=0.001)`、`MultiStepLR(milestones=[2,5], gamma=0.1)`、`num_epoch=70`、每 3 epoch 存檔。

---

## 🏎️ 訓練效能優化技巧

模型刻意用「省算力」的卷積塊堆疊，並搭配幾個訓練加速手段，讓這個逐幀 BPTT、序列又長的任務在有限算力下跑得動。

### A. DepthConv：深度可分離卷積（省參數的主力）

`DepthConv`（`layers.py:58-86`）把一個普通 3×3 卷積拆成三步，並加殘差：
```
x ─→ 1×1 conv (conv1) ─→ 3×3 depthwise conv (groups=C) ─→ 1×1 conv (conv2) ─→ + x
     逐點(只混通道)        逐通道(只混空間,各通道獨立)        逐點(只混通道)     殘差
```
- **depthwise**（`groups=dw_ch`）：每個通道用自己獨立的 3×3 filter、通道間不混 → 只負責「空間混合」。
- **兩個 1×1**：只負責「跨通道混合」、不碰空間。
- 把「空間混合」與「通道混合」拆開正是深度可分離的核心，大幅減參數。

**參數量對比**（輸入=輸出 C 通道、3×3）：

| | 參數量 | C=128 時 | 比例 |
|---|---|---|---|
| 普通 3×3 conv | 9·C² | 147,456 | 1× |
| DepthConv (1×1→3×3dw→1×1) | 2·C² + 9·C | 33,920 | **≈0.23（少約 4.3×）** |

FLOPs 比例類似 → 省參數也省計算。概念同 MobileNet 系列。

### B. ConvFFN：門控前饋（便宜又有表達力的通道混合）

`ConvFFN`（`layers.py:42-56`）全部用 1×1 卷積（只混通道、不碰空間），加門控與殘差：
```python
x1, x2 = self.conv(x).chunk(2, 1)     # 1×1 擴張到 4C，切成兩半(各 2C)
out = x1 * self.relu(x2)              # 門控：x2(經激活)當閥門，動態調控 x1
return identity + self.conv_out(out)  # 1×1 投影 2C→C + 殘差
```
- `x1 * ReLU(x2)` 是 **GLU 家族的門控**：x2 像閥門，決定 x1 每個位置放多少訊號通過 → 比「conv→ReLU」表達力強。
- 全 1×1 → 每像素成本低（參數約 6·C²，但完全無空間卷積）。

### C. DepthConvBlock = DepthConv + ConvFFN（MetaFormer 風格）

`DepthConvBlock`（`layers.py:31-40`）把兩者串起來：
```
DepthConvBlock = DepthConv(空間混合)  →  ConvFFN(通道混合)
```
這正是 **ConvNeXt / MetaFormer 的設計哲學**——「空間混合 + 通道混合」分工，用便宜的 depthwise + 1×1 取代昂貴的全卷積，兼顧效率與表達力。五大模組內部大量用它與 `ResidualBlock` 交替堆疊。

### D. fast_train：用部分資料加速早期收斂

`Trainer.py:train_dataloader` 內建一個訓練加速開關：
```
--fast_train          啟用
--fast_partial 0.4    前期只用 40% 訓練資料
--fast_train_epoch 5  前 5 個 epoch 套用，之後自動切回全量
```
前 5 個 epoch 只吃 40% 資料 → 每 epoch 約快 2.5× → 快速暖機到大致收斂，之後再用全量資料精修。「先粗後細」的省時策略。

### E. 其他內建的效能／穩定手段

- **Gradient clipping**（範數=1.0）：對抗 BPTT 長鏈的梯度爆炸，讓較大學習率也能穩定（見訓練機制）。
- **殘差連接**（ResidualBlock / DepthConv / ConvFFN 都有 skip）：緩解梯度消失、加速深層網路收斂。
- **低解析度 32×64**：直接砍掉大量像素級計算，是這作業能在有限算力跑完的前提。
- **`LeakyReLU(inplace=True)`**：就地運算省記憶體（`layers.py:10`）。
- **MultiStepLR 早期激進降溫**（第 2、5 epoch 各 ÷10）：前期大步快速收斂、後期小步精修。

---

## 🔬 核心技術詳解

### 1. ELBO：loss 的來源（不是硬湊的）
我們想最大化資料機率 `log p(x)`，但它含 intractable 積分。引入 encoder `q(z|x,c)` 後可證恆等式：
```
log p(x) = ELBO + KL(q ‖ 真後驗)        其中 KL ≥ 0
ELBO = E_q[log p(x|z,c)] − KL(q(z|x,c) ‖ N(0,1))
       └── 重建項(→MSE) ──┘   └──── 規整項(封閉解) ────┘
```
`log p(x)` 對固定 x 是定值（固定大餅）→ 推高 ELBO 必然壓縮 gap → 同時逼 encoder 逼近真後驗。最大化 ELBO ⟺ 最小化 `MSE + KL`。

### 2. Reparameterization Trick（讓抽樣可微）
`z ~ N(μ,σ²)` 的抽樣動作沒有導數、會斷梯度。改寫成把隨機性隔離到外部：
```
z = μ + σ ⊙ ε ,  ε ~ N(0,1)     （modules.py:78-83： std=exp(0.5*logvar); z=mu+eps*std）
```
ε 與參數無關 → z 是 μ、σ 的可微函數（∂z/∂μ=1, ∂z/∂σ=ε）→ 梯度順利回流 encoder。

### 3. KL 封閉解與「兩道牆」
```
KL(N(μ,σ²) ‖ N(0,1)) = −½ Σ(1 + logσ² − μ² − σ²)     （Trainer.py:kl_criterion）
```
- `−μ²` → 逼 μ 往 0；`logσ²` 那項 → σ→0 時 −logσ² 爆炸，**擋住 σ 塌縮（防退化成 AE）**；`−σ²` 那項 → 擋住 σ 膨脹。谷底在 σ=1。

### 4. z 到底是什麼？「細節的隨機種子」
給定「前一幀+骨架」，下一幀仍有殘餘不確定性（皺褶、被遮處、光影）。**z 只負責這些零頭**（架構上 z 僅 12 通道 vs 主力 160 通道）。
- 訓練：posterior 偷看答案 → 反推「重現這幀細節的種子」；decoder 學「種子→細節」；KL 把種子整齊填滿 N(0,1) 庫。
- 測試：沒答案 → **直接從 N(0,1) 抽一顆種子**，decoder 輸出某一版合理細節。大結構由前一幀+骨架鎖定。
- **「有意義」（decoder 真的用 z）與「服從 N(0,1)」（種子排列整齊）不衝突**——這正是 VAE 能隨機生成的橋樑。

### 5. KL Annealing（β 由輕變重，防 posterior collapse）
若一開始 `β=1`，模型還不會畫圖時，會發現「讓 q≈N(0,1) 使 KL→0」最省事 → 無視 z（**posterior collapse**）→ 只畫模糊平均。因 z 只是 12 通道的弱補充，特別容易被無視。
- **β 先小後大**：初期 β≈0 先逼 decoder 學會用 z；後期 β→1 才規整。**順序很重要**。
- **Monotonic**：β 單調 0→1（穩）；**Cyclical**：β 反覆 0→1（給 latent 喘息、較易跳出局部最優）。

### 6. Teacher Forcing（輔助輪 → 獨立行走）
與 z 無關的獨立機制，管「前一幀 context」用真圖還是自輸出。
- 初期用真圖（輔助輪）對抗 exposure bias，穩定收斂。
- 但測試無真圖可用，過度依賴會在 630 幀崩潰 → **隨 epoch 衰減 TF 比例**，逼模型練習吃自己的輸出（短鏈→長鏈）。

---

## 📈 評估方法（PSNR）

```
PSNR = 20·log₁₀(MAX) − 10·log₁₀(MSE)   正規化後 MAX=1 → PSNR = −10·log₁₀(MSE)
```
單位 dB、越高越像；直接量「跟 ground truth 的逐像素保真度」，與 VAE 重建式做法一致（這也是不選 GAN 的原因之一——GAN 重視真實感而非貼合答案，PSNR 反而吃虧）。checkpoint 檔名即 PSNR（如 `28.667.ckpt`）。

**測試生成與訓練的三差異**：z 從 **prior N(0,1)** 抽（不用 posterior）、**永遠自迴歸**（無 TF）、**不算 KL**。

---

## 💡 實驗發現

- **長影片預測**：訓練僅 16 幀、測試 630 幀；過度依賴 TF 會在長影片崩潰。調低 TF 比例強迫模型減少依賴 ground truth。
- **收斂性**：無 annealing → loss 混亂難收斂；Monotonic → 收斂最平滑、穩定利用潛在空間；Cyclical → 反覆探索、通常得到更好的特徵分佈。
- **KL 的意義**：本質是 z 分佈與標準高斯的差異，確保生成具一定多樣性（同骨架、不同 z → 不同但合理的細節）。

---

## 🚀 現代視角（延伸）

此 MSE+VAE 路線的天花板：**偏糊**（MSE 取平均）、**低解析度**（32×64）、**自迴歸 drift**。現代替代：
- **Diffusion**：兼具 VAE 的穩定與 GAN 的銳利（DDPM → Latent Diffusion → Stable Video Diffusion / DiT）。本 repo **Lab6 即 conditional DDPM**。
- **與本任務幾乎同題**：ControlNet（骨架條件）、**AnimateAnyone / DisCo / MagicAnimate**（參考圖+骨架→高清人物動畫影片），用擴散引擎做 Lab4 的事。
- 其他：VQ-VAE/VQGAN（離散 latent）、autoregressive transformer（MAGVIT）、flow matching（SD3/FLUX）。
- 指標進化：除 PSNR/SSIM，生成任務更看 **LPIPS、FVD**。
- **VAE 沒過時**：latent diffusion 內部就用 VAE 當壓縮骨幹——本文所有概念可直接轉移。
