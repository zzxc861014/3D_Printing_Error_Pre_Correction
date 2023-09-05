# 3D_Printing_Error_Pre_Correction
 
這個資料夾是主要的研究資料夾，也是我碩論的內容，請務必仔細看。

首先，我使用的資料以及程式碼都是放在6樓機房的資源跑的。先在6樓機房創一個帳號，請先跟有管理員權限的同學說請他幫忙，然後再遠端(Moba、VSCode等)到6樓機房操作

6樓機房使用的是Linux指令，請務必熟悉，不會上網找。
那接下來講解每個資料夾跟檔案內容。
這個計劃使用到的牙齒模型都放袃實驗室的NAS上，名稱為Teeth_Model，資料夾裡有詳細的介紹這些牙齒模型。

# GD_Net跟PG_Net
兩個內容物基本上一樣，只是輸入的跟預測的東西不同。
### Vox2vox
主要訓練網路的程式碼。
### main_3DGAN
主要的訓練程式碼。
### Data_generator_3Dpatch
產生Patch的程式碼，有點像把模型切成PATCH送進網路訓練的前處理程式碼，修改INPUT主要從這個檔案裡去修改。
### 3DGAN_testing
用來進行測試資料準確率與錯誤率的檔案
### Full_reconstruction
將切片模型送進訓練好的網路裡，進行預測預校正後，得到的預測後結果切片。 

# Cross_Validation
就是PG_Net跟GD_Net的交叉驗証pth檔案，我在驗證時選的都是第1500個epoch，所以都只有存這個pth下來。

# Positional_encoding
我在增加位置資訊後使用的訓練程式碼
### Model_rotation
是旋轉模型用的程式碼，旋轉三維模型的。
### Posi_encoding
是位置編碼的程式碼，會在main_3DGAN裡呼叫。

其餘就都一樣。

# ６樓機房訓練
訓練指令：

```
python -m torch.distributed.launch -nprock_per_node=4 main_3DGAN.py
```

# 相關資料
做這個計畫前有一些資料你可以先看一下，可以比較瞭解這個計畫的研究內容。

* Generative Advesarial Network (GAN)  [點我](https://arxiv.org/abs/1406.2661)
* Vox2Vox [點我](https://arxiv.org/abs/2003.13653)
* Pytorch 官方連結公開說明說(裡面有安裝教學、使用方法跟一些模型的範例和教學自己研究一下) [點我](https://pytorch.org/)
* Pytorch影片 [點我](https://www.youtube.com/playlist?list=PLqnslRFeH2UrcDBWF5mfPGpqQDSta6VK4)
* TensorFlow版本的pix2pix [點我](https://www.tensorflow.org/tutorials/generative/pix2pix)

看完之後可以大概了解整個網路架構的流程，如果Pytorch還是不大會用的話，可以去多看其他Youtube的教學影片，或是看看別人的網路架構怎麼寫怎麼訓練的，邏輯差不多相同。

# 本研究的目標
這個研究主要與陽明牙醫系李士元教授、台科大教授林宗翰、與北科教授（但我不知道到他的名子）合作，計畫開發一套光固化3D列印機台，包括列印機台、列印用樹脂、校正系統等等，全部自行開發。而我們主要負責校正系統的部分。

目前我們手上有的相關模型檔案主要是兩部分（1）光固化列印 （2）後固化曝光。之後可能還會有其他過程像是熱壓(thermal-suckdown)等後續流程的誤差模型。

當他們在對牙齒模型進行光固化3D列印與後固化曝光的時候，會因為許多複雜的原因（可以看我的碩論）而導致其表面上的幾何誤差。李老師表示，在口腔科學中他們的容許誤差值是正負0.1um。在0.1以內的視為無誤差；超過0.1視為誤差。

每組模型包含三個模型，原始設計STL檔（Digital, D模型）、光固化後STL檔（Green, G模型）、與後固化曝光後STL檔（Post-Curing, P模型）。

為了減少誤差，我們期望藉由『預校正』來達到預測效果。在光固化列印或後固化曝光前，預先對牙齒模型進行預校正，這樣一來列印後，就可以得到我們一開始想要的無誤差模型。

我們採用逆向訓練的方法嘗試做到『預校正』預測，及對Ｐ模型做預校正的概念，訓練兩階段模型，分別為Ｐ模型輸入、Ｇ模型真值輸出，與Ｇ模型輸入、Ｄ模型為真值輸出。
