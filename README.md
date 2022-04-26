# Stock AutoTrading HW_2

### 簡介
本專案欲實現[股票預測與自動交略]，使用Nasdaq所提供的IBM（International Business Machines Corporation）過去5年的歷史股價作為資料。
模型架構選用長短期記憶（Long Short-Term Memory，LSTM)架構來建立預測模型，用以預測未來一天的股票價格走向，並以預測價格推演交易策略，以收益最大化為目標。
最終的獲利評斷標準會使用助教提供的 獲利計算程式來評判 [StockProfitCalculator](https://github.com/NCKU-CCS/StockProfitCalculator)

### 資料準備與可視化
由於Nasdaq所提供的IBM的數據區分為Training data與Testing data兩者為連續性的資料，故將兩者合併方便執行後續的時間切割。
下圖表示此股票走勢圖(僅顯示 High Low Close )
![GITHUB](https://github.com/yudream0214/Stock_AutoTrading_HW2/blob/main/Stock%20Price%20Curve.png "Stock Price Curve")

### 特徵挑選依據
在特徵選定部分是採用收盤價(Close)來進行訓練，原因在於該特徵表達了市場參與者共同認可的價格。市場在開盤後會因為各種新聞、風聲或想法等因素，上下反覆變動，
以本專案的資料特性無法將即時交易列入考量。而在接近收盤時間，市場參與者接能夠充分了解行情變動的要因為何。思考持有的股票價值是高於或低於目前股價，並考量
該價格未來是否能繼續維持，再依狀況調整操作。


### 交易策略

#### 交易規則
  * 每一天只能進行一筆交易 買入[1] 不交易[0] 賣出[0]
  * 每筆交易以1單位為上限
  * 允許買空賣空
  * 持股上限為1單位，下限為-1單位
  * 收益均使用開盤價計算，唯最後一天使用收盤價
  * 最後一天將強制使用收盤價出清手中持股，持有1單位則賣出(1->0)，持有-1單位則買入(-1->0)，使手中持股歸零

#### 交易策略
  * 判斷目前的持股狀態(stock)為 0、 1 或 -1
  
  * 預測收盤價(Close) > 預測前一天收盤價 [漲]
    * 無持股       [0]    actoin =>  1   [Buy]
    * 持1單位      [1]    action => -1  [Sell]
    * 持-1單位    [-1]    action =>  0  [None]
  
  * 預測收盤價(Close) == 預測前一天收盤價 [平盤]
    * 所有狀態  [-1,0,1]  action =>  0  [None]

  * 預測收盤價(Close) < 預測前一天收盤價 [跌]
    * 無持股       [0]    actoin => -1  [Sell]
    * 持1單位      [1]    action =>  0  [Sell]
    * 持-1單位    [-1]    action =>  1   [Buy]
  示意圖
  

### 模型架構 LSTM
下圖表示本專案所使用的網路架構
![GITHUB](https://github.com/yudream0214/Stock_AutoTrading_HW2/blob/main/LSTM.png "LSTM")

參數設定為 
  * units = 16
  * batch_input_shape = (BATCH_SIZE, TIME_STEPS, INPUT_SIZE)  
  * optimizer = 'adam'
  * loss = 'mean_squared_error'
  * EarlyStopping[ monitor='loss']
  * output Dense = 1


### 股價預測結果
下圖表示收盤價(Close)的預測價格與實際價格的曲線分布。
![GITHUB](https://github.com/yudream0214/Stock_AutoTrading_HW2/blob/main/Stock%20Close%20Result.png "Stock Close Curve")

  * MSE 驗證結果
![GITHUB](https://github.com/yudream0214/Stock_AutoTrading_HW2/blob/main/MSE_Figure.png "MSE_Figure")

### 獲利驗證結果 
以獲利計算程式[StockProfitCalculator]進行評斷，結果如下圖所示。
![GITHUB](https://github.com/yudream0214/Stock_AutoTrading_HW2/blob/main/profit_result.png "Profit_Result")


### 環境要求

| Name| Version
|:---:|---:
|Python|3.6.7
|Numpy|1.15.4
|Pandas|0.23.4
|matplotlib|3.0.2
|keras|2.1.6
|tensorflow-gpu|1.13.1
|scikit-learn|0.22


### 命令參數

|Name|Input|Default
|:---:|---|---
|--training|training file|training.csv
|--testing|testing file |testing.csv
|--outpit|output file|output.csv


可於直接於終端機中執行以下指令，並將參數改成欲使用的資料集，或是直接使用預設值  

    python Autotrading_main.py --training "your weather data"  --testing "your weather data" --output  "your output data"
### 輸出
輸出之[**output.csv**](https://github.com/yudream0214/Stock_AutoTrading_HW2/blob/main/output.csv)格式如下表所示。

|Trading Strategies
| action | 
|:---:|
|0 |
|1 |
|0 |
|0 |
|0 |
|-1|
|1 |
|-1|
|-1|
|1 |
|-1|
|0 |
|1 |
|-1|
|1 |
|1 |
|-1|
|1 |
|-1|


