# 睡眠品質關鍵因素分析

## 1. 背景與概述

在現代快節奏的高壓社會中，焦慮與過勞已成為許多人的生活常態，「睡個好覺」逐漸從基本的生理需求，轉變為人們競相追求的奢侈品。然而，影響睡眠的因素錯綜複雜，從職業型態、運動習慣到生理數據（如血壓、心率），每一個變數都可能成為提升或阻礙睡眠品質的關鍵。本專案旨在透過系統性的分析，深入挖掘生活方式與睡眠健康之間的隱性關聯，透過數據驅動為提升睡眠品質提供具科學依據的改善建議與健康洞察。

## 2. 數據結構與預處理

資料集來自 Kaggle 公開資料集 **Sleep Health and Lifestyle Dataset**，**共 13 個欄位、374 筆樣本**。

| 類別 | 關鍵欄位 | 說明 |
|---|---|---|
| **個人特徵與職業** | `Person ID`, `Gender`, `Age`, `Occupation` | 包含受試者的基本背景與職業類別（如工程師、醫生、業務等）|
| **睡眠核心指標** | `Sleep Duration`, `Quality of Sleep`, `Sleep Disorder` | 評估睡眠表現的核心數據，包含每日睡眠時數、主觀睡眠品質評分（1-10分），以及是否有失眠或睡眠呼吸中止症等障礙 |
| **生活型態變數** | `Physical Activity Level`, `Daily Steps` | 記錄受試者的活動量，包含每日運動時間（分鐘）與步數 |
| **生理健康數據** | `Stress Level`, `BMI Category`, `Blood Pressure`, `Heart Rate` | 涵蓋壓力指數（1-10分）、身體質量指數、血壓與靜止心率 |

預處理過成包括：
* 檢查無缺漏值、無重複紀錄。
* 數據格式化：確保數據遵循一致的格式和標準。
* 數據離散化：將連續數據轉換為分類數據，提高模型的解釋性。
* 類別重編碼：將部分類別合併成一類，避免稀疏數據問題。


## 3. 數據相關性分析與統計分析

從相關係數矩陣中發現與 Sleep Quality 相關係數絕對值較⼤的前六個變量，從⼤到⼩分別是 Stress Level、Sleep Duration、Heart Rate、BMI Category、Age、Daily Steps Group。其中 Sleep Duration 與 Sleep Quality 呈強烈正相關。

<img width="600" src="img/corelation.png">


### 多變量敘述統計

已知 Sleep Duration 與 Sleep Quality 強烈正相關，因此針對另外五個變量 Stress Level、Heart Rate、BMI Category、Age、Daily Steps Group 做多變量敘述統計。

發現壓⼒程度越⼤，睡眠品質越差；⼼率越低，睡眠品質越⾼；壓⼒、⼼率與睡眠品質呈顯著負相關。

<img width="400" src="img/statistics5.png">

發現壓⼒程度越⼤，睡眠品質越差；⼼率越低，睡眠品質越⾼；壓⼒、⼼率與睡眠品質呈顯著負相關。

<img width="400" src="img/statistics6.png">

發現 BMI 落在正常範圍的群體，睡眠品質⾼的⼈較多；⽽ BMI 落在過重以上的群體，睡眠品質普通的⼈較多。每⽇步數達 7500 的群體，睡眠品質都偏⾼；⽽每⽇步數不到 5000 的群體，睡眠品質低的⼈較多。

<img width="400" src="img/statistics7.png">

<img width="400" src="img/statistics7.png">


## 4. 機器學習特徵重要性分析：隨機森林分類 Random Forest

**預測 Sleep Quality 的特徵重要性分析**

從以上特徵分析中發現，相較於其他變量，顯著重要的特徵為 Stress Level 和 Sleep Duration。那對 Stress Level 最重要的特徵是什麼呢？

<img width="400" src="img/featureimportance.png">

**預測 Sleep Level 的特徵重要性分析**

從以上特徵分析中發現，對 Stress Level 顯著重要的特徵為 Heart Rate。

<img width="400" src="img/featureimportance2.png">

## 5. 分析結果

綜上分析，識別出影響睡眠品質的三⼤關鍵因素為睡眠時⾧、壓⼒程度和⼼率，且三者間存在互相影響的關係，另外分析結果也顯示睡眠品質會影響壓力程度。

## 6. 建議

本研究提出兩個方法，在個⼈因素（性別、年齡、職業）不變的情況下，改善睡眠品質：
* 首先是直接調整睡眠的時間長度，不要過短或過長，在一個適合的範圍內。
* 其次是透過運動（身體活動）調整身體素質（BMI）和心血管健康，進而排解壓力、穩定心率，才能夠進一步提高睡眠品質。

**文獻支撐**

1. Bilal A. Chaudhry , et al.，The Relationship between Sleep Duration and Metabolic Syndrome Severity Scores in Emerging Adults.（2023）
   此研究表明短睡眠時間（<7 小時）和長睡眠時間（>9 小時）都與較高的代謝症候群嚴重程度評分相關，這表明 ​​ 最佳睡眠時間對於代謝健康和整體健康至關重要。
2. Mirjam Ekstedt , et al.，Microarousals during sleep are associated with increased levels of lipids, cortisol, and blood pressure.（2004）
   高壓力水平一直與較差的睡眠品質有關。壓力會激活下丘腦-垂體-腎上腺（HPA) 軸，增加皮質醇水平，從而擾亂睡眠模式並降低睡眠效率。
3. Yongbin Li , et al.，Research on the relationship between physical activity, sleep quality, psychological resilience, and social adaptation among Chinese college students: A cross-sectional study.（2023）
   研究發現體育活動可以顯著改善大學生的心理恢復力和社會適應，從而改善睡眠質量，表明參與體育活動可能有助於減少這個人群的睡眠問題
4. Hanne K J Gonnissen , et al.，Sleep duration, sleep quality and body weight: Parallel developments. (2013)
   研究討論了睡眠質量和體重之間的關係，強調睡眠短或受干擾與肥胖的增加有關，且青春期和成年期的 BMI 指數變化與睡眠時長的變化相反相關，表明更好的睡眠可能有助於管理體重。
5. Heart rate variability, sleep and sleep disorders.（2012）
   研究調查了失眠患者的心率變異性（HRV) 與睡眠品質之間的關係。結論是較低的 HRV（表示較高的心率）與較差的睡眠品質相關。
6. Amirreza Sajjadieh , et al.，The Association of Sleep Duration and Quality with Heart Rate Variability and Blood Pressure.（2020）
   這項研究評估了青少年心率變異性和睡眠效率之間的關聯。研究結果表明，較高的心率與較低的睡眠效率有關，導致睡眠品質較差。

