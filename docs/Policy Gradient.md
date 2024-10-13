---
title: '5.Policy Gradient'
description:
keywords: 'DRL Lecture 1: Policy Gradient (Review)'
---

# Policy Gradient
這篇文章將介紹一個強化學習中的進階技術，名為 Proximal Policy Optimization (PPO)。PPO 是一種政策梯度的升級版方法，被 OpenAI 當作他們的預設強化學習演算法。假如你要實作強化學習，PPO 很可能是你首先會考慮的算法之一。

我們將先快速回顧 Policy Gradient，讓大家再次熟悉這個基礎概念，隨後會深入探討如何從 on-policy 方法轉變為 off-policy 方法，並進一步解釋這兩者的區別。最後，我們將介紹如何在 off-policy 方法上加入一些約束條件，進而形成 PPO 演算法。

![](https://i.imgur.com/NSKoS8K.png)

## 回顧 Policy Gradient
首先，我們來快速回顧一下 Policy Gradient，因為 PPO 是 Policy Gradient 的變形，因此理解這個基礎非常重要。在強化學習（Reinforcement Learning, RL）中，主要有三個基本組件：

- Actor：即代理人，負責決策，選擇下一步應該執行的行動。
- Environment（環境）：環境決定代理人的行為會產生什麼樣的結果，並提供當前的狀態。
- Reward Function（回報函數）：根據代理人的行動，提供一個回報，衡量行動的好壞。

在本章節的實作練習中，目標是讓機器學會玩電子遊戲。此時，Actor 的角色就是操控遊戲中的搖桿，比如控制角色向左、向右或開火等動作。Environment 就是遊戲主機，它負責控制遊戲的畫面、怪物的移動以及顯示當前的狀態。而 Reward Function 則決定了當你執行某些行動時，會得到多少分數。比如，每當你擊敗一個怪物時，就可以獲得 20 分。

![](https://i.imgur.com/YmlU0Zf.png)

同樣的概念也適用於圍棋中。在這裡，Actor 就是 AlphaGo，負責決定每一步棋的落子位置。Environment 是對手，負責回應每次的行動。而 Reward Function 則根據圍棋規則決定，贏得比賽你會得到 1 分，輸掉則得到 -1 分。

在強化學習中，Environment 和 Reward Function 是事先給定的，也就是說，它們在學習開始前就已經確定，並且無法被控制或修改。我們唯一可以控制和調整的是 Actor，也就是代理人決策的 Policy（策略）。我們的目標是調整 Actor 的策略，使其能在給定的環境和回報函數下獲得最大的回報。

## Policy of Actor
在強化學習中，Actor（執行者）負責決定在當前狀態下要採取的行為，而這個決策過程是由一個 Policy（策略）來控制的。Policy 𝜋 是一個具有參數 𝜃 的神經網路，其輸入和輸出決定了 Actor 如何選擇行為。

- **Policy 的結構**：
    - **輸入（Input）**：這是機器觀察到的狀態，通常表示為向量或矩陣。如果是在玩電子遊戲，輸入就是遊戲的畫面，這些畫面通常是由像素（pixels）所組成。
    - **輸出（Output）**：每個可能的行為對應於輸出層中的一個神經元。比如在遊戲中，Actor 可能有三個可選的行為，分別是向左移動、向右移動和開火，這三個行為分別對應輸出層的三個神經元。

!!! note

    在使用**深度學習**技術來進行強化學習時，網路的參數 𝜃 會根據學習過程不斷調整，使 Policy 能夠在不同狀態下做出更優化的決策。

當網路接收到輸入後，它會根據當前狀態對每個行為分配一個分數，這些分數代表了採取該行為的機率分布。Actor 會根據這個機率分布來決定下一步的行為。例如，當輸出的機率分布為 70% 向左、20% 向右、10% 開火時，Actor 就有 70% 的機率選擇向左，20% 的機率選擇向右，10% 的機率選擇開火。

![](https://i.imgur.com/8Q2bDTj.png)

這樣的機率分布決定了 Actor 在不同情況下所採取的行為，並且根據學習過程的進行，這些機率會不斷調整，從而提升 Actor 的策略。

### Example: Playing Video Game
接下來我們來介紹 Policy 的部分，它實際上是一個神經網路。現在，我們用一個具體的例子快速說明 Actor 如何與環境進行互動。

首先，Actor 會看到遊戲的初始畫面，這個畫面我們用 𝑆1 來表示。當 Actor 看到這個初始畫面後，根據其內部的神經網路和策略（Policy），它會決定採取一個行動。例如，Actor 可能決定向右移動。決定行動後，Actor 就會根據該行動獲得一個回報（Reward），代表這個行動得到多少分數。

在這個過程中，我們可以用 𝑆1 表示初始狀態，用 𝐴1 來表示第一次執行的動作，並用 𝑅1 來表示這次行動後得到的回報。有些文獻中可能會用不同的符號，例如 𝑅2 來表示回報，這都可以，最重要的是你自己理解這個過程。

![](https://i.imgur.com/idYxfeP.png)

接下來，Actor 會看到新的遊戲畫面（我們用 𝑆2 來表示），然後 Actor 再次根據新的畫面決定下一個行動，比如開火。如果這個行動成功殺掉了一個怪物，它可能會獲得 5 分的回報。這個過程會不斷反覆進行。

直到某一時刻，當 Actor 執行了一個動作並獲得回報後，環境決定遊戲結束。例如，在某些遊戲中，Actor 控制一艘綠色的船殺怪物，如果船被殺死，遊戲就結束；或者當所有怪物都被消滅時，遊戲也會結束。

![](https://i.imgur.com/av9zul2.png)

整個遊戲過程稱為一個 episode，而在這個 episode 中所累積的所有回報總和就是 total reward，我們可以用 𝑅 來表示。Actor 存在的最終目的是最大化（maximize）它所能獲得的總回報（reward）。

### Actor, Environment, Reward 關係
我們可以用圖像化的方式來解釋 Environment（環境）、Actor（執行者）和 Reward（回報）之間的關係。首先，環境其實可以看作是一個函數，雖然遊戲主機內部可能不是神經網路，而是基於規則的系統，但我們仍然可以把它當作一個函數來理解。

這個函數一開始會給出一個初始狀態（例如遊戲的畫面），讓 Actor 進行觀察。當 Actor 看到這個初始畫面 𝑆1 後，根據策略決定並執行一個行動 𝐴1 。接著，環境會將這個行動 𝐴1 作為輸入，並返回下一個狀態 𝑆2 也就是新的遊戲畫面）。

![](https://i.imgur.com/L8kaf9D.png)  

Actor 再次根據這個新的畫面 𝑆2決定下一個行動 𝐴2，環境接收這個行動後，繼續返回新的狀態 𝑆3。這個過程會不斷重複，直到環境判斷遊戲應該結束為止。

在一場遊戲中，我們可以將 Environment（環境）輸出的狀態 𝑆 與 Actor（執行者）輸出的行為 𝐴 組合在一起，形成一條軌跡，這條軌跡稱為 Trajectory。每一個 Trajectory 都可以計算它發生的機率。如果我們假設 Actor 的參數 𝜃 已經設定好，那麼就可以根據這些設定來計算某個 Trajectory 出現的機率，也就是在一個回合或一個 Episode 中發生這樣情況的概率。

![](https://i.imgur.com/TJOSgId.png)

如何計算這個機率呢？我們可以按照以下步驟進行：

1. 首先，計算環境輸出初始狀態 𝑆1 的機率 𝑝(𝑆1)。
2. 然後，根據狀態 𝑆1 決定行為 𝐴1 的機率，這個機率是由 Actor 的策略（Policy）決定，並且依賴於網路的參數 𝜃，表示為 𝑝𝜃(𝐴1∣𝑆1)。這裡我們之前提到過，Actor 的輸出其實是一個機率分佈，根據這個分佈來選擇具體的行動。

![](https://i.imgur.com/wIgvjxJ.png)

接下來，環境根據 Actor 執行的行為 𝐴1 和當前狀態 𝑆1 輸出下一個狀態 𝑆2。這個過程也可能具有機率性，取決於環境本身的設定。如果環境是完全決定性的，那每次相同的行為和狀態都會導致相同的下一個狀態；但若環境內部帶有隨機性，那麼每次產生的下一個狀態可能會不同。

![](https://i.imgur.com/uLv2a5l.png)

這個過程會不斷重複下去，直到遊戲結束。根據這個過程，我們可以計算出整個 trajectory 的機率，這取決於兩個部分：

1. **環境的行為**：即環境如何根據前一個狀態 𝑆𝑡 和行為 𝐴𝑡 生成下一個狀態 𝑆𝑡+1。這部分通常是由環境的內部設計決定的，並且我們無法控制。
2. **Actor 的行為**：即 Actor 根據當前狀態 𝑆𝑡 選擇行為 𝐴𝑡 的機率。這由 Actor 的參數 𝜃 控制，是我們可以通過學習進行優化的部分。

!!! note
    隨著 Actor 的行為不同，每一個 trajectory 出現的機率也會有所不同。因此，我們的目標是調整 Actor 的參數，最大化 trajectory 中獲得的回報（reward）。

在強化學習中，除了環境（Environment）和執行者（Actor）之外，還有一個重要的角色叫做 Reward Function（回報函數）。Reward Function 的作用是根據某個狀態（State）下採取的某個行為（Action）來決定該行為能夠獲得多少分數，它是一個函數。舉例來說，當你給它狀態 𝑆1 和行為 𝐴1 時，Reward Function 會告訴你得到的回報是 𝑅1。同樣地，給它 𝑆2 和 𝐴2 ，它會返回 𝑅2。

![](https://i.imgur.com/faNUogq.png)

我們把所有小的回報 𝑟𝑡 加總起來，就得到了一個 總回報 𝑅。這裡的 𝑅 代表某一個 Trajectory（軌跡）在一場遊戲或一個回合（Episode）中累積的總回報。在強化學習中，我們的目標是調整 Actor 內部的參數 𝜃，使得總回報 𝑅 越大越好。然而，實際上 Reward 並不是一個固定的值，而是一個隨機變數。這是因為：

1. Actor 在相同狀態下選擇行為時具有隨機性。
2. Environment 在相同行為下生成觀察值（Observation）時也可能具有隨機性。

因此，總回報 𝑅 是一個隨機變數，而我們能計算的是它的期望值（Expected Reward）。這個期望值代表了在給定某組參數 𝜃 的情況下，能夠期望獲得的總回報。

我們如何計算這個期望值呢？具體來說，首先要列舉出所有可能的 Trajectory，每個 Trajectory 都有一個對應的機率，這個機率是由 Actor 的參數 𝜃 決定的。假如 Actor 的參數非常優化，且模型表現很好，那麼長壽命、高回報的 Episode 會有較高的機率出現，而短命、低回報的 Episode 機率會較低。

根據這些機率，我們可以計算每個 Trajectory 的總回報，然後對所有可能的 Trajectory 進行加權總和，這就是總回報的期望值。我們可以將這個過程寫作：

![](https://i.imgur.com/qVJdX8K.png)

這表示從 𝑝𝜃(𝜏) 的分佈中抽樣一個 Trajectory，然後計算其總回報 𝑅(𝜏) 的期望值。

## Policy Gradient學習方式
在強化學習中，我們的目標是最大化期望回報（Expected Reward）。為了實現這個目標，我們使用梯度上升法（Gradient Ascent），與常見的梯度下降法不同，梯度上升法是讓期望回報逐漸變大，因此在更新參數時我們進行加法而不是減法。

為了應用梯度上升法，我們需要計算期望回報 𝑅ˉ𝜃 的梯度。首先，我們來看期望回報的梯度計算。因為只有機率分佈 𝑝𝜃(𝜏) 與參數 𝜃 相關，所以梯度只作用在這個機率分佈上。此時，我們不需要考慮回報函數 𝑅(𝜏) 是否可微分，因為即使回報函數是不可微的（例如在一些情境下，回報函數可能是黑箱模型），我們仍然可以進行後續計算。接下來，我們可以利用常見的數學公式，即：

![](https://i.imgur.com/EuVn1ey.png)

來簡化梯度計算。當我們取 ∇𝑝𝜃(𝜏) 的梯度時，可以將其轉換為 𝑝𝜃(𝜏)×∇log𝑝𝜃(𝜏)，這樣我們可以將整個期望回報寫成期望值的形式。具體來說，我們可以從機率分佈 𝑝𝜃(𝜏) 中取樣 𝜏，然後計算 𝑅(𝜏)∇log𝑝𝜃(𝜏)，並對所有可能的 𝜏 進行加總。由於實際上無法列舉所有的 𝜏，因此我們會使用抽樣的方式，即從分佈中抽取大量的樣本，對每一個樣本計算其對應的值，然後將這些值加總，最終得到梯度。

![](https://i.imgur.com/eIEPnjY.png)

之後，我們就可以使用這個梯度來更新 Actor 的參數。需要注意的是，這個機率分佈 𝑝𝜃(𝜏) 包含兩個部分，一部分來自於環境，另一部分來自於 Actor。我們無法對環境部分進行梯度計算，因為它與參數 𝜃 無關。因此，我們真正計算梯度的是來自 Actor 的部分，即 log𝑝𝜃(𝑎𝑡∣𝑠𝑡)。

!!! note

    如果在某個狀態 𝑠𝑡 執行行為 𝑎𝑡，最終導致整個 Trajectory 的回報是正的，那麼我們應該增大在該狀態下選擇該行為的機率；反之，如果回報是負的，我們應該減少該行為的機率。這就是策略梯度的核心概念。

在實作時，我們將使用梯度上升法（Gradient Ascent）來更新參數。具體過程如下：你有一組初始的參數 𝜃，接下來你將其與梯度項相加，當然，還需要引入一個學習率（learning rate），學習率需要進行調整，比如可以使用 Adam 等優化算法進行調整。

那麼，這個梯度是如何計算的呢？實際上，我們要依據下方公式進行計算。首先，你需要收集大量的狀態（𝑠）與行動（𝑎）的配對資料，並且要知道每個配對在實際與環境互動時會得到的回報（reward）。這些資料需要透過讓 Agent 與環境進行互動來收集。具體來說，你會先拿已經訓練好的 Agent 與環境進行互動，讓它玩幾場遊戲，然後記錄遊戲過程。

![](https://i.imgur.com/CZZJkZD.png)

例如，在第一場遊戲中，你記錄下來的資料可能是：在狀態 𝑠1採取行動 𝑎1，在狀態 𝑠2採取行動 𝑎2，並得到一個回報 𝑅(𝜏1)。由於 Agent 有隨機性，這意味著在同樣的狀態 𝑠1，它可能不會每次都選擇 𝑎1，因此你需要記錄每次的選擇。

接下來，當第二場遊戲結束後，你會得到第二筆資料，比如在狀態 𝑠1 採取了某個行動，得到的回報是 𝑅(𝜏2)。當你收集到這些資料後，就可以將其代入梯度公式中計算出對應的梯度。具體來說，你要計算每個 𝑠 和 𝑎 配對的對數機率（log probability），然後對它取梯度，並加權該場遊戲的回報（reward）。

當你計算出這些梯度後，就可以用它們來更新模型的參數。更新完模型後，你需要重新收集新的數據，然後再次更新模型，這個過程將反覆進行。需要特別注意的是，Policy Gradient 中的資料（數據）通常只會使用一次。這意味着每次你收集完資料後，用它來更新參數後，這些數據就會被丟棄，然後重新收集新的數據來進行下一輪的參數更新。

## 實作重點
接下來我們討論實作的細節，尤其是在強化學習中，如何使用深度學習框架來實現這個過程。當你實際進行實作時，最簡單的方式就是將這個問題看作是一個分類問題。分類問題我們都非常熟悉，無論是使用 TensorFlow 還是其他深度學習框架來做圖像分類，你都已經學會了這些基礎技能。

![](https://i.imgur.com/fkRzoen.png)

在這裡，你可以將強化學習中的 State 當作分類問題的輸入，類似於圖像分類中的輸入圖片。不同的是，分類的結果不是決定圖片中有哪些物體，而是決定當前狀態下該採取什麼行為。比如，在這個問題中，我們的分類可能有三個 class：向左、向右、開火。每一個行為都對應一個 class。

那麼，這些訓練資料是從哪裡來的呢？資料來自於你在與環境互動過程中進行的 sampling。當你在某個狀態 𝑠𝑡 下進行行為 𝑎𝑡 的時候，這個行為就成為你的訓練數據的一部分。即便這個行為在該狀態下的機率不是最高的，但因為它被抽樣出來了，所以我們在訓練中會告訴機器：當看到這個狀態時應該採取這個行為。

![](https://i.imgur.com/blp9APK.png)

在常見的分類問題中，我們的目標函數（objective function）通常是最小化交叉熵（cross entropy）。實際上，最小化交叉熵就是最大化對數似然（log likelihood），這個概念在這裡同樣適用。我們的目標是最大化 log likelihood，而這樣的損失函數在像 TensorFlow 這樣的框架中已有現成的實現，你只需要調用相關的函數，框架會自動幫你計算。最後，你會計算出梯度，並使用它來更新模型參數。這就是強化學習中策略梯度方法的實作步驟，與傳統的分類問題相似，只是我們的分類結果是採取的行為，而不是物體識別。


當我們將強化學習（RL）應用於這個框架時，唯一不同的地方就是，在原本的 loss 前面，我們需要乘上一個 權重（weight）。這個 weight 是什麼呢？這個 weight 是指在某個狀態 𝑠 下採取某個行為 𝑎 時，你最終在整場遊戲中獲得的 總回報（total reward）。也就是說，這個權重並不是你在當下 𝑠 採取 𝑎 時立即獲得的回報，而是這個行為影響整場遊戲的最終回報 𝑅(𝜏)。

![](https://i.imgur.com/CInebGN.png)

換句話說，我們不是僅僅針對當前狀態下的回報，而是將整場遊戲的 total reward 作為每筆訓練數據的權重。當你在某個狀態 𝑠 採取了行為 𝑎，這筆數據的權重就是該場遊戲的最終回報 𝑅(𝜏)。

在實作上，你只需要將每筆訓練數據按照這個權重進行加權，然後交給 TensorFlow 或 PyTorch 來幫你計算梯度，接下來的更新步驟就與一般的分類問題幾乎相同。因此，整個實作流程與分類問題的差異不大，最主要的不同就是加入了這個基於 total reward 的權重。

## 實作小技巧
### Tip 1: Add a Baseline
在實作強化學習時，有一些技巧可以幫助你提升訓練效果，其中一個重要的技巧就是加入基準線（Baseline）。那麼，什麼是基準線呢？在訓練過程中，我們會遇到這樣的情況：當我們根據一個狀態 𝑠 採取行為 𝑎 後，可能會獲得一個正的回報，但這個回報總是正的，導致每個行為的機率都在上升。

例如，在一些遊戲中，回報值可能始終是正的，甚至最高只有 0 分，這樣我們就無法區分不同行為的效果。例如，如果你的回報分數介於 0 到 21 分之間，那麼所有的回報都是正數，直接使用這個回報進行更新會使模型將所有行為的機率都提高，這不符合我們的預期，因為有些行為並不應該被強化。

在理想狀況下，即便所有回報都是正的，因為每個行為的回報不同，有大有小，模型會根據這些回報進行適當的權重調整。例如，在某個狀態下，你有三個行為選擇：A、B 和 C，雖然每個行為的回報都是正的，但回報值不同，模型會根據回報的大小來調整每個行為的機率。這樣，得到高回報的行為機率會增加，而得到低回報的行為機率則會減少。

然而，問題在於，實際上我們使用的是 sample 而不是 expectation。在某些狀態下，可能某些行為從來沒有被抽樣到，比如行為 A 沒有被抽樣到，這樣其他行為的機率會因為上升而導致 A 的機率下降，這並不意味著 A 是一個不好的行為，它只是運氣不好沒被選中而已，這樣的結果顯然是不理想的。

![](https://i.imgur.com/TUgyx5u.png)

要解決這個問題，我們可以通過引入一個基準線 𝑏 來調整回報。我們將回報 𝑅(𝜏) 減去基準線 𝑏，這樣可以讓最終的回報值有正有負。例如，當回報大於基準線時，我們會提升該行為的機率，而當回報小於基準線時，即便回報是正的，但比較小的回報仍然會降低該行為的機率。

![](https://i.imgur.com/O6t2WQg.png)

那麼，這個基準線 𝑏 應該怎麼設置呢？一個簡單的方法是使用回報的期望值來作為基準線。具體來說，你可以計算所有回報的平均值，然後將這個平均值作為基準線 𝑏，這樣就可以確保回報值在基準線之上或之下進行適當的調整。

!!! note

    在實作時，你需要不斷記錄每次訓練的總回報，並計算它們的平均值來作為基準線，這樣可以確保在訓練過程中，模型能夠合理地區分哪些行為應該被強化，哪些應該被弱化。

### Tip 2: Assign Suitable Credit
第二個技巧是關於如何給每一個行為（action）合適的貢獻值（credit）。具體來說，在同一場遊戲的過程中，並不是每一個行為都應該被賦予相同的回報權重。回想我們前面的做法，當你在某個狀態 𝑠 下執行某個行為 𝑎 後，我們會根據整場遊戲的總回報 𝑅−𝑏 來給所有的狀態-行為對（state-action pair）加權。然而，這樣的方式並不公平，因為在同一場遊戲中，不同的行為可能對最終結果的貢獻並不相同。例如，某些行為可能是好的，而有些行為可能是壞的。

舉個例子來說，在一場只有三次互動的遊戲中，當你在狀態 𝑠𝑎 下執行行為 𝑎1，你得到 5 分；在狀態 𝑠𝑏 下執行行為 𝑎2，你得到 0 分；而在狀態 𝑠𝑐 下執行行為 𝑎3，你得到 -2 分。整場遊戲的總回報 𝑅 是 3 分。

![](https://i.imgur.com/TDjVFIS.png)

現在問題來了：這 3 分是否代表在狀態 𝑠𝑏 下執行行為 𝑎2 是好的呢？並不一定。這個正向的回報主要是由在狀態 𝑠𝑎 執行行為 𝑎1 所帶來的，與在狀態 𝑠𝑏 的行為 𝑎2 可能毫無關聯，甚至可能是由於在 𝑠𝑏 執行了 𝑎2 才進入了不利的狀態 𝑠𝑐，導致後續的扣分。因此，整場遊戲的結果是正向的，並不代表每一個行為都是正確的。如果我們按照之前的方式，在訓練時每一個狀態和行為對都被加權 3 分，那麼即使有些行為實際上是不好的，也會被錯誤地強化。

讓我們再看另一個例子，假設在另一場遊戲中，你在狀態 𝑠𝑏 下執行了行為 𝑎2，最終你得到了負 7 分。為什麼會得到這個負 7 分呢？這是因為在 𝑠𝑎 狀態下執行了 𝑎2，導致了扣 5 分的結果。但這個扣分不一定是因為 𝑠𝑏 下執行 𝑎2 的錯，因為這兩件事可能並無直接關聯，只是發生的順序上 𝑠𝑎 先於 𝑠𝑏。在 𝑠𝑏 下執行 𝑎2 可能只導致接下來的狀態扣了 2 分，而這 2 分與之前在 𝑠𝑎 的扣 5 分沒有直接關聯。如果我們單純根據整場遊戲的總回報來權衡每個行為，那麼會導致某些行為不當地承擔了整個負回報的責任。

然而，當我們進行多次抽樣，並累積足夠多的樣本時，這些個別行為的影響會逐漸顯現出來。通過統計多場遊戲的結果，我們可以更準確地評估每個行為對整場遊戲結果的真實貢獻，這樣就能避免將所有的責任不公平地分配給某些特定行為。


問題在於，我們的抽樣次數可能不足，因此在這種情況下，我們需要為每一個狀態和行為對（state-action pair）賦予合理的貢獻值（credit）。這樣可以反映每一個行為對最終結果的真實貢獻。如何合理地分配這些貢獻呢？一個做法是：當我們計算某一個狀態-行為對的回報時，應該只計算從該行為執行後所獲得的回報。這是因為在執行該行為之前發生的事情與這個行為無關，之前的回報不能算作該行為的功勞。只有該行為執行後至遊戲結束所獲得的回報，才是真正屬於該行為的貢獻。

![](https://i.imgur.com/7VCIiHH.png)

例如，在狀態 𝑠𝑏 下執行 𝑎2，或許它真正導致的回報應該是 -2，而不是正 3，因為先前的 +5 並不是 𝑎2 的功勞。實際上，執行 𝑎2 後，你只被扣了 2 分，因此這應該是其實際貢獻。同樣地，執行 𝑎2 也不應該負責 -7 分，因為之前的 -5 分與該行為無關。實際上，該行為只導致了接下來的 -2 分，因此合理的貢獻應該是 -2。

如果我們要將梯度更新的公式具體化，並考慮到每個行為的實際貢獻，該怎麼做呢？原本的權重是基於整場遊戲的總回報 𝑅(𝜏)，也就是所有行為的回報總和。然而，現在我們將這個權重進行修改，將其改為從某一時間點 𝑡 開始，僅考慮從該行為發生之後到遊戲結束時的所有回報總和。

![](https://i.imgur.com/YamlDTy.png)

具體來說，當某個行為 𝑎𝑡 在時間點 𝑡 被執行時，我們只會將從時間 𝑡 到遊戲結束之間所有的回報進行總和，這樣的回報總和才能真實反映這個行為的好壞。這個修改能更準確地為每個行為分配它應有的貢獻，而不是使用整場遊戲的總回報來做加權。

接下來我們要進一步介紹一個概念，稱為折扣因子（discount factor），這是用來調整未來回報的重要性。為什麼我們要對未來的回報進行折扣呢？雖然在某一個時間點執行的行為 𝑎𝑡 會影響後續的結果，但在現實中，隨著時間的推移，行為對最終結果的影響會逐漸減弱。

舉例來說，如果你在時間點 𝑡2 執行了一個行為，並在 𝑡3 時刻得到了回報，那麼這個回報很可能是 𝑡2 執行行為的直接結果。然而，如果在很久之後，比如在 𝑡100 時刻再次得到一個回報，那麼這個回報與 𝑡2 的行為之間的關聯就會變得非常小。因此，我們需要對這些未來的回報進行折扣，使得離當前行為較遠的回報權重較低。

![](https://i.imgur.com/vmXe4Fh.png)

具體來說，我們會在每個未來回報前面乘上一個折扣因子 𝛾，這個 𝛾 介於 0 和 1 之間，通常設置為 0.9 或 0.99。如果某個回報發生在較遠的時間點，這個回報的權重就會隨著 𝛾 的次數乘積而變得越來越小。這樣可以確保我們更重視接近當前行為的回報，而將遠未來的回報影響降到最小。最終，這樣的處理方式可以更準確地衡量每個行為對其後續回報的實際貢獻，並且更符合真實環境中的情況。

在這裡，我們引入了 baseline 𝑏，這個 𝑏 通常可以是依賴於狀態的 (state-dependent)，實際上它經常是通過一個神經網路來估計的，因此這部分內容比較複雜，我們會在之後詳細講解。現在，我們將 𝑟−𝑏 這個項目統稱為 Advantage Function，並用 𝐴 來表示。

![](https://i.imgur.com/LbWxsTU.png)

!!! note

    Advantage Function 的核心目的是幫助我們判斷在某一個狀態 𝑠𝑡 下執行某一個行為 𝑎𝑡 時，這個行為相較於其他可能的行為有多好。這並不是一個絕對的好壞判斷，而是相對的，意思是說在同樣的狀態下，執行行為 𝑎𝑡 相較於其他可能的行為是否更優越。

此外，𝐴𝜃(𝑠𝑡,𝑎𝑡) 中的上標 𝜃 代表的是當前模型的參數，這表示我們使用帶有參數 𝜃 的模型與環境互動後，計算得出的 Advantage。換句話說，當我們計算 Advantage Function 時，我們考慮的是這個模型與環境的互動結果，並根據執行行為後的回報來衡量這個行為的好壞。

Advantage Function 的意義在於：它幫助我們不僅看行為是否好，還看這個行為相對於其他行為是否更好。我們會在稍後介紹 Actor-Critic 方法時進一步探討，並解釋如何通過一個叫 Critic 的神經網路來估計這個 Advantage Function。
