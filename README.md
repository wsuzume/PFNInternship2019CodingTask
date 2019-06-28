# PFNInternship2019CodingTask

PFNインターンシップ2019選考課題（コーディングタスク）の提出物です。今回私は最終的に落選となり残念な結果となってしまいましたが、この提出物は1次選考を通過しましたので、次年度以降挑戦される方々の参考になればと思います。

[PFNインターンシップ選考課題](https://github.com/pfnet/intern-coding-tasks)のリポジトリはこちらのリンクから参照してください。私が挑戦した課題は[MachineLearning](https://github.com/pfnet/intern-coding-tasks/tree/master/2019/machine_learning)のタスクです。

## 公開理由
1. 上記の理由。
2. PFN側から選考期間終了後に公開することを明示的に禁止されてはいないため。
3. 課題への解答にそれなりの時間がかかっているので、公開しないと私が単に時間を無駄にしたことになるため。
4. こいつをGitHubに置いとけば他社でたまにやらされる1時間くらいのFizzBuzzレベルのWebコーディングテストがあわよくば免除にならないかなって。

## 感想
評価基準が明記されており非常にフェアで、課題や問題文自体も解答者を悩ませることなく「必要な情報はすべて与えますので、純粋にあなたのコーディング能力とそれをわかりやすく説明する力だけが見たいです」「余計な忖度は必要ありません」といった誠実さを感じる構成になっている。おそらく減点一切なしの完全加点方式。

この課題のように、社員に対してもフェアで誠実で問題解決に際して仲間への助力を惜しまない社風なんだろうなと感じた。めっちゃ魅力的。もっと勉強して来年もっかい応募する。

## 詳細
下書きに Jupyter notebook を用い、整理したものをモジュール化していった。レポートに添付した図も Jupyter notebook から matplotlib で描画したもの。

### 実装した範囲
課題3までは課題中で要求されている水準（PFNの人がこれくらいの性能にはなるよと課題中で言っている数値）にまで達している。時間の都合で課題4に解答できなかったことをレポートに明記。

### 解答に使った時間
課題に「想定所要時間は最大2日です」と明記されているため、自分がフルパワーで2日間働いた場合に使用可能な時間、オーバーしたとしても3日間分までと決めて取り組んだ。嘘をついてもっと時間をかけてもよかった（あと1日あればADAMを実装し完答できたと思う）が、仮に選考に通った場合にお互い困ったことになるので正直に申告した。あと実際忙しかった。

* コーディングとテスト：1日目(6時間)+2日目(6時間)+3日目(4時間)=20時間
* レポート作成：3日目(4時間)
* レポート推敲・README微調整など：徹夜（4時間程度）
* 合計24時間

## 結果
1. コーディングタスクにおける選考：通過
2. 面接後：落選

落選通知メールに「応募書類とコーディング課題提出物及び面接内容を慎重に検討した結果」とあったことから、コーディング課題は通過後も評価対象になることが伺える。すなわち面接のみではなく総合評価での落選であることがわかるので、この提出物は「合格点」であっても「上位者ではない」可能性には注意されたい。

## その他
### 私のバックグラウンド（2019年時点）
某大学院M1。情報基礎科学専攻。プログラミング歴10年。数学歴1年。某ベンチャー企業でデータサイエンティストインターン2年。卒論以外の論文なし。目立った業績や成果物なし。バイト先で作ったものは公開不可能。仕事以外ではゴミとガラクタしか作らない。

### コーディングしながら考えてたこと
#### 初見
難易度は高くない。やるべきことはすべて課題の文中にこれ以上ないくらい丁寧に書かれている。問題自体もGNNというひねりを加えてきてはいるが『[ゼロから作るDeep Learning](https://www.amazon.co.jp/dp/4873117585)』を読んでシンプルなニューラルネットワークを作ったことがあれば難なく解けるだろう。論文読んでアルゴリズムを実装するほうがよっぽどキツい。

問題はバグを直してる時間があるかどうか。数式はとりわけデバッグしにくいので自分がヤバいバグを仕込むかどうかは経験上運ゲー。そして問題に比して所要時間はかなり短く設定されているので、私でも時間内にすべて実装しきることが不可能であることはこの時点で判断がついた。

#### 解答の方針
評価基準は課題中に明記されているので提出物はそれを満たすように工夫した。

* ソースコードが他人が読んで理解しやすいものになっていること。
    * → リファクタリングはかなり重視した。
* ソースコードが正しいものであることを検証するためにある程度の単体テストや検証のためのコードが書かれていること。
    * → これ見よがしに単体テスト書いた。図を描画するコードとかも添付した。
* 提出物を見て追試がしやすい形になっていること。
    * → README.md に実行方法やファイルの説明をまとめた。
* レポートが要点についてわかりやすくまとまっていること。
    * → A4紙2枚以内という制約を厳守した。

他に「想定所要時間は最大2日です。全課題が解けていなくても提出は可能ですので、学業に支障の無い範囲で取り組んでください。」という注意があったので、なるべく時間を守るように心がけた。

上記の注意書きと課題の内容的に課題3までは解答必須項目、課題4は加点要素という感じがした。特に課題4の「Adamを実装してください」はぶっちゃけ課題3までできていれば毛を生やすだけなので、解答しても加点は微々たるものだと判断し、むしろ評価基準として明記されているレポートやREADMEの整理に時間を費やすべきだと判断した。課題4は選択式だが、選択肢の1番目、2番目、3番目の順で難易度が上がるので、大きな加点が入るのはおそらく2番目以降だと思う。

あと1日あれば完答、さらに追加で2日あればほぼ満点の提出物を作れただろうが、PFNが想定している2〜3倍の時間を使ってしまってはPFN側が評価に困るだろうと思ったため(無限に時間を使っていいならば誰でもいいアウトプットを出せるが、インターンでは限られた期間でのアウトプットを求められているため)になるべく時間を守った。またかなり簡単な課題であることも相まって「最低限のコーディングができるかどうかの足切りにしか使わないだろう」と判断していた。

今考えればこの判断はおそらく間違いである。コーディング課題は面接後も評価対象に含まれているし、解答に使った時間自体は自己申告なのでPFN側もあまりあてにしていないだろう。したがって時間オーバーしてもクオリティが高いほうが多分いい。

すべて解答しなかった理由、および時間があれば解答できたかどうかについては面接時に聞かれたので正直に話している。反応は「そうですか」みたいな感じだったので本当のところはよくわからない（評価基準を課題に明記するくらいなので、時間をかけてもすべて解答してほしければ「全部解いてもらったほうがよかったですね」くらいは言うはずだが、そう言わなかったということは全解答は重視していないか、あるいは選考にシビアに関わるのであえて言わないようにしているかのどちらかである。想定所要時間を課題に明記している以上、面接官は時間オーバーについて肯定しづらいが、応募者の能力の最大値を見るためには否定もしづらいといったところか）。

#### 解いてみて
時間が足りない。私は経験が長いのでかなりコーディングが速いほう（Pythonだと2000行以内ならほぼ全部覚えていてコーディングしながらリファクタリング可能。バイト先にも仕事が速いって言われるくらい）だが、それでも「最大2日」というのはかなりタイトな時間設定であった。私は才能には恵まれていない常人の部類だからこそ言うが、これを2日で完答したら間違いなくバケモノの部類である。うん。次回は時間オーバーしてもいいや。

### 反省
なんだかんだ言ってもコーディング課題は応募者の中でも上位層だと思う（最小限のコメントで読めるくらいには整理してある）。数学が好きって書いちゃったので面接でご親切に数学ガチ勢の方を割り当てていただき、好きではあるが数学初心者の私は塵と消えた、もとい多少しどろもどろになってしまった部分もあり、面接評価は応募者の中では平均ちょい下くらいかと思われる。

わざわざこのレベルのコーディング課題を用意しているので業績や成果物は加点要素ではあっても過剰な重視はしていなさそうという印象があるため、応募書類の評価部分は「実際にインターンでやりたいことの研究計画」がどれくらい具体的に書けていて、かつPFN側がやりたいこととマッチしているかという部分だろうと思われる。私は第2希望でこのコーディング課題と同じGNNに近い分野の研究を提案していて、かつそれに近い研究を大学でやっていたのでマッチ度は高そうだが、第1希望のほうの動画解析に関しては幾分勉強不足と詰めの甘さがあったので、やはりこちらも応募者の平均よりは下かもしれない。

したがって主な落選の理由としては「勉強不足」「上には上がいる」という元も子もない感じになるだろう。今回は応募者も多かったようで、上位層でも内容のマッチ度というラックで選別された人もいるくらいだろうから、私程度の能力では仕方あるまい。
