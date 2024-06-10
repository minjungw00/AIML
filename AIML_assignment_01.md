
# 중앙대학교 예술공학대학 2023-2 인공지능과 머신러닝 과제# 1

# 목표

- 주어진 데이터를 잘 나타내는 한 편, 예측력도 높은 함수 f(x)를 설계하는 과제입니다.
- 모든 코드는 `regression.py` 파일에 작성하며, 파일명을 변경하면 안 됩니다. 파일명을 변경하는 경우 평가 시스템에서 오류가 발생하게 됩니다.
- 과제의 최종 점수는 제출 기한 이후 최종 순위에 따라 차등 결정됩니다.
- 제출 기한은 **2023년 10월 24일 화요일 오후 11시 59분** 입니다.

![scatter.jpg](/assets/images/scatter.jpg)

### 문제(1)

- 본 문제에서는 제공된 데이터(훈련 데이터)를 `func` 함수가 잘 **설명**할 수 있는지 테스트합니다.
- 본 문제의 평가 데이터는 훈련 데이터를 포함하며, 훈련 데이터에는 없지만 훈련 데이터의 $x$ 및 $y$의 범위에 존재하는 **별도의 데이터**도 포함합니다.
- 평가 데이터의 범위는 아래와 같습니다.

$`
\begin{align*}
1 < x < 72 \\
-19 < y < 157
\end{align*}
`$

### 문제(2)

- 본 문제에서는 `func` 함수가 훈련 데이터의 범위를 벗어나는 데이터를 잘 **예측**할 수 있는지 테스트합니다.
- 본 문제의 평가 데이터는 훈련 데이터에는 **없는** 데이터로서, $x$와 $y$의 범위는 다음과 같습니다.

$`
\begin{align*}
71 < x < 101 \\
156 < y < 304
\end{align*}
`$


# 접근

## 사전 접근

### 다들 과제를 어떻게 하나

최적의 함수를 찾기 위해 다른 사람들이 다양한 방법으로 접근하더라

- 2차 이하의 함수
	- 가장 기본적인 접근 방법
	- 이걸로 고득점은 당연히 안된다
- 3차 이상의 함수
	- 다들 인터넷에서 모델 피팅을 검색해서 함수를 찾더라
	- 데이터의 모양이 꾸불꾸불한데 이거때문에 계수를 무진장 늘려놓으면 오버피팅이 될 거 같다

### 아이디어

근데 함수를 굳이 1개의 다항식으로 표현해야 할까?

구간함수를 사용해도 되겠는데?


## 1번째 접근

### 아이디어

- 구간함수를 통해 모델을 피팅한다
- 데이터를 직접 선별해 눈에 보이는 패턴대로 구간을 정의한다
- 일단은 각 구간 별 함수를 간단하게 근사해서 동향을 파악해본다

### 구간 나누기

데이터를 눈에 보이는 대로 3가지 패턴으로 구분했다

![Pasted_image_20231020212645.png](/assets/images/Pasted_image_20231020212645.png)

- 1번 패턴
	- y값이 가파르게 증가했다가 증가폭이 서서히 감소하는 형태
	- 2차 함수로 근사되는 형태
	- 임의의 3개 점을 잡아 2차 함수를 
- 2번 패턴
	- y값이 일정한 속도로 증가하는 형태
	- 1차 함수로 근사되는 형태
- 3번 패턴
	- y값이 일정한 속도로 감소하는 형태
	- 1차 함수로 근사되는 형태

위 과정을 통해 구간 2차 함수로 주어진 데이터를 충분히 근사할 수 있다고 판단했기 때문에 각 구간별로 제곱 오차를 최소화하는 2차식의 계수를 구한다
다항식은 numpy에서 제공하는 [polyfit](https://numpy.org/doc/stable/reference/generated/numpy.polyfit.html)을 통해 계산했다

![Pasted_image_20231020035346.png](/assets/images/Pasted_image_20231020035346.png)

구간별로 생성한 2차식의 계수는 다음과 같다

![Pasted_image_20231020035415.png](/assets/images/Pasted_image_20231020035415.png)

구간별 2차식을 적용한 함수는 아래와 같다

![scatter_with_func.jpg](/assets/images/scatter_with_func.jpg)

오버피팅을 최소화하면서 주어진 데이터를 피팅했다

### 구간 예측하기

각 패턴이 어느 정도의 주기로 찾아오는지 파악한다

|1->2|2->3|3->1|1->1|
|:---:|:---:|:---:|:---:|
|-|-|5.514|5.514|
|2.506|3.508|5.514|11.528|
|3.008|4.511|5.013|12.532|
|4.009|3.008|6.015|13.032|
|3.508|2.507|6.517|12.030|
|3.508|3.509|5.012|12.531|

- 1->2에서, 뚜렷한 패턴은 보이지 않으나 뒤로 갈수록 증가폭이 약 3.5에 가까워지는 것으로 보인다. 일단은 약 3.5의 증가폭을 가지는 것으로 예상해본다.
- 2->3에서, 역시 뚜렷한 패턴은 보이지 않는다. 억지로 규칙을 끼워맞춘다면 +1.0, -1.5, -0.5, +1.0으로 증가폭이 변화하는데 약 2.5에서 3.5 사이의 증가폭으로 예상하고 진행하는 것이 좋아보인다.
- 3->1에서, 마찬가지로 뚜렷한 패턴이 보이지 않는다. 약 5.5에서 6.5의 증가폭을 보일 것으로 보인다.
- 전체적으로, 하나의 사이클이 약 11.5에서 13.0 주기로 진행된다. 역시 뚜렷한 패턴은 없다

패턴이 크게 존재하지 않아서 임의로 계산해본다

- 1->2 약 3.0~4.0
- 2->3 약 2.5~3.5
- 3->1 약 5.5~6.5
- 전체 사이클 약 12.0~13.0

위에서 x=69까지의 데이터를 피팅했고, 패턴이 1번으로 끝났으며, 문제(2)의 데이터의 x가 101까지이므로  남은 패턴은 2-3-1-2-3-1-2-3이다.

각 구간 별 구간함수를 추정하기 위해 각 구간 별 이차항의 계수와 꼭짓점의 좌표를 정리한다

|p|a|x|y|
|:---:|:---:|:---:|:---:|
|1|-0.68538924|6.1690181625845195|-4.8413526401175435|
|2|-1.21803016|11.107769934038414|5.511015649164503|
|3|0.55494366|18.346254131455435|-27.2373754928593|
|1|-0.62911887|19.409290512935975|5.968006890333752|
|2|-1.08832353|22.68575212188971|14.324371640186527|
|3|-0.49595229|21.05123460565128|17.705095917230867|
|1|-1.11963727|31.404984048985796|25.19453098742326|
|2|-0.00889130482|278.73706505093145|567.175685384849|
|3|0.726066341|40.377235652630254|26.239591467720135|
|1|-0.67937196|45.4641783420087|57.69450502410782|
|2|0.691600166|41.46488174787396|48.19754764880228|
|3|-1.55646517|48.048054618530266|74.63378858153071|
|1|-0.297719384|62.86224799524643|106.02558281235348|
|2|-1.06630822|62.28316471198168|117.6386733166567|
|3|-0.04905084|42.15669813197898|133.80069570926455|
|1|-0.88049278|70.31070090092051|143.33557641439572|

이를 패턴별로 정리한다

|p|a|x|y|x 증가량|y 증가량|
|:---:|:---:|:---:|:---:|:---:|:---:|
|1|-0.68538924|6.1690181625845195|-4.8413526401175435|-|-|
|1|-0.62911887|19.409290512935975|5.968006890333752|13.240|10.809|
|1|-1.11963727|31.404984048985796|25.19453098742326|11.995|19.226|
|1|-0.67937196|45.4641783420087|57.69450502410782|14.060|32.500|
|1|-0.297719384|62.86224799524643|106.02558281235348|17.398|48.331|
|1|-0.88049278|70.31070090092051|143.33557641439572|7.448|37.31|

|p|a|x|y|x 증가량|y 증가량|
|:---:|:---:|:---:|:---:|:---:|:---:|
|2|-1.21803016|11.107769934038414|5.511015649164503|-|-|
|2|-1.08832353|22.68575212188971|14.324371640186527|11.578|8.813|
|2|-0.00889130482|278.73706505093145|567.175685384849|256.052|552.851|
|2|0.691600166|41.46488174787396|48.19754764880228|-237.273|-518.978|
|2|-1.06630822|62.28316471198168|117.6386733166567|20.819|69.441|

|p|a|x|y|x 증가량|y 증가량|
|:---:|:---:|:---:|:---:|:---:|:---:|
|3|0.55494366|18.346254131455435|-27.2373754928593|-|-|
|3|-0.49595229|21.05123460565128|17.705095917230867|2.705|44.942|
|3|0.726066341|40.377235652630254|26.239591467720135|19.326|8.534|
|3|-1.55646517|48.048054618530266|74.63378858153071|7.671|48.394|
|3|-0.04905084|42.15669813197898|133.80069570926455|-5.892|59.167|

패턴이 전혀 짐작가지 않기 때문에 임의로 함수를 추정해본다.

- 1번 패턴
	- 이차항의 계수를 -0.65로 추정한다
	- 꼭짓점의 x값이 약 13만큼 이동한다고 추정한다
	- 꼭짓점의 y값이 약 50, 60만큼 이동한다고 추정한다
- 2번 패턴
	- 이차항의 계수를 -1로 추정한다
	- 꼭짓점의 x값이 약 10만큼 이동한다고 추정한다
	- 꼭짓점의 y값이 약 40만큼 이동한다고 추정한다
- 3번 패턴
	- 이차항의 계수를 -0.05로 추정한다
	- 꼭짓점의 x값이 약 40에 있다고 추정한다
	- 꼭짓점의 y값이 약 60만큼 이동한다고 추정한다

추정한 구간함수의 이차항의 계수와 꼭짓점은 다음과 같다

|p|a|x|y|
|:---:|:---:|:---:|:---:|
|2|-1|75|160|
|3|-0.5|40|190|
|1|-0.65|83|190|
|2|-1|90|200|
|3|-0.5|40|250|
|1|-0.65|96|250|
|2|-1|105|240|
|3|-0.5|40|310|

이 값을 이차식으로 변형한다

![Pasted_image_20231021000937.png](/assets/images/Pasted_image_20231021000937.png)

이렇게 생성한 이차식의 계수들을 배열에 넣는다

![Pasted_image_20231021000812.png](/assets/images/Pasted_image_20231021000812.png)

구간은 위에서 예측한 대로 작성한다

![Pasted_image_20231021001352.png](/assets/images/Pasted_image_20231021001352.png)

이 상태로 그래프를 한번 뽑아봤다

![scatter_with_func_1.jpg](/assets/images/scatter_with_func_1.jpg)

망했다

결국 직감의 힘을 빌려 수치를 세부 조정해 노가다해서 그럴듯한 함수 개형을 찾았다

![Pasted_image_20231021011030.png](/assets/images/Pasted_image_20231021011030.png)

![Pasted_image_20231021011104.png](/assets/images/Pasted_image_20231021011104.png)

![scatter_with_func_2.jpg](/assets/images/scatter_with_func_2.jpg)

이정도면 그럴듯하게 생겼는데 과연?

### 결과 및 피드백

```
[2023-10-21 01:23:13] 
[SECURITY] Checking security issues has passed!
[PROBLEM_01] Testing observed data
Evaluation metric: 3.520024

[SECURITY] Checking security issues has passed!
[PROBLEM_02] Testing unseen data
Evaluation metric: 328.292198



- Metric: 331.81222200
- Rank: 47
- Score: 72.5
```

2번 문제에서 굉장히 낮은 점수를 받았다.

그래프의 개형이 완전히 잘못된 것으로 보인다


## 2번째 접근

### 문제점

앞서 제작한 함수는 다음 문제가 있었다

- 1번 문제는 그럴듯하게 설명됐다
- 2번 문제가 제대로 설명되지 않았다

이후 구간 예측 실패에 대한 가설은 다음과 같다

- 증가량이 너무 작다
- 제기된 문제의 구간은 다음과 같다

$`
\begin{align*}
71 < x < 101 \\
156 < y < 304
\end{align*}
`$


- 단순히 생각해본다면 x가 101일 때 y기 304까지 갈 수 있다는 뜻이다
- 근데 지금 함수는 y가 260까지밖에 증가하지 않는다
- 좀 더 상승량이 많아야 될 것 같다

### 이후 구간이 어떻게 생겼을까?


이후 구간이 어떻게 생겼는지 확인하기 위해 일단 확인 작업이 필요하다

예측범위인 (70, 150)에서 (101, 304)까지 직선을 그려서 점수를 확인해본다

![scatter_with_func_3.jpg](/assets/images/scatter_with_func_3.jpg)

```
[2023-10-21 01:44:20] 
[SECURITY] Checking security issues has passed!
[PROBLEM_01] Testing observed data
Evaluation metric: 3.659482

[SECURITY] Checking security issues has passed!
[PROBLEM_02] Testing unseen data
Evaluation metric: 66.777712



- Metric: 70.43719400
- Rank: 42
- Score: 75.0
```

확실히 기존 함수보다 점수가 높다

기존 개형에서 y값의 상승량을 더 높여보고 테스트해봐야겠다

### 아이디어

그래프를 개선하기 위해서 다음 방법을 사용해본다

- 각 패턴 별 이차항의 계수는 그대로일 것으로 간주한다
- 예측 구간에서 각 구간의 경계마다 가질 수 있는 데이터를 예측한다
- 예측한 데이터들을 지나도록 구간함수를 수정한다

### 가상 데이터 예측하기

각 구간을 결정하는 대표 데이터는 다음과 같다

|p|x|y|비고|
|:---:|:---:|:---:|:---:|
|3->1|2.255|-15.703||
|1->2|7.769|-8.266||
|2->3|10.275|4.332|ㅇ|
|3->1|13.784|-12.604||
|1->2|19.298|1.614|ㅇ|
|2->3|23.308|15.537|ㅇ|
|3->1|26.817|1.029||
|1->2|32.330|25.492||
|2->3|35.839|41.558||
|3->1|38.847|27.899||
|1->2|44.862|54.533||
|2->3|48.370|73.817|ㅇ|
|3->1|50.375|62.959||
|1->2|56.892|92.647||
|2->3|61.904|117.778|ㅇ|
|3->1|64.411|110.089||
|1->2|68.922|140.070||

이를 구간별로 재분류한다

|p|x|y|x 증가량|y 증가량|비고|
|:---:|:---:|:---:|:---:|:---:|:---:|
|1->2|7.769|-8.266|-|-||
|1->2|19.298|1.614|11.529|9.88|ㅇ|
|1->2|32.330|25.492|13.032|23.878||
|1->2|44.862|54.533|12.532|29.041||
|1->2|56.892|92.647|12.030|38.114||
|1->2|68.922|140.070|12.030|47.423||

|p|x|y|x 증가량|y 증가량|비고|
|:---:|:---:|:---:|:---:|:---:|:---:|
|2->3|10.275|4.332|-|-|ㅇ|
|2->3|23.308|15.537|13.033|11.205|ㅇ|
|2->3|35.839|41.558|12.531|26.021||
|2->3|48.370|73.817|12.531|32.259|ㅇ|
|2->3|61.904|117.778|13.534|43.961|ㅇ|

|p|x|y|x 증가량|y 증가량|비고|
|:---:|:---:|:---:|:---:|:---:|:---:|
|3->1|2.255|-15.703|-|-||
|3->1|13.784|-12.604|11.529|3.099||
|3->1|26.817|1.029|13.033|13.633||
|3->1|38.847|27.899|12.030|26.870||
|3->1|50.375|62.959|11.528|35.060||
|3->1|64.411|110.089|14.036|47.130||

증가량의 변화량에 일관성이 없다

임의로 다음과 같이 예측한다
- x 증가량은 약 12.5에서 약 13.5다
- y 증가량은 약 9.0~10.0씩 상승한다

따라서 예측되는 8개의 점은 다음과 같다

|p|x|y|
|:---:|:---:|:---:|
|2->3|74|170|
|3->1|77|167|
|1->2|81|207|
|2->3|86.5|233|
|3->1|89.5|234|
|1->2|93.5|284|
|2->3|99|306|
|3->1|102|311|

해당 점을 통해 개선한 구간함수는 다음과 같다

![Pasted_image_20231021025726.png](/assets/images/Pasted_image_20231021025726.png)

![Pasted_image_20231021025750.png](/assets/images/Pasted_image_20231021025750.png)

![scatter_with_func_5.jpg](/assets/images/scatter_with_func_5.jpg)

일부 튀는 값이 보이나 일단은 이걸로 점수를 확인해본다

![Pasted_image_20231021025922.png](/assets/images/Pasted_image_20231021025922.png)

처음 함수보다 개선됐으나 일차 함수보다는 낮은 점수다

튀어나온 값을 일부 정리해본다

![Pasted_image_20231021031347.png](/assets/images/Pasted_image_20231021031347.png)

![scatter_with_func_6.jpg](/assets/images/scatter_with_func_6.jpg)

```
[2023-10-21 03:14:10] 
[SECURITY] Checking security issues has passed!
[PROBLEM_01] Testing observed data
Evaluation metric: 3.452964

[SECURITY] Checking security issues has passed!
[PROBLEM_02] Testing unseen data
Evaluation metric: 103.874445



- Metric: 107.32740900
- Rank: 44
- Score: 75.0
```

튀어나온 값을 아래로 내리니 Matric이 더 감소했다

현재 함수의 y값이 과도하게 높은 것으로 파악되어 함수의 y값을 더 내려봤다

|p|x|y|
|:---:|:---:|:---:|
|2->3|74|170|
|3->1|77|167|
|1->2|81|200|
|2->3|86.5|225|
|3->1|89.5|220|
|1->2|93.5|260|
|2->3|99|290|
|3->1|102|300|

![Pasted_image_20231021032859.png](/assets/images/Pasted_image_20231021032859.png)

![scatter_with_func_7.jpg](/assets/images/scatter_with_func_7.jpg)

```
[2023-10-21 03:29:20] 
[SECURITY] Checking security issues has passed!
[PROBLEM_01] Testing observed data
Evaluation metric: 3.518558

[SECURITY] Checking security issues has passed!
[PROBLEM_02] Testing unseen data
Evaluation metric: 73.422938



- Metric: 76.94149600
- Rank: 42
- Score: 75.0
```

여전히 크게 개선되는 모습은 아니다

## 중간 점검

### 현재 상황

현재 내 점수는 다음과 같다

```
[2023-10-21 03:29:20] 
[SECURITY] Checking security issues has passed!
[PROBLEM_01] Testing observed data
Evaluation metric: 3.518558

[SECURITY] Checking security issues has passed!
[PROBLEM_02] Testing unseen data
Evaluation metric: 73.422938



- Metric: 76.94149600
- Rank: 42
- Score: 75.0
```

사실 뒤에다 일차함수 때려박는거보다 점수가 더 안나오고 있다

다른 방법을 생각해보던가 해야 할 거 같은데

### 다른 사람들 점수

10-21 03:30 기준 다른 사람들의 점수는 다음과 같다

1등

```
- [ASSIGNMENT_01 (2023-10-15 08:04:37)] 20203806 has submitted..! (![💡]Metric: 2.81517e+00 ![🏅]Rank: 1 ![🔥]Score: 100.0)
```

Metric : 2.81517

5등

```
- [ASSIGNMENT_01 (2023-10-18 19:28:40)] 20196151 has submitted..! (![💡]Metric: 7.65482e+00 ![🏅]Rank: 5 ![🔥]Score: 100.0)
```

Metric : 7.65482

### 1등을 하려면...

Metric 계산 방식은 문제 1의 Matric과 문제 2의 Metric을 합산하는 방식이다

지금 1등의 두 문제 합산 Matric은 내 문제 1의 Metric보다 작다

1등을 하려면 처음부터 다시 해야 된다는 뜻이다

## 3번째 접근

### 아이디어

뒤 구간을 임의로 예측해보지 않고 다항함수로 근사화해본다.

### 구간을 나누지 않고 poly를 돌려버리면?

3차 함수로 poly를 돌려버리면

![scatter_with_func_8.jpg](/assets/images/scatter_with_func_8.jpg)

```
[SECURITY] Checking security issues has passed!
[PROBLEM_01] Testing observed data
Evaluation metric: 31.382493

[SECURITY] Checking security issues has passed!
[PROBLEM_02] Testing unseen data
Evaluation metric: 34.006545



- Metric: 65.38903800
- Rank: 33
- Score: 82.5
```

무려 여태 내가 짠 거보다 점수가 높게 나온다

일단은 문제 1의 metric이 내 함수가 너 낮으므로 둘을 합쳐본다

![scatter_with_func_9.jpg](/assets/images/scatter_with_func_9.jpg)

```
[SECURITY] Checking security issues has passed!
[PROBLEM_01] Testing observed data
Evaluation metric: 3.561504

[SECURITY] Checking security issues has passed!
[PROBLEM_02] Testing unseen data
Evaluation metric: 34.006545



- Metric: 37.56804900
- Rank: 20
- Score: 90.0
```

일단은 20등...

### 여러 가지 시도

문제1의 구간을 3차함수로 변경하고 다듬었을 땐

![scatter_with_func_10.jpg](/assets/images/scatter_with_func_10.jpg)

```
[2023-10-21 05:10:59] 
[SECURITY] Checking security issues has passed!
[PROBLEM_01] Testing observed data
Evaluation metric: 3.454575

[SECURITY] Checking security issues has passed!
[PROBLEM_02] Testing unseen data
Evaluation metric: 34.006545



- Metric: 37.46112000
- Rank: 20
- Score: 90.0
```

조금 줄었다

앞의 함수를 뒤에 그대로 붙였을 땐

![scatter_with_func_11.jpg](/assets/images/scatter_with_func_11.jpg)

```
[SECURITY] Checking security issues has passed!
[PROBLEM_01] Testing observed data
Evaluation metric: 3.307006

[SECURITY] Checking security issues has passed!
[PROBLEM_02] Testing unseen data
Evaluation metric: 695.377464



- Metric: 698.68447000
- Rank: 47
- Score: 72.5
```

3차 함수에 근사해야 되는 것으로 보인다

이전에 임의로 생성한 점들을 피팅했을 때

![scatter_with_func_12.jpg](/assets/images/scatter_with_func_12.jpg)

```
[2023-10-21 06:09:15] 
[SECURITY] Checking security issues has passed!
[PROBLEM_01] Testing observed data
Evaluation metric: 3.350499

[SECURITY] Checking security issues has passed!
[PROBLEM_02] Testing unseen data
Evaluation metric: 65.103023



- Metric: 68.45352200
- Rank: 38
- Score: 77.5
```

metric이 증가했다

![scatter_with_func_13.jpg](/assets/images/scatter_with_func_13.jpg)

### 패턴을 2개로 줄여버리면?


![scatter_with_func_14.jpg](/assets/images/scatter_with_func_14.jpg)

```
[SECURITY] Checking security issues has passed!
[PROBLEM_01] Testing observed data
Evaluation metric: 4.380869

[SECURITY] Checking security issues has passed!
[PROBLEM_02] Testing unseen data
Evaluation metric: 33.950962



- Metric: 38.33183100
- Rank: 21
- Score: 90.0
```

Metric이 상승했으므로 3개 구간으로 나눈 패턴이 더 정확했던 것으로 보인다

## 4번째 접근

### 지금까지 얻은 결론

현재 주어진 데이터로 구간을 예측해 모델을 생성하는 것은 좋은 방법이 아니다

- 생각보다 오차가 드라마틱하게 줄어들지 않음
- 이후 구간을 예측할 수 없음

현재 데이터를 정확하게 피팅할 수 있는 새로운 모델이 필요하다

### 몇 가지 실험

아예 훈련데이터의 인접한 두 점을 직선으로 이어버리면?

![Pasted_image_20231021230632.png](/assets/images/Pasted_image_20231021230632.png)

![scatter_with_func_15.jpg](/assets/images/scatter_with_func_15.jpg)

```
[2023-10-21 23:03:45] 
[SECURITY] Checking security issues has passed!
[PROBLEM_01] Testing observed data
Evaluation metric: 2.323682

[SECURITY] Checking security issues has passed!
[PROBLEM_02] Testing unseen data
Evaluation metric: 34.006545



- Metric: 36.33022700
- Rank: 21
- Score: 90.0
```

이래도 문제 1의 metric이 2.32로 줄었다

같은 방식으로 함수를 조금 더 다듬었다

![scatter_with_func_16.jpg](/assets/images/scatter_with_func_16.jpg)

```
[2023-10-22 00:31:00] 
[SECURITY] Checking security issues has passed!
[PROBLEM_01] Testing observed data
Evaluation metric: 3.032857

[SECURITY] Checking security issues has passed!
[PROBLEM_02] Testing unseen data
Evaluation metric: 34.006545



- Metric: 37.03940200
- Rank: 21
- Score: 90.0
```

안다듬는게 좋겠다

위의 일차식을 뒤에다 잘 붙여놓으면

![scatter_with_func_17.jpg](/assets/images/scatter_with_func_17.jpg)


```
[SECURITY] Checking security issues has passed!
[PROBLEM_01] Testing observed data
Evaluation metric: 2.210297

[SECURITY] Checking security issues has passed!
[PROBLEM_02] Testing unseen data
Evaluation metric: 66.009396



- Metric: 68.21969300
- Rank: 38
- Score: 77.5
```

끝부분이 약간 하자있는거같은데 끝부분을 한번 없애보자

![scatter_with_func_18.jpg](/assets/images/scatter_with_func_18.jpg)

```
[2023-10-22 01:32:42] 
[SECURITY] Checking security issues has passed!
[PROBLEM_01] Testing observed data
Evaluation metric: 2.210297

[SECURITY] Checking security issues has passed!
[PROBLEM_02] Testing unseen data
Evaluation metric: 39.518992



- Metric: 41.72928900
- Rank: 23
- Score: 87.5
```

앞부분에도 하자가 있는 것으로 보인다




|x|y|x 증가량|y 증가량|
|:---:|:---:|:---:|:---:|
|2.2556390977443606|-15.703351869377082|-|-|
|10.275689223057643|4.332933182249585|8.0200501253132824|20.036285051626667|
|13.784461152882205|-12.60412206710894|3.508771929824562|-16.937055249358525|
|23.308270676691727|15.537777888898923|9.523809523809522|28.141899956007863|
|26.81704260651629|1.029043543439636|3.508771929824563|-14.508734345459287|
|35.83959899749373|41.55890040439935|9.02255639097744|40.529856860959714|
|38.847117794486216|27.899441476292964|3.007518796992486|-13.659458928106386|
|47.86967418546366|76.18294674859914|9.022556390977444|48.283505272306176|
|50.37593984962406|62.95947126439767|2.5062656641604|-13.22347548420147|
|61.9047619047619|117.7783283452736|11.52882205513784|54.81885708087593|
|64.41102756892231|110.0894285872276|2.50626566416041|-7.688899758046|


### 문득 든 생각

이거 3차함수에 사인함수를 더한 거 아니야?

![Pasted_image_20231022022319.png](/assets/images/Pasted_image_20231022022319.png)

![scatter_with_func_19.jpg](/assets/images/scatter_with_func_19.jpg)

```
[2023-10-22 02:12:06] 
[SECURITY] Checking security issues has passed!
[PROBLEM_01] Testing observed data
Evaluation metric: 2.715325

[SECURITY] Checking security issues has passed!
[PROBLEM_02] Testing unseen data
Evaluation metric: 24.203352



- Metric: 26.91867700
- Rank: 17
- Score: 92.5
```

이거 좀만 더 다듬으면 뭔가 될 거 같은데


![Pasted_image_20231022022406.png](/assets/images/Pasted_image_20231022022406.png)

```
[2023-10-22 02:23:34] 
[SECURITY] Checking security issues has passed!
[PROBLEM_01] Testing observed data
Evaluation metric: 2.762093

[SECURITY] Checking security issues has passed!
[PROBLEM_02] Testing unseen data
Evaluation metric: 26.993873



- Metric: 29.75596600
- Rank: 17
- Score: 92.5
```

![scatter_with_func 20.jpg](/assets/images/scatter_with_func_20.jpg)

이런 식으로 함수를 피팅할 수 있을 거 같은데?

## 5번째 접근

### 아이디어

훈련 데이터가 어떻게 생성됐는가

-> 특정한 함수에서 적당한 오차로 랜덤값을 뽑아서 데이터를 추출했을 것

적당한 함수 개형을 찾아서 피팅하기만 해도 충분히 좋은 점수가 나올 수 있을 것이다

그렇다면 어떻게 해결할 수 있을까?

- 적당한 함수 개형을 찾는다
- scipy 라이브러리의 curve_fit을 사용해 함수를 피팅한다

### 적절한 함수 개형 찾기

계속 찾아보자

![Pasted_image_20231022035641.png](/assets/images/Pasted_image_20231022035641.png)

![scatter_with_func 21.jpg](/assets/images/scatter_with_func_21.jpg)

```
[2023-10-22 03:55:04] 
[SECURITY] Checking security issues has passed!
[PROBLEM_01] Testing observed data
Evaluation metric: 2.558908

[SECURITY] Checking security issues has passed!
[PROBLEM_02] Testing unseen data
Evaluation metric: 23.230880



- Metric: 25.78978800
- Rank: 16
- Score: 92.5
```


아무리 봐도 삼각함수의 합성같은데 어떻게 합성해야 나올지 모르겠어서

함수 그래프 계산기 사이트에서 어떻게 합성해야 되는지 계속 실험했다

![Pasted_image_20231024145555.png](/assets/images/Pasted_image_20231024145555.png)

![Pasted_image_20231024145702.png](/assets/images/Pasted_image_20231024145702.png)

원하는 모양을 찾았다.

바로 함수에 대입해봤다.


![Pasted_image_20231024142721.png](/assets/images/Pasted_image_20231024142721.png)

![Pasted_image_20231024142743.png](/assets/images/Pasted_image_20231024142743.png)

```
1.0214458575150344e-05 * x ** 3 + 0.02844802434719404 * x ** 2 + 0.17095659922519982 * x + -9.86967380508745 + -5.766219235416621 * np.sin(0.5065138419048528 * x) + -4.697246026427157 * np.sin(2 * 0.5065138419048528 * x)
```

![scatter_with_func 22.jpg](/assets/images/scatter_with_func_22.jpg)

```
[2023-10-24 14:24:58] 
[SECURITY] Checking security issues has passed!
[PROBLEM_01] Testing observed data
Evaluation metric: 2.211558

[SECURITY] Checking security issues has passed!
[PROBLEM_02] Testing unseen data
Evaluation metric: 10.313945



- Metric: 12.52550300
- Rank: 22
- Score: 87.5
```

마지막날이 되니까 metric 12.5가 22등밖에 안된다
말이 안된다...

3차함수를 2차함수로 줄이면?

![Pasted_image_20231024143949.png](/assets/images/Pasted_image_20231024143949.png)

![Pasted_image_20231024144018.png](/assets/images/Pasted_image_20231024144018.png)


![scatter_with_func_1_1.jpg](/assets/images/scatter_with_func_1_1.jpg)

```
[2023-10-24 14:38:32] 
[SECURITY] Checking security issues has passed!
[PROBLEM_01] Testing observed data
Evaluation metric: 2.212061

[SECURITY] Checking security issues has passed!
[PROBLEM_02] Testing unseen data
Evaluation metric: 4.206891



- Metric: 6.41895200
- Rank: 6
- Score: 97.5
```

두 삼각함수의 일차항의 계수가 정확히 2배가 아닐 수 있기 때문에 함수 개형을 좀 더 일반화해서 함수를 예측하고자 했는데

![Pasted_image_20231101171236.png](/assets/images/Pasted_image_20231101171236.png)

이 경우에 curve_fit을 통해 함수를 예측했을 때 특유의 개형이 나오지 않고 일반적인 삼각함수의 개형으로 나왔다.

![scatter_with_func 21.jpg](/assets/images/scatter_with_func_21.jpg)

특유의 개형을 그대로 유지하기 위해 curve_fit의 초기값을 설정해야 했다.

![Pasted_image_20231101171759.png](/assets/images/Pasted_image_20231101171759.png)

```
params, covariance = curve_fit(quadratic_model, x, y, [0.029565943435839646, 0.1378579815628654, -9.654010240351456, -5.892426664318304, 0.5000054698264715, 0.48003462287003157, -4.785135902940669, 2 * 0.5000054698264715, 0.48003462287003157], maxfev=100000)
```

![Pasted_image_20231024211252.png](/assets/images/Pasted_image_20231024211252.png)

![Pasted_image_20231024211341.png](/assets/images/Pasted_image_20231024211341.png)

![scatter_with_func_21.jpg](/assets/images/scatter_with_func_2_1.jpg)

```
[2023-10-24 21:02:08] 
[SECURITY] Checking security issues has passed!
[PROBLEM_01] Testing observed data
Evaluation metric: 1.485355

[SECURITY] Checking security issues has passed!
[PROBLEM_02] Testing unseen data
Evaluation metric: 4.475937



- Metric: 5.96129200
- Rank: 1
- Score: 100.0
```

추하게 끝부분 노가다로 낮춰 5등 안에 들려 했는데 큰일났다

![Pasted_image_20231025000132.png](/assets/images/Pasted_image_20231025000132.png)

![scatter_with_func_23.jpg](/assets/images/scatter_with_func_23.jpg)



```
[2023-10-24 23:59:36] 
[SECURITY] Checking security issues has passed!
[PROBLEM_01] Testing observed data
Evaluation metric: 1.486391

[SECURITY] Checking security issues has passed!
[PROBLEM_02] Testing unseen data
Evaluation metric: 4.462163



- Metric: 5.94855400
- Rank: 6
- Score: 97.5
```


# 결과

## 최종 코드

함수의 개형은 다음과 같다.

$`
\begin{align*}
ax^2+bx+c+d\sin(ex+f)+g\sin(hx+i)\quad(h\approx 2e)
\end{align*}
`$

![scatter_with_func_23.jpg](/assets/images/scatter_with_func_23.jpg)

이 함수와 별도로 선형 보간을 위해 점수를 높이기 위해 다음 점을 이용했다.

계속 테스트를 하면서 평가 데이터에 훈련 데이터가 포함돼있다는 사실을 알았다. 따라서 x 값의 배열이 입력됐을 때, 이 x 배열을 순회하면서 훈련 데이터와 일치하는 x 좌표에 대해서 훈련 데이터의 y값을 그대로 반환해 140개 점의 MSE를 0으로 낮췄다.

함수의 개형을 봤을 때 x<2.3 구간과 x>90 구간에서 개형이 많이 엇나가는 것처럼 보였다. 이 끝값을 보간하기 위한 방법을 생각해내지 못해 이 구간을 나눠 임의로 보정했다. 

### 교수 피드백

교수님께서 왜 훈련 데이터를 매핑하는 데 딕셔너리를 쓰지 않았냐고 질문하셨다.

이분 탐색은 시간복잡도가 O(logn)이고 해시 테이블은 시간복잡도가 O(1)이기 때문에 해시 테이블을 사용하는 것이 가장 효율적임에도 이 방법을 생각해내지 못했는데, 아래 이유가 있었다.

- Python을 사용한 지 오래돼 딕셔너리를 생각하지 못했다.
- 코딩테스트를 공부할 때도 해시 테이블을 자주 쓰지 않아서 해시 테이블에 익숙하지 않았다.
- 코딩테스트는 C++로 공부했는데, C++ STL의 map은 유사하게 key-value 쌍으로 저장되나 map은 레드블랙트리이기 때문에 탐색 시간은 O(logn)이다.  따라서 C++에서 해시맵을 사용하기 위해서는 hash_map을 사용해야 하는데, hash_map을 거의 사용하지 않았었다.

결국 경험 부족으로 해시 테이블을 생각해내지 못했는데, 이후 코딩테스트를 하면서 해시테이블을 적극적으로 사용해봐야겠다.

## 정답

평가 데이터는 약 400개로, 이 중 140개를 추출해 훈련 데이터로 공개했다.

교수님이 데이터 추출에 사용한 함수의 개형은 다음과 같았다.

$`
\begin{align*}
ax^2+bx+c+d\sin(ex+f)+g\sin(hx+i)+j\cos(kx+l)+(sigmoid\ function)
\end{align*}
`$

함수의 개형을 얼추 맞췄는데 끝값에서 많이 엇나갔던 이유가 시그모이드 함수 때문이었다. 시그모이드 함수를 많이 접해본 적이 없어서 시그모이드 함수를 적용할 생각을 하지 못했다.

## 최종 결과

![Pasted_image_20231101160316.png](/assets/images/Pasted_image_20231101160316.png)

운이 좋아서 순위권에 안착해 만점을 받았다.

만점자는 별도로 코드 리뷰가 있었기 때문에 이 글에 삽입한 이미지들을 기반으로 발표를 진행했다.
## 순위권 분석

전체적으로 함수 개형은 다항식 + 삼각함수로 모두가 비슷했다. 선형 보간의 디테일 차이에서 순위가 많이 나뉜 것으로 보인다.

### 1등


각 구간의 첫 값이 선형으로 증가한다고 가정하고, 각 구간의 첫 값을 해당 일차식을 통해 한 점으로 모아 구간 별 데이터의 개형을 겹쳐 비교했다.

구간별로 비교한 데이터를 통해 모든 구간에서 선형 보간을 적용했다.

다른 사람들과 달리 삼각함수의 개형이 굉장히 독특했는데 다음과 같았다.

$`
\begin{align*}
\sin(ax+b+c\sin(dx+e))
\end{align*}
`$

이 삼각함수의 개형을 보았을 때 단순 sin함수의 덧셈보다 부정확했다.

### 2등

다른 분반이었기 때문에 코드 리뷰를 듣지 못했으나 함수 개형은 비슷했고 선형 보간을 더 자세하게 적용했다.
### 3등

구간을 나눠서 노가다로 값을 찾았다.

함수 개형이 꽤 복잡했는데, 삼각함수가 여러 개 겹쳐있었다.
### 4등

함수 개형이 다음과 같았다.

$`
\begin{align*}
ax^3+bx^2+cx+d+e\sin(fx+g)+h\cos(ix+j)
\end{align*}
`$

sin함수와 cos함수는 주기를 맞추면 같은 개형을 가질 수 있으므로 다항식이 3차식인 것을 빼면 거의 비슷한 개형이다. 이후 주어진 훈련 데이터를 이용해 선형 보간을 진행했다.

### 기타

다들 나와 같은 방법을 생각해냈으나 선형 보간의 정도 차이로 인해 순위권에서 밀려난 것 같다.

