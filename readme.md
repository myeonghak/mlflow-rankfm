Recommender HPO and Serving with MLflow
==============================================

<br/>

<center><img src="/img/mlflow_logo.png" align="center" alt="drawing" width="400"/></center>   


**[TL-DR]**   
An example of [MLflow](https://mlflow.org/) [Tracking](https://mlflow.org/docs/latest/tracking.html#tracking) and [Models](https://mlflow.org/docs/latest/models.html#models) Using [Factorization Machine Recommender](https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf) model library, [rankfm](https://github.com/etlundquist/rankfm).

----

<br>

Contents
--------

1.	[What is MLflow?](#intro)
2.	[Structure](#structure)
3.  [Usage](#usage)
4.	[HPO with MLflow](#tracking)
5.  [Model serving using MLflow](#models)


<br>

<a id="intro"></a>
## What is MLflow?  

요약하자면, 머신러닝 모델의 학습(하이퍼파라미터 튜닝), 패키지 환경, 배포 등의 전반적인 사이클을 손쉽게 관리하게 해주는 MLOps 툴입니다.  

Mlflow의 [공식 문서](https://mlflow.org/docs/latest/index.html)에서, MLflow는 end-to-end machine learning lifecycle을 관리하는 오픈소스 플랫폼이라고 소개하고 있습니다.

MLflow는 라이브러리나 언어에 구애받지 않고, 클라우드를 비롯한 어떤 환경에서든 동일하게 작동합니다. 확장가능성 역시 훌륭한데, 사용 조직의 규모가 1명이든 1,000명 이상이든 유용하게 사용할 수 있도록 고안되었습니다.  

Mlflow는 크게 4가지의 컴포넌트로 나눌 수 있습니다.  

1) [MLflow Tracking](https://mlflow.org/docs/latest/tracking.html#tracking): 실험 기록을 추적하며, 파라미터와 그 결과를 비교합니다.  
2) [MLflow Projects](https://mlflow.org/docs/latest/projects.html#projects): ML code를 재사용, 재구현 가능한 형태로 패키징하여 다른 데이터 과학자들과 공유하거나 프로덕션으로 변환합니다.  
3) [MLflow Models](https://mlflow.org/docs/latest/models.html#models): 다양한 ML 라이브러리로 만들어진 모델을 관리하고 다양한 모델 서빙과 추론 플랫폼으로 배포합니다.  
4) [MLflow Registry](https://mlflow.org/docs/latest/model-registry.html#registry): 중앙 모델 스토어를 제공함으로써, 한 MLflow 모델의 전체 라이프 사이클을 협동적으로 관리합니다. 이러한 관리 작업에는 모델 버전 관리(versioning), 모델 스테이지 관리(stage transitions), 주석 처리등을 포함합니다.

더 자세한 내용은 [제 블로그 포스트](https://myeonghak.github.io/mlops/MLOps-MLFlow%EB%A1%9C-%EB%AA%A8%EB%8D%B8-%ED%95%99%EC%8A%B5-%EA%B4%80%EB%A6%AC%ED%95%98%EA%B8%B0/)에서 소개하고 있습니다.


<br>   


<a id="structure"></a>
## Structure  


```bash

├── dataset                                       # 모델 학습 및 데모에 사용되는 데이터셋
├── rankfm                                        # 과거 버전의 rankfm 레포지터리를 클론한 폴더
├── demo  
│   ├── model.py                                  # rankfm의 FM 모델을 MLflow에 맞추어 수정한 모델                             
│   ├── preprocess.py                             # movielens 데이터셋의 interaction 데이터를 전처리하는 모듈
│   ├── serve.py                                  
│   ├── serve_module.py                           # mlflow models의 serving 기능을 위해 수정한 모델
│   ├── train.py                                  
└── readme.md                                       

```



<a id="usage"></a>
### Usage

<br>

손쉬운 재현을 위해 docker container 환경에 구현했습니다.  


``` bash
# docker image를 pull한 후, container를 run 합니다.

>> docker pull nilsine11202/mlflow_rankfm # 생략 가능

>> docker run -it --name mlflow -p 8885:8885 -p 8886:8886 -e GUNICORN_CMD_ARGS="--bind=0.0.0.0" -v "${PWD}":/workspace -d nilsine11202/mlflow_rankfm /bin/bash

>> docker exec -it mlflow bash

>> cd /workspace/demo

```



<br>


<a id="tracking"></a>
### HPO(Hyper Parameter Optimization) with MLflow   

머신러닝 모델 개발자, 혹은 분석가의 입장에서 모델링 과정에 상당한 시간과 노력을 기울이는 작업 중 하나는 하이퍼파라미터 최적화 과정일 것입니다. 만약, 원하는 하이퍼 파라미터 (범주형이든, 수치형이든)를 손쉽게 실험해주고 그 결과를 실험마다 비교해주는 툴이 있다면 편리하지 않을까요?  

그러한 기능을 [MLflow Tracking](https://mlflow.org/docs/latest/tracking.html#tracking)에서 제공하고 있습니다. 이 프로젝트에서는, rankfm 모델의 하이퍼파라미터 실험을 손쉽게 수행하고 그 결과를 웹앱으로 시각화해 보여주는 MLflow의 기능에 대해 살펴보겠습니다.  


``` bash

>> conda activate mlflow

>> python train.py
```
아래와 같은 로그를 확인하실 수 있습니다. 학습과 실험이 정상적으로 수행되고 있다는 것을 의미합니다.  

```
interactions shape: (1000209, 3)
interactions unique users: 6040
interactions unique items: 3706
user features users: 6040
item features items: 3706

...


train.py:71: ExperimentalWarning: MLflowCallback is experimental (supported from v1.4.0). The interface can change in the future.
  metric_name='precision'

[I 2021-09-06 22:44:06,918] A new study created in memory with name: FM_movie

training epoch: 0
log likelihood: -409541.6875

training epoch: 1
log likelihood: -407007.15625

training epoch: 2
log likelihood: -392451.78125
[I 2021-09-06 22:44:21,623] Trial 0 finished with value: 0.12442861874792979 and parameters: {'max_samples': 5, 'bootstrap': 'constant', 'sigma': 0.5, 'alpha': 0.03}. Best is trial 0 with value: 0.12442861874792979.
INFO: 'FM_movie' does not exist. Creating a new experiment

training epoch: 0
log likelihood: -431483.875

...

```

학습 후, `mlruns` 폴더가 포함된 디렉터리에서 다음 커맨드를 실행해 줍니다.

``` bash
>> mlflow ui --host 0.0.0.0 --port 8885
```

브라우저로 이동해서, 다음 주소로 이동합니다. (mlflow 웹 앱의 기본 포트 번호는 5000번입니다.)

```
http://127.0.0.1:8885
```


<br>

<center><img src="/img/mlflow_01.png" align="center" alt="drawing" width="400"/></center>   

docker bash 내에서 mlflow ui ~ 코드를 실행하면, 위와 같은 화면을 확인할 수 있습니다.  
비교하고 싶은 실험을 선택하고, Compare를 실행하면, 다음과 같이 실험 비교를 확인할 수 있습니다.  

<br>

<center><img src="/img/mlflow_02.png" align="center" alt="drawing" width="400"/></center>   

<br>

각 실험마다 사용된 하이퍼 파라미터와 그 성능 지표를 확인할 수 있습니다.  

<br>

<center><img src="/img/mlflow_03.png" align="center" alt="drawing" width="400"/></center>   

<br>

alpha가 증가함에 따라 MRR이 감소하는 경향을 보이는 것을 확인할 수 있습니다.  

가장 성능지표가 좋았던 버전을 optuna study 객체의 best_trial, 혹은 best_params로 재현할 수 있습니다.  


<a id="models"></a>
### Model serving using MLflow

MLflow는 이렇게 완성된 모델을 API로 제공하는 기능을 제공합니다. 여기에서는 [MLflow Models](https://mlflow.org/docs/latest/models.html#models)의 기능으로 추천 결과를 서빙하는 방법에 대해 실습합니다.  

<br>

1) **serve.py 실행**

먼저, 다음의 커맨드를 실행합니다.

``` bash
>> python serve.py
```

<br>

**(optional: rankfm package error 발생시 아래와 같이 처리해 주세요.)**

생성된 폴더로 이동합니다. `77b3d67a709e49549563ff585ddd5187`라고 생성된 폴더는 실행마다 무작위하게 생성되는 폴더명입니다.  
본인의 폴더명은 demo 디렉터리 아래에 있는 mlruns 하위에서 확인할 수 있습니다.  

``` bash
>> cd ./mlruns/0/77b3d67a709e49549563ff585ddd5187/artifacts/example_mlflow_model/conda.yaml
```
pip 아래에 rankfm 패키지를 추가해 주면 해결됩니다.  


<br>


2) **mlflow models 실행**

``` bash
>> mlflow models serve -m ${PWD}/mlruns/0/e4b251d575b84ab3ab947bed86b0c9ca/artifacts/example_mlflow_model/ -h 0.0.0.0 -p 8886
```
<br>  


3) **API 호출**  


새로운 터미널을 열어, 도커 컨테이너 안에서 다음 커맨드를 실행합니다.  


``` bash

>> docker exec -it mlflow bash

>> curl -X POST -H "Content-Type:application/json-numpy-split" --data '{"index":[0],"data":[1],"columns":[0]}' http://0.0.0.0:8886/invocations

```  


<br>


4) **결과 확인**  


```bash
[{"title": "Toy Story (1995)", "genres": "Animation|Children's|Comedy"}, {"title": "Babe (1995)", "genres": "Children's|Comedy|Drama"}, {"title": "Shawshank Redemption, The (1994)", "genres": "Drama"}, {"title": "Lion King, The (1994)", "genres": "Animation|Children's|Musical"}, {"title": "Aladdin (1992)", "genres": "Animation|Children's|Comedy|Musical"}, {"title": "Snow White and the Seven Dwarfs (1937)", "genres": "Animation|Children's|Musical"}, {"title": "Beauty and the Beast (1991)", "genres": "Animation|Children's|Musical"}, {"title": "Wizard of Oz, The (1939)", "genres": "Adventure|Children's|Drama|Musical"}]
```
