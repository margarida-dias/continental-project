# MlServer Project

DevOps pipeline to determine the performance of Continental Tire's and deployment using a XGBoost pre-packaged server to Seldon Core



## How to deploy to Seldon Core

#### Open a terminal and leave running the port-fowarding
````
kubectl port-forward -n istio-system svc/istio-ingressgateway 8080:80
````

#### Run the following commands in another terminal
````
kubectl delete namespace seldon-xgboost-namespace
kubectl create namespace seldon-xgboost-namespace
kubectl apply -f seldon-xgboost-deployment.yaml
kubectl get pods -n seldon-xgboost-namespace
```
