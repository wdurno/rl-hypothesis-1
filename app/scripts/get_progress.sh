MASTER=$(kubectl get pods | tail -n+2 | grep master | awk '{print $1;}')
kubectl exec $MASTER cat logs
