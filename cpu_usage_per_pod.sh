# get pods
kubectl get pods | tail -n+2 > /tmp/pods 
# print cpu usage 
while read p; do 
	POD=$(echo $p | head -n1 | awk '{print $1;}') 
	USAGE=$(kubectl exec $POD grep 'cpu ' /proc/stat | awk '{usage=($2+$4)*100/($2+$4+$5)} END {print usage "%"}'); 
	echo $POD $USAGE; 
done < /tmp/pods
