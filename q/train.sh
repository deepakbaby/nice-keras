#!/bin/bash
cd /remote/idiap.svm/user.active/dbaby/github/nice-keras
. ./path.sh
( echo '#' Running on `hostname`
  echo '#' Started at `date`
  echo -n '# '; cat <<EOF
nice_mnist.py 
EOF
) >train.log
time1=`date +"%s"`
 ( nice_mnist.py  ) 2>>train.log >>train.log
ret=$?
time2=`date +"%s"`
echo '#' Accounting: time=$(($time2-$time1)) threads=1 >>train.log
echo '#' Finished at `date` with status $ret >>train.log
[ $ret -eq 137 ] && exit 100;
touch ./q/sync/done.31054
exit $[$ret ? 1 : 0]
## submitted with:
# qsub -v PATH -cwd -S /bin/bash -j y -l arch=*64* -o ./q/train.log -l q_gpu -l cuda9 -v LD_LIBRARY_PATH    /remote/idiap.svm/user.active/dbaby/github/nice-keras/./q/train.sh >>./q/train.log 2>&1
