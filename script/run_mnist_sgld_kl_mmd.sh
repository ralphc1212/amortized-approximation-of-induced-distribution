python ../src/train_teacher_sgld_mnist.py --from-model 0

python ../src/amt_approx_mnist_sgld_KL.py --from-approx-model 0 --test-ood-from-disk 1

python ../src/amt_approx_mnist_sgld_mmd.py --from-approx-model 0 --test-ood-from-disk 1