python ../src/train_teacher_mcdp_mnist.py --from-model 0

python ../src/amt_approx_mnist_mcdp_KL.py --from-approx-model 0 --test-ood-from-disk 0

# Running with EMD is slow and not scalable in this problem, so not suggested (also stated in Sec.4 experiment).
# We will add it later if readers are interested.

python ../src/amt_approx_mnist_mcdp_mmd.py --from-approx-model 0 --test-ood-from-disk 1
