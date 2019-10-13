python ../src/train_teacher_mcdp_mnist.py --from-model 0

python ../src/amt_approx_mnist_mcdp_KL.py --from-approx-model 0 --test-ood-from-disk 0

python ../src/amt_approx_mnist_mcdp_mmd.py --from-approx-model 0 --test-ood-from-disk 1

# As the previous implementation of EMD is not scalable (not suggested in the paper),
# we provides a more scalable way using the Sinkhorn divergence which approximates EMD.
python ../src/amt_approx_mnist_mcdp_emd.py --from-approx-model 0 --test-ood-from-disk 1