.PHONY: fedzero fedzero-gpu upper-limit

fedzero:
	python main.py --scenario global --dataset cifar100 --approach fedzero_1_1 --cpu

fedzero-gpu:
	python main.py --scenario global --dataset cifar100 --approach fedzero_1_1

upper-limit:
	python main.py --scenario unconstrained --dataset cifar100 --approach random