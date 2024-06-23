.PHONY: fedzero

fedzero:
	python main.py --scenario global --dataset cifar100 --approach fedzero_1_1 --cpu
