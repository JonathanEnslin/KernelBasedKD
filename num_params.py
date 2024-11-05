from models.resnet import *

r20 = resnet20(num_classes=10)
r56 = resnet56(num_classes=10)
r110 = resnet110(num_classes=10)

total_params = sum(p.numel() for p in r20.parameters())
print(f"Total parameters for resnet20: {total_params}")
total_params = sum(p.numel() for p in r56.parameters())
print(f"Total parameters for resnet56: {total_params}")
total_params = sum(p.numel() for p in r110.parameters())
print(f"Total parameters for resnet110: {total_params}")

r20 = resnet20(num_classes=100)
r56 = resnet56(num_classes=100)
r110 = resnet110(num_classes=100)


total_params = sum(p.numel() for p in r20.parameters())
print(f"Total parameters for resnet20: {total_params}")
total_params = sum(p.numel() for p in r56.parameters())
print(f"Total parameters for resnet56: {total_params}")
total_params = sum(p.numel() for p in r110.parameters())
print(f"Total parameters for resnet110: {total_params}")

r20 = resnet20(num_classes=200)
r56 = resnet56(num_classes=200)
r110 = resnet110(num_classes=200)

total_params = sum(p.numel() for p in r20.parameters())
print(f"Total parameters for resnet20: {total_params}")
total_params = sum(p.numel() for p in r56.parameters())
print(f"Total parameters for resnet56: {total_params}")
total_params = sum(p.numel() for p in r110.parameters())
print(f"Total parameters for resnet110: {total_params}")
