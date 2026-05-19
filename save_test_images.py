import torchvision
import os

dataset = torchvision.datasets.CIFAR100(root='data', train=False, download=True)
os.makedirs('test_images', exist_ok=True)

demo_classes = ['cat', 'butterfly', 'rose', 'shark', 'elephant', 
                'bicycle', 'mushroom', 'sunflower', 'tiger', 'dolphin']

saved = {c: 0 for c in demo_classes}
for i in range(len(dataset)):
    img, label = dataset[i]
    class_name = dataset.classes[label]
    if class_name in demo_classes and saved[class_name] < 5:
        fname = f'test_images/{class_name}_{saved[class_name]+1}.png'
        img.save(fname)
        saved[class_name] += 1
    if all(v == 5 for v in saved.values()):
        break

print(f"Saved {sum(saved.values())} images to test_images/")
for cls, count in saved.items():
    print(f"  {cls}: {count} images")
