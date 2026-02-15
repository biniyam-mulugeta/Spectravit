from torchvision import transforms

def get_transforms(img_size: int = 224):
    """
    Returns (train_transform, val_transform, normalize)
    Mixup typically happens after these but before normalization in the logic if using manual mixup,
    but here we provide the standard augmentation chain.
    """
    pre_train = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.ToTensor(),
    ])

    pre_val = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])

    # Standard ImageNet normalization
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    return pre_train, pre_val, normalize
