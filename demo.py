import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50, vit_b_32
from transformers import LevitForImageClassification, logging

logging.set_verbosity_error()

class PlantDiseaseClassifier:
    class_names = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']
    def __init__(self, model_type, model_path, batch_size=32):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        self.model_type = model_type

        # Initialize and load the appropriate model
        self.model = self._load_model(model_type, model_path)
        self.model = self.model.to(self.device)
        self.model.eval()

        # Data transformation pipeline
        self.data_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def _load_model(self, model_type, model_path):
        if model_type == "resnet":
            model = resnet50(pretrained=False)
            model.fc = torch.nn.Linear(model.fc.in_features, len(self.class_names))
            model.load_state_dict(torch.load(model_path, map_location=self.device))
        elif model_type == "levit":
            model = LevitForImageClassification.from_pretrained(
                "facebook/levit-128S", num_labels=len(self.class_names), ignore_mismatched_sizes=True,
            )
            state_dict = torch.load(model_path, map_location=self.device)
            filtered_state_dict = {k: v for k, v in state_dict.items() if not k.startswith("classifier_distill")}
            model.load_state_dict(filtered_state_dict)
        elif model_type == "vit":
            model = vit_b_32(pretrained=False, num_classes=len(self.class_names))
            model.load_state_dict(torch.load(model_path, map_location=self.device))
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        return model

    class _PlantDiseaseDataset(Dataset):
        def __init__(self, directory_path, transform=None):
            self.directory_path = directory_path
            self.transform = transform

            # Collect all images and their respective class labels
            self.image_files = []
            self.labels = []
            for class_name in os.listdir(directory_path):
                class_dir = os.path.join(directory_path, class_name)
                if os.path.isdir(class_dir) and class_name in PlantDiseaseClassifier.class_names:
                    for img_file in os.listdir(class_dir):
                        if img_file.lower().endswith(('.jpg', '.png')):
                            self.image_files.append(os.path.join(class_dir, img_file))
                            self.labels.append(PlantDiseaseClassifier.class_names.index(class_name))

        def __len__(self):
            return len(self.image_files)

        def __getitem__(self, idx):
            img_path = self.image_files[idx]
            image = Image.open(img_path).convert('RGB')
            label = self.labels[idx]
            if self.transform:
                image = self.transform(image)
            return image, label

    def calculate_accuracy(self, test_dir):
        dataset = self._PlantDiseaseDataset(test_dir, transform=self.data_transforms)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(inputs)
                logits = outputs.logits if self.model_type in ["levit"] else outputs
                _, preds = torch.max(logits, 1)

                correct += (preds == labels).sum().item()
                total += labels.size(0)

        accuracy = (correct / total) * 100 if total > 0 else 0.0
        return accuracy

    def predict_image(self, image_path):
        # Load and transform the image
        image = Image.open(image_path).convert('RGB')
        transformed_image = self.data_transforms(image).unsqueeze(0).to(self.device)

        # Predict
        with torch.no_grad():
            outputs = self.model(transformed_image)
            logits = outputs.logits if self.model_type in ["levit"] else outputs
            _, predicted_idx = torch.max(logits, 1)

        predicted_class = self.class_names[predicted_idx.item()]
        return predicted_class

def predict_image_with_all_models(image_path, classifiers):
    actual_disease = os.path.basename(os.path.dirname(image_path))
    print(f"Actual disease: {actual_disease}\n")
    for model_name, classifier in classifiers.items():
        predicted_class = classifier.predict_image(image_path)
        print(f"Model: {model_name}, Predicted Class: {predicted_class}")


if __name__ == "__main__":
    model_types = ["resnet", "resnet", "vit", "vit", "levit", "levit"]
    model_paths = ["resnet50_org.pth", "resnet50_aug.pth", "vit32b_org.pth", "vit32b_aug.pth", "levit_org.pth", "levit_aug.pth"]
    model_names = [path.split('.')[0] for path in model_paths]
    test_data_dir = "test"

    # Initialize classifiers
    classifiers = {
      model_name: PlantDiseaseClassifier(model_type, model_path, batch_size=32)
      for model_name, model_type, model_path in zip(model_names, model_types, model_paths)
    }

    for model_name, classifier in classifiers.items():
        accuracy = classifier.calculate_accuracy(test_data_dir)
        print(f"Model: {model_name}, Test Accuracy: {accuracy:.2f}%")

