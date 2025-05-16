🛍️ Multimodal AI for Product Detection and Recommendation

Fine-Tuned ResNet-50 + YOLOv8 + NLP Chatbot (2.5M+ Products)

This is a full-scale AI solution combining image classification, object detection, and natural language processing (NLP) to detect and recommend fashion products from large e-commerce datasets. The system was trained on over 2.5 million product images, covering 30+ product categories, and includes an intelligent chatbot interface for product search.

🚀 Project Highlights
🧠 ResNet-50 Fine-Tuning
Replaced the default classifier with a custom head for 30+ category classification. Achieved ~80% accuracy on a noisy real-world dataset.

🔎 YOLOv8 Object Detection
Enhanced image localization and bounding box prediction for product tagging and extraction.

💬 NLP-Powered Chatbot
Built using NLTK and rule-based NLP to understand customer queries like:

“Show me jeans under 1000”
“List black shirts above 500 with hood”

🧠 Multimodal Fusion
Integrated vision + language understanding to match user queries with product images and metadata.

📈 Scalable Training Pipeline
Efficiently trained on 2.5M+ product images with batch augmentation, custom loss functions, and label smoothing.

🧰 Technologies Used
Component	Framework / Library
Image Classification	TensorFlow/Keras, ResNet-50
Object Detection	YOLOv8 (Ultralytics)
NLP/NLU	NLTK, Regex, Custom Intent Parser
Dataset Management	Pandas, NumPy, OpenCV
Visualization	Matplotlib, Seaborn
Deployment	Python CLI / Web-ready backend

📦 Dataset Summary
Source: Amazon Product Dataset (internal/local structured format)

Size: ~2.5 million labeled images

Categories: 30+ (e.g., Shirts, Pants, Shoes, Jackets, Bags, etc.)

Challenges: Label noise, occlusion, low-resolution images, inconsistent lighting

🧠 Model Architecture
🔹 ResNet-50 (Backbone)
Frozen early layers

Removed final dense layer

Added:

GlobalAveragePooling2D

Dense layers with Dropout

Final softmax for 30+ class output

🔹 YOLOv8 (Detection)
Used pretrained weights (yolov8n.pt)

Fine-tuned on cropped product regions for bounding boxes

Used to auto-tag and refine regions before classification

💬 Chatbot Interface
The chatbot understands natural language for filtering and retrieval:

Examples:

Show me t-shirts below 1000

I want blue jeans above 2000

Display winter jackets under 5000

How it works:

Tokenizes and parses query using NLTK

Extracts intent, category, price range,rating and color

Maps to filtered results using metadata

📈 Results
Task	Accuracy / mAP
ResNet-50 Classification	~80% Top-1 Accuracy
YOLOv8 Detection	~84.3% mAP (IoU=0.5)
Chatbot Intent Detection	~92% Precision

🗂️ Folder Structure
kotlin
Copy
Edit

📍 How to Run
🔧 Installation
bash
Copy
Edit
git clone https://github.com/Muhammad-junaid-mujtaba/Multimodal-Product-Classification-Query-System.git
cd multimodal-ai-product-detection
pip install -r requirements.txt
🧠 Run Image Classifier
bash
Copy
Edit
python classification/train_classifier.py
🔍 Run YOLOv8 Detection
bash
Copy
Edit
python detection/yolov8_train.py
💬 Start Chatbot
bash
Copy
Edit
python chatbot/chatbot_engine.py

📬 Contact
Author: Junaid

Email:junaidqazi705@gmail.com

LinkedIn: https://www.linkedin.com/in/junaid-qazi-4a5299323/


