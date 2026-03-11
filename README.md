# Weeb Backend — Django 5 + Django REST Framework API
## Apps
- **blog**: CRUD for articles (`ArticleViewSet`)
- **contact**: Contact form submissions + ML satisfaction scoring
---
## Requirements
Python 3.11
- Create virtual environment and install dependencies: 
```bash
python -m venv .venv
pip install -r requirements.txt
```
---
## Database Setup
```bash
python manage.py makemigrations
python manage.py migrate
```
---
## Run Locally
```bash
python manage.py runserver
```
---
## Machine Learning — Contact Satisfaction
Training script: `contact/train_model.py`
Models tested: Logistic Regression, Decision Tree, Random Forest (TF-IDF + classifier pipeline)
Best model saved to: `contact/satisfaction_model.joblib`
Train via shell:
```bash
python manage.py shell
```
Shell command:
```bash
from contact.train_model import train_model
train_model()
```
---
## API Endpoints
```bash
GET /api/articles/
POST /api/articles/
GET /api/articles/<id>/
PUT /api/articles/<id>/
DELETE /api/articles/<id>/
POST /api/contact/ → returns satisfaction (0 or 1)
```

