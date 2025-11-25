from rest_framework import generics, status
from rest_framework.response import Response
from .models import ContactMessage
from .serializers import ContactSerializer
import joblib
import os


class ContactCreateView(generics.CreateAPIView):
    queryset = ContactMessage.objects.all().order_by('-created_at')
    serializer_class = ContactSerializer

    def create(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        message_text = serializer.validated_data['message']
        satisfaction = predict_satisfaction(message_text)
        instance = serializer.save(satisfaction=satisfaction)
        return Response(ContactSerializer(instance).data, status=status.HTTP_201_CREATED)

_satisfaction_model = None

def load_model():
    """Charge le modèle de satisfaction (chargé une seule fois)"""
    global _satisfaction_model
    if _satisfaction_model is None:
        model_path = os.path.join(
            os.path.dirname(__file__),
            'satisfaction_model.joblib'
        )
        if os.path.exists(model_path):
            _satisfaction_model = joblib.load(model_path)
        else:
            # Si le modèle n'existe pas, retourner 0 par défaut
            print(f"Attention: Modèle non trouvé dans {model_path}")
            print("Exécutez 'python manage.py shell' puis 'from contact.train_model import train_model; train_model()' pour entraîner le modèle")
    return _satisfaction_model

def predict_satisfaction(message):
    """
    Prédit la satisfaction à partir du message
    Retourne 1 si satisfait, 0 sinon
    """
    model = load_model()
    
    if model is None:
        return 0
    
    try:
        prediction = model.predict([str(message)])
        return int(prediction[0])
    except Exception as e:
        print(f"Erreur lors de la prédiction: {e}")
        return 0