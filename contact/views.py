from rest_framework import generics, status
from rest_framework.response import Response
from .models import ContactMessage
from .serializers import ContactSerializer
import joblib
import os

# View for creating ContactMessage with satisfaction prediction
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
    """
    Load the pre-trained satisfaction prediction model.
    """
    global _satisfaction_model
    if _satisfaction_model is None:
        model_path = os.path.join(
            os.path.dirname(__file__),
            'satisfaction_model.joblib'
        )
        if os.path.exists(model_path):
            _satisfaction_model = joblib.load(model_path)
        else:
            # If the model file does not exist, log an error
            print(f"Modèle non trouvé dans {model_path}")
    return _satisfaction_model

def predict_satisfaction(message):
    """
    Predict satisfaction score for a given message.
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