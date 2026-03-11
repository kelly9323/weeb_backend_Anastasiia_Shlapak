from rest_framework import serializers
from .models import ContactMessage

# Serializer for the ContactMessage model
class ContactSerializer(serializers.ModelSerializer):
    class Meta:
        model = ContactMessage
        fields = ['id', 'first_name', 'last_name', 'phone', 'email', 'message', 'created_at', 'satisfaction']
        read_only_fields = ['satisfaction','created_at']
