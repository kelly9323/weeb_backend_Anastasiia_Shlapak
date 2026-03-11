from rest_framework import serializers
from .models import Article

# Serializer for the Article model
class ArticleSerializer(serializers.ModelSerializer):
    class Meta:
        model = Article
        fields = ['id', 'title', 'author', 'content', 'created_at', 'updated_at']
