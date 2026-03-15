from rest_framework import serializers
from .models import Article

# Serializer for the Article model
class ArticleSerializer(serializers.ModelSerializer):
    class Meta:
        model = Article
        fields = ['id', 'title', 'author', 'author_user', 'content', 'created_at', 'updated_at']
        read_only_fields = ['author', 'author_user']
