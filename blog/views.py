from rest_framework import viewsets
from .models import Article
from .serializers import ArticleSerializer

# ViewSet for managing Articles
class ArticleViewSet(viewsets.ModelViewSet):
    queryset = Article.objects.all().order_by('-created_at')
    serializer_class = ArticleSerializer