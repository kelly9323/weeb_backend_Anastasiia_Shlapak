from rest_framework import viewsets
from .models import Article
from .serializers import ArticleSerializer

# ViewSet for managing Articles
class ArticleViewSet(viewsets.ModelViewSet):
    queryset = Article.objects.all().order_by('-created_at')
    serializer_class = ArticleSerializer

    def perform_create(self, serializer):
        user = self.request.user
        author = f"{user.first_name} {user.last_name}".strip() or user.email
        serializer.save(author=author)

    def perform_update(self, serializer):
        serializer.save()