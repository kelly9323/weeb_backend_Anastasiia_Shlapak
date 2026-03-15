from rest_framework import viewsets, permissions
from rest_framework.exceptions import PermissionDenied
from .models import Article
from .serializers import ArticleSerializer


class IsAuthorOrReadOnly(permissions.BasePermission):
    def has_object_permission(self, request, view, obj):
        if request.method in permissions.SAFE_METHODS:
            return True
        return obj.author_user == request.user


# ViewSet for managing Articles
class ArticleViewSet(viewsets.ModelViewSet):
    queryset = Article.objects.all().order_by('-created_at')
    serializer_class = ArticleSerializer
    permission_classes = [permissions.IsAuthenticatedOrReadOnly, IsAuthorOrReadOnly]

    def perform_create(self, serializer):
        user = self.request.user
        author = f"{user.first_name} {user.last_name}".strip() or user.email
        serializer.save(author=author, author_user=user)

    def perform_update(self, serializer):
        serializer.save()