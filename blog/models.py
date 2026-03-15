from django.db import models
from django.conf import settings

# Article model representing a blog article
class Article(models.Model):
    title = models.CharField(max_length=255)
    author = models.CharField(max_length=100, default='Anonymous')
    author_user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='articles',
    )
    content = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.title
