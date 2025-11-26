from django.db import models

# Article model representing a blog article
class Article(models.Model):
    title = models.CharField(max_length=255)
    author = models.CharField(max_length=100, default='Anonymous')
    content = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.title
