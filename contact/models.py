from django.db import models

class ContactMessage(models.Model):
    name = models.CharField(max_length=200)
    email = models.EmailField()
    message = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    satisfaction = models.IntegerField(null=True, blank=True)

    def __str__(self):
        return self.email

