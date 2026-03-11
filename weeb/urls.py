from django.contrib import admin
from django.urls import path, include
from rest_framework.routers import DefaultRouter
from blog.views import ArticleViewSet
from contact.views import ContactCreateView

router = DefaultRouter()
router.register(r'articles', ArticleViewSet, basename='article')

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/', include(router.urls)),
    path('api/contact/', ContactCreateView.as_view(), name='contact-create'),
    path('api/users/', include('users.urls', namespace='users')),
]
