from django.contrib.auth import get_user_model
from django.test import TestCase
from django.urls import reverse

from rest_framework.test import APIRequestFactory, APITestCase
from rest_framework_simplejwt.tokens import RefreshToken

from .models import Article
from .serializers import ArticleSerializer
from .views import IsAuthorOrReadOnly

User = get_user_model()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_active_user(email='user@test.com', password='Str0ng!Pass99',
                     first_name='Ada', last_name='Lovelace'):
    user = User.objects.create_user(
        email=email, first_name=first_name, last_name=last_name,
        password=password,
    )
    user.is_active = True
    user.save()
    return user


def auth_client(client, user):
    access = str(RefreshToken.for_user(user).access_token)
    client.credentials(HTTP_AUTHORIZATION=f'Bearer {access}')


def make_article(title='Test Article', content='Some content', author_user=None):
    return Article.objects.create(
        title=title, content=content, author_user=author_user,
    )


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------

class ArticleModelTests(TestCase):

    def test_author_defaults_to_anonymous(self):
        article = Article.objects.create(title='T', content='C')
        self.assertEqual(article.author, 'Anonymous')

    def test_author_user_set_null_on_user_delete(self):
        user = make_active_user()
        article = Article.objects.create(
            title='T', content='C', author_user=user,
        )
        user.delete()
        article.refresh_from_db()
        self.assertIsNone(article.author_user)


# ---------------------------------------------------------------------------
# Serializer tests
# ---------------------------------------------------------------------------

class ArticleSerializerTests(TestCase):

    def test_author_and_author_user_read_only(self):
        data = {'title': 'T', 'content': 'C', 'author': 'Injected', 'author_user': 999}
        s = ArticleSerializer(data=data)
        self.assertTrue(s.is_valid(), s.errors)
        self.assertNotIn('author', s.validated_data)
        self.assertNotIn('author_user', s.validated_data)

    def test_title_required(self):
        s = ArticleSerializer(data={'content': 'C'})
        self.assertFalse(s.is_valid())
        self.assertIn('title', s.errors)

    def test_content_required(self):
        s = ArticleSerializer(data={'title': 'T'})
        self.assertFalse(s.is_valid())
        self.assertIn('content', s.errors)


# ---------------------------------------------------------------------------
# List / Create view tests
# ---------------------------------------------------------------------------

class ArticleListCreateViewTests(APITestCase):

    LIST_URL = reverse('article-list')

    def test_list_unauthenticated_returns_200(self):
        make_article('A1')
        make_article('A2')
        response = self.client.get(self.LIST_URL)
        self.assertEqual(response.status_code, 200)
        self.assertIn('results', response.data)
        self.assertEqual(response.data['count'], 2)

    def test_list_empty_returns_200(self):
        response = self.client.get(self.LIST_URL)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data['count'], 0)

    def test_create_authenticated_sets_author_from_fullname(self):
        user = make_active_user(first_name='Ada', last_name='Lovelace')
        auth_client(self.client, user)
        response = self.client.post(self.LIST_URL, {'title': 'My Article', 'content': 'Body'})
        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.data['author'], 'Ada Lovelace')
        self.assertEqual(response.data['author_user'], user.pk)

    def test_create_author_falls_back_to_email_if_names_blank(self):
        user = make_active_user(
            email='noname@test.com', first_name='', last_name='',
        )
        auth_client(self.client, user)
        response = self.client.post(self.LIST_URL, {'title': 'T', 'content': 'C'})
        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.data['author'], 'noname@test.com')

    def test_create_unauthenticated_returns_401(self):
        response = self.client.post(self.LIST_URL, {'title': 'T', 'content': 'C'})
        self.assertEqual(response.status_code, 401)


# ---------------------------------------------------------------------------
# Detail view tests
# ---------------------------------------------------------------------------

class ArticleDetailViewTests(APITestCase):

    def setUp(self):
        self.user_a = make_active_user(email='a@test.com')
        self.user_b = make_active_user(email='b@test.com')
        self.article = Article.objects.create(
            title='Article A', content='Content A', author_user=self.user_a,
        )
        self.url = reverse('article-detail', kwargs={'pk': self.article.pk})

    def test_retrieve_unauthenticated_returns_200(self):
        response = self.client.get(self.url)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data['title'], 'Article A')

    def test_update_own_article_returns_200(self):
        auth_client(self.client, self.user_a)
        response = self.client.patch(self.url, {'title': 'Updated'})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data['title'], 'Updated')

    def test_update_other_user_article_returns_403(self):
        auth_client(self.client, self.user_b)
        response = self.client.patch(self.url, {'title': 'Hacked'})
        self.assertEqual(response.status_code, 403)

    def test_delete_own_article_returns_204(self):
        auth_client(self.client, self.user_a)
        response = self.client.delete(self.url)
        self.assertEqual(response.status_code, 204)
        self.assertFalse(Article.objects.filter(pk=self.article.pk).exists())

    def test_delete_other_user_article_returns_403(self):
        auth_client(self.client, self.user_b)
        response = self.client.delete(self.url)
        self.assertEqual(response.status_code, 403)
        self.assertTrue(Article.objects.filter(pk=self.article.pk).exists())


# ---------------------------------------------------------------------------
# IsAuthorOrReadOnly permission tests
# ---------------------------------------------------------------------------

class IsAuthorOrReadOnlyPermissionTests(TestCase):

    def setUp(self):
        self.factory = APIRequestFactory()
        self.permission = IsAuthorOrReadOnly()
        self.user_a = make_active_user(email='perm_a@test.com')
        self.user_b = make_active_user(email='perm_b@test.com')
        self.article = Article.objects.create(
            title='T', content='C', author_user=self.user_a,
        )

    def _make_request(self, method, user):
        req_fn = getattr(self.factory, method.lower())
        request = req_fn('/')
        request.user = user
        return request

    def test_safe_method_always_allowed(self):
        request = self._make_request('GET', self.user_b)
        self.assertTrue(self.permission.has_object_permission(request, None, self.article))

    def test_write_allowed_for_author(self):
        request = self._make_request('PATCH', self.user_a)
        self.assertTrue(self.permission.has_object_permission(request, None, self.article))

    def test_write_denied_for_non_author(self):
        request = self._make_request('PATCH', self.user_b)
        self.assertFalse(self.permission.has_object_permission(request, None, self.article))
