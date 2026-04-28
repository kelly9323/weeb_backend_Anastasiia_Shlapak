from django.contrib.auth import get_user_model
from django.test import TestCase
from django.urls import reverse

from rest_framework.test import APIRequestFactory, APITestCase
from rest_framework_simplejwt.tokens import RefreshToken

from .permissions import IsActiveMember
from .serializers import UserRegistrationSerializer

User = get_user_model()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_active_user(email='test@test.com', password='Str0ng!Pass99',
                     first_name='Ada', last_name='Lovelace'):
    user = User.objects.create_user(
        email=email, first_name=first_name, last_name=last_name,
        password=password,
    )
    user.is_active = True
    user.save()
    return user


def get_access_token(user):
    return str(RefreshToken.for_user(user).access_token)


def get_refresh_token(user):
    return str(RefreshToken.for_user(user))


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------

class UserModelTests(TestCase):

    def test_create_user_is_inactive_by_default(self):
        user = User.objects.create_user(
            email='inactive@test.com',
            first_name='A', last_name='B',
            password='Str0ng!Pass99',
        )
        self.assertFalse(user.is_active)

    def test_create_user_without_email_raises(self):
        with self.assertRaises(ValueError):
            User.objects.create_user(
                email='', first_name='A', last_name='B',
                password='Str0ng!Pass99',
            )

    def test_create_superuser_is_active_and_staff(self):
        user = User.objects.create_superuser(
            email='admin@test.com',
            first_name='A', last_name='B',
            password='Str0ng!Pass99',
        )
        self.assertTrue(user.is_active)
        self.assertTrue(user.is_staff)
        self.assertTrue(user.is_superuser)

    def test_email_unique_constraint(self):
        User.objects.create_user(
            email='dup@test.com', first_name='A', last_name='B',
            password='Str0ng!Pass99',
        )
        with self.assertRaises(Exception):
            User.objects.create_user(
                email='dup@test.com', first_name='C', last_name='D',
                password='Str0ng!Pass99',
            )


# ---------------------------------------------------------------------------
# Serializer tests
# ---------------------------------------------------------------------------

class UserRegistrationSerializerTests(TestCase):

    def _valid_data(self, email='new@test.com'):
        return {
            'email': email,
            'first_name': 'Ada',
            'last_name': 'Lovelace',
            'password': 'Str0ng!Pass99',
            'password_confirm': 'Str0ng!Pass99',
        }

    def test_valid_data_creates_inactive_user(self):
        s = UserRegistrationSerializer(data=self._valid_data())
        self.assertTrue(s.is_valid(), s.errors)
        user = s.save()
        self.assertIsInstance(user, User)
        self.assertFalse(user.is_active)

    def test_mismatched_passwords_invalid(self):
        data = self._valid_data()
        data['password_confirm'] = 'Different99!'
        s = UserRegistrationSerializer(data=data)
        self.assertFalse(s.is_valid())
        self.assertIn('password_confirm', str(s.errors))

    def test_duplicate_email_invalid(self):
        User.objects.create_user(
            email='taken@test.com', first_name='X', last_name='Y',
            password='Str0ng!Pass99',
        )
        s = UserRegistrationSerializer(data=self._valid_data(email='taken@test.com'))
        self.assertFalse(s.is_valid())
        self.assertIn('email', s.errors)

    def test_short_password_invalid(self):
        data = self._valid_data()
        data['password'] = 'abc'
        data['password_confirm'] = 'abc'
        s = UserRegistrationSerializer(data=data)
        self.assertFalse(s.is_valid())


# ---------------------------------------------------------------------------
# Register view tests
# ---------------------------------------------------------------------------

class RegisterViewTests(APITestCase):

    URL = reverse('users:register')

    def _valid_payload(self, email='reg@test.com'):
        return {
            'email': email,
            'first_name': 'Ada',
            'last_name': 'Lovelace',
            'password': 'Str0ng!Pass99',
            'password_confirm': 'Str0ng!Pass99',
        }

    def test_successful_registration_returns_201(self):
        response = self.client.post(self.URL, self._valid_payload())
        self.assertEqual(response.status_code, 201)
        self.assertIn('message', response.data)
        self.assertIn('email', response.data)
        user = User.objects.get(email='reg@test.com')
        self.assertFalse(user.is_active)

    def test_duplicate_email_returns_400(self):
        User.objects.create_user(
            email='dup@test.com', first_name='A', last_name='B',
            password='Str0ng!Pass99',
        )
        response = self.client.post(self.URL, self._valid_payload(email='dup@test.com'))
        self.assertEqual(response.status_code, 400)

    def test_missing_email_returns_400(self):
        data = self._valid_payload()
        del data['email']
        response = self.client.post(self.URL, data)
        self.assertEqual(response.status_code, 400)
        self.assertIn('email', response.data)

    def test_password_mismatch_returns_400(self):
        data = self._valid_payload()
        data['password_confirm'] = 'Different99!'
        response = self.client.post(self.URL, data)
        self.assertEqual(response.status_code, 400)


# ---------------------------------------------------------------------------
# Login view tests
# ---------------------------------------------------------------------------

class LoginViewTests(APITestCase):

    URL = reverse('users:login')

    def setUp(self):
        self.user = make_active_user()

    def test_active_user_login_returns_200_with_token_and_cookie(self):
        response = self.client.post(self.URL, {
            'email': 'test@test.com',
            'password': 'Str0ng!Pass99',
        })
        self.assertEqual(response.status_code, 200)
        self.assertIn('access', response.data)
        self.assertIn('user', response.data)
        self.assertEqual(response.data['user']['email'], 'test@test.com')
        self.assertIn('refresh_token', response.cookies)
        self.assertTrue(response.cookies['refresh_token']['httponly'])

    def test_inactive_user_login_returns_401(self):
        User.objects.create_user(
            email='inactive@test.com', first_name='A', last_name='B',
            password='Str0ng!Pass99',
        )
        response = self.client.post(self.URL, {
            'email': 'inactive@test.com',
            'password': 'Str0ng!Pass99',
        })
        self.assertEqual(response.status_code, 401)

    def test_wrong_password_returns_401(self):
        response = self.client.post(self.URL, {
            'email': 'test@test.com',
            'password': 'WrongPass99!',
        })
        self.assertEqual(response.status_code, 401)


# ---------------------------------------------------------------------------
# Cookie token refresh view tests
# ---------------------------------------------------------------------------

class CookieTokenRefreshViewTests(APITestCase):

    URL = reverse('users:token-refresh')

    def setUp(self):
        self.user = make_active_user()

    def test_valid_cookie_returns_new_access_token(self):
        refresh = get_refresh_token(self.user)
        self.client.cookies['refresh_token'] = refresh
        response = self.client.post(self.URL)
        self.assertEqual(response.status_code, 200)
        self.assertIn('access', response.data)
        self.assertIn('refresh_token', response.cookies)

    def test_missing_cookie_returns_401(self):
        response = self.client.post(self.URL)
        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.data['detail'], 'Refresh token missing.')

    def test_invalid_token_returns_401(self):
        self.client.cookies['refresh_token'] = 'notavalidtoken'
        response = self.client.post(self.URL)
        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.data['detail'], 'Invalid or expired refresh token.')


# ---------------------------------------------------------------------------
# Logout view tests
# ---------------------------------------------------------------------------

class LogoutViewTests(APITestCase):

    URL = reverse('users:logout')

    def setUp(self):
        self.user = make_active_user()

    def test_authenticated_logout_returns_205_and_clears_cookie(self):
        access = get_access_token(self.user)
        refresh = get_refresh_token(self.user)
        self.client.credentials(HTTP_AUTHORIZATION=f'Bearer {access}')
        self.client.cookies['refresh_token'] = refresh
        response = self.client.post(self.URL)
        self.assertEqual(response.status_code, 205)
        # Cookie should be cleared (empty value after delete_cookie)
        cookie = response.cookies.get('refresh_token')
        self.assertTrue(cookie is None or cookie.value == '')

    def test_unauthenticated_returns_401(self):
        response = self.client.post(self.URL)
        self.assertEqual(response.status_code, 401)

    def test_missing_refresh_cookie_returns_400(self):
        access = get_access_token(self.user)
        self.client.credentials(HTTP_AUTHORIZATION=f'Bearer {access}')
        response = self.client.post(self.URL)
        self.assertEqual(response.status_code, 400)


# ---------------------------------------------------------------------------
# IsActiveMember permission tests
# ---------------------------------------------------------------------------

class IsActiveMemberPermissionTests(TestCase):

    def setUp(self):
        self.factory = APIRequestFactory()
        self.permission = IsActiveMember()

    def _make_request(self, user):
        request = self.factory.get('/')
        request.user = user
        return request

    def test_active_user_has_permission(self):
        user = make_active_user(email='active@test.com')
        request = self._make_request(user)
        self.assertTrue(self.permission.has_permission(request, None))

    def test_inactive_user_denied(self):
        user = User.objects.create_user(
            email='inactive@test.com', first_name='A', last_name='B',
            password='Str0ng!Pass99',
        )
        request = self._make_request(user)
        self.assertFalse(self.permission.has_permission(request, None))

    def test_anonymous_user_denied(self):
        from django.contrib.auth.models import AnonymousUser
        request = self._make_request(AnonymousUser())
        self.assertFalse(self.permission.has_permission(request, None))
