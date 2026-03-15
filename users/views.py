from django.conf import settings
from django.contrib.auth import get_user_model
from rest_framework import generics, status
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework_simplejwt.views import TokenObtainPairView
from rest_framework_simplejwt.tokens import RefreshToken
from rest_framework_simplejwt.exceptions import TokenError, InvalidToken

from .serializers import (
    UserRegistrationSerializer,
    UserInfoSerializer,
    PasswordResetRequestSerializer,
    PasswordResetConfirmSerializer,
)

User = get_user_model()


class RegisterView(generics.CreateAPIView):
    """
    POST /api/users/register/
    Creates an inactive user account. An admin must activate it before login is possible.
    """
    queryset = User.objects.all()
    serializer_class = UserRegistrationSerializer
    permission_classes = [AllowAny]

    def create(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        user = serializer.save()
        return Response(
            {
                'message': 'Registration successful. Your account is pending activation by an administrator.',
                'email': user.email,
            },
            status=status.HTTP_201_CREATED,
        )


class CustomTokenObtainPairView(TokenObtainPairView):
    """
    POST /api/users/login/
    Returns access token and user info in body. Sets refresh token as HttpOnly cookie.
    simplejwt rejects inactive users automatically (AuthenticationFailed).
    """
    permission_classes = [AllowAny]

    def post(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        try:
            serializer.is_valid(raise_exception=True)
        except TokenError as e:
            raise InvalidToken(e.args[0])

        user = serializer.user
        response = Response(
            {
                'access': serializer.validated_data['access'],
                'user': UserInfoSerializer(user).data,
            },
            status=status.HTTP_200_OK,
        )
        response.set_cookie(
            key='refresh_token',
            value=serializer.validated_data['refresh'],
            httponly=True,
            secure=not settings.DEBUG,
            samesite='None' if not settings.DEBUG else 'Lax',
            max_age=7 * 24 * 60 * 60,
            path='/',
        )
        return response


class CookieTokenRefreshView(APIView):
    """
    POST /api/users/token/refresh/
    Reads refresh token from HttpOnly cookie, returns new access token.
    Sets a new refresh cookie (token rotation).
    """
    permission_classes = [AllowAny]

    def post(self, request):
        refresh_token = request.COOKIES.get('refresh_token')
        if not refresh_token:
            return Response({'detail': 'Refresh token missing.'}, status=status.HTTP_401_UNAUTHORIZED)
        try:
            token = RefreshToken(refresh_token)
            access = str(token.access_token)
            new_refresh = str(token)
        except TokenError:
            return Response({'detail': 'Invalid or expired refresh token.'}, status=status.HTTP_401_UNAUTHORIZED)

        response = Response({'access': access}, status=status.HTTP_200_OK)
        response.set_cookie(
            key='refresh_token',
            value=new_refresh,
            httponly=True,
            secure=not settings.DEBUG,
            samesite='None' if not settings.DEBUG else 'Lax',
            max_age=7 * 24 * 60 * 60,
            path='/',
        )
        return response


class LogoutView(APIView):
    """
    POST /api/users/logout/
    Blacklists the refresh token. The short-lived access token expires naturally.
    Body: { "refresh": "<refresh_token>" }
    """
    permission_classes = [IsAuthenticated]

    def post(self, request):
        refresh_token = request.COOKIES.get('refresh_token')
        if not refresh_token:
            return Response(status=status.HTTP_400_BAD_REQUEST)
        try:
            token = RefreshToken(refresh_token)
            token.blacklist()
        except TokenError:
            return Response(status=status.HTTP_400_BAD_REQUEST)
        response = Response(status=status.HTTP_205_RESET_CONTENT)
        response.delete_cookie('refresh_token', path='/')
        return response


class PasswordResetRequestView(APIView):
    """
    POST /api/users/password-reset/
    Sends a password reset email. Always returns 200 to prevent user enumeration.
    Body: { "email": "user@example.com" }
    """
    permission_classes = [AllowAny]

    def post(self, request):
        serializer = PasswordResetRequestSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        serializer.save()
        return Response(status=status.HTTP_200_OK)



class PasswordResetConfirmView(APIView):
    """
    POST /api/users/password-reset/confirm/
    Validates the uid + token and sets the new password.
    Body: { "uid": "...", "token": "...", "new_password": "...", "new_password_confirm": "..." }
    """
    permission_classes = [AllowAny]

    def post(self, request):
        serializer = PasswordResetConfirmSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        serializer.save()
        return Response(status=status.HTTP_200_OK)

