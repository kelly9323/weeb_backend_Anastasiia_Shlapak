from rest_framework.permissions import BasePermission


class IsActiveMember(BasePermission):
    """
    Grants access only to authenticated users whose account has been activated by an admin.

    Usage:
        permission_classes = [IsActiveMember]
    """
    message = 'Your account is not active. Please contact an administrator.'

    def has_permission(self, request, view):
        return bool(
            request.user
            and request.user.is_authenticated
            and request.user.is_active
        )
