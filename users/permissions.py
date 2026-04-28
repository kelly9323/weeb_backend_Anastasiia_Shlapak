from rest_framework.permissions import BasePermission


class IsActiveMember(BasePermission):
    """
    Grants access only to authenticated users whose account has been activated by an admin.

    Usage:
        permission_classes = [IsActiveMember]
    """
    message = 'Le compte doit être activé par un administrateur pour accéder à cette ressource.'

    def has_permission(self, request, view):
        return bool(
            request.user
            and request.user.is_authenticated
            and request.user.is_active
        )
