// Convertible: a single conditioned permit.
permit (
    principal is User,
    action == Action::"pull",
    resource is Repo
)
when { resource.isPrivate == false };
