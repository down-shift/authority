// Convertible: group-membership scope, action list, and a forbid.
@id("admins_edit")
permit (
    principal in Group::"admins",
    action in [Action::"view", Action::"edit"],
    resource is Doc
);

@id("no_archived_edit")
forbid (
    principal,
    action == Action::"edit",
    resource is Doc
)
when { resource.archived };
