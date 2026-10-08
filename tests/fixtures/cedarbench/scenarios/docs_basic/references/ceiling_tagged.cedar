// Excluded: set membership, `has` on an optional attribute, and `unless`.
permit (
    principal is User,
    action == Action::"view",
    resource is Doc
)
when { resource.tags.contains("public") || (principal has nickname && principal.nickname == "boss") }
unless { resource.archived };
