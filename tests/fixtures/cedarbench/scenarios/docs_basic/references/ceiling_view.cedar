// Convertible: conjunctive conditions, attribute-attribute comparison,
// a negated and a bare boolean (exact rewrites to == false / == true).
permit (
    principal is User,
    action == Action::"view",
    resource is Doc
)
when {
    principal.role == "editor" &&
    principal.level >= resource.classification &&
    !resource.archived &&
    context.mfa
};
