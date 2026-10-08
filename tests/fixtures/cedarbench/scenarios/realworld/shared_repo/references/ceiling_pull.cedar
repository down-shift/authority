// Excluded: `in` over an entity-valued attribute, entity comparison, disjunction.
permit (
    principal is User,
    action == Action::"pull",
    resource is Repo
)
when { principal in resource.readers || resource.owner == principal };
