from typing import List

USER_COLS = [
    "COUNTRY",
    "REGION",
    "LATEST_CLICK_CLIENT_TYPE",
    "LATEST_CLICK_CLIENT_NAME",
    "LATEST_CLICK_CLIENT_OS_FAMILY",
    "TOTAL_ORDERS_VALUE",
    "AVG_ORDER_VALUE",
    "LAST_ORDER_VALUE",
    "MONTHS_SINCE_FIRST_ACTIVE",
    "CLICK",
    "FIRST_UTM_SOURCE",
    "FIRST_UTM_CONTENT",
    "FIRST_UTM_CAMPAIGN",
    "LAST_UTM_SOURCE",
    "LAST_UTM_CONTENT",
    "LAST_UTM_CAMPAIGN",
    "CITY",
    "TIMEZONE",
]
VARIATION_COLS = [
    "Q1_CREATIVE",
    "Q2_CREATIVE",
    "Q3",
    "Q4",
    "Q5",
    "Q6",
    "Q7",
    "Q8",
    "Q9",
    "Q10",
    "Q11",
    "Q12",
    "Q13",
    "Q14",
    "Q15",
    "Q16",
]
SUBJECT_LINE_COLS = [
    "Q1_SBL",
    "Q2_SBL",
]

CATEGORICAL_COLS = [
    "COUNTRY",
    "REGION",
    "CITY",
    "TIMEZONE",
    "LATEST_CLICK_CLIENT_TYPE",
    "LATEST_CLICK_CLIENT_NAME",
    "LATEST_CLICK_CLIENT_OS_FAMILY",
    "FIRST_UTM_SOURCE",
    "FIRST_UTM_CONTENT",
    "FIRST_UTM_CAMPAIGN",
    "LAST_UTM_SOURCE",
    "LAST_UTM_CONTENT",
    "LAST_UTM_CAMPAIGN",
    "Q1_CREATIVE",
    "Q2_CREATIVE",
    "Q3",
    "Q4",
    "Q5",
    "Q6",
    "Q7",
    "Q8",
    "Q9",
    "Q10",
    "Q11",
    "Q12",
    "Q13",
    "Q14",
    "Q15",
    "Q16",
    "Q1_SBL",
    "Q2_SBL",
]

NUMERICAL_COLS = [
    "TOTAL_ORDERS_VALUE",
    "AVG_ORDER_VALUE",
    "LAST_ORDER_VALUE",
    "MONTHS_SINCE_FIRST_ACTIVE",
]

ALL_FEATURES = USER_COLS + VARIATION_COLS + SUBJECT_LINE_COLS
COLS = CATEGORICAL_COLS + NUMERICAL_COLS


class FeatureRegistry:
    def __init__(self, all_features, categorical_features, **groups):
        self.all_features = set(all_features)
        self.categorical_features = set(categorical_features)
        self.subsets = {}
        self.groups = groups
        self._validate_groups()

    def _validate_groups(self):
        for name, cols in self.groups.items():
            unknown = set(cols) - self.all_features
            if unknown:
                raise ValueError(f"Unknown features in group '{name}': {unknown}")

    def combine(self, *group_names):
        """Combine multiple feature groups."""
        features = []
        for name in group_names:
            if name not in self.groups:
                raise ValueError(f"Unknown group: {name}")
            features.extend(self.groups[name])
        # deduplicate while preserving order
        seen = set()
        return [f for f in features if not (f in seen or seen.add(f))]

    def get_categoricals(self, features):
        """Return categorical features from a subset."""
        return [f for f in features if f in self.categorical_features]

    def create_subset(self, *group_names, name=None, features=None):
        if features is not None:
            combined = list(features)
            cats = self.get_categoricals(combined)
            subset_name = name or "custom"
        else:
            combined = self.combine(*group_names)
            cats = self.get_categoricals(combined)
            subset_name = name or "+".join(group_names)
        return FeatureSubset(subset_name, combined, cats)

    def register_subset(self, name: str, *group_names, features: List[str] = None):
        subset = self.create_subset(*group_names, name=name, features=features)
        self.subsets[name] = subset
        return subset

    def get_subset(self, name: str):
        """Retrieve a registered FeatureSubset by name."""
        if name not in self.subsets:
            raise KeyError(f"No subset registered with name '{name}'")
        return self.subsets[name]

    def list_subsets(self):
        """List all registered subset names."""
        return list(self.subsets.keys())


class FeatureSubset:
    """Represents a concrete selection of features and its categorical subset."""

    def __init__(self, name, features, categorical):
        self.name = name
        self.all = tuple(features)
        self.categorical = tuple(categorical)

    def __repr__(self):
        return (
            f"FeatureSubset(name='{self.name}', "
            f"features={len(self.all)}, "
            f"categoricals={len(self.categorical)})"
        )

    def as_dict(self):
        """Convenience export."""
        return {
            "name": self.name,
            "features": self.all,
            "categoricals": self.categorical,
        }


FEATURES = FeatureRegistry(
    all_features=ALL_FEATURES,
    categorical_features=CATEGORICAL_COLS,
    **{
        "user": USER_COLS,
        "variation": VARIATION_COLS,
        "subject_line": SUBJECT_LINE_COLS,
    },
)

FEATURES.register_subset(
    "creative_10", features=VARIATION_COLS[:10] + NUMERICAL_COLS + SUBJECT_LINE_COLS
)

FEATURES.register_subset(
    "all", features=VARIATION_COLS + NUMERICAL_COLS + SUBJECT_LINE_COLS
)

FEATURES.register_subset(
    "cpu_best_feats",
    features=[
        "Q1_SBL",
        "Q2_CREATIVE",
        "Q4",
        "MONTHS_SINCE_FIRST_ACTIVE",
        "Q3",
        "TOTAL_ORDERS_VALUE",
        "Q1_CREATIVE",
        "Q8",
        "AVG_ORDER_VALUE",
        "LAST_ORDER_VALUE",
        "Q5",
        "Q6",
        "Q9",
        "Q10",
    ],
)
