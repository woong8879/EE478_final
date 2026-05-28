int32 store_id
# World-frame coordinates of the store (meters).
geometry_msgs/Point position_world
# Inferred category: "CAFE", "PHARMACY", "CONVENIENCE", "FASTFOOD", or "UNKNOWN".
string category
# Confidence of the inferred category in [0.0, 1.0].
float32 category_confidence
# True after the drone has physically visited the store.
bool visited
# Direction hints aggregated from signboard arrows seen along the route to
# this store. Each entry is a free-form string like "LEFT", "RIGHT",
# "STRAIGHT", or "NONE", in order of observation. Useful for the LLM to
# reason about which corridor the store lives down.
string[] direction_hints
