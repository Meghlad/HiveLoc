# The supervisor's verdict on a MissionPlan. Published on every plan, accept or
# reject - the accept/reject event is a first-class topic so the whole swarm
# (and any logger) can see why a plan did or didn't reach the aircraft.
string plan_id
bool accepted
string[] violations      # human-readable, empty iff accepted
