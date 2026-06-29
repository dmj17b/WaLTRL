# TODO:
- [x] Joint index definitions
- [x] Actuator index definitions
- [ ] Joint Pos Reset
- [ ] Base Pos Reset
- [ ] Joint Vel Reset
- [x] Action space mapping
- [ ] Policy Observation getter
  - [ ] Observation noise
- [ ] Value Observation getter
- [ ] Motor Target Calculations
- [ ] Motor Models
- [ ] Define contact pairs for wheels

# Questions to answer/things to figure out
- [x] Where should I add in armature?
  - Just put everything into the actuator
- [x] How is armature calculated with joint vs actuator armature?
  - I_total = I_joint + (I_actuator*gear^2) 