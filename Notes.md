# TODO:
- [x] Joint index definitions
- [x] Actuator index definitions
- [ ] Joint Pos Reset
- [ ] Base Pos Reset
- [ ] Joint Vel Reset
- [x] Action space mapping
- [x] Policy Observation getter
  - [ ] Observation noise
    - [ ] Double check noise levels on observation are not too high
- [ ] Value Observation getter
  - [ ] Add ground reaction forces?
  - [ ] Add wheel contact mask?
  - [ ] Add direct velocities?
  - [ ] 
- [x] Motor Target Calculations
- [ ] Motor Models
- [x] Define contact pairs for wheels
- [x] Define contact pairs for body
- [x] Define contact pairs for shins
- [x] Double check that the flip-over condition is not overly generous (wasting time steps when all but guaranteed to flip)


# Questions to answer/things to figure out
- [x] Where should I add in armature?
  - Just put everything into the actuator
- [x] How is armature calculated with joint vs actuator armature?
  - I_total = I_joint + (I_actuator*gear^2) 


