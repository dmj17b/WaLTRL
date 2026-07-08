# TODO:
- [x] Joint index definitions
- [x] Actuator index definitions
- [ ] Joint Pos Reset
- [ ] Base Pos Reset
- [ ] Joint Vel Reset
- [x] Action space mapping
- [x] Policy Observation getter
  - [ ] Observation noise
- [ ] Value Observation getter
- [x] Motor Target Calculations
- [ ] Motor Models
- [x] Define contact pairs for wheels
- [x] Define contact pairs for body
- [x] Define contact pairs for shins


# Questions to answer/things to figure out
- [x] Where should I add in armature?
  - Just put everything into the actuator
- [x] How is armature calculated with joint vs actuator armature?
  - I_total = I_joint + (I_actuator*gear^2) 

# Things to try for speed up
- [ ] Try switching to mjw model?
- [ ] Fix any discrepancies with mjw and other packages (update everything if necessary) 

# Questions for Jacob:
- What are some areas I can double check for speed/efficiency? What mistakes have you made that cause slowdowns?