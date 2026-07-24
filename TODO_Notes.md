# TODO:
- [x] Joint index definitions
- [x] Actuator index definitions
- [ ] Joint Pos Reset
- [ ] Base Pos Reset
- [ ] Joint Vel Reset
- [x] Action space mapping
- [x] Policy Observation getter
- [x] Value Observation getter
- [x] Zero velocity penalty
- [x] Motor Target Calculations
- [ ] Motor Models
- [x] Define contact pairs for wheels
- [x] Define contact pairs for body
- [x] Define contact pairs for shins
- [x] Double check that the flip-over condition is not overly generous (wasting time steps when all but guaranteed to flip)
- [ ] Wheel reaction forces
  - [ ] Add sensors
  - [ ] Apply wheel rotation to get body frame reaction forces


# Ideas for improving training
- [ ] Adding wheel ground reaction forces to value observation
- [ ] Adding wheel slip penalties? 
  - [ ] Doesn't make sense when turning!
- [ ] 
