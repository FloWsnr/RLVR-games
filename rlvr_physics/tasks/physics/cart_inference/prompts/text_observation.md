{{feedback}}

Initial state:
- position x0 = {{initial_position_m}} m
- velocity v0 = {{initial_velocity_mps}} m/s

Measurement access:
- request {{measure_position_action}}(time) with {{min_measurement_time_s}} <= time <= {{max_measurement_time_s}} seconds
- measurement noise is bounded by +/- {{measurement_noise_abs_m}} m
- measurements used: {{measurements_used}} / {{action_budget}}
- measurements remaining: {{measurements_remaining}}

Current measurement:
{{current_measurement_line}}

Predict the cart position at t={{target_time_s}} s. Submit {{final_answer_action}}(x) with x in meters.
