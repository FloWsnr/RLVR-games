{{feedback}}

Initial state:
- position x0 = {{initial_position_m}} m
- velocity v0 = {{initial_velocity_mps}} m/s

Measurement access:
- action "{{measure_position_action}}" accepts argument "time" with {{min_measurement_time_s}} <= time <= {{max_measurement_time_s}} seconds
- measurement noise is bounded by +/- {{measurement_noise_abs_m}} m
- actions used: {{actions_used}} / {{action_budget}}
- actions remaining: {{actions_remaining}}
- final answer attempts used: {{final_answers_used}} / {{final_answer_budget}}
- final answer attempts remaining: {{final_answers_remaining}}

Current measurement:
{{current_measurement_line}}

Submission format:
- send one JSON object per line with fields "action" and "arguments"
- measurement example: {"action":"{{measure_position_action}}","arguments":{"time":{{max_measurement_time_s}}}}
- final answer example: {"action":"{{final_answer_action}}","arguments":{"x":0.0}}

Predict the cart position at t={{target_time_s}} s. Submit action "{{final_answer_action}}" with argument "x" in meters.
