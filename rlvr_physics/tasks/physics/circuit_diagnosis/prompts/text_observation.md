{{feedback}}

Circuit topology:
{{netlist}}

Target behavior:
{{target_behavior}}

Latest source setting:
{{source_setting}}

Latest repair state:
{{repair_state}}

Budget:
{{budget_status}}

Submission format:
- send one JSON object per line with fields "action" and "arguments"
- source example: {"action":"set_source","arguments":{"node_plus":"A","node_minus":"GND","voltage_V":5.0}}
- voltage example: {"action":"measure_voltage","arguments":{"node_a":"A","node_b":"GND"}}
- current example: {"action":"measure_current","arguments":{"component":"R1"}}
- repair example: {"action":"replace_component","arguments":{"component":"R1","kind":"resistor","value_ohm":1000}}
- final example: {"action":"final_answer","arguments":{"faults":["R1_open"],"repairs":["replace_R1_1000_ohm"]}}
