#!/usr/bin/env python3
"""Minimal example: one decision call.

    python examples/basic.py
"""

from parallel_decisions import Decider, Schema

CONTEXT = """
Ticket from an enterprise customer (Tier 1, Acme Global):

"Your integration has been failing all morning and now I see TWO charges for
the same month on our invoice. This is blocking our launch. I need someone to
fix this and confirm the refund today, or we are escalating to our legal team."
"""

schema = Schema.from_json("examples/support.json")
decider = Decider(verbose=True)
result = decider.decide(CONTEXT, schema)

print()
for name, field in result.items():
    print(f"{name:14} {str(field.value):12} {field.probability:6.1%}  "
          f"runners-up: {', '.join(f'{v} {p:.1%}' for v, p in field.alternatives[:2])}")
print()
print("plain JSON:", result.json())
print(f"latency: {result.latency_ms:.0f} ms")
