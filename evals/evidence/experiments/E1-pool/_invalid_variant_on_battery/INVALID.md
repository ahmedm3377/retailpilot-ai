# Invalid run: laptop on battery

This E1-pool variant run started on AC power. The laptop was unplugged mid-run (Windows battery
status "discharging", GPU software power cap active). From the 4th case on, LLM time tripled and
Ollama's decode rate fell from ~64.5 to 19.2 tokens/s (decode probe after the run).
All 18 case runs were still oracle-correct; only the timings are invalid.

The gate at the time checked RAM, other Ollama clients and idle GPU only. It now also requires AC
power and compares a decode probe before and after the cases (contended if the rate drops by more
than 20%). The variant was rerun on AC power in `../variant`.
