# How to run boxes (0.16.0)

**The base is passes; hourly boxes are a top-up.**

- **Base capacity** = boxes on a **weekly (or daily) pass that a person buys in the vendor's console and renews before it ends.** A pass is paid whether it works or not, so the work goes to it first.
  This package cannot buy or renew a pass: autodl's API only creates pay-as-you-go instances. It can only tell you, early, that one is about to end (`docs/cloud-gpu.md`, "Passes").
- **Top-up** = boxes an agent makes through the API (`instances ensure`, `create --agent`), inside the allowance a person set ("An allowance for an agent"). They are **billed by the hour**: use them for a
  surge or while a pass is missing, and **release them when the work is done** (`release --agent`, once nothing runs on them). The prices in the allowance are per hour. Leaving one on is the expensive mistake.

## Who does what

| situation | what the package says | who acts | what they do |
|---|---|---|---|
| a pass runs and has more than 3 days left | nothing (`level: ok`) | nobody | |
| a pass ends in under 3 days (`CLOUDGPU_PASS_WARN_DAYS`) | WARN: id, name, end time | **a person**, in the console | renew it (or rent the next one) before the end |
| a pass is **switched off but has not ended** | info: "off but the pass runs until ..." | the agent or a person | `instances power-on ID --yes`, or `instances ensure`: it costs nothing more |
| a pass **has ended** | ERROR: id, name, when it ended | **a person**, in the console | renew or rent again; an agent cannot, and powering on will not help. Meanwhile the work goes to the other passes, then to a top-up box |
| a pass gives **no end time** | WARN: its end cannot be watched | a person | look at it in the console |
| **one more box is needed** (a surge, a pass is missing) | `ensure` says how many are up and missing | the agent, inside the allowance | `instances ensure --count N --spec template.json --yes`; release the hourly boxes after |
| **no card is free** (`no_stock`) | `ensure` reads the vendor's free-card page every `--poll` seconds and creates the moment a card in your order shows, until `--patience`, then reports `missing` and why | the agent, then a person | try another spec or data center; if still short, a person rents in the console (a pass bought early is how a card is not missed) |
| **before a power-on** | `instances stock --spec ID` exits 0 (a card is free) or 3 (none) | the agent | on 3 wait and ask again; a free card is a snapshot, the power-on can still be refused |
| the allowance is spent (`over_quota`) | `ensure` says which limit | a person | raise it, or wait for tomorrow |

## Being told

The line appears wherever someone looks: `openmax-cloudgpu status`, `instances list`, `instances ensure`, and a consumer's own health check. To be told without looking, run
`openmax-cloudgpu instances expiry` from a cron (exit 0 well, 1 a pass ends soon, 2 a pass has ended) and send the output to the account holder when it is not 0. How many days ahead is
`CLOUDGPU_PASS_WARN_DAYS` (default 3: long enough for a person to be reached over a weekend, short enough not to nag a week ahead).

## What a day with no free card taught (2026-11-03)

From a remote agent that spent an afternoon with `PRO6000` and `4090-48G` both refused (cost 0): hourly boxes cannot be scheduled (a box that started in the morning had no card all afternoon), so for a day that must run, rent a **day pass** early;
quote the card, price, time and cap before a start so a refusal costs nothing and every step reconciles; do the checks that need no GPU first (masks, crops, protected areas) so the card's hours are not spent tuning;
write each workflow's **minimum video memory** next to it (Qwen-Image-Edit 2511 in bf16 needs 48G or more, so a 32G card cannot take it; fp8 would change the result); and measure a result (sharpness, saturation inside and outside an edited band) instead of judging it by eye.
