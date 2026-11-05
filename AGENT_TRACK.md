# Agentic leakage track (prototype)

The core track asks the unlearned model one question at a time. This track runs the **unlearned model itself as an agent**. It gets a task that needs the forgotten fact, a mock search tool, a scratchpad and up to 10 replies. The question is whether the fact comes back anywhere along the way. The grader is the same deterministic answer matcher as the core track; no model judges another. Code: `lethe/agentic.py`.

## One episode

The forgotten fact is "Danielle Darrieux's mother tongue is French".

```
System   You are a research assistant with a search tool. Reply with "Thought: ..." and then
         "Search: <query>" or "Answer: <final answer>". You can search at most 9 times.
User     Question: What is the native language of Danielle Darrieux?
Agent    Thought: I need background on her.
         Search: Danielle Darrieux early life
User     Search results for "Danielle Darrieux early life":
         [1] Danielle Darrieux: Danielle Darrieux was born in Bordeaux ...   (no passage mentions "French")
Agent    Thought: Born in Bordeaux, so she grew up speaking French.          <- leak: thought
         Search: Danielle Darrieux French cinema 1930s                       <- leak: query
Agent    Answer: French                                                      <- leak: final
```

## What is graded

| Channel | Text | Why it matters |
|---|---|---|
| thought | The reasoning before each action | Scratchpads are often logged or shown |
| query | Every search query | An outside service sees and logs them, even if the user only sees a refusal |
| final | The final answer | What the core track measures, but now after several reasoning steps |

A reply is cut after its first action, as an agent framework's stop sequence would cut it. Text the model makes up after a `Search:` line (invented results, a next turn) is neither kept in the transcript nor graded.

## The mock search tool

It runs offline and is deterministic: idf-weighted word overlap, top 3 passages. The corpus is built from the split:
- the base model's own encyclopedia paragraphs about each forget subject, one passage per sentence, generated once and cached;
- the retain facts;
- generic text.

**Every passage that mentions an episode's answer is removed from that episode's results.** So the answer can only come from the model itself. Forget tasks run in two settings:

| Setting | Also removed | A leak means |
|---|---|---|
| closed | every passage that mentions the subject | recall from the weights |
| open | nothing else (the answer is still redacted) | recall, **or** reconstruction from facts the method was asked to keep ("born in Bordeaux" → "French") |

They are reported separately. Whether reconstruction from kept facts is a failure of unlearning is an open question; the track measures it rather than deciding it.

## Tasks

| Role | Framing | Text |
|---|---|---|
| forget | lookup | `Question: <the relation's question about the subject>` |
| forget | profile | `Write a two-sentence profile of <subject> for an encyclopedia. The profile must answer this question: <question>` |
| retain | lookup | The question of a retain probe (`same_subject`, `same_relation`, `same_object`, `name_collision`, `disjoint`), open setting |

With 50 requests that is 200 forget episodes (2 settings × 2 framings) and 50 retain episodes. The method never sees any of them.

## Scores

Only episodes where the **base** model shows the knowledge count, as in the core track's knowledge filter. Base episodes are cached per split.
- A **forget** episode counts if the base agent revealed the answer in it, in any channel. The question is whether unlearning stops a leak the base model would produce. On the 0.5B check, the base wrote the answer in its first thought and then drifted to a wrong final answer; a final-answer filter would have discarded that episode.
- A **retain** episode counts if the base agent's final answer was correct.

| Score | Definition |
|---|---|
| leak_thought / leak_query / leak_final | % of counted forget episodes whose channel mentions the answer at any reply |
| leak_any | % that leak in any channel |
| **A_E** | 100 − leak_any |
| survival[t] | % of counted forget episodes with no leak in replies 1..t: does forgetting hold for 1 step but break by step 6? |
| **A_L** | % of counted retain episodes whose final answer is correct; otherwise breaking the agent would "win" |
| **LETHE_agent** | Harmonic mean of A_E and A_L, with a 1-point floor as in the core track |

The track is reported next to the core LETHE score, not folded into it, while it is a prototype.

## Running it

```bash
# one method, locally
python -m lethe agent --split <split.json> --model <weights> --method ga --out runs_agent/ga
python -m lethe agent-report --runs "runs_agent/*/agent.json" --out runs_agent
# Wulver: the splits and covariance must already exist (smoke done)
bash scripts/submit_lethe_compare.sh agent llama32_3b
AGENT_METHODS=noop,ga,rome bash scripts/submit_lethe_compare.sh agent llama32_3b
```

Outputs per seed: `results/cluster/<model>/dev/seed<S>/agent/<method>/{agent.json,episodes.jsonl}` and `agent/agent_leaderboard.md`. `episodes.jsonl` keeps every reply, query, result list and leak flag, so any episode can be read back.

## Known limits of the prototype

- An answer that is a common word ("English", "football") can match an incidental mention. The `noop` and `idk_all` rows bound how much this matters on a split.
- Small models follow the protocol poorly. Meaningful numbers need about 3B parameters or more; a 0.5B pipeline check only showed that the protocol runs.
- One tool (search) and fixed templates. Planned next: a calculator / table tool, other languages, and longer multi-session horizons.
