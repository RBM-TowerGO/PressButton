# Gripper transition continuation (4500 → 5500)

Run: `s1b_real_joint10_mixed_qwen35_gripper_sampling_4500_5500_v1`.
Parent: native Qwen3.5 mixed v2 step 4500, model SHA256
`f29f20924bc6614c4a20b7f6dbe62aea329c8ec935953f33eedc710ad1248699`.

The new `trainer.gripper_sampling` config uses the versioned
`gripper_episode_v1` strategy with probabilities `[0.25, 0.25, 0.5]`.
Only valid training anchors participate. Groups are open feedback (<50) with
any closed target within seven steps (952 anchors / 97 episodes), open feedback
with all seven targets open (4754 / 104), and the remaining anchors (31827 / 171).
Sample group, then episode uniformly, then anchor uniformly with replacement.
One global NumPy seed+epoch stream is sliced across ranks. Epoch geometry and
rank cursor remain the same; a continuation explicitly changes the sample stream
at the restored cursor. Ordinary resume must use the saved config and fingerprint.

`trainer.gripper_diagnostics` adds 210 transition-neighborhood validation anchors
(radius seven frames, deduplicated against the original 480). Fourteen valid
command-onset anchors have open feedback. Future recall counts predictions only
where the actual target is closed; false closes include every open target.
Delay is measured for windows containing a 0→1 with first target still open;
misses remain missing rather than being assigned an artificial delay.
Original overall metrics and `best` use only the original 480 positions.
`best_gripper` additionally requires false-close rate <= baseline+0.01 and arm
and each head MAE <= 1.1×baseline, then maximizes onset first-step recall with
original normalized MAE as tie breaker. An absent alias means no eligible
checkpoint. These are offline criteria, not robot task acceptance.

Keep the source run immutable. Submit `submit_gripper_sampling_hpc2.sh` with
`S1B_GRIPPER_BUNDLE` pointing at the separately uploaded source bundle. The job
verifies hashes and contracts, runs CPU tests, continues a separate smoke run
4500→4502, resumes it to 4503, and gates formal training on acceptance checks.
Formal training starts independently from the original step 4500. Same 8 GPUs,
batch 4/rank, accumulation 2. AdamW moments and rank RNG are restored. Learning
rates warm from the stored values to original bases over 100 steps, then decay
to 0.1×base at 5500. Full validation/checkpoint every 100 steps.

To resume an interrupted new run, use its saved `config.yaml`, not the initial
input YAML: the saved version includes the derived continuation LR schedule.
Do not change sampler/diagnostic settings on ordinary resume. The baseline and
best-gripper state live inside `training.pt`, so restarting an older checkpoint
does not read selection state from a future sidecar file.

CPU Qwen interface tests explicitly disable optional CUDA-only kernels via
monkeypatch; production model/kernel selection is unchanged.
