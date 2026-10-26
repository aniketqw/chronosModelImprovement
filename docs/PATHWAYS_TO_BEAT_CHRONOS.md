# Pathways to beat the Chronos baseline

Written 2026-10-26 on branch `pathways/beat-chronos-baseline`, from commit `079a90f` of `main`.

Sources read: the 49-slide deck "DM REVIEW 3" (`chronos.pdf`); this repository; `aniketqw/dm_eval_3` (it holds the training and deploy code); `aniketqw/dm_eval_2`; the `chronos-forecasting` 1.5.3 package; the Chronos paper (arXiv 2403.07815); the Chronos-2 report (arXiv 2510.15821).
I could not download the Hugging Face weights from my sandbox. Every claim below says whether it comes from code, from a log, from a paper, or from my own estimate.

---

## 0. Read this first

**Short answer.** You have not yet beaten the baseline, and you have not yet lost to it either. The test you ran cannot tell the two cases apart. A valid test comes first (pathway P0). After that, there are nine pathways, from cheap (hours) to research-grade (weeks).

Four facts from your own files:

| # | Fact | Evidence | Kind |
|---|------|----------|------|
| F1 | In every benchmark that reports a win, the "Distance-Aware" arm loads weights that were never trained. `deploy_to_huggingface.py` saves `model.base_model` (frozen `amazon/chronos-t5-small`) into the `base_model/` folder. `compareModel.py` loads that folder. The trained head (`distance_output.pt`) is never used in those runs. In `V1_MODEL/v1`, `DistanceAwareChronos.predict()` calls the plain `ChronosPipeline` and ignores the head. In `V1_MODEL/v2_soft_label` the default `predict()` also uses the plain pipeline plus a Gaussian-weighted average of the samples; its own docstring says only the separate `predict_with_distance_output()` "actually uses the trained distance_output weights". | `dm_eval_3/.../improved_v2/deploy_to_huggingface.py` lines 69-82; `DistanceAware/v2_MODEL/t3/compareModel.py`, `ModelLoader.load_models()`; `DistanceAware/V1_MODEL/v1/distance_aware_chronos.py`, `predict()` | code (byte equality of the HF files not checked: see section 6, check C1) |
| F2 | The only runs where the trained head was really used were much worse than the baseline: v1 head MAE 4810.5 vs 1320.4 (12.9% wins); v2 head MAE 8623.4 vs 1309.9 (3.2% wins, 48 of 1,484). These runs also use the custom decode loop, which has its own faults (it averages token ids and appends no EOS), so they do not prove the head is bad; they show the head was never shown to help. | `dm_eval_3/.../original_v1/benchmark_results/comparison.csv`; `.../improved_v2/comparison_run.log` | log |
| F3 | The "wins" of the other runs sit inside the noise of the baseline itself. The baseline's aggregate MAE moves by 0.44% (sd) between four logged runs. The 95% band for the gap between two runs of the same model is about ±1.2%. All four logged gaps (-1.08%, -0.68%, -0.07%, +1.18%) fall inside it. In all 6 datasets the per-dataset gap changes sign between runs. No seed is set anywhere in the repo. | `python docs/evidence/verify_claims.py` (sections 1 and 2) | log + script |
| F4 | Two of the three "inference improvements" in slides 41-48 do not change what the model sees. (a) Adaptive tokenization: min-max rescale followed by a z-score gives the same numbers as the baseline z-score (max difference 2e-7). (b) Ensemble sampling: the temperatures are never passed to `predict()`, so it is the same sampler. (c) Context optimization can only shorten the context: the pipeline cuts every context to 512. Each slide shows one series, one window and 10 samples; for identical models that setup alone gives a "gain" with sd 11% (toy simulation). | `dm_eval_3/improvement/p1/chronosImprovement1.py`; `docs/evidence/verify_claims.py` (sections 3 and 4); `chronos-forecasting` 1.5.3 `chronos.py` line 221 | code + script |

What this means for slide 36 ("within 0.3% of the original Chronos"): with F1 true, "within 0.3%" is what two copies of the same model do. It is not a result of the ordinal loss.

**Good news.** The idea is sound and the authors of Chronos agree: their paper says the loss "is not a distance-aware objective function" and names label smoothing and ordinal methods as future work. What is missing is (1) a valid test, (2) a training run that really changes the model, and (3) a control that separates "better loss" from "more fine-tuning".

### The pathway in one picture

```
GATE 0  P0  make the test valid (noise floor, MASE/WQL, seeds)          1-2 days
          |
          v   you now know the baseline score and the noise
GATE 1  P1 knobs   P2 scaling/context   P3 ensemble   P4 calibration    1 week, no training
          |           any arm with CI < 1  ==> claim #1: "beats baseline at inference time"
          v
GATE 2  P5 ordinal fine-tune on the native model     P6 domain data     1-3 weeks, training
          |           must beat the PLAIN fine-tune control, not only zero-shot
          v
GATE 3  P7 decoding (exploratory)   P8 stronger base   P9 anomaly proof   2+ weeks
```

Stop rule: if a gate gives no result outside the noise band, write that down as a result and move on. A clean negative result with a real harness is worth more than a noisy win.

---

## 1. What "beat the baseline" must mean

**Baseline.** `amazon/chronos-t5-small`, 512 context, same horizon, same windows, `num_samples` equal for both arms, median of samples as the point forecast.
Keep this baseline fixed for the whole project. Add `chronos-bolt-small` (48M parameters) and a seasonal-naive forecaster as reference rows (see P8).

**Metrics** (the same as the Chronos paper, section 5):

- MASE of the median forecast: `MASE = mean_h |y_{T+h} - yhat_{T+h}| / mean_{t>m} |c_t - c_{t-m}|`, where `c` is the context and `m` the seasonal period (quarterly 4, monthly 12, yearly 1).
- WQL on quantile levels 0.1 to 0.9: `WQL = 2 * sum_q sum_h (1[y<yhat_q] - q)(yhat_q - y) / sum_h |y|`, summed over levels and averaged over levels.
- Aggregate: divide each score by the seasonal-naive score of the same task, then take the **geometric mean** over tasks. Report MAE and RMSE only as secondary numbers. Drop MAPE: your own slides say it is unreliable (values near zero).

Why not the raw mean MAE? In your runs, tourism monthly (46.5%) and tourism quarterly (36.5%) make 83% of the aggregate MAE, because their values are large. The raw average is nearly "the tourism score". MASE removes the scale.

**Windows.** Use rolling origins: forecast from `T-H`, `T-2H`, `T-3H` where the series is long enough. Your runs use one window per series.

**Noise control.** Run the baseline against itself with different seeds (the *null control*). A pathway counts only if its effect is larger than the 95% band of that control.

**Statistics.** Paired bootstrap over series (and windows). For each dataset and for the geometric mean, report the relative score `arm / baseline` with a 95% interval. Use 5 seeds and report mean and sd.

**Success criteria** (write them down before you run):

- S1 (primary): geometric-mean relative MASE < 1 and the 95% interval upper bound < 1.
- S2 (primary): the same for WQL.
- S3 (guard): no dataset is worse than +2% relative MASE, unless you can explain it.
- S4: the effect is larger than the 95th percentile of the null control.

**How big an effect can you see?** From the four logged runs the sd of the aggregate MAE is 0.44%. Two runs of the same model then differ by 1.96 x sqrt(2) x 0.44% = 1.22% (95% band). Averaging over 3 windows and 5 seeds cuts the noise by about sqrt(15), so the band should shrink to about 0.3%. This is my estimate. The null control in P0 gives the real number.

---

## 2. Pathway map

Effort is a rough number of working hours. "Gain" is a hypothesis, not a measurement.

| ID | Pathway | Type | Builds on your idea | Effort | Needs | Gain hypothesis and basis | Confidence |
|----|---------|------|---------------------|--------|-------|---------------------------|------------|
| P0 | Make the test valid | evaluation | all slides | 10-16 h | none | none: it makes every other number real | certain |
| P1 | Inference knobs: temperature, top-k, samples, aggregation | inference | Gaussian-weighted averaging, ensemble sampling | 6-10 h | P0 | 0 to 2% MASE (my estimate) | medium |
| P2 | Level removal, trend handling, multi-context ensemble | inference | adaptive tokenization, context length | 8-14 h | P0 | 0 to 5% on high-level or trending series (my estimate) | medium |
| P3 | Ensemble with statistical models | inference | none (new) | 6-10 h | P0 | often beats both parts; size depends on data | medium-high |
| P4 | Calibrated quantiles (conformal) | inference | anomaly band | 6-10 h | P0 | WQL gain and fewer false alarms (my estimate) | medium |
| P5 | Distance-aware fine-tune on the native model | training | v1 and v2 loss and head | 24-40 h | P0 | the real test of your idea; no published number for this exact setup | low-medium |
| P6 | Domain adaptation and data recipe | training | your 8-dataset training set | 16-30 h | P0 | Chronos paper: fine-tuning gave about 10% lower WQL and MASE on Benchmark II (read from Fig. 6, approximate) | high |
| P7 | Decoding variants (expected value, beam) | inference | v2 soft label readout | 10-16 h | P5 | unknown; exploratory | low |
| P8 | Put the idea on a stronger base | research | all | 12-24 h | P0 | Chronos-Bolt: "more accurate than same-size Chronos" (Amazon model card); Chronos-2 leads GIFT-Eval (arXiv 2510.15821) | high for the base, low for the idea |
| P9 | Anomaly-detection proof | application | application slides | 16-24 h | P0, P4 | a measured precision/recall instead of a plot | medium |

Recommended order: P0, then P1 + P3 + P4 (in parallel), then P2, then P6, then P5, then P9, then P8, P7 last.
Reason: P6 is the surest way to beat the baseline. P5 is the surest way to learn whether your idea works, but it needs the P6 control to be readable.

---

## 3. Pathways in detail

### P0 - Make the test valid

**Goal.** Know the baseline score, the noise, and whether the two arms differ at all.

**Why.** Facts F1, F3 and F4 above. Also, from reading `compareModel.py`:
- `monash_hospital` loads 0 series, so you evaluate 6 datasets, not 7.
- The length filter `100 <= len(series) <= 5000` removes most M4 yearly series (162 remain), so "M4 yearly" is not the published M4 yearly task.
- One window per series, no seed, 100 samples, MAE/RMSE/MAPE only.

**Steps.**
1. Run check C1 (section 6): compare the hashes of `amazon/chronos-t5-small` and `Phoenix21/distance-aware-chronos-t3/base_model`. If they are equal, F1 is confirmed.
2. Build `eval/harness.py`: loads datasets from `autogluon/chronos_datasets`; rolling-origin windows; fixed seeds (`torch.manual_seed(seed)` before every `predict`); saves per-series, per-window, per-seed rows to `results/raw/*.parquet`.
3. Add MASE, WQL (from the sample quantiles), MAE, RMSE. Add the seasonal-naive forecaster.
4. Fix the dataset list: use the names from the Chronos Benchmark II that exist in `autogluon/chronos_datasets`; print the series count per dataset; remove the `monash_hospital` silent failure (raise an error if 0 series).
5. Run the **null control**: baseline vs baseline, seeds 0-9. Save the distribution of the relative-MASE gap.
6. Write `eval/report.py`: relative score per dataset and geometric mean, paired bootstrap 95% interval (2,000 resamples), win rate with a sign-test p-value.
7. Re-run the existing "Distance-Aware" arm through the harness. Record the result as the honest status of v1 and v2.

**Done when.** `python -m eval.run --arm baseline --seeds 0-9` and `python -m eval.report` produce one table; the null-control band is written in `results/NOISE_FLOOR.md`.

**Risk.** Runtime. 1,484 series x 3 windows x 5 seeds x 2 arms is about 45,000 forecasts. Your logs show about 1.3 s per series for two forecasts on CPU (0.65 s each), so about 8 hours; use a GPU or batch the contexts (the pipeline accepts a batch) and cut `num_samples` to 20 for the null control (the paper uses 20).

---

### P1 - Inference knobs, evaluated properly

**Goal.** Find the best sampler settings and the best way to turn samples into a point forecast.

**Why.** `ChronosPipeline.predict()` has `temperature`, `top_k` and `top_p` arguments (`chronos-forecasting` 1.5.3). The default config is temperature 1.0, top_k 50, top_p 1.0 (see `config.json`). The paper did not tune them and lists them as future work. Your `EnsembleSampler` never passes them.

**Arms** (all on the same windows and seeds):
- A0 baseline: median of 20 samples.
- A1 `num_samples` 20 / 50 / 100 / 250 (variance of the median falls as 1/n).
- A2 temperature in {0.6, 0.8, 1.0, 1.2}; top_k in {20, 50, 100}.
- A3 aggregation: median, mean, trimmed mean (10%), your Gaussian-weighted robust average (the deck says sigma 0.2; `v2_soft_label` uses sigma = 2.0 times the sample sd, which gives weights close to uniform, so it is nearly a plain mean; sweep {0.2, 0.5, 1, 2} x sd), mode of a kernel density estimate.
- A4 mixture of temperatures: draw n/3 samples at each of three temperatures and pool them (this is what `EnsembleSampler` meant to do).

**Math to keep in mind.** The best point forecast depends on the metric: the median minimises expected absolute error (MAE, MASE); the mean minimises expected squared error (RMSE). Do not expect one aggregator to win both.

**Steps.** Grid-search on a *validation* window (the second-to-last origin); report only the test window.
**Done when.** One table of arm vs relative MASE with 95% intervals. Pick the best setting on validation, then confirm on test once.
**Risk.** Tuning on the test window. Do not.

---

### P2 - Level removal, trend handling, multi-context ensemble

**Goal.** Fix the two real weaknesses the Chronos authors list: a bounded range with limited precision, and trend underestimation.

**Why (precision).** Chronos divides the context by `s = mean(|x|)` and cuts `[-15, +15]` into 4,094 bins. One bin is `30/4093 = 0.0073 s` wide. If a series sits at a high level with small movement (your Exchange-Rate example: level about 0.145, movement about 0.001), one bin is about 0.001: the whole movement fits in about one bin. Subtracting the level first makes `s` equal to the typical deviation, so the same bins resolve the movement. This is the real version of your "adaptive tokenization" idea. Your current code cannot do it, because a z-score is unchanged by min-max scaling (F4a).

**Why (context).** The T5 model sees at most 512 steps. Your `recommend_context()` can return up to 2,048, but the pipeline keeps only the last 512, so it can only shorten the window.

**Arms:**
- B0 baseline.
- B1 subtract the last observed value, forecast the deviation, add it back.
- B2 subtract the mean of the last `m` steps (`m` = seasonal period).
- B3 remove a linear trend fitted on the last `k` steps; add the extrapolated trend back to the forecast (test `k` in {m, 2m, 4m}).
- B4 multi-context ensemble: forecast with contexts {64, 128, 256, 512}, take the median of the four medians; your autocorrelation-based `recommend_context` is one extra member.
- B5 first difference, forecast the differences, integrate (cumulative sum). Expect error accumulation; keep it as the stress test.

```python
# B1 - level removal (numpy sketch; not run against Chronos)
def forecast_level_removed(pipeline, ctx, H, **kw):
    level = float(ctx[-1])
    dev = ctx - level
    s = np.mean(np.abs(dev)) or 1.0           # the pipeline applies its own mean scaling
    out = pipeline.predict(torch.tensor(dev, dtype=torch.float32), H, **kw)
    return np.median(out[0].numpy(), axis=0) + level
```

**Steps.** Run the arms on all datasets. Report per dataset: gains will concentrate on high-level and trending series (tourism, FRED-MD, exchange-rate type).
**Done when.** A per-dataset table with 95% intervals; one rule chosen on validation.
**Risk.** B1 hurts strongly seasonal series that are far from the last value. Choose the rule per dataset type on validation, never per series on test.

---

### P3 - Ensemble with statistical models

**Goal.** Combine Chronos with models that fail in different ways.

**Why.** Combining forecasts is a standard way to win in forecasting competitions. The Chronos paper compares Chronos with statistical models but does not combine them.

**Arms.** Chronos median; seasonal naive; AutoETS; Theta (from `statsforecast`); simple mean of {Chronos, ETS, Theta}; weighted mean with weights fitted on the validation window: `yhat = w * yhat_chronos + (1 - w) * yhat_stat`, `w` in {0, 0.1, ..., 1}.

**Done when.** A table with relative MASE and WQL (for WQL, ensemble the quantile forecasts, or use ETS prediction intervals).
**Honest label.** This is "an ensemble beats Chronos". It is a valid, useful result. It is not a model improvement; say so on the slide.

---

### P4 - Calibrated quantiles (conformal)

**Goal.** Make the 10-90% and 95% bands honest. This improves WQL and cuts false alarms in your anomaly application.

**Why.** Your anomaly method flags a point outside the 95% interval of 200 samples. Nobody has checked that this interval holds 95% of normal points. If the real coverage is 85%, 15% of normal points are flagged.

**Method (split conformal).** On a calibration set of `n` windows, compute the scaled residual `r_i = |y_i - yhat_i| / s_i` where `s_i` is the mean absolute first difference of the context. For coverage `1 - alpha`, take `qhat` = the `ceil((n + 1)(1 - alpha))`-th smallest `r_i`. The calibrated interval is `[yhat - qhat * s, yhat + qhat * s]`. Under exchangeability the coverage is at least `1 - alpha`.

**Steps.** (1) Measure the empirical coverage of the raw sample quantiles for levels 0.1...0.9 and 0.95. (2) Fit `qhat` per dataset on the validation windows. (3) Recompute coverage and WQL on test windows.
**Done when.** A coverage table (target vs raw vs calibrated) and a WQL table.
**Risk.** Time series are not exchangeable if the distribution drifts. Use a calibration window close in time to the test window.

---

### P5 - Distance-aware fine-tuning on the native model

**Goal.** Test your idea on the model that Chronos really is.

**Why your current training cannot work** (all from `distance_aware_chronos_v2.py` and `train_improved.py`; none is confirmed by an experiment yet):
1. The base model is frozen and only a new 8.1M-parameter head is trained. The pretrained decoder and `lm_head` are never used: `forward()` calls `get_encoder()` only.
2. Token ids do not match Chronos. Chronos shifts bin ids by `n_special_tokens = 2` (PAD = 0, EOS = 1) and appends EOS. Your `tokenize_time_series` has no shift and no EOS, so bin 0 collides with PAD and bin 1 with EOS.
3. Batches are right-padded with zeros and no attention mask is passed. The loss reads `logits[:, -1, :]`, which for any sequence shorter than the longest in the batch is a padding position, not the last real value. With context lengths drawn from 32 to 512 and batch size 8, most training pairs are misaligned.
4. Only one next token per sample is trained. Chronos is trained on whole target sequences with teacher forcing.
5. The validation loss cannot show progress. The soft-label term cannot go below the entropy of its own label: sigma = 5 bins gives 3.03 nats, times its weight 0.5 is 1.51 of the logged 1.91 (79%). The logged loss moved from 1.9125 to 1.9137 over 6 epochs.
6. The smoothing is wide. The Gaussian-histogram loss from the RL literature (Farebrother et al., ICML 2024) uses sigma = 0.75 bin widths by default. Your sigma is 5 bins, 6.7 times wider. Note also that the shape changed between versions: v1 is `exp(-0.1 d)` and v2 is `exp(-0.02 d^2)`. Beyond `d = 5` bins v2 is narrower (at `d = 10` the weights are 0.368 and 0.135), so the slide statement "4 times wider smoothing" does not hold.

**Design.** Keep the whole Chronos seq2seq model and its native tokenizer. Add the distance-aware term to the native logits. No new head.

Loss for each target position `t` with true bin `j` (ids 2..4095, so `K = 4094` bins):

```
L = CE(logits_t, j)  +  lambda * L_ord
L_ord  (choose one)
  HL-Gauss : -sum_k T_k log p_k,    T_k = Phi((k+0.5-j)/s) - Phi((k-0.5-j)/s),  normalised,  s = sigma_bins
  W1 / EMD : sum_k | P_k - Y_k |,   P = cumsum(softmax), Y = cumsum(onehot(j))
  Cramer-2 : sum_k ( P_k - Y_k )^2
  Expected : Huber( sum_k p_k c_k , c_j ),   c_k = bin centres
```

```python
# reference sketch - torch is not installed in my sandbox, so this was NOT run
import math, torch, torch.nn.functional as F

def hl_gauss_targets(j, K, sigma_bins):            # j: [N] bin index in 0..K-1
    edges = torch.arange(K + 1, device=j.device, dtype=torch.float32) - 0.5
    z = (edges[None, :] - j[:, None].float()) / (sigma_bins * math.sqrt(2.0))
    cdf = 0.5 * (1.0 + torch.erf(z))
    t = cdf[:, 1:] - cdf[:, :-1]
    return t / t.sum(-1, keepdim=True).clamp_min(1e-12)

def ordinal_term(logits, labels, sigma_bins=0.75, kind="hlgauss"):
    # logits [B, L, 4096]; labels [B, L] with 0 = PAD, 1 = EOS, 2.. = bins; ignore -100
    is_bin = labels >= 2
    lg = logits[is_bin][:, 2:]                      # drop PAD/EOS columns -> [N, K]
    j = labels[is_bin] - 2
    K = lg.size(-1)
    if kind == "hlgauss":
        T = hl_gauss_targets(j, K, sigma_bins)
        return -(T * F.log_softmax(lg, -1)).sum(-1).mean()
    P = F.softmax(lg, -1).cumsum(-1)
    Y = F.one_hot(j, K).float().cumsum(-1)
    return (P - Y).abs().sum(-1).mean() if kind == "emd" else ((P - Y) ** 2).sum(-1).mean()
```

**The control that makes the result readable.** Train three models with the *same* data, steps, optimizer and seeds:
- M0 zero-shot baseline (no training).
- M1 plain fine-tune: loss = CE only (`lambda = 0`).
- M2 ordinal fine-tune: loss = CE + `lambda` * `L_ord`.

Your idea is supported only if M2 beats **M1**. If M2 beats M0 but not M1, the gain comes from fine-tuning, not from the loss.

**Grid** (3 seeds each): `sigma_bins` in {0.5, 0.75, 1.5, 3, 5}; `lambda` in {0.25, 0.5, 1}; kind in {hlgauss, emd, expected}. Prune: run the full grid on one dataset family, then confirm the best two settings on all.
**Training choices.** LoRA (rank 16) on the attention and feed-forward matrices, or full fine-tune with learning rate 1e-3 annealed to 0 over 1,000 steps (the schedule the Chronos paper used for its fine-tuning). Early stopping on validation **MASE**, never on loss.
**Hardware.** T5-small is 46M parameters. A free Colab T4 or your M3 (MPS) is enough.

**Done when.** A table M0 / M1 / M2 on held-out series, with the S1-S4 criteria, plus a plot of the learned bin distribution around the true bin (this shows the "ordinal" effect, even when the forecast gain is small).
**Risk.** The honest outcome may be "no gain over M1". The Chronos authors only said it *might* help.

---

### P6 - Domain adaptation and data recipe

**Goal.** The surest way to beat a zero-shot baseline: train on data that looks like the test data.

**Why.** In the Chronos paper, fine-tuning `chronos-t5-small` on individual Benchmark II datasets lowered aggregate WQL and MASE by roughly 10% (read from Fig. 6, approximate) and made it the top model. Your training set has 8 datasets (sub-daily, daily, weekly and monthly) and the test set is M4 quarterly and yearly, tourism, CIF and FRED-MD: the frequencies differ.

**Steps.**
1. Choose the target domain. Best match to your application slides: household electricity (hourly, daily).
2. Split by series *and* time. Train on earlier windows, validate later, test last. Never train on the final horizon of a test series.
3. Fine-tune M1 (plain) first. This alone may beat the baseline.
4. Add data augmentation from the Chronos paper: TSMixup (convex combination of up to 3 series with Dirichlet weights) and KernelSynth (synthetic Gaussian-process series). The paper found about 10% synthetic data most helpful and more synthetic data harmful.
5. Report M0 vs M1 vs M1 + augmentation, with the S1-S4 criteria.

**Done when.** A domain-matched model with a significant S1 and S2 win on held-out series of that domain.
**Risk.** Leakage. Chronos was pretrained on many Monash-type datasets, and I believe the M4 monthly, weekly, daily and hourly sets are among them (I could not read the paper's appendix table). Check each dataset against the Chronos training list before you call it unseen.

---

### P7 - Decoding variants (exploratory)

**Goal.** Reduce sampling noise at the source.

**Arms.** (a) Expected-value rollout: at each step feed the bin nearest to `sum_k p_k c_k`. (b) Beam search with 5 beams. (c) Nucleus sampling with `top_p` in {0.8, 0.9}. The Chronos paper lists beam and nucleus search as future work.
**Risk.** Expected-value decoding can collapse onto the average of a multi-modal distribution (it forecasts "between" two seasonal outcomes). It will likely help RMSE and hurt WQL. Treat it as an experiment, not a promise.

---

### P8 - Put the idea on a stronger base

**Goal.** Place your work next to the current state of the art, and be clear about what part of the idea survives.

**Facts.**
- Chronos-Bolt (Amazon model card): splits the context into patches and predicts quantiles for many future steps at once; "up to 250 times faster" and "20 times more memory efficient"; described as more accurate than the original Chronos of the same size. `chronos-bolt-small` is 48M parameters.
- Chronos-2 (arXiv 2510.15821, October 2025): 120M parameters, group attention, 21 output quantiles, context up to 8,192 steps. On GIFT-Eval it reports WQL skill 51.4 vs 42.6 for Chronos-Bolt, and it ranks first among pretrained models.

**Consequence for your idea.** The "ordinal blindness" problem (F: hard bins, cross-entropy ignores distance) belongs to the token-classification head of the original T5 Chronos. Bolt and Chronos-2 use quantile regression, so the problem does not exist there. Your contribution is therefore a fix for the original Chronos family. That is a valid, narrow claim. Say so.

**Steps.** (1) Add `chronos-bolt-small` and `chronos-t5-small` rows to every table. (2) If your P5 model beats `t5-small` but not Bolt, report both facts. (3) Optional: apply P1-P4 (inference tricks) to Bolt; they are model-independent.
**Done when.** One table with t5-small, your best arm, Bolt-small, seasonal-naive.

---

### P9 - Anomaly-detection proof

**Goal.** Turn the application slides into a measured result.

**Why.** Slides 37-39 show plots. A plot cannot be graded; precision, recall and false-alarm rate can.

**Steps.**
1. Use a household-electricity dataset. Mark the normal part by a rule you fix before testing.
2. Inject labelled anomalies into a copy: spikes (3x), drops to zero, level shifts, a missing-day pattern. Keep the injection seed and positions in a file.
3. Detector A: your method (96-step context, many samples, flag outside the interval).
4. Detectors B-D: rolling z-score; STL decomposition + median absolute deviation; Isolation Forest.
5. Metrics: precision, recall, F1 at a fixed false-alarm rate (for example 1 per 1,000 points), and the empirical coverage of the interval on normal points (from P4).
6. Compare raw Chronos interval vs conformal interval (P4) vs fine-tuned model (P6).

**Done when.** One table of detectors with precision and recall, plus a statement of the cost (seconds per 1,000 points).
**Risk.** Sampling 200 paths per point is slow. Batch the contexts, use 50 samples first, and report the cost.

---

## 4. Your existing suggestions: where each one goes

| Your suggestion (slides) | Status from the evidence | Where it goes now |
|--------------------------|--------------------------|-------------------|
| v1 head: Gaussian soft binning, hybrid mixing, ordinal embedding | Head trained but never used in the winning benchmarks; real runs 3.6x worse (F2) | P5 (native logits, no new head). Ordinal embedding is not needed once the loss acts on the native logits |
| v1 loss: ordinal CE 0.5 + smooth label 0.3 + EMD 0.2 | Not evaluated against a plain fine-tune | P5 grid (`kind`, `lambda`) with the M1 control |
| v2: self-attention, deep projection, skip path, regularisation, focal loss | 8.1M new parameters on frozen encoder states; no result | P5: try only if the simple loss works; ablate each part |
| Inference: Gaussian-weighted robust averaging | Gains inside noise (F3) | P1 arm A3, with sigma tied to sample spread |
| Adaptive tokenization | No-op as coded (F4a) | P2 arms B1-B3 (level removal) |
| Ensemble sampling + temperature | No-op as coded (F4b) | P1 arms A2, A4 |
| Context length optimization | Can only shorten the context (F4c) | P2 arm B4 (multi-context ensemble) |
| Anomaly detection with prediction bands | Plot only | P4 (calibration) + P9 (measured) |

---

## 5. Fix list before you reuse anything

1. `dm_eval_3/.../VERSION_COMPARISON.md` reports v2 "RMSE 1597.27, MAPE 23456789.12, MAE TBD". No log in the repository contains these numbers (the v2 log shows MAE 8623.4). Remove or replace them.
2. The deck slide 36 and the Hugging Face model cards `distance-aware-chronos-t2` and `-t3` describe trained behaviour that the uploaded `base_model` folder does not contain (after check C1, correct the cards).
3. Set seeds (`torch.manual_seed`, `numpy.random.seed`) in every benchmark script.
4. Raise an error when a dataset loads 0 series.
5. Report horizons, window counts and series counts in every result table.
6. Use the Chronos tokenizer from the pipeline (`pipeline.tokenizer`) instead of re-implementing it.
7. Delete `__pycache__/*.pyc` and add a `.gitignore` entry.
8. Do not call a result "unseen" without checking the Chronos training dataset list.

---

## 6. Checks you can run now

**C1 - Are the "trained" weights the baseline weights?** (needs internet and `pip install huggingface_hub safetensors`; I could not run it)

```python
from huggingface_hub import hf_hub_download
import hashlib
def sha(repo, filename, **kw):
    p = hf_hub_download(repo, filename, **kw)
    return hashlib.sha256(open(p, "rb").read()).hexdigest()
a = sha("amazon/chronos-t5-small", "model.safetensors")
b = sha("Phoenix21/distance-aware-chronos-t3", "base_model/model.safetensors")
print(a == b)    # True  -> the "distance-aware" arm in the benchmark is the baseline
```
If the file names differ, list the repo files first (`HfApi().list_repo_files(repo)`). If the hashes differ, compare tensors one by one (`safetensors.torch.load_file`) and tell me: F1 then changes.

**C2 - Evidence script** (runs offline, numpy only): `python docs/evidence/verify_claims.py`. It prints the tables behind F3 and F4 and the loss floor.

---

## 7. Instructions for Claude (hand this file to Claude to implement)

Work on branch `pathways/beat-chronos-baseline` (or a child branch per pathway).

- Never invent a number. Every number in a README, a slide or a resume line comes from a file in `results/`.
- Implement P0 first. Do not start P1-P9 until `results/NOISE_FLOOR.md` exists.
- Create `eval/` (harness, metrics, report), `experiments/pXX_name/` (one folder per pathway, with `config.yaml`, `run.py`, `RESULTS.md`), `results/raw/` (parquet), `results/tables/` (csv).
- Fix all seeds; log package versions and the git commit in every result file.
- Use the pipeline's own tokenizer; keep context at 512 for t5-small.
- For every pathway, stop and write `RESULTS.md` with: arms, relative MASE and WQL with 95% intervals, the S1-S4 verdict, and one sentence on what to do next.
- Never tune on the test window. Never train on a test series.
- Ask the owner before running anything that takes more than 8 hours of compute.
- Open one pull request per pathway; keep `main` unchanged until the owner approves.

---

## 8. Questions for you

1. Which baseline do you want to beat in the end: `chronos-t5-small` zero-shot (assumed here), a bigger Chronos, or Chronos-Bolt?
2. Is the target a general zero-shot gain (P1-P5) or the best result on one domain such as electricity (P6, P9)? The second is much easier to win.
3. What hardware do you have for training (M3 only, Colab, a GPU)?
4. Is there a deadline (for example a review date)? With a short deadline, do P0, P1, P3, P4 and P9 only.

---

## 9. Sources

- Chronos paper: https://arxiv.org/abs/2403.07815 (tokenization, loss "not distance-aware", 20 samples, WQL/MASE, TSMixup, KernelSynth, fine-tuning, limitations).
- Chronos-2 report: https://arxiv.org/abs/2510.15821 and the Amazon Science blog "Introducing Chronos-2" (20 Oct 2025).
- Chronos-Bolt model card: https://huggingface.co/amazon/chronos-bolt-base (speed, memory, sizes).
- Farebrother et al., "Stop Regressing: Training Value Functions via Classification for Scalable Deep RL" (ICML 2024), https://arxiv.org/abs/2403.03950 (HL-Gauss, sigma = 0.75 bin widths). This is evidence from reinforcement learning, not from forecasting; treat the transfer as a hypothesis.
- `chronos-forecasting` 1.5.3 source (`chronos.py`: `predict()` arguments, token shift, EOS, context cut at `context_length`).
- Your files: this repository; `dm_eval_3` (`improvement/p1`, `improvement/p3_model/distanceAware/v1`).
