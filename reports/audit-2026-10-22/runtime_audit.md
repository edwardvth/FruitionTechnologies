# Runtime, actuation, recording and deployment audit

Date: 2026-10-22. Reviewed Git HEAD `b44d6e69708eff6057818973a453b18e1dfd5db5` using `git show` because checkout was still materializing. No source modifications, model deserialization, camera access, GPIO access, or service installation occurred. No applicable tracked AGENTS.md was found; checked ancestor locations also had none.

Severity describes potential impact if the provided runtime is deployed with GPIO-connected laser hardware. The module says hardware is not yet connected (inference_pipeline.py:14–17); this audit cannot establish real deployment or physical hazard severity. Findings below concern implementation and do not establish exploitation, actual beam emission, or measured detector performance.

## R1 — High: last laser-enable decision survives camera loss and blocking work

**Confidence: high; reproduced with extracted code and mock GPIO.**

Evidence: `goose_detection/inference/inference_pipeline.py:297–308, 311–345, 351–355, 385–407`; laser state helpers at `105–124`; exit hooks at `127–141`.

After one goose-only frame sets output HIGH, `cap.read()` returning false only prints a warning and continues. No LOW is issued, so repeated camera failures preserve the previous HIGH command indefinitely while the process remains alive. Our two consecutive failed reads both observed HIGH; LOW happened only when the probe deliberately raised an exception to leave the loop.

Likewise, the previous HIGH remains asserted while the next read, goose inference, person inference, and debug predictions/image write execute. The probe observed HIGH on entry to the next tracking call. There is no independent expiration of an enable decision, freshness check, maximum activation period, or hardware watchdog represented in this repository. A blocking native operation can therefore delay the next safety decision without a Python exception. The existing finally/atexit/signal paths are useful for ordinary exits but do not make the stated “any exit path” guarantee true for process termination that cannot run Python cleanup or loss of power; pin state in those cases depends on hardware and is unverified.

**Remediation:** disable immediately on acquisition failure, bound consecutive failures and stop/recover capture, enforce frame age and safety-result age, and make every enable command a short-lived permission with independently enforced expiry. Specify and verify the default-off driver, independent watchdog, interlocks, and emergency stop for the actual hardware. Add mock regression cases for camera loss, stalled inference and output failures before enabling actuation. An in-loop timeout alone cannot cover a hung loop.

## R2 — High: hardware activation is automatic when GPIO is available

**Confidence: high; static evidence.**

Evidence: `goose_detection/inference/inference_pipeline.py:91–102, 253–260, 351–355, 438–478`; `goose_detection/scripts/goose-detector.service:11–14`; `goose_detection/scripts/goose-detector.desktop:5`.

The normal live-camera command always calls setup_gpio. If Jetson.GPIO imports and pin setup succeeds, detections drive the real output; simulation is only an exception fallback. There is no explicit arm/enable-hardware option. Thus the same command used for demonstrations or the installed auto-start service can actuate newly connected hardware without any change to runtime arguments. The comment that hardware is not yet connected is not an enforced mode.

**Remediation:** default to explicit simulation, require deliberate hardware mode and arming after validated configuration/self-checks, expose unmistakable mode/status, and make GPIO setup failures fatal in hardware mode rather than silently switching behavior. Reboots must have a defined safe arming policy.

## R3 — High: safety model identity and class mapping are never validated

**Confidence: high for missing validation; deployment-specific consequence.**

Evidence: `goose_detection/inference/inference_pipeline.py:48–53, 149–153, 190–207, 440–447, 460–468`.

Both custom goose and stock person models use class ID 0, with meaning supplied only by a constant/comment. The configurable `--safety-weights` is loaded without verifying task, class names, expected model identity, or an approved artifact digest. Passing a valid detector whose person class has another ID filters out those people, while returning an empty result is interpreted as permission when the goose model reports a goose. No assertion checks that `model.names[PERSON_CLASS_ID]` is `person`. The default is intended to be the COCO model; this finding does not claim its actual class mapping is wrong. Only the goose weight path gets a preflight existence check; the safety default explicitly permits downloading on first use.

**Remediation:** pin and verify the approved safety artifact, assert task and full class mapping before GPIO setup, reject missing/incompatible weights, and validate confidence thresholds as finite values in the intended range. Derive class IDs from verified metadata. Keep independent safety validation tied to the exact deployed artifact/version.

## R4 — Medium: recorder exceptions can silently end recording and block shutdown

**Confidence: high; exception propagation and unbounded enqueue reproduced.**

Evidence: `goose_detection/inference/clip_recorder.py:65–71, 90–114, 148–158, 178–180`; pipeline cleanup `inference_pipeline.py:401–407`; service `goose-detector.service:17–18`.

The worker loop has no exception handling around `_process`. An encoder exception can end the daemon worker while inference keeps accepting frames until the queue fills and then drops them. `close()` performs a blocking queue put before its 30-second join: if the worker is dead or stalled and the queue is full, the join timeout is never reached. The mock probe confirmed a simulated encoder OSError escapes and the sentinel put has no timeout. The service allows only 10 seconds to stop, less than even the documented join budget, so pending clips may be abandoned at forced termination. This is primarily a recording/recovery issue: the normal loop finally attempts laser_off before recorder.close, so it should not be misreported as a direct HIGH-on-recorder-close issue.

**Remediation:** catch/report worker errors through an explicit health channel, make shutdown enqueue and joins bounded, ensure failures unblock producers/cleanup, and align service stop budgets with the recorder design. Test a stopped worker, full queue, failed writer, and disk-full behavior with mocks and isolated filesystem tests.

## R5 — Medium: unattended clips, debug frames and log files have no storage limits

**Confidence: high for missing limits in reviewed implementation; exhaustion not exercised.**

Evidence: `goose_detection/inference/clip_recorder.py:129–139, 143–158, 193–230`; `inference_pipeline.py:344–345, 357–363`; `goose-detector.service:15–16`; `goose-detector.desktop:5`.

Continuous detections keep a clip open without a maximum duration or size; finalized clips have no deletion/quota policy. Debug mode saves frames repeatedly. Both launch modes append logs to a regular file, and no rotation configuration is provided. A long-running appliance can consume its storage, compromise recordings and affect other services. External system policy could mitigate this, but was not inspected. Camera failure also produces an unthrottled warning on every failed read (pipeline:299–302), accelerating log growth for fast-failing reads.

**Remediation:** configure bounded retention, free-space thresholds, maximum clip durations, log rotation or bounded journald output, error-rate limiting, and observable recording/storage health. Define what the actuator must do when recording/storage health is degraded.

## R6 — Low: clip timing metadata assumes a full pre-roll and fixed inference FPS

**Confidence: high; static code calculation.**

Evidence: `goose_detection/inference/clip_recorder.py:57, 82–88, 141–169, 198–202`; pipeline default FPS `inference_pipeline.py:58–60` and recorder input `385–392`.

A deque holds `fps * pre_roll_sec` frames, but inference arrival rate is variable and only frames (not timestamps) enter the ring. `_start_clip` always assigns `now - pre_roll_sec`, even when the ring is empty at startup. The first immediate detection therefore claims 10 seconds of pre-roll despite having none. At a different actual processing rate, the nominal frame count corresponds to a different wall-clock duration; video playback FPS is still fixed. This undermines incident reconstruction and duration interpretation.

**Remediation:** buffer timestamped frames by elapsed duration, record actual earliest/latest captured timestamps and actual pre-roll, and explicitly distinguish video playback duration from wall-clock coverage. Test startup detections, low FPS and dropped frames.

## Safety validation gaps and useful improvements

- A person correctly detected in the next frame does disable the output; this positive control passed. However, one frame without a person immediately re-enables it when a goose is present (`inference_pipeline.py:347–355`). There is no post-person exclusion period or requirement for multiple fresh clear frames. That policy should be explicitly specified and tested; a lower confidence threshold alone cannot establish safe person recall.
- Run a separately labeled safety benchmark for people at the installation's distances, lighting, occlusion, motion and crowd conditions, including simultaneous geese and people. Report false-negative rate with uncertainty, and test camera freezes/occlusion and out-of-view hazards. This source review cannot certify a vision-only physical safety interlock.
- Recorder metadata includes goose confidence/tracks but no machine-readable person detections, laser transitions, model hashes, frame age, or safety reason (`clip_recorder.py:160–175` and pipeline recorder call `385–392`). Add structured safety-event evidence for diagnosis.
- Default ring storage is 150 frames, with a queue capacity of 60. At the specified 1280×720 BGR format, their nominal raw frame storage is about 415 MiB + 158 MiB when full, before models/encoders and other buffers. Frame copies also occur before queue-full detection. Profile peak memory and inference latency on the Jetson under sustained recording; this arithmetic is not a measured memory footprint.
- Service deployment uses a non-root user (positive), but hardcodes one username/path, a system Python interpreter, and lacks repository-provided watchdog/resource/retention configuration. Document a tested environment, install-time path validation and deliberate privilege/device permissions. Do not add sandboxing directives blindly that prevent CUDA/camera/GPIO access.

## Verification artifacts and limitations

Run `python3 reports/audit-2026-10-22/runtime_probes.py` from the repository root. `runtime_probe_output.txt` contains the observed results. The probe extracts selected functions/classes with AST, supplies mock dependencies, and executes only those definitions; it does not import repository modules. It demonstrates control-flow/output-command behavior, exception propagation and shutdown call arguments. It does not reproduce a real native hang or blocked queue, assert actual GPIO electrical behavior, load weights, measure person recall, test a camera, or run systemd. Source line ranges refer to the recorded HEAD, not an independently inspected deployment.
