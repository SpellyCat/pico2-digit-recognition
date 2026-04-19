import board
import digitalio
import time
import sys
import json
import math

try:
    from pico_model_balanced import SCALER_MEAN, SCALER_SCALE, WEIGHTS1, BIASES1, WEIGHTS2, BIASES2
except Exception as e:
    print(json.dumps({"error": "model_import_failed", "detail": str(e)}))
    raise

led = digitalio.DigitalInOut(board.LED)
led.direction = digitalio.Direction.OUTPUT


def blink_led():
    for _ in range(2):
        led.value = True
        time.sleep(0.03)
        led.value = False
        time.sleep(0.03)


def tanh(x):
    if x > 8.0:
        return 1.0
    if x < -8.0:
        return -1.0
    e_pos = math.exp(x)
    e_neg = math.exp(-x)
    denom = e_pos + e_neg
    if denom == 0.0:
        return 0.0
    return (e_pos - e_neg) / denom


def softmax(logits):
    m = logits[0]
    for v in logits[1:]:
        if v > m:
            m = v
    exps = [math.exp(v - m) for v in logits]
    total = 0.0
    for e in exps:
        total += e
    if total == 0.0:
        return [0.0 for _ in exps]
    return [e / total for e in exps]


def predict(features):
    x = [0.0] * 64
    for i in range(64):
        x[i] = (float(features[i]) - SCALER_MEAN[i]) / SCALER_SCALE[i]

    hidden = [0.0] * len(BIASES1)
    for i in range(len(BIASES1)):
        acc = BIASES1[i]
        row = WEIGHTS1[i]
        for j in range(64):
            acc += x[j] * row[j]
        hidden[i] = tanh(acc)

    logits = [0.0] * len(BIASES2)
    for i in range(len(BIASES2)):
        acc = BIASES2[i]
        row = WEIGHTS2[i]
        for j in range(len(hidden)):
            acc += hidden[j] * row[j]
        logits[i] = acc

    probs = softmax(logits)
    best = 0
    best_p = probs[0]
    for i in range(1, len(probs)):
        if probs[i] > best_p:
            best_p = probs[i]
            best = i
    return best, best_p * 100.0


print("Ready")

while True:
    try:
        line = sys.stdin.readline()
        if not line:
            continue
        line = line.strip()
        if not line:
            continue

        try:
            features = json.loads(line)
        except Exception as e:
            print(json.dumps({"error": "bad_json", "detail": str(e)}))
            continue

        if not isinstance(features, list) or len(features) != 64:
            print(json.dumps({
                "error": "bad_feature_length",
                "expected": 64,
                "got": len(features) if isinstance(features, list) else None
            }))
            continue

        blink_led()
        digit, conf = predict(features)
        print(json.dumps({"digit": digit, "confidence": conf}))

    except Exception as e:
        print(json.dumps({"error": str(e)}))
