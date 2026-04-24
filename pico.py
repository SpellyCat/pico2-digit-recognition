import board
import digitalio
import time
import sys
import json
import math
import gc

try:
    with open("pico_mnist_model.json", "r") as f:
        model_data = json.load(f)
        
    HIDDEN_UNITS = model_data["h_units"]
    LOGIT_SCALE = model_data["log_s"]
    WEIGHTS1 = model_data["w1"]
    BIASES1 = model_data["b1"]
    W1_SCALE = model_data["w1_s"]
    WEIGHTS2 = model_data["w2"]
    BIASES2 = model_data["b2"]
    W2_SCALE = model_data["w2_s"]
    
    # 强制释放解析大型 JSON 树时产生的极其庞大的中间字典内存占用
    del model_data
    gc.collect()
    
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


def relu(x):
    return x if x > 0.0 else 0.0


def softmax(logits):
    m = logits[0]
    for v in logits[1:]:
        if v > m: m = v
    exps = [math.exp(v - m) for v in logits]
    total = sum(exps)
    if total == 0.0:
        return [0.0 for _ in exps]
    return [e / total for e in exps]


def predict(features):
    hidden = [0.0] * HIDDEN_UNITS
    for i in range(HIDDEN_UNITS):
        acc = BIASES1[i]
        row = WEIGHTS1[i]
        # 直接利用融合后的参数处理 784 个未降维的像素输入，一步完成全部变换与网络传播
        for j in range(784):
            acc += features[j] * (row[j] * W1_SCALE)
        hidden[i] = relu(acc)

    logits = [0.0] * 10
    for i in range(10):
        acc = BIASES2[i]
        row = WEIGHTS2[i]
        for j in range(HIDDEN_UNITS):
            acc += hidden[j] * (row[j] * W2_SCALE)
        logits[i] = acc * LOGIT_SCALE

    probs = softmax(logits)
    best = 0
    best_p = probs[0]
    for i in range(1, 10):
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

        if not isinstance(features, list) or len(features) != 784:
            print(json.dumps({
                "error": "bad_feature_length",
                "expected": 784,
                "got": len(features) if isinstance(features, list) else None,
            }))
            continue

        blink_led()
        digit, conf = predict(features)
        print(json.dumps({"digit": digit, "confidence": conf}))

    except Exception as e:
        print(json.dumps({"error": str(e)}))