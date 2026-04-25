# pico2-digit-recognition

> A handwritten digit recognizer running on Pi Pico 2, vibe-coded with:
>
> - Gemini 3.1 Pro - template, model & UI refinement, also cleaning the spaghetti code left by GPT
> - DeepSeek v4-flash - UI enhancement & port selector
> - ChatGPT - writing code for the initial version of using MNIST model (code in Legacy Release)

> [!TIP]
> env:
>
> * PC: Python 3.10
>
> * Pico: CircuitPython 10.1.4

## Screenshot
<img width="422" height="266" alt="image" src="https://github.com/user-attachments/assets/a9a90188-5dec-4783-95c2-507759cbebf1" />
>
<img width="521" height="630" alt="image" src="https://github.com/user-attachments/assets/ae8a04dc-a597-49aa-968e-3abfa961f8ef" />


## Setup

First, clone this repo:

```bash
git clone https://github.com/MeowCata/pico2-digit-recognition
cd pico2-digit-recognition
```

Install necessary libs: 

```python
pip install numpy==1.26.4 scikit-learn==1.4.2 pyserial==3.5 Pillow==10.3.0
```

Now let's train the model:

```python
python train_model.py
```

After training, `pico_mnist_model.json` will be generated

> Want pre-trained model? Download it clicking [here]()

Finally, upload the codes. Please drag-and-drop `code.py`(**renamed from pico.py**) and `pico_mnist_model.json` to Pico's disk(displayed as `CIRCUITPY`)

Once the codes uploaded, run `ui.py` on PC, select the serial port, and check out how this little model worked!

### Info
- [v1.1](https://github.com/MeowCata/pico2-digit-recognition/releases/tag/MLP_hu32_0): Nice version
- [Legacy](https://github.com/MeowCata/pico2-digit-recognition/releases/tag/MLP_hu64): A heavier model with a simplified UI
