# Emo-TTS

Emo-TTS is an ensemble-based system designed to generate emotionally aligned speech responses. It is divided into three sections:

1. **Emotion & Sentiment Understanding**
2. **Emotion-Conditioned Response Generation**
3. **Emotionally Expressive Speech Synthesis**

---

<!-- ## Section 1: Emotion & Sentiment Detection

This section evaluates the user’s emotional state using multimodal input.

### Input Processing
- User speech is captured and converted to text.
- ASR models such as Whisper can be used, though improved alternatives may be explored.

### Sentiment & Emotion Modeling
Various approaches have been tested:
- K-means
- K-NN
- Neural Networks
- LLM-based analysis

All experiments use BERT-based embeddings to represent user text. These baseline methods focus only on semantic content and do not yet incorporate vocal tone, prosody, or facial expressions.

### Personalization
A future enhancement involves leveraging concepts inspired by neurogenesis and neuroplasticity to adapt emotional understanding and personalization over time. Implementation details are in progress.

---
 -->

---
## Section 2: Emotion-Aligned Response Generation

This section produces:
- A textual response to the user
- A set of target emotional attributes for speech synthesis

It can use:
- An LLM
- An Ollama-hosted model
- Third-party APIs (e.g., Gemini)

The output includes both the response itself and the emotion distribution needed by Section 3.

---

## Section 3: Emotionally Expressive Speech Synthesis

This component converts the output from Section 2 into expressive audio.

### Input Format

Example input:

```json
{
  "text": "Hello, I am happy today",
  "emotion": ["Cheerful", "Warm"],
  "percentage": ["77%", "23%"]
}
```

The synthesis model outputs speech with the specified emotional blend.

### Voice Modeling
- Training data is based on the dataset laion/Emilia-with-Emotion-Annotations.
- Emotional percentages guide the selection of voice samples matching the desired mix.

### Core TTS Model
- We have tested on multiple different models and based on human perception, we have chosen CosyVoice V3
- [CosyVoice V3](https://funaudiollm.github.io/cosyvoice3/)