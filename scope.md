Here is a complete summary of everything we have designed so far. You now have a comprehensive, end-to-end architectural blueprint and the Python codebase to perform Representation Engineering on a state-of-the-art Vision-Language Model.

### What We Have Accomplished

We have successfully mapped the Contrastive Activation Addition (CAA) methodology onto the Gemma-3-12b-it model using the CLEVR visual reasoning dataset. Specifically, we have:

1. **Defined the Research Target:** We are isolating the concept of **"Counting"** (specifically an "Overcount by 1" hallucination).
2. **Selected the Evaluation Strategy:** We abandoned brittle free-form text generation in favor of **Token-Level Logit Masking**, allowing us to mathematically measure exactly how the probability of specific CLEVR vocabulary tokens shifts.
3. **Built the Core Tools:** We wrote the PyTorch functions to extract activations, hook into Gemma's residual stream, and mask the vocabulary.
4. **Designed the Batch Pipelines:** We created dedicated PyTorch `Dataset` and `DataLoader` structures that maintain exact token-alignment (using Batch Size = 1) for both extraction and evaluation.

---

### The Master Chronological Execution Plan

To actually run this experiment on your machine or cluster, you must execute the code we wrote in this exact, four-phase order:

#### Phase 1: Data Preparation (Creating the Splits)

*You start with the raw, downloaded CLEVR dataset.*

1. **Run the Preparation Script:** Execute the script we just wrote (`prepare_counting_extraction_dataset`).
2. **What it does:** It scans the `CLEVR_train_questions.json`, finds valid counting questions, calculates the `+1` target answer, and formats the exact text strings the model needs to see.
3. **Output:** It generates a flat file named `clevr_caa_extract.jsonl`.
4. **Next Step:** You will need to write/run a nearly identical script on the `CLEVR_val_questions.json` to generate your `clevr_caa_eval.jsonl` (the held-out test set).

#### Phase 2: Vector Extraction (Mapping the Concept)

*You now have your extraction data and your Gemma 3 model loaded.*

1. **Load the Extraction DataLoader:** Feed `clevr_caa_extract.jsonl` into the `CLEVRExtractionDataset` class.
2. **Run the Extraction Loop:** Execute the `calculate_dataset_steering_vector` function.
3. **What it does:** It pushes the positive ("Answer: 3") and negative ("Answer: 2") pairs through Gemma, grabs the activations at the last token for a specific layer (e.g., Layer 15), and computes the Mean Difference vector across the whole dataset.
4. **Output:** You save the resulting tensor to disk (e.g., `clevr_overcount_plus1_layer15.pt`).

#### Phase 3: Intervention & Evaluation (Testing the Vector)

*You now have a saved steering vector and your evaluation data.*

1. **Load the Evaluation DataLoader:** Feed your test set (`clevr_caa_eval.jsonl`) into the `CLEVRCAADataset` class.
2. **Load the Vector & Define Tokens:** Load your saved `.pt` vector from disk. Define your `allowed_token_ids` to mask everything except valid CLEVR answers.
3. **Run the Evaluation Loop:** Execute the `evaluate_caa_dataset` function, passing in your `multiplier` (e.g., `1.5`).
4. **What it does:** It loops through unseen images. The `SteeringHook` intercepts the forward pass and injects your vector. The logit masker forces the model to pick the most likely CLEVR answer.
5. **Output:** A dictionary of raw counts (how many times it output the base truth vs. the target hallucination).

#### Phase 4: Analysis (The Results)

*You have finished running the model.*

1. **Run the Metrics Function:** Pass your results dictionary into the `print_final_metrics` function.
2. **What it does:** It calculates the Base Accuracy, Shifted Accuracy, and average token probabilities, sliced by question type.
3. **The Final Goal:** You will plot these numbers to prove that as you increased the steering multiplier, Gemma's natural counting accuracy plummeted, and your artificially injected "+1 Overcount" accuracy spiked—all without breaking the model's ability to answer color or shape questions.

You now have the full theoretical and practical pipeline. Are you planning to run this locally on a multi-GPU setup, or are you deploying this to a cloud cluster? Knowing the hardware might dictate if we need to add DeepSpeed or specific quantization wrappers to the Gemma loading code.