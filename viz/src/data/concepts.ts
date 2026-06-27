export interface InlineLink {
  match: string;
  url: string;
}

export interface ConceptSource {
  url: string;
  label: string;
  badge: string;
  type: "video" | "article" | "wiki";
}

export interface ConceptDef {
  short: string;
  analogy?: string;
  source: ConceptSource;
  inlineLinks?: InlineLink[];
  detail?: string;
}

const CONCEPTS: Record<string, ConceptDef> = {
  parameter: {
    short:
      "A number the model learns during training. Like a dial the model turns to make better predictions. This model has about 4,200 of them.",
    analogy:
      "A calculator has fixed formulas — this model has 4,224 adjustable ones it figured out on its own.",
    source: {
      url: "https://www.3blue1brown.com/lessons/neural-networks",
      label: "3blue1brown · Neural Networks ↗",
      badge: "▶ 3blue1brown",
      type: "video",
    },
    detail:
      "Each parameter starts as a random number and is gradually adjusted so the model's output gets closer to the correct answer. The total (4,224) comes from: token embeddings (432), positional embeddings (256), two RMSNorm layers (32), attention weights (1,024), MLP weights (2,048), and the LM head (432).",
  },
  transformer: {
    short:
      "A type of neural network that figures out which parts of the input matter most using a mechanism called 'attention.' It's the architecture behind modern language models.",
    analogy:
      "Like a very fast reader that highlights the most relevant words before making a decision.",
    source: {
      url: "https://www.3blue1brown.com/lessons/attention",
      label: "3blue1brown · Attention ↗",
      badge: "▶ 3blue1brown",
      type: "video",
    },
    detail:
      "Unlike older models that process words one by one in order, a transformer looks at all positions at once and uses attention scores to decide which earlier characters are most relevant for predicting the next one. This makes it both more powerful and more parallelisable than recurrent networks.",
  },
  embedding: {
    short:
      "Converting a character (like 'e') into a list of numbers the model can do math with. The model learns a good list for each character during training.",
    analogy:
      "Like describing a colour with RGB: 3 numbers. Use 16 and you capture finer distinctions.",
    source: {
      url: "https://jalammar.github.io/illustrated-word2vec/",
      label: "jalammar.github.io · The Illustrated Word2vec ↗",
      badge: "jalammar.github.io",
      type: "article",
    },
    detail:
      "Think of it as a lookup table with one row per vocabulary token. The row for 'a' contains 16 numbers, the row for 'b' has a different set of 16, and so on. Similar characters end up with somewhat similar number lists.",
  },
  positionalEncoding: {
    short:
      "Adds position information (1st, 2nd, 3rd…) so the model knows the order of characters - otherwise 'emma' and 'amme' would look identical.",
    analogy:
      "Like numbering the pages of a book so chapters stay in order.",
    source: {
      url: "https://www.3blue1brown.com/lessons/attention",
      label: "3blue1brown · Attention ↗",
      badge: "▶ 3blue1brown",
      type: "video",
    },
    detail:
      "This uses sine and cosine waves at different frequencies, inspired by the original 'Attention Is All You Need' paper. Each position gets a unique pattern that the model can learn to recognise.",
  },
  attention: {
    short:
      "A mechanism that lets each position look at the positions before it and decide which ones are most relevant for predicting the next character.",
    analogy:
      "Like a spotlight that brightens the most relevant words on the page.",
    source: {
      url: "https://www.3blue1brown.com/lessons/attention",
      label: "3blue1brown · Attention ↗",
      badge: "▶ 3blue1brown",
      type: "video",
    },
    detail:
      "Each position computes a 'query' and compares it to the 'keys' of all earlier positions via a dot product. The resulting scores are turned into weights via softmax, and the output is a weighted sum of 'value' vectors - positions with higher weight contribute more.",
  },
  softmax: {
    short:
      "A mathematical operation that turns a list of scores into probabilities that add up to 1.0, so the model can pick one option.",
    analogy:
      "Like ranking contestants by applause — the loudest gets first place, but everyone gets some share.",
    source: {
      url: "https://en.wikipedia.org/wiki/Softmax_function",
      label: "Wikipedia · Softmax function ↗",
      badge: "Wikipedia",
      type: "wiki",
    },
    detail:
      "Softmax exponentiates each score (making them all positive), then divides by their sum. The highest score gets the highest probability. 'Temperature' controls how sharp or flat the distribution is.",
  },
  logits: {
    short:
      "The raw scores the model produces for each possible next character - before they're turned into probabilities. Higher score = the model thinks it's more likely.",
    analogy:
      "Like athletes' raw scores before the final ranking is calculated.",
    source: {
      url: "https://en.wikipedia.org/wiki/Softmax_function",
      label: "Wikipedia · Softmax function ↗",
      badge: "Wikipedia",
      type: "wiki",
    },
    detail:
      "The LM head (a linear projection) converts the 16-dimensional hidden state into 27 logits - one per vocabulary token. These logits then pass through softmax to become probabilities.",
  },
  loss: {
    short:
      "A score that measures how wrong the model's prediction was. Lower is better. The model trains by trying to make this number smaller.",
    analogy:
      "Like a penalty score in a game — the model plays 10,000 rounds and gets less wrong each time.",
    source: {
      url: "https://en.wikipedia.org/wiki/Cross-entropy",
      label: "Wikipedia · Cross-entropy ↗",
      badge: "Wikipedia",
      type: "wiki",
    },
    detail:
      "This model uses 'cross-entropy loss.' If the model predicts a 10% chance for the correct next character, the loss is higher than if it predicted 90%. The perfect score would be 0 (100% confidence on every correct answer).",
  },
  forwardPass: {
    short:
      "Running an input through the model from start to finish - from character → embedding → attention → prediction - to get an output.",
    analogy:
      "Like a factory assembly line — each station transforms the product and passes it to the next.",
    source: {
      url: "https://www.3blue1brown.com/lessons/neural-networks",
      label: "3blue1brown · Neural Networks ↗",
      badge: "▶ 3blue1brown",
      type: "video",
    },
    detail:
      "In the forward pass, data flows through each layer in order: token lookup, positional encoding, RMSNorm, attention, residual add, more RMSNorm, MLP, another residual add, and finally the LM head and softmax. No learning happens here - just computation.",
  },
  backwardPass: {
    short:
      "Working backwards through the model to calculate how much each dial (parameter) contributed to the error, so we know how to adjust it.",
    analogy:
      "Like retracing your steps to find where you dropped your keys.",
    source: {
      url: "https://www.3blue1brown.com/lessons/neural-networks",
      label: "3blue1brown · Neural Networks ↗",
      badge: "▶ 3blue1brown",
      type: "video",
    },
    detail:
      "The backward pass uses the chain rule from calculus to compute gradients - one for every parameter. Each gradient tells us the direction and magnitude to adjust that parameter to reduce the loss. This is the core of how neural networks learn.",
  },
  rmsnorm: {
    short:
      "A gentle rescaling that keeps numbers from getting too big or too small as they pass through the network's layers.",
    analogy:
      "Like an automatic volume limiter — keeps everything audible without clipping.",
    source: {
      url: "https://en.wikipedia.org/wiki/Layer_normalization",
      label: "Wikipedia · Layer normalisation ↗",
      badge: "Wikipedia",
      type: "wiki",
    },
    detail:
      "RMSNorm divides each value by the root-mean-square of all values in that vector, then multiplies by a learned scale factor (gamma). It's a simpler variant of LayerNorm that doesn't recenter the mean.",
  },
  MLP: {
    short:
      "A small feed-forward network that processes each position independently, expanding from 16 numbers to 64 then back to 16.",
    analogy:
      "Attention decides what to look at; the MLP decides what to do with it.",
    source: {
      url: "https://www.3blue1brown.com/lessons/neural-networks",
      label: "3blue1brown · But what is a neural network? ↗",
      badge: "▶ 3blue1brown",
      type: "video",
    },
    detail:
      "The MLP (multi-layer perceptron) consists of two linear layers with a ReLU activation in between. The first layer projects 16→64 (giving the model more capacity to transform the representation) and the second compresses 64→16 (returning to the residual stream size).",
  },
  residual: {
    short:
      "A shortcut that adds the original input directly to a layer's output. This helps information flow through the network without degrading.",
    analogy:
      "Like taking notes in the margin instead of rewriting the whole page — the original is always preserved.",
    source: {
      url: "https://mbrenndoerfer.com/writing/residual-connections-gradient-highways-deep-transformers",
      label: "mbrenndoerfer.com · Residual Connections ↗",
      badge: "mbrenndoerfer.com",
      type: "article",
    },
    detail:
      "Residual connections (also called skip connections) solve the vanishing-gradient problem: if a layer doesn't learn anything useful, the gradient can still flow through the shortcut. The output of each sub-layer becomes: output = layer(input) + input.",
  },
  autoregressive: {
    short:
      "Generating one character at a time, feeding each new output back in as input - like writing a sentence word by word.",
    analogy:
      "Like texting one letter at a time: each character influences what feels natural next.",
    source: {
      url: "https://www.youtube.com/watch?v=kCc8FmEb1nY",
      label: "Karpathy · Let's build GPT ↗",
      badge: "▶ Karpathy",
      type: "video",
    },
    detail:
      "The model starts with just the BOS (beginning-of-sequence) token, predicts the next character, appends it to the input, predicts the next one, and repeats until it outputs the EOS (end-of-sequence) token. Each step only sees the characters it has generated so far.",
  },
  overfitting: {
    short:
      "When the model memorises the training data instead of learning general patterns - it does well on names it has seen but worse on new ones.",
    analogy:
      "Like memorising the answer key instead of learning the subject — you ace that test but fail the next one.",
    source: {
      url: "https://en.wikipedia.org/wiki/Overfitting",
      label: "Wikipedia · Overfitting ↗",
      badge: "Wikipedia",
      type: "wiki",
    },
    detail:
      "Signs of overfitting include the training loss continuing to drop while the validation loss plateaus or rises. In this model, the gap between train (2.50) and val (2.86) loss suggests mild overfitting. Solutions include more training data, regularisation, or a smaller model.",
  },
  vocabulary: {
    short:
      "The set of symbols the model knows. Here: the 26 lowercase letters plus a boundary marker (·), for 27 total.",
    analogy:
      "Like a 27-key typewriter — every name has to be spelled with just those keys.",
    source: {
      url: "https://en.wikipedia.org/wiki/Tokenization",
      label: "Wikipedia · Tokenization ↗",
      badge: "Wikipedia",
      type: "wiki",
    },
    detail:
      "The boundary marker acts as both beginning-of-sequence (BOS) and end-of-sequence (EOS). The model always starts generating from BOS and stops when it predicts BOS again. Anything outside the vocabulary (like capital letters or punctuation) is ignored.",
  },
  d_model: {
    short:
      "The size of the model's internal 'working space' for each position - here 16 numbers. Larger d_model means more capacity but more parameters.",
    analogy:
      "Like the width of a highway — wider roads handle more traffic but cost more to build.",
    source: {
      url: "https://jalammar.github.io/illustrated-word2vec/",
      label: "jalammar.github.io · The Illustrated Word2vec ↗",
      badge: "jalammar.github.io",
      type: "article",
    },
    detail:
      "Every token's representation in the residual stream is a vector of 16 numbers. All the layers (attention, MLP, RMSNorm) operate on and produce 16-dimensional vectors. The choice of 16 keeps the model small enough to train in a browser.",
  },
  relu: {
    short:
      "An activation function that zeroes out negative numbers and keeps positive numbers as-is. It adds non-linearity so the model can learn patterns.",
    analogy:
      "Like a gate that only lets positive people through — negatives are stopped at the door.",
    source: {
      url: "https://en.wikipedia.org/wiki/Rectifier_(neural_networks)",
      label: "Wikipedia · ReLU ↗",
      badge: "Wikipedia",
      type: "wiki",
    },
    detail:
      "ReLU (Rectified Linear Unit) is defined as f(x) = max(0, x). It's simple, fast, and helps prevent vanishing gradients. The non-linearity is crucial - without it, stacking linear layers wouldn't be more powerful than a single layer.",
  },
  regularization: {
    short:
      "Techniques that prevent the model from memorising and help it generalise better to new data.",
    analogy:
      "Like studying with both hands tied behind your back — harder to learn, but you learn what truly matters.",
    source: {
      url: "https://en.wikipedia.org/wiki/Regularization_(mathematics)",
      label: "Wikipedia · Regularization ↗",
      badge: "Wikipedia",
      type: "wiki",
    },
    detail:
      "Common regularisation methods include L2 weight decay (penalising large parameters), dropout (randomly turning off neurons during training), and data augmentation. This model doesn't use explicit regularisation, which partly explains the train-val gap.",
  },
  head: {
    short:
      "An 'attention head' is one independent attention computation. This model runs 4 heads in parallel, each focusing on different patterns.",
    analogy:
      "Like 4 readers skimming the same sentence, each highlighting different things, then comparing notes.",
    source: {
      url: "https://www.3blue1brown.com/lessons/attention",
      label: "3blue1brown · Attention in Transformers ↗",
      badge: "▶ 3blue1brown",
      type: "video",
    },
    detail:
      "Multi-head attention splits the 16-dimensional space into 4 groups of 4 dimensions each. Each head independently computes queries, keys, values, and attention weights on its subset. The outputs are concatenated and projected back to 16 dimensions. Each head can learn to attend to different types of relationships.",
  },
};

export default CONCEPTS;
