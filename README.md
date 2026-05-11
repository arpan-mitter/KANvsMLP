KAN vs MLP — Benchmarking Kolmogorov-Arnold Networks
Against Multi-Layer Perceptrons 
A systematic comparison of Kolmogorov-Arnold Networks (KANs) against traditional Multi-Layer Perceptrons (MLPs) for function
approximation tasks, with a focus on parameter efficiency, interpretability, and symbolic regression capability — evaluated through
the lens of edge AI deployment constraints. 
 Background 
In 2024, Liu et al. proposed KANs as a fundamentally different neural architecture based on the Kolmogorov-Arnold representation
theorem. Unlike MLPs — where fixed activation functions sit on nodes and learnable weights sit on edges — KANs place learnable
activation functions on edges (implemented as B-splines), with no fixed nonlinearities on nodes. 
The key claims from the original paper: - KANs achieve comparable or better accuracy than MLPs with fewer parameters - KANs are
more interpretable — learned activations can be visualised and symbolically simplified into mathematical expressions - KANs are
particularly suited to scientific discovery tasks where symbolic function recovery matters 
This project tests those claims empirically and explores implications for low-power edge deployment. 
 What This Project Does 
1. data_gen.py — Synthetic Function Generation 
Generates ground-truth datasets from known mathematical functions for controlled benchmarking. Using synthetic data means we
know the exact underlying function — allowing us to measure not just predictive accuracy but whether the model recovers the true
symbolic relationship. 
2. kan_model.py — KAN Architecture Implementation 
Implements the KAN architecture with B-spline activation functions on edges. Includes configurable grid size (spline resolution) and
order parameters that control the expressiveness vs. computational cost tradeoff. 
3. main.py — Benchmarking Pipeline 
Trains and evaluates both KAN and MLP architectures under controlled conditions: - Matched parameter counts for fair comparison -
Identical training data, optimiser, and schedule - Metrics: MSE, parameter count, training time, convergence speed 
4. auto_scientist_1.py — Symbolic Regression via KAN 
The most experimental component. Leverages KAN’s interpretability to attempt automated symbolic function discovery — after
training, the learned B-spline activations are pruned and simplified to extract a symbolic mathematical expression approximating the
underlying data-generating function. 
This is the “auto-scientist” capability described in the original KAN paper: the network doesn’t just fit data, it attempts to discover the
formula. 
 Key FindingsMetric KAN MLPParameter efficiency (low-dim functions) ✅ Better ❌ WorseTraining speed ❌ Slower ✅ FasterInterpretability ✅ High (symbolic) ❌ Black boxHigh-dimensional scaling ❌ Degrades ✅ Scales wellSymbolic recovery accuracy ✅ Possible ❌ Not applicable 
Results are consistent with the independent “KAN or MLP: A Fairer Comparison” benchmark (Yu et al., 2024) — KANs
outperform MLPs primarily in symbolic formula representation; MLPs remain stronger for most ML tasks at scale. 
 Relevance to Edge AIThe parameter efficiency and interpretability advantages of KANs are specifically relevant to TinyML and edge deployment: 
Parameter efficiency: Embedded hardware (STM32, ESP32) has severe memory constraints. A model achieving equivalent accuracy
with fewer parameters has direct deployment advantages. 
Interpretability: In safety-critical embedded applications (industrial sensors, medical devices), being able to inspect and verify the
learned function matters. A KAN that symbolically recovers f(x) = sin(x) * x2 is auditable in a way a deep MLP is not. 
Compute cost tradeoff: KANs are slower to train but inference cost depends on grid size — small-grid KANs may be competitive for
edge inference on simple sensor-to-output mappings. 
 Limitations 
Benchmarks limited to low-dimensional synthetic functions
No evaluation on real sensor datasets
Auto-scientist symbolic recovery works best on clean, low-noise data
KAN training instability at higher grid sizes not fully characterised 
 Quickstart 
pip install torch numpy matplotlib pykan
 # Generate benchmark data
python data_gen.py
 # Run KAN vs MLP comparison
python main.py
 # Run symbolic regression experiment
python "auto scientist_1.py"

 Related Work 
Liu et al., KAN: Kolmogorov-Arnold Networks (2024) — arxiv.org/abs/2404.19756
Yu et al., KAN or MLP: A Fairer Comparison (2024) — arxiv.org/abs/2407.16674
KindXiaoming/pykan — github.com/KindXiaoming/pykan 
 Relation to Other Projects 
The parameter efficiency and interpretability findings here directly informed the architecture choices in the MATD3 swarm project —
where edge deployment constraints require careful model size management: 
KANvsMLP (this repo) — which architecture fits edge constraints better?
↓ informed model design decisions
MATD3 swarm — bandwidth-aware gated execution on resource-constrained hardware
