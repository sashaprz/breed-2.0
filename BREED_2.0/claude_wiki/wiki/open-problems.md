# Open Problems

Living document: bottlenecks and active research questions in SSE + ML research.

---

## Data scarcity for rare material classes
- Description: Training sets for ML models on SSEs are typically small (tens to low hundreds of materials), especially for less-studied classes (sulfides, halides, specific NASICON compositions). This limits model accuracy and generalizability to new chemical spaces.
- Why it's hard: Experimental data is spread across heterogeneous literature sources with inconsistent measurement conditions. Computational data (DFT, AIMD) is expensive to generate at scale.
- Approaches tried: Curated databases (Hargreaves et al. 820-entry Li-ion dataset); Materials Project subsets; high-throughput DFT generation; transfer learning across material families.
- Sources: [[machine-learning-assisted-property-prediction-solid-state]], [[unsupervised-machine-learning-accelerates-solid]], [[data-science-approach-advanced-solid-polymer]]
- Status: partial progress

---

## Feature engineering subjectivity and scalability
- Description: Most ML models for SSEs use manually curated composition or structure features, which reflect researcher intuition. This is laborious, potentially biased, and doesn't automatically scale to new material families.
- Why it's hard: Optimal features are property- and crystal-structure-specific. There's no universal descriptor set that works well across all SSE classes.
- Approaches tried: HECS (hierarchically encoded crystal structure-based) descriptors for argyrodites; behavior-based descriptors from MD trajectories; minimal 3-input-per-atom neural networks (Lu et al.) that auto-learn secondary features; GNNs (CGCNN, MEGNet) that learn from graphs; geometry-based descriptors (ionic radii, octahedral factors, space filling) for garnets (Kireeva 2017); site-specific features (ionization energy, electronegativity, electron affinity per crystallographic site) outperform material-level averages (Ma et al. 2024).
- Sources: [[machine-learning-assisted-property-prediction-solid-state]], [[materials-space-of-solid-state-electrolytes]], [[improving-ionic-conductivity-garnet-gradient-boosting]]
- Status: partial progress

---

## Multi-property trade-offs in SSE design
- Description: Optimizing a single SSE property (e.g., ionic conductivity) typically degrades others (mechanical stability, chemical stability, wide electrochemical window). No framework reliably finds the Pareto front across all relevant properties.
- Why it's hard: The property space is high-dimensional, the objectives partially conflict, and labeled multi-property data is even sparser than single-property data. High Li-ion conductivity empirically correlates with metastability and poor ambient stability (Famprikis 2019): the fastest conductors (LGPS, argyrodites) are kinetically trapped phases that are difficult to synthesize and sensitive to moisture/air.
- Approaches tried: Multi-objective Bayesian optimization; simultaneous prediction of conductivity + modulus; design of full-cell metrics rather than single properties.
- Sources: [[machine-learning-assisted-property-prediction-solid-state]], [[fundamentals-of-inorganic-solid-state-electrolytes]], [[designing-solid-state-electrolytes-safe-energy]]
- Status: open

---

## Model transferability across crystal structure families
- Description: ML models trained on one structure type (e.g., anti-perovskites) don't straightforwardly transfer to structurally dissimilar families. Features that matter for vacancy migration in one structure may be irrelevant in another.
- Why it's hard: The physics of ion transport is structure-dependent; general descriptors that capture all relevant physics don't exist yet.
- Approaches tried: Transfer learning with domain adaptation; hierarchical models; universal graph neural network potentials (M3GNet, MACE) as feature extractors.
- Sources: [[machine-learning-assisted-property-prediction-solid-state]]
- Status: open

---

## Random k-fold CV as the dominant evaluation methodology
- Description: Almost all ML papers on SSE properties use random k-fold cross-validation, which overestimates generalization performance when training and test materials share similar compositions or structures (data leakage).
- Why it's hard: Defining a meaningful "scaffold split" for crystalline inorganic materials is non-trivial — there's no canonical structural fingerprint analogous to molecular scaffolds. Chemical space clustering methods exist but aren't standard.
- Approaches tried: Leave-one-cluster-out (LOCO) CV; scaffold splits based on structure type; leave-one-composition-out. Rarely used in practice. OBELiX (2025) quantifies the inflation for the first time on a large SSE dataset, using Monte Carlo optimization over composition+paper groupings to define a leakage-free split; the performance gap between random CV and leakage-free evaluation is substantial.
- Sources: [[machine-learning-assisted-property-prediction-solid-state]], [[materials-space-of-solid-state-electrolytes]], [[improving-ionic-conductivity-garnet-gradient-boosting]], [[data-science-approach-advanced-solid-polymer]], [[harnessing-artificial-intelligence-holistic-design]], [[obelix]]
- Status: open (methodological gap in the field, not just one paper)

---

## Data heterogeneity and measurement irreproducibility
- Description: Experimental ionic conductivity for the same compound can vary by orders of magnitude across different studies, even using the same synthesis method. Adding more literature data to an ML dataset can reduce model accuracy if the new data was collected under different, unreported conditions. The same systematic measurement bias problem also affects stability window data: CV measurements overestimate ESWs by 2–3 V versus DFT-calculated and carbon-composite-cell results (Xiao 2020) — a reminder that this is a field-wide data quality problem, not specific to conductivity.
- Why it's hard: Synthesis conditions (sintering temperature, atmosphere, pressure, precursor purity), measurement setup (impedance frequency range, electrode geometry), and sample quality (grain boundary content, relative density) all affect measured conductivity but are inconsistently reported. There is no standard reporting template for SSE conductivity papers. Critically, whether a paper reports total conductivity (bulk + grain boundary) or grain-boundary-free bulk conductivity is rarely stated explicitly; in LLZO, grain boundaries account for 40–50% of total resistance (Zhao et al. 2020), creating a systematic ambiguity in training labels. Miao et al. (2023) quantify the GB penalty systematically: LLTO total σ is 2 orders of magnitude below bulk (10⁻⁵ vs 10⁻³ S/cm); NASICON is 1 order below bulk; LLZO is exceptional — GB and bulk conductivities are both ~10⁻⁴ S/cm. For BREED, this means LLTO and NASICON-class training labels may be 1–2 orders below intrinsic material values (systematic label bias), while LLZO labels are reliable.
- Approaches tried: Manual data curation with ad hoc deduplication (Kireeva & Pervov 2017); supplementing literature data with independently controlled experiments (Liu et al. 2021); outlier detection via Cook's distance (Ma et al. 2024, identified 5 outliers in 398-point garnet dataset). OBELiX (2025) quantifies the experimental reproducibility floor at MAD=0.41, RMSD=0.63 log₁₀(S/cm), establishing a hard lower bound on achievable test MAE from heterogeneous literature data.
- Sources: [[materials-space-of-solid-state-electrolytes]], [[data-science-approach-advanced-solid-polymer]], [[improving-ionic-conductivity-garnet-gradient-boosting]], [[obelix]], [[fundamentals-of-inorganic-solid-state-electrolytes]], [[designing-solid-state-electrolytes-safe-energy]], [[role-of-interfaces-solid-state-batteries]], [[understanding-interface-stability-solid-state]]
- Status: open

---

## Partial occupancy as a barrier to structure-based ML
- Description: Approximately 75% of CIF entries in OBELiX have partial site occupancy (crystallographic disorder). Standard GNN architectures require integer site occupancies and round fractional values to integers, destroying the disorder physics that is the primary enabler of Li⁺ conduction in many high-performing SSEs (argyrodites, Li-stuffed garnets).
- Why it's hard: Crystallographic disorder is not a simple property to encode — it represents a statistical average over many local configurations, and the relevant quantity for conductivity is the site-to-site correlation function, not just the mean occupancy. This requires either explicit supercell enumeration (exponentially expensive) or a descriptor that encodes disorder statistics compactly.
- Approaches tried: Disorder-aware GNN variants (dis-CGCNN, dis-SO3Net) improve marginally but remain far below RF/MLP on OBELiX. LLMs (Kim et al. 2026) encode fractional occupancies as text and achieve the best current performance, but without mechanistic interpretability. BVSE-based methods (Hashizume 2026) address a different aspect (polymorphism) but not occupancy disorder directly.
- Sources: [[obelix]], [[data-driven-prediction-ionic-conductivity-llm]], [[prediction-3d-potential-landscape-bvse]]
- Status: open

---

## Polymorphism blind spot in composition-only ML
- Description: Composition-based ML models assign identical predictions to all polymorphs of a composition, regardless of crystal structure. For cases like LLZO (cubic σ ≈ 10⁻³ S/cm vs. tetragonal σ ≈ 10⁻⁶ S/cm — a ~1000× difference), this is a catastrophic failure mode for virtual screening.
- Why it's hard: Structure-based models that could distinguish polymorphs require CIFs, and only ~54% of OBELiX entries have CIFs. Even with CIFs, partial occupancy limits GNN performance. The BVSE-derived 3D potential landscape (Hashizume 2026) addresses polymorphism but requires CIF input and adds computational overhead.
- Approaches tried: BVSE-encoded 3DCNN (Hashizume 2026) correctly ranks LLZO polymorphs. LLMs may partially resolve this if structural text cues are included. Scalar BVSE features (bottleneck radius, minimum Eb) could be added to composition-based models as a low-overhead partial fix.
- Sources: [[prediction-3d-potential-landscape-bvse]], [[obelix]]
- Status: open

---

## Bulk conductivity ≠ device performance (interface resistance gap)
- Description: ML models like BREED predict bulk ionic conductivity σ_bulk. However, in practical solid-state batteries, the rate-limiting step is typically the solid/solid interface resistance between SSE and electrode — not bulk conductivity. A material with excellent σ_bulk may fail to deliver useful current density due to interface resistance orders of magnitude larger than bulk resistance. This gap manifests at multiple scales: (1) grain boundary resistance within the SE pellet (LLTO loses 2 orders of magnitude, NASICON loses 1 order); (2) space-charge layers at SE/cathode interfaces (sulfide/oxide); (3) mechanical delamination at electrode/SE contact under cycling-induced volume change.
- Why it's hard: Interface resistance depends on contact geometry, surface chemistry, electrochemical decomposition products, and mechanical stress — none of which are captured by bulk composition or structure. Even when SSE bulk conductivity matches liquid electrolyte (~10 mS/cm), achievable operation current for solid-state Li metal cells is 1–3 orders of magnitude below practical targets (Cheng et al. 2019). Delamination nucleates at just 7.5% volumetric change (Bucci 2018) — exceeded by most intercalation electrodes. No SSE is thermodynamically stable against both Li metal and high-voltage cathodes (Xiao 2020) — all practical cells rely on passivation layers. This means σ_bulk prediction is a necessary but far-from-sufficient condition for practical device performance. Expert note (Ismat, 2026-05-24): ceramics face this problem especially acutely because solid-solid contact cannot conform to electrode surfaces the way polymers can — the interface gap is geometrically baked in for rigid ceramics.
- Approaches tried: Interfacial coating layers (LiNbO₃, Al₂O₃ ALD) reduce interface resistance significantly but add processing steps. 3D electrode architectures improve contact area. Nanostructuring electrode particles does not prevent crack nucleation (Bucci 2018). No ML framework currently predicts interface resistance from bulk composition/structure.
- Sources: [[fundamentals-of-inorganic-solid-state-electrolytes]], [[lithium-battery-chemistries-enabled-solid-state]], [[recent-advances-energy-chemistry-solid-state]], [[designing-solid-state-electrolytes-safe-energy]], [[understanding-interface-stability-solid-state]], [[interfaces-in-solid-state-lithium-batteries]], [[role-of-interfaces-solid-state-batteries]], [[mechanical-instability-electrode-electrolyte-interfaces]]; [[interviews/ismat-may2026]]
- Status: open (not an ML problem yet — materials science bottleneck limiting the practical relevance of bulk σ prediction)

---

## MLFF stability and transferability for SSE conductivity
- Description: Machine learning force fields (MLFFs) used in MLFF-MD to compute ion diffusivity exhibit trajectory instability (atom fusion, lattice mismatch) even when energy/force accuracy metrics (R²>0.96) appear acceptable. Published MLFF-MD conductivity predictions may be physically unreliable if stability validation is skipped.
- Why it's hard: Energy and force accuracy on held-out DFT snapshots does not guarantee stable dynamics in long trajectories. Failure modes accumulate over millions of MD steps in ways not captured by static benchmarks. No community standard exists for MLFF stability validation.
- Approaches tried: Radial distribution function (RDF) comparison to reference DFT-AIMD trajectories (Duangdangchote 2024); Duangdangchote identify DimeNet and DimeNet++ as the only architectures passing the full stability pipeline for sulfide SSEs (LGPS → Li₃PS₄ → Li₄GeS₄).
- Sources: [[stability-transferability-mlff]], [[obelix]]
- Status: partial progress (DimeNet++ validated for sulfides; other families untested)

---

## Manufacturing cost and scalability of ceramic SSEs
- Description: Ceramic solid electrolytes are prohibitively expensive at current production scales. LLZO pellets (14 mm diameter, 0.7 mm thick) cost ~$181.95 per disk commercially (as of 2026). Ceramics require extremely high sintering temperatures maintained at very precise tolerances, making manufacturing energy-intensive and equipment-demanding.
- Why it's hard: The cost barrier is not primarily a materials science problem — it's a process engineering and economies-of-scale problem. High-temperature, high-precision firing is intrinsic to achieving the dense, low-grain-boundary microstructure that gives ceramics their conductivity advantage. Lowering temperature (e.g., cold sintering, flash sintering) risks sacrificing the very microstructure that makes them competitive.
- Approaches tried: Cold sintering; spark plasma sintering; tape casting for thin films to reduce material per cell. None have achieved cost parity with liquid electrolytes or polymer SSEs at scale.
- Sources: [[interviews/ismat-may2026]]
- Status: open (engineering/economics bottleneck, largely independent of ML progress)

---

## Synthesis pathway prediction
- Description: Even if a target SSE composition and structure are known, predicting the synthesis protocol that reliably produces it remains largely unsolved. The synthesis space (temperature, atmosphere, precursor ratios, mixing order, time) is high-dimensional, sparsely explored, and exhibits non-linear and emergent behavior — small changes in conditions can produce qualitatively different phases or microstructures. Additionally, lab-scale results are unreliable guides to production-scale behavior due to scale-dependent effects in heat/mass transfer and atmosphere control.
- Why it's hard: Published literature reports successful protocols, not systematic parameter sweeps — negative results are almost never published. This means training data for synthesis ML is extremely sparse and heavily selection-biased. MD and DFT operate at time and length scales (ns, nm) far below bulk synthesis (minutes–hours, mm–cm), and bridging these scales requires multiscale frameworks that don't yet exist in general form. Disorder, grain boundaries, and intermediate phases further complicate atomistic simulation.
- Approaches tried: Inverse design via generative models (composition → structure, not synthesis); high-throughput combinatorial synthesis (addresses data sparsity but not scale gap); cold sintering / flash sintering (reduce temperature requirements without necessarily predicting the pathway).
- Open question: Can a polymer SSE be simulated across all relevant scales — from quantum bond formation through chain conformation and phase separation to bulk processing? No current framework connects these.
- Sources: [[personal-research/synthesis-pathway-prediction]], [[interviews/ismat-may2026]]
- Status: open

---

## Polymer SSE synthesis-to-property control
- Description: For polymer solid electrolytes, the core challenge is not identifying promising compositions — it is figuring out the synthesis conditions that reliably produce the target properties. Translating a known formulation to a reproducible, high-performance process is not solved.
- Why it's hard: Polymer SSE properties (conductivity, mechanical stiffness, stability) are highly sensitive to molecular weight, chain architecture, crosslink density, plasticizer distribution, and processing conditions. Small changes in synthesis can produce large property variation. There is no predictive model that maps synthesis protocol → final material properties reliably.
- Approaches tried: Systematic design-of-experiments on synthesis parameters; ML models trained on composition → property (bypass synthesis), but these don't help practitioners who need to know how to make the material. Multi-property optimization is complicated by the conductivity-stiffness trade-off.
- Sources: [[data-science-approach-advanced-solid-polymer]], [[interviews/ismat-may2026]]
- Status: open

---

## Computational tools accessibility for experimental SSE researchers
- Description: Computational screening tools (BVSE, AIMD, ML models) exist and could accelerate experimental SSE research, but adoption among experimentalists is low. The bottleneck is not capability — it is usability, compute access, and learning curve.
- Why it's hard: AIMD and high-throughput DFT require HPC access and domain expertise in setting up calculations. Even simpler tools (BVSE, ML APIs) require familiarity with Python ecosystems and crystallographic input formats that experimentalists are not trained in. Compute budget is a separate constraint for researchers at smaller institutions.
- Approaches tried: Web interfaces for some DFT tools; Materials Project API; pre-trained ML models as drop-in tools. Uptake remains limited in practice among experimental groups.
- Sources: [[interviews/ismat-may2026]] — expert (Ismat, 2026-05-24) stated directly: would have used computational tools, had compute access and learning curve been lower; paraphrased: "if it was easier to use, 1000%."
- Status: open (usability gap, not a scientific gap)
