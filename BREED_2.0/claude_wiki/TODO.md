# TODO

Ambiguities and items flagged during ingestion.

---

## machine-learning-assisted-property-prediction-solid-state.md

- The source note for this review references several primary papers whose results appear in benchmarks.md but have not been directly ingested. The benchmark rows are attributed via "(via [[source]])" notation. Confirm results by ingesting primary papers when available:
  - Zhao et al. 2021, *Sci. Bull.* 66, 1401 (HECS descriptors, argyrodite Ea)
  - ~~Wang et al. 2021, *Nano Energy* 89, 106337 (XGB for garnet Eg)~~ — **INGESTED** as [[harnessing-artificial-intelligence-holistic-design]]; benchmarks.md row updated with confirmed R²=0.866 and source link.
  - Kim & Siegel 2022, *J. Mater. Chem. A* 10, 15169 (anti-perovskite migration barriers)
  - Hargreaves et al. 2023, *npj Comput. Mater.* 9, 9 (820-entry Li-ion conductor dataset)
  - Sendek et al. 2018, *Chem. Mater.* 31, 342 (12,000+ material screening)

- ~~Concept stubs for [[aimd]] and [[bvse]] are linked in index.md and concepts but not yet created (only one source so far, threshold not met for standalone articles — revisit after next ingestion).~~ — **RESOLVED**: [[concepts/aimd]] created (3 sources now). ~~[[concepts/bvse]] still a stub~~ — **RESOLVED**: [[concepts/bvse]] created as full article after ingesting prediction-3d-potential-landscape-bvse and obelix (3 sources now). [[concepts/mlff]] also created (3 sources).

- The Extra Trees CTE result (Kumar et al. 2023) R² value was read from a figure (~0.91) and is approximate; flag if ingesting that paper directly.

---

## unsupervised-machine-learning-accelerates-solid.md

- **Research highlight, not a primary paper**: The raw PDF (`Unsupervised-machine-learning-accelerates-solid-ele_2021_Green-Energy---Envi.pdf`) is a 2-page commentary summarizing Zhang et al. 2019 (Nat. Commun. 10, 5260). The primary paper has NOT been ingested. If available in raw/, ingest directly and update [[unsupervised-machine-learning-accelerates-solid]] accordingly.

---

## data-science-approach-advanced-solid-polymer.md

- **RMSE discrepancy**: The abstract and conclusion of Liu et al. 2021 report RMSE=0.332 log(S/cm) for the RF model on Dataset 3. The model comparison section reports RF RMSE=0.289, MAE=0.229 on the same dataset. Canonical value 0.332 is used (abstract/conclusion), consistent with benchmarks.md entry. Flag if this paper is re-read.

---

## improving-ionic-conductivity-garnet-gradient-boosting.md

- Work temperature (WT) is the #1 SHAP feature — this is a measurement condition, not a material property. Check whether OBELiX records measurement temperature for each entry. If it does, add WT as a BREED feature; if temperatures are missing or inconsistent, the irreducible error floor is partly explained by this omission.

---


## data-driven-prediction-ionic-conductivity-llm.md

- **DOI unknown**: The Kim et al. 2026 paper DOI was not locatable in the PDF. Dataset is available at zenodo.17157647. Update the frontmatter `doi:` field when the published DOI is confirmed.

---

## Missing concept stubs (threshold met — create on next ingestion)

- ~~**[[argyrodite-electrolytes]]**~~ — **RESOLVED**: created as full article after ingesting Famprikis 2019 and Zhao 2020 (4+ sources now).
- ~~**[[nasicon-electrolytes]]**~~ — **RESOLVED**: created as full article after ingesting Famprikis 2019, Manthiram 2017, Zhao 2020 (4+ sources now).
- **[[ion-diffusivity]]**: Mentioned in ionic-conductivity.md as a Related concept. Closely related to ionic-conductivity via Nernst-Einstein; may not need its own page — consider folding into ionic-conductivity explanation rather than creating a stub. Formerly a broken link (removed).

---

## Unreadable PDFs in raw/

- **`interfaces-and-interphases-in-all-solid-state-batteries-with-inorganic-solid-electrolytes.pdf`**: File exceeds 20 MB; Read tool fails on full read and fails even on page-range reads. Cannot be ingested without external extraction. Likely a comprehensive review covering chemomechanical, electrochemical, and space-charge interface physics — high potential relevance given current interface-focused ingestion session. Retry when available in a smaller/compressed format.

- **`garnet-type-solid-state-electrolytes-materials-interfaces-and-batteries.pdf`**: Previously flagged >20 MB. Has not been retried. Skip until available in a smaller format.
