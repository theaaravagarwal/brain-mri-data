# ISEF compliance checklist — public, de-identified MRI study

This is a project-management checklist, not a rules determination, research
plan, abstract, or approval form. The student must write all competition
materials in their own words and obtain a decision from the local affiliated
fair's Adult Sponsor/SRC **before work that counts as experimentation**.
Recheck the current rules for the actual competition year.

## Before the study

- [ ] Ask the Adult Sponsor and affiliated-fair SRC whether this exact public,
  retrospective MRI-data study is exempt from human-participant review. Do not
  infer an exemption merely because a repository calls data de-identified.
- [ ] Retain written evidence from each data provider that the supplied data
  are pre-existing and appropriately de-identified, plus the applicable data
  terms/license and access date. Keep it in the project log, not in the public
  source tree.
- [ ] Obtain the SRC's written determination that the documentation satisfies
  the fair's requirements. If any data are identifiable, any participant is
  contacted, or any output is returned to a person, stop and seek IRB/SRC
  direction before proceeding.
- [ ] Complete the current-cycle paperwork requested by the fair before
  experimentation. The ISEF general rules list Forms 1, 1A, 1B, and 2A for
  every project; the SRC determines whether other forms apply.
- [ ] Write the Research Plan/Project Summary yourself. It must state the
  question, hypothesis, materials, procedures, risks/safety, analysis, and
  bibliography. If an adult mentor contributed, delineate their contribution.
- [ ] Record the project start/end dates and keep the study within the current
  ISEF time window. Treat a material protocol change as requiring an addendum
  and SRC guidance before data work resumes.

## Data and model safeguards

- [ ] Keep raw MRIs and any provider access credentials off the exhibit,
  repository, and language-model worker. Do not publish case IDs or images.
- [ ] Preserve provider license/terms snapshots, source manifests, QC reports,
  frozen study lock, code revision, runtime profile, and output hashes in the
  private project log.
- [ ] Use only the locked four-sequence whole-lesion protocol. Report boxes as
  mask-derived secondary geometry, never as an independent ground truth.
- [ ] Keep the external cohort inaccessible for model selection. Do not call
  any segmentation result a diagnosis, treatment recommendation, or clinical
  decision.
- [ ] Keep the AMD worker to the bounded language benchmark. It receives only
  validated structured outputs and no raw MRI images or patient-level data.

## Competition materials

- [ ] Create the research plan, abstract, poster, and citations in the
  student's own words. ISEF permits AI as a project resource only when it is
  acknowledged; it does not permit generative AI to write those materials or
  create citations.
- [ ] Cite every chart, external image, dataset description, software tool,
  and model/library version displayed. Use aggregate, non-identifying metrics
  on the board; do not display raw participant data.
- [ ] Have the Adult Sponsor/SRC review the final title and claims. The valid
  claim is comparative external segmentation robustness on a locked research
  cohort—not a diagnostic product.

## Official references

- [ISEF Rules for All Projects](https://www.societyforscience.org/isef/international-rules/rules-for-all-projects/)
- [ISEF Human Participants rules](https://www.societyforscience.org/isef/international-rules/human-participants/)
- [ISEF Forms](https://www.societyforscience.org/isef/forms/)
- [ISEF Overview of Forms and Dates](https://www.societyforscience.org/isef/overview-of-forms-and-dates/)

