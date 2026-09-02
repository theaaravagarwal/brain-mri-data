---
name: Brain MRI Research Workspace
description: A calm ruled research docket for validated four-volume MRI segmentation and exact provenance.
colors:
  canvas: "#f3f4f6"
  surface: "#ffffff"
  surface-muted: "#eef0f3"
  ink: "#20242a"
  muted: "#646b75"
  rule: "#e2e4e8"
  rule-strong: "#c8ccd2"
  active: "#255fbc"
  complete: "#167351"
  warning: "#a6600a"
  failure: "#b43838"
typography:
  display:
    fontFamily: "-apple-system, BlinkMacSystemFont, Segoe UI, sans-serif"
    fontSize: "clamp(2.35rem, 5vw, 4.8rem)"
    fontWeight: 690
    lineHeight: 0.98
    letterSpacing: "-0.038em"
  heading:
    fontFamily: "-apple-system, BlinkMacSystemFont, Segoe UI, sans-serif"
    fontSize: "1.7rem"
    fontWeight: 670
    lineHeight: 1.1
    letterSpacing: "-0.025em"
  body:
    fontFamily: "-apple-system, BlinkMacSystemFont, Segoe UI, sans-serif"
    fontSize: "16px"
    fontWeight: 400
    lineHeight: 1.5
  label:
    fontFamily: "-apple-system, BlinkMacSystemFont, Segoe UI, sans-serif"
    fontSize: "0.7rem"
    fontWeight: 650
    lineHeight: 1.35
    letterSpacing: "0.055em"
rounded:
  progress: "3px"
  tab: "6px"
  control: "8px"
  loading: "10px"
  circle: "50%"
spacing:
  xs: "3px"
  sm: "8px"
  md: "18px"
  lg: "24px"
  xl: "32px"
  section: "48px"
components:
  button-primary:
    backgroundColor: "{colors.active}"
    textColor: "{colors.surface}"
    rounded: "{rounded.control}"
    padding: "0 16px"
    height: "42px"
  button-secondary:
    backgroundColor: "{colors.surface}"
    textColor: "{colors.ink}"
    rounded: "{rounded.control}"
    padding: "0 16px"
    height: "42px"
  file-action:
    backgroundColor: "{colors.canvas}"
    textColor: "{colors.muted}"
    rounded: "{rounded.control}"
    padding: "8px 10px"
  workflow-step-active:
    backgroundColor: "{colors.active}"
    textColor: "{colors.surface}"
    rounded: "{rounded.circle}"
    size: "27px"
  progress-track:
    backgroundColor: "{colors.surface-muted}"
    rounded: "{rounded.progress}"
    height: "5px"
---

# Design System: Brain MRI Research Workspace

## Overview

**Creative North Star: "The Validated Research Docket"**

The workspace treats MRI research as a controlled docket: calm, ruled, factual, and explicit about what may happen next. Cool paper, graphite text, thin structural rules, and generous whitespace carry the visual world. Cobalt identifies the current or permitted action, green confirms completion, amber calls for attention, and red marks a hard failure or unavailable gate.

The primary surface is an ordered four-stage workflow: select exactly T1, T1ce, T2, and FLAIR NIfTI volumes; validate modality identity and shared geometry; run the fixed local CNN; then return a binary mask, exact receipt, and research-only metadata explanation. Operations, model evidence, explanations, and proposals are secondary views in the same shell. The system rejects generic AI-dashboard styling: no decorative gradients, glows, imagery, card mosaics, or theatrical motion.

**Key Characteristics:**
- A four-stage validation gate is the signature composition and primary product surface.
- Ruled rows and ledgers replace floating card mosaics.
- System sans typography combines an unusually large workflow headline with compact factual labels.
- Semantic color is sparse, paired with visible words, and tied to state rather than decoration.
- Model readiness, validation facts, provenance, retention, and research limitations remain adjacent to the actions or results they qualify.
- Five hash-addressed views collapse to a native select on small screens; New study always leads.

## Colors

The palette is cool paper and graphite with four restrained semantic accents; the frontmatter values are the normative light-mode tokens, with corresponding darker preference values supplied by the implementation.

### Primary
- **Cobalt Active** (`{colors.active}`): Current navigation and workflow steps, enabled primary actions, progress, active metric series, focus outlines, and selected states.

### Secondary
- **Completion Green** (`{colors.complete}`): Validated studies, ready model state, completed inference, selected files, and completed operational states.
- **Attention Amber** (`{colors.warning}`): Waiting, stale data, research-scope notices, deterministic fallback context, and attention states.
- **Failure Red** (`{colors.failure}`): Checkpoint mismatch, inference unavailability, rejected or failed states, alert rules, and destructive clear actions.

### Neutral
- **Cool Canvas** (`{colors.canvas}`): The continuous page background; the workflow is composed directly on it rather than inside a hero card.
- **Paper Surface** (`{colors.surface}`): Native controls, selected tabs, and restrained local control fills.
- **Muted Surface** (`{colors.surface-muted}`): Disabled actions, progress tracks, tab groups, and loading placeholders.
- **Graphite Ink** (`{colors.ink}`): Headlines, identifiers, primary copy, and validated values.
- **Quiet Gray** (`{colors.muted}`): Supporting explanations, filenames, units, freshness, and secondary metadata.
- **Fine Rule** (`{colors.rule}`) and **Strong Rule** (`{colors.rule-strong}`): Row separation and major structural boundaries.

**The Semantic-Without-Decoration Rule.** Accent colors communicate workflow state, permission, result, or metric identity; never use them as ambient decoration.

**The Hard-Gate Rule.** A failure state uses red text plus an explicit explanation and boundary rule; never imply readiness through neutral styling when validation or the checkpoint fails.

## Typography

**Display Font:** Native system sans (`-apple-system`, `BlinkMacSystemFont`, `"Segoe UI"`, sans-serif)

**Body Font:** The same native system sans stack

**Character:** Direct, compact, and utilitarian. Hierarchy comes from scale, weight, and tracking rather than an ornamental display face. Scientific values, byte counts, geometry, hashes, timestamps, and operational measurements use tabular figures.

### Hierarchy
- **Display** (690, `clamp(2.35rem, 5vw, 4.8rem)`, `0.98`): The New study proposition; large, tightly tracked, balanced, and never decorative.
- **Heading** (670, `1.7rem`, `1.1`): Major secondary-view titles; result headings sit nearby at `1.65rem`.
- **Section title** (650, `1rem`, `1.25`): Study volumes, validation ledger, host identities, and section headings.
- **Body** (400, primarily `0.78–0.92rem` in the dense interface): Instructions, limitations, file details, and factual explanations; primary explanatory lines stay within roughly `68–72ch`.
- **Label** (650–680, `0.66–0.72rem`, `0.05–0.055em`, uppercase when denoting state or ledger keys): Status, units, table headings, and contract facts.

**The Exact-Facts Rule.** Units, modality names, model identifiers, geometry, hashes, retention, and unavailable marks remain visible; labels must not make users infer scale or provenance.

## Layout

The shell is centered at a maximum width of `1440px`, with `28px 32px 40px` outer padding on wide screens. The masthead, research-use notice, and five-column navigation form a quiet ruled header. New study leads the navigation, followed by Operations, Model evidence, Explanations, and Proposals.

The primary study view opens with a two-column introduction (`1.55fr / 0.65fr`) pairing the large task statement with fixed-model readiness. A full-width four-column stage rail follows. The working area uses a `1.65fr / 0.65fr` split: the exact four modality selectors occupy the primary column and the validation ledger occupies the secondary column, divided by a strong vertical rule. Successful results reuse a `1.35fr / 0.65fr` summary-and-receipt split. Major vertical gaps are `38–48px`; row-level facts use `12–24px` rhythm and rules instead of card padding.

At `1050px`, the intro, workspace, and result grids become single-column and the validation ledger loses its vertical divider. At `700px`, outer padding becomes `20px 18px 28px`, navigation becomes a full-width native select with a `44px` minimum height, the stage rail becomes two columns, and actions stack where needed. At `430px`, padding tightens to `14px 16px 24px`, the headline becomes `2.15rem`, the stage rail returns to four compact icon-and-label columns, supporting step copy disappears, model-readiness detail recedes, and modality rows become a concise two-column list. The minimum body width is `320px`.

**The Workflow-First Rule.** The select–validate–run–return sequence holds the strongest hierarchy. Operational telemetry and review surfaces may share the shell, but they must not visually outrank the active study task.

## Elevation & Depth

The system is flat by default. Continuous paper, muted fills, fine rules, strong rules, and whitespace create structure; page sections and workflow content have no ambient shadow. The selected metric tab alone receives a small local lift (`0 1px 3px rgba(22, 28, 36, 0.12)` in light mode), reinforcing control selection without creating a card stack.

**The Flat-By-Default Rule.** Use rules and tonal change before elevation. Shadows are local state feedback, never page-level atmosphere.

## Shapes

Forms are gently squared and restrained. Progress tracks use a `3px` radius, tabs and compact selects use `6px`, actions and file controls use `8px`, and loading placeholders use `10px`. Major sections, modality rows, ledgers, tables, alerts, and result areas remain unrounded. Circular geometry is reserved for `7px` state dots and numbered workflow markers (`24–27px`).

**The Ruled-Surface Rule.** Large regions derive shape from alignment and borders; do not wrap workflow stages, volumes, validation facts, or results in rounded containers.

## Components

### Workflow stage rail
- **Character:** An explicit contract sequence rather than a decorative stepper.
- **Shape:** Four equal columns on wide screens; each stage has a circular numbered marker and a `3px` bottom state rule.
- **State:** The current marker is cobalt with white text and the column receives a cobalt rule. Inactive markers remain canvas-colored with a strong neutral border.
- **Responsive behavior:** Two columns below `700px`; four compact columns again below `430px`, with secondary descriptions hidden.

### Modality file rows
- **Character:** Exact, calm selectors for T1, T1ce, T2, and FLAIR; all four remain simultaneously visible.
- **Shape:** Unrounded `82px` rows separated by fine rules, with modality code, filename or description, and a compact outlined file action.
- **State:** Hover and focus move the outline to cobalt; a selected file changes the action to completion green and the copy to Replace. The invisible native file input preserves the full-row target.

### Buttons
- **Shape:** Gently squared controls (`8px`) with a `42px` minimum height and `0 16px` horizontal padding.
- **Primary:** Cobalt fill, white text, and cobalt border; used only for the next valid workflow action or primary artifact download.
- **Secondary:** Paper fill, graphite text, and strong neutral border; hover shifts border and text to cobalt.
- **Disabled:** Muted surface, muted text, strong rule border, and not-allowed cursor. Disabled styling is mandatory before all four files, model readiness, and validation gates pass.
- **Destructive text action:** Transparent and failure red, underlined only on hover.
- **Focus / Press:** A `2px` cobalt focus-visible outline with `3px` offset; enabled actions compress slightly (`scale(0.98)`) on press.

### Validation ledger
- **Character:** Contract facts only: modalities, geometry, spacing, and geometry receipt.
- **Shape:** A ruled definition list; wide layouts add a strong left divider, while narrow layouts remove it.
- **Empty state:** A bordered quiet field states that no study is validated and explains that the CNN cannot start before the ledger passes.
- **Running / failure:** Inference uses a restrained `5px` cobalt activity line; failure uses explicit red text and a red structural rule.

### Model readiness and status indicators
- **Style:** Compact uppercase word plus a `7px` semantic dot. Ready/complete is green, waiting is amber, failed/unavailable is red.
- **Accessibility:** Color never stands alone. The fixed model identifier remains visible, and unavailable readiness expands into a red ruled alert with a concrete recheck action.

### Result and receipt
- **Character:** The mask, explanation, exact receipt, retention, and research limitation form one inseparable result.
- **Layout:** A large tabular output count and explanation pair with an unrounded receipt ledger; artifact actions follow below.
- **Explanation state:** Validated local LLM rendering is green and subordinate. If unavailable or rejected, amber fallback language explicitly states that validated deterministic metadata is shown.
- **Safety:** Research-only limitations appear in the result heading, never in a tooltip or detached legal footer.

### Navigation
- **Style:** Five equal ruled links with label and short detail. The active view receives cobalt text and a `3px` bottom rule.
- **Order:** New study, Operations, Model evidence, Explanations, Proposals.
- **Mobile:** Below `700px`, replace the link row with a full-width native select. Hash navigation remains deep-linkable, and the skip link targets the current view.

## Do's and Don'ts

### Do:
- **Do** make the four-stage study workflow the first and strongest product surface.
- **Do** require all four named modalities and successful validation before presenting Run fixed CNN as available.
- **Do** use the semantic palette for state and pair every color cue with visible language.
- **Do** keep model readiness, exact checkpoint provenance, geometry, units, retention, deterministic fallback, and research limitations visible beside the evidence they qualify.
- **Do** preserve ruled rows, ledgers, generous whitespace, keyboard focus rings, native file and select affordances, reduced-motion behavior, dark mode, increased contrast, and reduced transparency.
- **Do** keep Operations, Model evidence, Explanations, and Proposals secondary, factual, and non-clinical.
- **Do** retain last-known operational values with freshness and explicit stale/error context.

### Don't:
- **Don't** introduce decorative gradients, glows, illustrations, glass effects, generic AI-dashboard styling, or rounded card mosaics.
- **Don't** allow inference to appear runnable before model readiness and validation pass.
- **Don't** obscure the exact T1, T1ce, T2, and FLAIR contract behind a generic multi-file upload.
- **Don't** separate the segmentation from its model/checkpoint receipt, geometry evidence, explanation source, retention, or research-only limitation.
- **Don't** imply diagnosis, treatment, clinical use, automatic promotion, proposal execution, or language-model access to scan voxels.
- **Don't** use color alone for status, hide failures or provenance behind tooltips, or replace stale values with a success-looking empty state.
- **Don't** add theatrical transitions, auto-animated charts, or unsafe remote-worker mutation controls.
