---
name: Training Monitor
description: A calm, read-only scheduler docket for two private training workers.
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
  body:
    fontFamily: "-apple-system, BlinkMacSystemFont, Segoe UI, sans-serif"
    fontSize: "16px"
    fontWeight: 400
    lineHeight: 1.5
  heading:
    fontFamily: "-apple-system, BlinkMacSystemFont, Segoe UI, sans-serif"
    fontSize: "1.7rem"
    fontWeight: 670
    lineHeight: 1.1
    letterSpacing: "-0.025em"
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
spacing:
  xs: "3px"
  sm: "8px"
  md: "18px"
  lg: "24px"
  xl: "32px"
components:
  metric-tab:
    backgroundColor: "{colors.surface-muted}"
    textColor: "{colors.muted}"
    rounded: "{rounded.tab}"
    padding: "6px 10px"
  metric-tab-selected:
    backgroundColor: "{colors.surface}"
    textColor: "{colors.ink}"
    rounded: "{rounded.tab}"
    padding: "6px 10px"
  progress-track:
    backgroundColor: "{colors.surface-muted}"
    rounded: "{rounded.progress}"
    height: "5px"
---

# Design System: Training Monitor

## Overview

**Creative North Star: "The Research Scheduler Docket"**

The interface treats training state as a working docket: quiet, ruled, factual, and easy to scan. Cool paper and graphite rules establish the page; cobalt marks active work, green completion, amber attention or stale data, and red failure. The wide composition gives live work and validation history the authority they need while keeping provenance and freshness visible.

This is an operate surface for a local, read-only monitor. It favors dependable information density over dashboard spectacle: no gradients, glows, decorative imagery, excessive cards, or theatrical motion. Measurements use tabular figures and retain units; missing data is represented by an em dash or an explicit unavailable state.

**Key Characteristics:**
- Ruled scheduler-docket structure instead of floating metric-card mosaics.
- Semantic color is sparse and never the only status signal.
- System sans typography, compact labels, and tabular scientific values.
- Last-known values remain visible when connectivity is stale.

## Colors

The palette is a cool neutral paper with graphite text and thin rules; four restrained semantic accents carry operational meaning.

### Primary
- **Cobalt Active** (`{colors.active}`): Training, running queue state, progress bars, and active chart series.

### Secondary
- **Completion Green** (`{colors.complete}`): Reporting/completed states and Box IoU history.
- **Attention Amber** (`{colors.warning}`): Stale values, attention queues, incomplete runs, and HD95 history.
- **Failure Red** (`{colors.failure}`): Unavailable workers and gateway/inline errors.

### Neutral
- **Cool Canvas** (`{colors.canvas}`): Page background in light mode.
- **Paper Surface** (`{colors.surface}`): Selected controls and chart tooltip surface.
- **Muted Surface** (`{colors.surface-muted}`): Progress tracks, tab groups, and loading placeholders.
- **Graphite Ink** (`{colors.ink}`): Primary text and selected values.
- **Quiet Gray** (`{colors.muted}`): Supporting copy, labels, and freshness metadata.
- **Fine Rule** (`{colors.rule}`) and **Strong Rule** (`{colors.rule-strong}`): Structural separators.

**The Semantic-Without-Decoration Rule.** Accent colors communicate state or metric identity; they are not used as ambient decoration.

## Typography

**Display Font:** System sans (`-apple-system`, `BlinkMacSystemFont`, `"Segoe UI"`, sans-serif)

**Body Font:** The same system sans stack

**Character:** Native, compact, and utilitarian. Weight and spacing create hierarchy; there is no display face or ornamental type treatment. Numeric values always use tabular figures.

### Hierarchy
- **Heading** (670, `1.7rem`, `1.1`): Masthead title; tight tracking (`-0.025em`).
- **Section title** (650, `1rem`, `1.25`): Host and section headings.
- **Metric value** (620, `1rem` or `2rem` for the latest chart value): High-salience measurements with tabular figures.
- **Body** (400, `0.82–0.875rem` supporting copy): Roles, hostnames, freshness, and explanatory text.
- **Label** (650, `0.68–0.72rem`, `0.05–0.055em`, uppercase): Measure labels, table headers, and state labels.

**The Unit-Visible Rule.** Scientific values keep their unit or an explicit unavailable mark; labels should not force users to infer scale.

## Layout

The shell is centered at a maximum width of `1440px`, with `28px 32px 40px` outer padding on wide screens. A masthead and timestamp sit above two full-width host bands. Each band is a three-column grid: identity, four measures, and current work/progress. A strong rule separates the host docket from a `1.75fr / 0.75fr` work grid containing validation history and queues. Recent runs use a ruled, horizontally scrollable table; footer provenance closes the page.

At `1050px`, host work moves to a full-width row and the work grid becomes one column. At `700px`, padding reduces to `20px 18px 28px`, host bands stack, measures become two columns, metric tabs fill the width, charts reduce from `258px` to `220px`, and the footer stacks. The minimum body width is `320px`.

Spacing is deliberately open for a data-dense surface: common structural gaps are `20px`, `24px`, and `32px`; rules provide grouping rather than card chrome.

## Elevation & Depth

Depth is primarily tonal and structural: paper surfaces, muted control fills, and fine/strong rules do the work. There are no ambient shadows on the page. The selected metric tab alone receives a small local lift (`0 1px 3px rgba(22, 28, 36, 0.12)` in light mode; a slightly deeper dark-mode shadow), reinforcing selection without turning the UI into a card stack.

**The Flat-By-Default Rule.** Keep surfaces flat at rest; use rules and tonal changes before adding elevation.

## Shapes

Forms are gently squared and restrained. Progress tracks use a `3px` radius, tabs use `6px`, controls use `8px`, and loading placeholders use `10px`. Host bands, sections, tables, and chart frames are not rounded; their silhouettes come from whitespace and horizontal rules. Status dots are the sole circular geometry (`7px`).

## Components

### Host bands
- **Character:** The signature docket row: identity, freshness, compact measures, and active work in one scan.
- **Shape:** Full-width, unrounded row with a bottom rule; `24px` vertical padding and `28px` column gap.
- **States:** Training uses cobalt; stale uses a faint amber wash while preserving values; unavailable collapses to an explicit error row; status words and dots accompany color.

### Status indicators
- **Style:** Inline label with a `7px` dot and `6px` gap. Reporting is green, Training cobalt, Attention/Stale amber, and Unavailable red.
- **Accessibility:** Never rely on the dot alone; the visible state word is always present.

### Progress track
- **Style:** `5px` muted track with a `3px` radius and a cobalt fill scaled from the left edge.
- **Behavior:** Fill changes use a short `180ms` ease-out transition; reduced-motion collapses this to `1ms`.

### Metric tabs
- **Style:** A muted `8px` group with `3px` inset padding; each button has a `6px` radius and `6px 10px` padding.
- **States:** Unselected is muted; selected is paper/ink with a small local shadow; hover darkens text; active compresses to `scale(0.97)`; focus-visible uses a `2px` cobalt outline.

### Validation chart
- **Style:** A `258px` plot area with horizontal rules, quiet axes, and no animation. Dice is cobalt, Box IoU green, and HD95 amber. Empty state uses ruled bounds and plain explanatory copy.

### Queue rows and recent-runs table
- **Style:** Unrounded rows separated by rules. Queue state is uppercase, letter-spaced, and semantic; session names are compact code-like text. The table is tabular-figure dense, right-aligns numeric columns, and scrolls horizontally when needed.

## Do's and Don'ts

### Do:
- **Do** preserve the ruled docket structure and let active work precede historical detail.
- **Do** use the semantic palette for operational meaning and pair it with text labels.
- **Do** keep freshness, latency, units, and stale/error context visible.
- **Do** maintain keyboard focus rings, semantic headings/landmarks, progress semantics, and reduced-motion behavior.
- **Do** support dark mode, increased contrast, and reduced transparency as implemented.

### Don't:
- **Don't** introduce gradients, glows, decorative illustrations, or generic AI-dashboard styling.
- **Don't** replace stale last-known values with an empty success-looking state.
- **Don't** turn each measure into a rounded floating card or add ornamental shadows.
- **Don't** use color alone for status, omit units, or hide errors behind tooltips.
- **Don't** add theatrical transitions, auto-animated charts, or controls that mutate remote workers.
