# Security and responsible disclosure

This project concerns former nuclear test sites. Some information it can produce is sensitive,
and some kinds of "bug" are really disclosure risks. Please read before opening an issue.

## Never post publicly

Do **not** put any of the following in issues, pull requests, discussions, commits or logs:

- Coordinates, maps, tile IDs or image crops of **newly detected candidate areas** (anything
  derived from `outputs/restricted/`). Only the public reference features in
  `config/reference_features.geojson` (IAEA 2005) may be discussed openly.
- Credentials or tokens (USGS, Google Earth Engine, Anthropic, `RESTRICTED_KEY`).
- Instructions for visiting, accessing or excavating the sites. This project is desk analysis
  only; the sites are restricted military zones and contaminated.

## How to report privately

Use GitHub's **private vulnerability reporting** on this repository
("Security" tab → "Report a vulnerability"). Use it for:

- a leak path (a code path that could write candidate locations outside `outputs/restricted/`,
  or a public artefact that `sbf report check-public` fails to catch);
- a committed secret;
- results or candidate locations you believe should reach the competent authorities.

Findings from this pipeline are "candidate areas for verification", never confirmed burial
sites. Verification is a matter for authorised Algerian teams and the competent authorities.

## Supported versions

Only the `main` branch is maintained.
