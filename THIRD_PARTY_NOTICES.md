# Third-party sources and rights

The project's Apache-2.0 licence applies to its original code and model artifacts to the extent of the publisher's rights. It does not relicense third-party data. This source review is documentary due diligence, not legal advice or blanket permission for every use.

## Sri Lanka surveillance counts — research rows withheld

The Epidemiology Unit's [WER archive](https://www.epid.gov.lk/weekly-epidemiological-report) carries an All Rights Reserved footer. The inspected page did not establish an affirmative open-data redistribution grant. [NDCU weekly reports](https://www.dengue.health.gov.lk/weekly-report/) likewise carry a copyright notice without a verified redistribution grant. Public access to a report is not used as a substitute for permission.

The release does not redistribute the case tables, extracted observations, lag/target training rows, raw bulletins or saved observed-outcome tables. See the Epidemiology Unit [Data Request Form](https://www.epid.gov.lk/data-request-form/data-request-form) for a permission route. NDCU and WER reporting weeks are not interchangeable; NDCU was indexed for context rather than silently merged into the WER training series.

## CHIRPS v3.0 final rnl — attribution

Rainfall source: Climate Hazards Center, University of California, Santa Barbara, [CHIRPS v3](https://www.chc.ucsb.edu/data/chirps3). The version-specific page uses both public-domain wording and an explicit [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) licence. This release conservatively retains the attribution and licence notice rather than relabelling the product CC0.

Research transformations include spatial subsetting, geographic weighting/aggregation, reporting-week aggregation, and derived lag/rolling features. These are modified/derived inputs, not an unmodified CHIRPS product. No endorsement is implied.

## Open-Meteo and ERA5 — data licence versus service terms

Weather data by [Open-Meteo.com](https://open-meteo.com/), using ERA5/Copernicus historical temperature and relative humidity. [Open-Meteo's data licence](https://open-meteo.com/en/licence) is [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), which permits redistribution and adaptation with attribution and modification notices. Historical API documentation and citations: https://open-meteo.com/en/docs/historical-weather-api . ERA5 reference: https://doi.org/10.24381/cds.adbb2d47 .

Contains modified Copernicus Climate Change Service information. Neither the European Commission nor ECMWF is responsible for any use that may be made of this information. Processing includes sampling/aggregation into districts and reporting weeks and derived temporal features.

The [free API service terms](https://open-meteo.com/en/terms) separately restrict free-endpoint use to non-commercial purposes and impose rate limits. CC BY on downloaded data does not authorise every use of the free service. The API service terms are not a CC BY-NC licence on the downloaded data. Open-Meteo's software licence is also distinct from its data licence.

## Exact district-boundary source — ODbL 1.0

Sri Lanka ADM2 source: geoBoundaries `gbOpen/LKA/ADM2`, revision `9469f09`, boundary ID `LKA-ADM2-46371173`, boundary source **OpenStreetMap, Wambacher**.

[Exact-version metadata](https://media.githubusercontent.com/media/wmgeolab/geoBoundaries/9469f09/releaseData/gbOpen/LKA/ADM2/geoBoundaries-LKA-ADM2-metaData.json) specifies **Open Data Commons Open Database License 1.0**, not merely the generic geoBoundaries website licence.

© [OpenStreetMap contributors](https://www.openstreetmap.org/copyright); attribution also to geoBoundaries and Wambacher. [ODbL 1.0](https://opendatacommons.org/licenses/odbl/1-0/) governs the database. Geometry is not redistributed here. Downstream spatial tables must be assessed as derivative databases, collective databases or produced works as appropriate; do not blindly apply the project's Apache licence to them. Preserve source credit and any applicable share-alike/access obligations. The software code itself is not automatically relicensed by importing geographic data.

## Population context

Department of Census and Statistics [2024 population tables](https://www.statistics.gov.lk/Population/StaticalInformation/CPH2024/Population_Tables) were contextual sources. Their redistribution permission was not established in this bounded review. Raw tables and population-reference rows are not included. The four released models' feature lists exclude population, density, incidence and area features.

## Model artifacts

The models were fitted by this project; they are not downloaded upstream pretrained weights. The artifact review distinguishes fitted coefficients/trees and preprocessing aggregates from original feature/target tables. No third-party dataset licence is claimed merely because fitted artifacts are offered for download. Checksums, artifact contents and limitations are documented alongside the model release. A technical contents review is not a proof against all inference or reconstruction attacks.
