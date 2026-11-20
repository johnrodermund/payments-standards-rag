# Public source documents

Every document ingested must be listed here with its URL and licence/redistribution terms.
Files in `data/raw/` are git-ignored; ingestion downloads them from these URLs.
Terms were read on 2026-11-20 in a browser. Nothing has been downloaded yet.

## Approved for Phase 1 (kept small: 3 documents)

| # | Document | Source | Terms (read) | File URL / size | Status |
|---|---|---|---|---|---|
| 1 | ISO 20022 message definition: pacs.008 (FI to FI customer credit transfer) | iso20022.org message definitions | [Terms of use](https://www.iso20022.org/terms-use): material "intended to be used and reproduced freely" under the ISO 20022 IPR policy. Replicas must state they are not the official site and point to https://www.iso20022.org/. | XSD schema: `https://www.iso20022.org/message/23500/download` (pacs.008.001.14). Returns HTTP 403 to scripted requests, so size unknown. The prose definitions (MDR) are per message set and I haven't found the payments set's link. | Terms OK; scripted download blocked |
| 2 | ISO 20022 message definition: pacs.002 (payment status report) | iso20022.org message definitions | Same as #1 | XSD schema: `https://www.iso20022.org/message/23496/download` (pacs.002.001.16). Same 403 as #1. | Terms OK; scripted download blocked |
| 3 | BIS CPMI report: harmonised ISO 20022 data requirements for cross-border payments (d218) | bis.org | [Terms](https://www.bis.org/about/terms-conditions): non-commercial download and redistribution allowed without permission; limited extracts (<=400 words) may be reproduced with BIS cited as source. | `https://www.bis.org/publications/harmonised-iso-20022-data-requirements-enhancing-cross-border-payments-original-report.pdf`, PDF, 2,532,898 bytes (about 2.5 MB), HTTP 200 | Terms OK; link and size verified, not downloaded |

Finding (2026-11-20): the ISO pacs.008/pacs.002 XSDs download fine from inside a browser session (63,598 and 62,059 bytes; the pacs set zip is 49,527 bytes) but contain no `xs:documentation` text, only element names and types. They are too thin for Q&A on their own. The prose element definitions are in the ISO 20022 e-Repository (https://www.iso20022.org/iso20022-repository/e-repository), whose terms and size are not checked yet. No ISO file has been saved to disk.

e-Repository check (2026-11-20): `https://www.iso20022.org/sites/default/files/media/file/20260904_ISO20022_2013_eRepository.zip`, 10,293,494 bytes, HTTP 200, same ISO terms of use as above. It holds one 118 MB XML file (EMF/XMI format, `.iso20022`) with every message definition, current and previous versions, plus the data dictionary. I inspected it in memory in the browser, and it contains real prose definitions, for example a Scope/Usage text on `FIToFICustomerCreditTransferV14` (pacs.008.001.14) and a `definition` on each data-dictionary entry. Plan: stream-parse it and keep only the latest pacs.008 and pacs.002 definitions and their elements. It may include draft candidate versions, so pin the version used. Not downloaded yet.

Conditions that follow from the terms: the project is non-commercial and says so; the README states it is not the official ISO 20022 site; answers quote short extracts and cite the source.

## Rejected

| Document | Why |
|---|---|
| FedNow Operating Circular 8, FedNow Operating Procedures and FedNow ISO 20022 explainer pages (frbservices.org) | [Terms](https://www.frbservices.org/terms-of-use/) say the site is copyrighted and prohibit reproducing, publishing or creating derivative works from its content without express authorisation. Indexing and quoting it in a public service is not clearly allowed. Three of the four URLs also returned 404 after a site restructure. |
| CBPR+ usage guidelines and FedNow ISO 20022 specs on Swift MyStandards | Need a login and Swift terms; not checked. Out of scope for now. |

FedNow and CBPR+ are therefore out of scope unless permission or clearer terms are found. The README must say so.
