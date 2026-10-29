# Reviewed GBA field-view anchor additions — 2026-10-29

Four-direction route attempts produced visually recognizable Cornelia exterior screenshots with different camera compositions. The images added here are the candidate outputs from the two route-start audits and were reviewed for outdoor terrain, town/castle landmarks, and player sprite. They are scene references only; they do not label coordinates, direction, progress, or route success.

The fixtures were added to `SceneAnchorClassifier` field variants and the offline regression confirms each reference is detected while title/new-game, Cornelia town, party menu, and Castle transition controls remain non-field or their expected labels.

Live limitation: the fresh benchmark still could not confirm all route-derived reset starts across two consecutive observations. Some otherwise field-looking routes transitioned to screens outside the reviewed anchors, including an observed battle screen. These anchors match only reviewed views and do not solve moving-frame coverage. The benchmark did not evaluate any policy or open the final test set.
