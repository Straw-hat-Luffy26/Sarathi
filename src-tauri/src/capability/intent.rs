//! The capability taxonomy.
//!
//! One enum, naming the specializations Sarathi can route to. It lives here
//! rather than beside a classifier because it is the vocabulary *every* part of
//! the capability layer shares: the classifier produces one, the switch policy
//! compares them, and the manifest keys adapters by their capability name.
//!
//! It previously lived in `model_intelligence::intent` alongside a first-match
//! substring scanner. That scanner was superseded by
//! [`crate::capability::classifier`] — which scores every intent independently
//! and reports a calibrated confidence — but the enum stayed behind, leaving the
//! capability layer importing its core type from the module it replaced.

use serde::{Deserialize, Serialize};

/// What a prompt is asking for.
///
/// `GeneralChat` is not a specialization: it is the explicit absence of one, and
/// resolves to the unmodified base model.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PromptIntent {
    Coding,
    Reasoning,
    Mathematics,
    ToolCalling,
    Research,
    GeneralChat,
}

impl PromptIntent {
    /// The manifest capability key for this intent.
    ///
    /// These strings are persisted in `manifest.json` and matched against
    /// adapter capability slots, so they are stable identifiers rather than
    /// display text.
    pub fn to_capability_name(&self) -> &'static str {
        match self {
            Self::Coding => "coding",
            Self::Reasoning => "reasoning",
            Self::Mathematics => "mathematics",
            Self::ToolCalling => "tool-calling",
            Self::Research => "research",
            Self::GeneralChat => "general",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The capability keys are persisted in every model package's manifest and
    /// are matched against adapter slots, so renaming one silently unbinds every
    /// adapter already installed under the old name.
    #[test]
    fn capability_keys_are_stable_identifiers() {
        assert_eq!(PromptIntent::Coding.to_capability_name(), "coding");
        assert_eq!(PromptIntent::Reasoning.to_capability_name(), "reasoning");
        assert_eq!(PromptIntent::Mathematics.to_capability_name(), "mathematics");
        assert_eq!(PromptIntent::ToolCalling.to_capability_name(), "tool-calling");
        assert_eq!(PromptIntent::Research.to_capability_name(), "research");
        assert_eq!(PromptIntent::GeneralChat.to_capability_name(), "general");
    }
}
