//! HuggingFace provider module

pub mod adapter_discovery;
pub mod brands;
pub mod resolve_upstream;
pub mod resolver;
pub mod catalog_provider;
pub mod adapter_provider;
pub mod discovery;
pub mod live_catalog;
pub mod card;
pub mod catalog_cache;
pub mod curation;
pub mod moe_fit;
pub mod moe_geometry;
pub mod probe;
pub mod runtime_arch;
pub mod use_case;

use crate::model_providers::provider::{ModelProvider, ProviderType};

pub struct HuggingFaceProvider;

impl ModelProvider for HuggingFaceProvider {
    fn name(&self) -> &'static str { "HuggingFace" }
    fn provider_type(&self) -> ProviderType { ProviderType::HuggingFace }
}
