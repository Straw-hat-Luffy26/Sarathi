//! LoRA adapter validation and format conversion.
//!
//! Runtime binding lives in [`crate::ai_engine::lora_binding`] and adapter
//! registration in [`crate::adapter_manager`]; this module is only concerned
//! with getting a downloaded adapter into a shape llama.cpp can load.

pub mod convert;
pub mod validator;

pub use convert::{convert_adapter, ConversionSummary};
pub use validator::{AdapterRuntimeStatus, AdapterValidationResult};
