//! Model manager module

pub mod classify;
pub mod traits;
pub mod manager;
pub mod store;

pub use manager::ModelManager;
pub use store::ModelStore;
pub use traits::*;
