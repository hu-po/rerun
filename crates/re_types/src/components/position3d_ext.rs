use crate::datatypes::Vec3D;

use super::Position3D;

// ---

impl Position3D {
    pub const ZERO: Self = Self::new(0.0, 0.0, 0.0);
    pub const ONE: Self = Self::new(1.0, 1.0, 1.0);

    #[inline]
    pub const fn new(x: f32, y: f32, z: f32) -> Self {
        Self(Vec3D::new(x, y, z))
    }

    #[inline]
    pub fn x(&self) -> f32 {
        self.0.x()
    }

    #[inline]
    pub fn y(&self) -> f32 {
        self.0.y()
    }

    #[inline]
    pub fn z(&self) -> f32 {
        self.0.z()
    }
}

#[cfg(feature = "glam")]
impl From<Position3D> for glam::Vec3 {
    #[inline]
    fn from(pt: Position3D) -> Self {
        Self::new(pt.x(), pt.y(), pt.z())
    }
}
