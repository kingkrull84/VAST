use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

/// Represents an element in the Z/9Z ring (integers modulo 9: {0, 1, ..., 8}).
/// Used for discrete, deterministic energy, pressure, and flux updates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct Z9 {
    value: u8,
}

impl Z9 {
    pub const ZERO: Z9 = Z9 { value: 0 };
    pub const ONE: Z9 = Z9 { value: 1 };

    /// Creates a new Z9 element from an i32 value using Euclidean modulo 9.
    pub fn new(val: i32) -> Self {
        let rem = val.rem_euclid(9);
        Z9 { value: rem as u8 }
    }

    /// Returns the inner value in 0..=8.
    pub fn value(&self) -> u8 {
        self.value
    }

    /// Computes the flux state transition given current energy and pressure states.
    /// Discrete Z/9Z flux update function: (energy + pressure * 2) mod 9.
    pub fn update_flux(energy: Z9, pressure: Z9) -> Z9 {
        energy + (pressure * Z9::new(2))
    }
}

impl From<i32> for Z9 {
    fn from(val: i32) -> Self {
        Z9::new(val)
    }
}

impl From<u8> for Z9 {
    fn from(val: u8) -> Self {
        Z9::new(val as i32)
    }
}

impl From<Z9> for u8 {
    fn from(z: Z9) -> Self {
        z.value
    }
}

impl From<Z9> for i32 {
    fn from(z: Z9) -> Self {
        z.value as i32
    }
}

impl Add for Z9 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        Z9::new((self.value + rhs.value) as i32)
    }
}

impl AddAssign for Z9 {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl Sub for Z9 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        Z9::new((self.value as i32) - (rhs.value as i32))
    }
}

impl SubAssign for Z9 {
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl Mul for Z9 {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        Z9::new((self.value as u32 * rhs.value as u32) as i32)
    }
}

impl MulAssign for Z9 {
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl Neg for Z9 {
    type Output = Self;
    fn neg(self) -> Self::Output {
        Z9::new(-(self.value as i32))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_z9_ring_arithmetic() {
        assert_eq!(Z9::new(0).value(), 0);
        assert_eq!(Z9::new(9).value(), 0);
        assert_eq!(Z9::new(10).value(), 1);
        assert_eq!(Z9::new(-1).value(), 8);
        assert_eq!(Z9::new(-9).value(), 0);

        assert_eq!(Z9::new(4) + Z9::new(6), Z9::new(1));
        assert_eq!(Z9::new(2) - Z9::new(5), Z9::new(6));
        assert_eq!(Z9::new(3) * Z9::new(4), Z9::new(3));
        assert_eq!(-Z9::new(2), Z9::new(7));
    }

    #[test]
    fn test_flux_update() {
        let energy = Z9::new(5);
        let pressure = Z9::new(3);
        // 5 + 3 * 2 = 11 mod 9 = 2
        let flux = Z9::update_flux(energy, pressure);
        assert_eq!(flux, Z9::new(2));
    }
}
