#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Tpes {
    pub tier: u8,      // 4 bits (0-15)
    pub positrons: u16,// 12 bits (0-4095)
    pub electrons: u16,// 12 bits (0-4095)
    pub phase: u8,     // 4 bits (0-15)
}

impl Tpes {
    pub fn new(tier: u8, positrons: u16, electrons: u16, phase: u8) -> Self {
        Self {
            tier: tier & 0x0F,
            positrons: positrons & 0x0FFF,
            electrons: electrons & 0x0FFF,
            phase: phase & 0x0F,
        }
    }

    /// Packs T.P.E.S. components into a 32-bit u32 integer.
    /// Bits 28-31: Tier [T]
    /// Bits 16-27: Positrons [P]
    /// Bits 4-15:  Electrons [E]
    /// Bits 0-3:   Phase [S]
    pub fn pack(&self) -> u32 {
        ((self.tier as u32 & 0x0F) << 28)
            | ((self.positrons as u32 & 0x0FFF) << 16)
            | ((self.electrons as u32 & 0x0FFF) << 4)
            | (self.phase as u32 & 0x0F)
    }

    /// Unpacks a 32-bit u32 integer into a Tpes struct.
    pub fn unpack(code: u32) -> Self {
        let tier = ((code >> 28) & 0x0F) as u8;
        let positrons = ((code >> 16) & 0x0FFF) as u16;
        let electrons = ((code >> 4) & 0x0FFF) as u16;
        let phase = (code & 0x0F) as u8;
        Self {
            tier,
            positrons,
            electrons,
            phase,
        }
    }

    /// Net charge: Positrons [P] - Electrons [E]
    pub fn net_charge(&self) -> i32 {
        self.positrons as i32 - self.electrons as i32
    }

    /// Net exhaust calculation based on particle total activity, tier, and phase.
    /// Calculated as total particles (P + E) scaled by phase and tier factor.
    pub fn net_exhaust(&self) -> i32 {
        let total_particles = self.positrons as i32 + self.electrons as i32;
        total_particles * (self.phase as i32 + 1) * (self.tier as i32 + 1)
    }
}

impl From<u32> for Tpes {
    fn from(code: u32) -> Self {
        Tpes::unpack(code)
    }
}

impl From<Tpes> for u32 {
    fn from(tpes: Tpes) -> Self {
        tpes.pack()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tpes_packing_unpacking() {
        let original = Tpes::new(2, 2, 1, 1); // Proton triad (2:2:1:1)
        let packed = original.pack();
        let unpacked = Tpes::unpack(packed);
        assert_eq!(original, unpacked);

        let code: u32 = 0x20020011; // T=2, P=2, E=1, S=1
        let tpes = Tpes::from(code);
        assert_eq!(tpes.tier, 2);
        assert_eq!(tpes.positrons, 2);
        assert_eq!(tpes.electrons, 1);
        assert_eq!(tpes.phase, 1);
        assert_eq!(u32::from(tpes), code);
    }

    #[test]
    fn test_net_charge() {
        let proton = Tpes::new(2, 2, 1, 1);
        assert_eq!(proton.net_charge(), 1);

        let electron_like = Tpes::new(1, 0, 1, 1);
        assert_eq!(electron_like.net_charge(), -1);

        let neutral = Tpes::new(1, 2, 2, 0);
        assert_eq!(neutral.net_charge(), 0);
    }

    #[test]
    fn test_net_exhaust() {
        let proton = Tpes::new(2, 2, 1, 1);
        // (2 + 1) * (1 + 1) * (2 + 1) = 3 * 2 * 3 = 18
        assert_eq!(proton.net_exhaust(), 18);
    }
}
