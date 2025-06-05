#![allow(missing_docs)]
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize, Valid};
use ark_std::rand::RngCore;

/// --------- field ----------------------------------------------------------
pub trait Field:
    Sized
    + Clone
    + Copy
    + PartialEq
    + Send
    + Sync
    + CanonicalSerialize
    + CanonicalDeserialize
    + Valid
{
    fn zero() -> Self;
    fn one() -> Self;
    fn is_zero(&self) -> bool;

    fn add(&self, rhs: &Self) -> Self;
    fn sub(&self, rhs: &Self) -> Self;
    fn mul(&self, rhs: &Self) -> Self;
    fn inv(&self) -> Option<Self>;

    fn random<R: RngCore>(rng: &mut R) -> Self;
    
    fn from_u64(val: u64) -> Self;
    fn from_i64(val: i64) -> Self;
}

/// --------- group ----------------------------------------------------------
pub trait Group:
    Sized + Clone + PartialEq + Send + Sync + CanonicalSerialize + CanonicalDeserialize + Valid
{
    type Scalar: Field;

    fn identity() -> Self;
    fn add(&self, rhs: &Self) -> Self;
    fn neg(&self) -> Self;
    fn scale(&self, k: &Self::Scalar) -> Self;

    fn random<R: RngCore>(rng: &mut R) -> Self;
}

/// -------------------------------- pairing ----------------------------------
pub trait Pairing: Sized + Send + Sync {
    type G1: Group;
    type G2: Group;
    type GT: Group;

    /// e : G1 × G2 → GT
    fn pair(p: &Self::G1, q: &Self::G2) -> Self::GT;

    /// Multi-pairing: computes the product of pairings
    /// Π e(p_i, q_i)
    fn multi_pair(ps: &[Self::G1], qs: &[Self::G2]) -> Self::GT {
        assert_eq!(
            ps.len(),
            qs.len(),
            "multi_pair requires equal length vectors"
        );

        if ps.is_empty() {
            return Self::GT::identity();
        }

        ps.iter()
            .zip(qs.iter())
            .fold(Self::GT::identity(), |acc, (p, q)| {
                acc.add(&Self::pair(p, q))
            })
    }
}

pub trait MultiScalarMul<G: Group> {
    fn msm<'a>(bases: &'a [G], scalars: &MultilinearPolynomial<'a, G::Scalar>) -> G;
}

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MultilinearPolynomial<'a, F: Field> {
    LargeScalars(&'a [F]),
    U8Scalars(&'a [u8]),
    U16Scalars(&'a [u16]),
    U32Scalars(&'a [u32]),
    U64Scalars(&'a [u64]),
    I64Scalars(&'a [i64]),
}

impl<'a, F: Field> MultilinearPolynomial<'a, F> {
    pub fn len(&self) -> usize {
        match self {
            MultilinearPolynomial::LargeScalars(evals) => evals.len(),
            MultilinearPolynomial::U8Scalars(evals) => evals.len(),
            MultilinearPolynomial::U16Scalars(evals) => evals.len(),
            MultilinearPolynomial::U32Scalars(evals) => evals.len(),
            MultilinearPolynomial::U64Scalars(evals) => evals.len(),
            MultilinearPolynomial::I64Scalars(evals) => evals.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn get(&self, index: usize) -> Option<F> {
        match self {
            MultilinearPolynomial::LargeScalars(evals) => {
                evals.get(index).copied()
            }
            MultilinearPolynomial::U8Scalars(evals) => {
                evals.get(index).map(|&x| F::from_u64(x as u64))
            }
            MultilinearPolynomial::U16Scalars(evals) => {
                evals.get(index).map(|&x| F::from_u64(x as u64))
            }
            MultilinearPolynomial::U32Scalars(evals) => {
                evals.get(index).map(|&x| F::from_u64(x as u64))
            }
            MultilinearPolynomial::U64Scalars(evals) => {
                evals.get(index).map(|&x| F::from_u64(x))
            }
            MultilinearPolynomial::I64Scalars(evals) => {
                evals.get(index).map(|&x| F::from_i64(x))
            }
        }
    }

    pub fn iter(&self) -> MultilinearPolynomialIter<'_, '_, F> {
        MultilinearPolynomialIter {
            poly: self,
            index: 0,
        }
    }

    pub fn chunks(&self, chunk_size: usize) -> MultilinearPolynomialChunks<'_, F> {
        MultilinearPolynomialChunks {
            poly: self,
            chunk_size,
            index: 0,
        }
    }
}

pub struct MultilinearPolynomialIter<'b, 'a, F: Field> {
    poly: &'b MultilinearPolynomial<'a, F>,
    index: usize,
}

impl<'b, 'a, F: Field> Iterator for MultilinearPolynomialIter<'b, 'a, F> {
    type Item = F;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index < self.poly.len() {
            let result = self.poly.get(self.index);
            self.index += 1;
            result
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.poly.len().saturating_sub(self.index);
        (remaining, Some(remaining))
    }
}

pub struct MultilinearPolynomialChunks<'a, F: Field> {
    poly: &'a MultilinearPolynomial<'a, F>,
    chunk_size: usize,
    index: usize,
}

impl<'a, F: Field> Iterator for MultilinearPolynomialChunks<'a, F> {
    type Item = MultilinearPolynomialSlice<'a, F>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.poly.len() {
            return None;
        }

        let start = self.index;
        let end = (self.index + self.chunk_size).min(self.poly.len());
        self.index = end;

        Some(MultilinearPolynomialSlice {
            poly: self.poly,
            start,
            end,
        })
    }
}

pub struct MultilinearPolynomialSlice<'a, F: Field> {
    poly: &'a MultilinearPolynomial<'a, F>,
    start: usize,
    end: usize,
}

impl<'a, F: Field> MultilinearPolynomialSlice<'a, F> {
    pub fn len(&self) -> usize {
        self.end - self.start
    }

    pub fn get(&self, index: usize) -> Option<F> {
        if index < self.len() {
            self.poly.get(self.start + index)
        } else {
            None
        }
    }

    pub fn iter(&self) -> MultilinearPolynomialSliceIter<'_, 'a, F> {
        MultilinearPolynomialSliceIter {
            slice: self,
            index: 0,
        }
    }

    pub fn to_multilinear_polynomial(&self) -> MultilinearPolynomial<'a, F> {
        match self.poly {
            MultilinearPolynomial::LargeScalars(scalars) => {
                MultilinearPolynomial::LargeScalars(&scalars[self.start..self.end])
            }
            MultilinearPolynomial::U8Scalars(scalars) => {
                MultilinearPolynomial::U8Scalars(&scalars[self.start..self.end])
            }
            MultilinearPolynomial::U16Scalars(scalars) => {
                MultilinearPolynomial::U16Scalars(&scalars[self.start..self.end])
            }
            MultilinearPolynomial::U32Scalars(scalars) => {
                MultilinearPolynomial::U32Scalars(&scalars[self.start..self.end])
            }
            MultilinearPolynomial::U64Scalars(scalars) => {
                MultilinearPolynomial::U64Scalars(&scalars[self.start..self.end])
            }
            MultilinearPolynomial::I64Scalars(scalars) => {
                MultilinearPolynomial::I64Scalars(&scalars[self.start..self.end])
            }
        }
    }
}

pub struct MultilinearPolynomialSliceIter<'b, 'a, F: Field> {
    slice: &'b MultilinearPolynomialSlice<'a, F>,
    index: usize,
}

impl<'b, 'a, F: Field> Iterator for MultilinearPolynomialSliceIter<'b, 'a, F> {
    type Item = F;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index < self.slice.len() {
            let result = self.slice.get(self.index);
            self.index += 1;
            result
        } else {
            None
        }
    }
}