use std::{fmt::Debug, iter::Sum};

use ipm::{
    ConvexConstraints, CostFunction, Gradient, Hessian, LinearConstraints,
    alg::{
        barrier::{BarrierParams, barrier_method},
        newton::NewtonsMethodSolution,
    },
};
use nalgebra::{
    ComplexField, Const, DMatrix, DVector, Dyn, Matrix, RawStorage, Scalar, StorageMut, Vector,
    Vector2,
};
use num_traits::{Float, FromPrimitive, Inv, Num, NumAssign, real::Real};

pub struct LinearDP<F> {
    pub xs: Vec<Vector2<F>>,
    pub ys: Vec<Vector2<F>>,
    pub gamma: F,
}

impl<T> LinearDP<T>
where
    T: Debug
        + Float
        + Scalar
        + NumAssign
        + ComplexField<RealField = T>
        + PartialOrd
        + Real
        + Sum
        + Inv<Output = T>
        + FromPrimitive,
{
    pub fn solve(&mut self, params: &BarrierParams<T>) -> ([T; 2], T) {
        let x0 = self.find_x0();

        let sol: NewtonsMethodSolution<T> = barrier_method(self, &x0, params);

        let a = [sol.arg[0], sol.arg[1]];
        let b = sol.arg[2];

        (a, b)
    }

    fn find_x0(&self) -> DVector<T> {
        let nrows = 3 + self.xs.len() + self.ys.len();
        let mut x0 = DVector::from_element(nrows, T::from_f64(1e4).unwrap());
        x0[0] = T::one();
        x0[1] = T::one();
        x0[2] = T::zero();
        x0
    }
}

impl<T> CostFunction for LinearDP<T>
where
    T: Copy + Scalar + ComplexField<RealField = T>,
{
    type F = T;

    fn cost<S>(&mut self, param: &Vector<Self::F, Dyn, S>, out: &mut Self::F)
    where
        S: RawStorage<Self::F, Dyn> + Debug,
    {
        let a = param.rows(0, 2);
        // let b = param.rows(2, 1);
        let u = param.rows(3, self.xs.len());
        let v = param.rows(3 + self.xs.len(), self.ys.len());

        *out = a.norm() + self.gamma * (u.sum() + v.sum())
    }

    fn dims(&self) -> usize {
        self.xs.len() + self.ys.len() + 3
    }
}

impl<T> Gradient for LinearDP<T>
where
    T: Copy + Scalar + ComplexField<RealField = T> + Num,
{
    fn gradient<S1, S2>(
        &mut self,
        param: &Vector<Self::F, Dyn, S1>,
        out: &mut Vector<Self::F, Dyn, S2>,
    ) where
        S1: RawStorage<Self::F, Dyn> + Debug,
        S2: StorageMut<Self::F, Dyn> + Debug,
    {
        out.fill(T::zero());

        // layout: a (2) | b (1) | u (nx) | v (ny)
        let a = param.rows(0, 2).into_owned();
        let nx = self.xs.len();
        let ny = self.ys.len();

        // gradient wrt a: a / ||a||  (choose 0 when ||a|| == 0)
        let norm_a = a.norm();
        if norm_a != T::zero() {
            let grad_a = &a / norm_a;
            out.rows_mut(0, 2).copy_from(&grad_a);
        } // else leave zeros (subgradient choice)

        // b has no contribution -> gradient 0 (already zero)

        // u and v: gamma * ones
        let start_u = 3;
        let start_v = 3 + nx;
        if nx > 0 {
            let ones_u = DVector::from_element(nx, self.gamma);
            out.rows_mut(start_u, nx).copy_from(&ones_u);
        }
        if ny > 0 {
            let ones_v = DVector::from_element(ny, self.gamma);
            out.rows_mut(start_v, ny).copy_from(&ones_v);
        }
    }
}

impl<T> Hessian for LinearDP<T>
where
    T: Copy + Scalar + ComplexField<RealField = T>,
{
    fn hessian<S1, S2>(
        &mut self,
        _param: &Vector<Self::F, Dyn, S1>,
        out: &mut Matrix<Self::F, Dyn, Dyn, S2>,
    ) where
        S1: RawStorage<Self::F, Dyn> + Debug,
        S2: StorageMut<Self::F, Dyn, Dyn> + Debug,
    {
        out.fill(T::zero());

        // Only a-block (top-left 2x2) is nonzero:
        // H_a = (I / ||a||) - (a a^T / ||a||^3)
        let a = _param.rows(0, 2).into_owned();
        let norm_a = a.norm();

        if norm_a != T::zero() {
            // build 2x2 identity and aa^T
            let i2 = DMatrix::identity(2, 2);
            let aa_t = &a * a.transpose(); // 2x2

            let ha = (&i2) / norm_a - (&aa_t) / (norm_a * norm_a * norm_a);

            // copy ha into top-left block of h
            let mut block = out.view_mut((0, 0), (2, 2));
            block.copy_from(&ha);
        }
        // all other second derivatives are zero because cost is linear in b,u,v and uses only ||a|| for quadratic part.
    }
}

impl<T> ConvexConstraints for LinearDP<T>
where
    T: Copy + Scalar + ComplexField<RealField = T>,
{
    fn number_of_constraints(&self) -> usize {
        2 * (self.xs.len() + self.ys.len())
    }

    fn convex_constraints<S>(&self, param: &Vector<Self::F, Dyn, S>, out: &mut [Self::F])
    where
        S: RawStorage<Self::F, Dyn> + Debug,
    {
        let n = self.xs.len();
        let m = self.ys.len();

        let a = param.fixed_rows::<2>(0);
        let b = param.fixed_rows::<1>(2);
        let u = param.rows(3, n);
        let v = param.rows(3 + n, m);

        for i in 0..n {
            out[i] = b[0] - a.dot(&self.xs[i]) - u[i] + Self::F::one();
        }
        for i in 0..m {
            out[n + i] = a.dot(&self.ys[i]) - b[0] - v[i] + Self::F::one();
        }
        for i in 0..n {
            out[n + m + i] = -u[i];
        }
        for i in 0..m {
            out[n + m + n + i] = -v[i];
        }
    }

    fn convex_gradients<S1, S2>(
        &self,
        _param: &Vector<Self::F, Dyn, S1>,
        out: &mut [Vector<Self::F, Dyn, S2>],
    ) where
        S1: RawStorage<Self::F, Dyn> + Debug,
        S2: StorageMut<Self::F, Dyn> + Debug,
    {
        let n = self.xs.len();
        let m = self.ys.len();

        // b[0] - a.dot(x_i) - u_i + 1
        for i in 0..n {
            let g = &mut out[i];
            g.rows_generic_mut(0, Const::<2>).copy_from(&-self.xs[i]);
            g[2] = T::one();
            g[3 + i] = -T::one();
        }

        // a.dot(y_i) - b[0] - v_i + 1
        for i in 0..m {
            let g = &mut out[n + i];
            g.rows_generic_mut(0, Const::<2>).copy_from(&self.ys[i]);
            g[2] = -Self::F::one();
            g[3 + n + i] = -Self::F::one();
        }

        // -u_i
        for i in 0..n {
            let g = &mut out[n + m + i];
            // g.fill(0.0);
            g[3 + i] = -Self::F::one();
        }

        // -v_i
        for i in 0..m {
            let g = &mut out[n + m + n + i];
            // g.fill(0.0);
            g[3 + n + i] = -Self::F::one();
        }
    }

    fn convex_hessians<S1, S2>(
        &self,
        _param: &Vector<Self::F, Dyn, S1>,
        _out: &mut [Matrix<Self::F, Dyn, Dyn, S2>],
    ) where
        S1: RawStorage<Self::F, Dyn> + Debug,
        S2: StorageMut<Self::F, Dyn, Dyn> + Debug,
    {
        // We assume the matrices are already initialized with zeros.
        // out.iter_mut().for_each(|x| x.fill(0.0));
    }
}

impl<T> LinearConstraints for LinearDP<T>
where
    T: Copy + Scalar + ComplexField<RealField = T>,
{
    fn mat_a(&self) -> DMatrix<Self::F> {
        DMatrix::zeros(0, self.dims())
    }

    fn vec_b(&self) -> DVector<Self::F> {
        DVector::zeros(0)
    }
}

#[cfg(test)]
mod tests {
    use ipm::alg::{barrier::BarrierParams, line_search::LineSearchParams, newton::NewtonParams};
    use nalgebra::Vector2;

    use crate::linear::LinearDP;

    #[test]
    fn test_square() {
        let xs = vec![Vector2::new(0.0, 0.0), Vector2::new(0.0, 1.0)];
        let ys = vec![Vector2::new(1.0, 0.0), Vector2::new(1.0, 1.0)];

        let mut prob = LinearDP {
            xs,
            ys,
            gamma: 1.0f64,
        };

        let lsp = LineSearchParams::new(0.3, 0.6);
        let center = NewtonParams::new(1e-1, lsp, 8, 20);
        let params = BarrierParams::new(10.0, 30.0, 1e-3, center);

        let (a, b) = prob.solve(&params);

        assert!((a[0] + 2.0).abs() < 1e-3); // we specified a tolerance of 1e-3
        assert!((a[1] + 0.0).abs() < 1e-3);
        assert!((b + 1.0).abs() < 1e-3);
    }
}
