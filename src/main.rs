use std::time::Instant;

use classer_opt::linear::LinearDP;
use ipm::alg::{
    descent::newton::NewtonsMethod,
    ipm::{barrier::BarrierMethod, infeasible::InfeasibleIpm},
    line_search::guarded::GuardedLineSearch,
};
use nalgebra::Vector2;

#[derive(Clone, Debug)]
pub struct LinearDiscrimination {
    pub vec_a: [f64; 2],
    pub scl_b: f64,
}

fn linear_discriminate(
    black_points: &[[f32; 2]],
    white_points: &[[f32; 2]],
) -> Result<LinearDiscrimination, String> {
    let mut problem = LinearDP {
        xs: black_points
            .iter()
            .map(|x| Vector2::new(x[0] as f64, x[1] as f64))
            .collect(),
        ys: white_points
            .iter()
            .map(|y| Vector2::new(y[0] as f64, y[1] as f64))
            .collect(),
        gamma: 1.0,
    };

    let lsp = GuardedLineSearch::new(0.3, 0.7).unwrap();

    let center = NewtonsMethod::new(1e-8, lsp.clone(), 128, 1024).unwrap();
    let params = BarrierMethod::new(1e-1, 10.0, 1e-8, center).unwrap();

    let aux_center = NewtonsMethod::new(1e-5, lsp, 16, 32).unwrap();
    let aux_params = BarrierMethod::new(1e-1, 1.5, 1e-3, aux_center).unwrap();

    let inf_ipm = InfeasibleIpm::new(aux_params, params);

    let (a, b) = problem.solve(&inf_ipm)?;

    Ok(LinearDiscrimination { vec_a: a, scl_b: b })
}

fn main() {
    let xs = [
        [165.0664062500000000, 135.7421875000000000],
        [147.5078125000000000, 158.0898437500000000],
    ];
    let ys = [
        [227.3203125000000000, 241.8945312500000000],
        [260.0429687500000000, 223.5351562500000000],
    ];

    let t0 = Instant::now();
    let sol = linear_discriminate(&xs, &ys);
    let dur = t0.elapsed();

    println!("Sol: {sol:?}");
    println!("Elapsed: {dur:?}");
}
