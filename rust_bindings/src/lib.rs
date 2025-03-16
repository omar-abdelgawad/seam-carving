// /// Formats the sum of two numbers as string.
// #[pyfunction]
// fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
//     Ok((a + b).to_string())
// }
use numpy::ndarray::prelude::*;
use numpy::PyArray2;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[pyfunction]
fn update_cumulative_energy(cumulative_energy: &PyArray2<i32>) -> PyResult<()> {
    // Safety: we trust that the caller passes a proper 2D array.
    let mut cum = unsafe { cumulative_energy.as_array_mut() };
    let shape = cum.shape();
    if shape.len() != 2 {
        return Err(PyValueError::new_err("Input array must be 2D"));
    }
    let height = shape[0];
    let width = shape[1];

    // Process each row from the second row onward.
    for i in 1..height {
        let prev_row = cum.slice(s![i - 1, ..]);

        // Create "left" vector: rolled version of prev_row to the right.
        let mut left = Vec::with_capacity(width);
        left.push((1 << 15) - 1); // left[0] is set to 2^15 - 1
        for j in 1..width {
            left.push(prev_row[j - 1]);
        }

        // Create "right" vector: rolled version of prev_row to the left.
        let mut right = Vec::with_capacity(width);
        for j in 0..(width - 1) {
            right.push(prev_row[j + 1]);
        }
        right.push((1 << 15) - 1); // right[-1] is set to 2^15 - 1

        // Update the current row by adding the minimum of left, center, and right.
        for j in 0..width {
            let min_val = prev_row[j].min(left[j]).min(right[j]);
            cum[(i, j)] += min_val;
        }
    }
    Ok(())
}

// #[pymodule]
// fn seam_carving(_py: Python, m: &PyModule) -> PyResult<()> {
//     m.add_function(wrap_pyfunction!(compute_cumulative_energy, m)?)?;
//     Ok(())
// }

/// A Python module implemented in Rust.
#[pymodule]
fn rust_bindings(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(update_cumulative_energy, m)?)?;
    Ok(())
}
