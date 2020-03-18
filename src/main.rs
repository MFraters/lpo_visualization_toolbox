extern crate meshgrid;
extern crate ndarray;
extern crate palette;

use ndarray::Array2;
//use ndarray::Array3;
use ndarray::Zip;
use std::fs;

//use std::error::Error;
use std::fs::File;
//use std::io::prelude::*;
use std::path::Path;

use ndarray::prelude::*;
use palette::{Gradient, LinSrgb};
use plotters::prelude::*;
use std::error::Error;
use std::io::prelude::*;
use std::path::PathBuf;
use std::time::Instant;
use structopt::StructOpt;

use serde_derive::{Deserialize, Serialize};
//use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize, Debug)]
enum LaticeAxes {
    AAxis,
    BAxis,
    CAxis,
}

#[derive(Debug)]
struct Lambert {
    X: Array<f64, Dim<[usize; 2]>>,
    Z: Array<f64, Dim<[usize; 2]>>,
    R: f64,
    x: Array<f64, Dim<[usize; 2]>>,
    y: Array<f64, Dim<[usize; 2]>>,
    z: Array<f64, Dim<[usize; 2]>>,
}

#[derive(Deserialize)]
struct Config {
    base_dir: String,
    experiment_dirs: Vec<String>,
    pole_figures: PoleFigures,
}
#[derive(Deserialize)]
struct PoleFigures {
    time_steps: Vec<u64>,
    particle_ids: Vec<u64>,
    axes: Vec<LaticeAxes>,
}

#[derive(StructOpt, Debug)]
#[structopt(name = "basic")]
struct Opt {
    /// Files to process
    #[structopt(name = "config file", parse(from_os_str))]
    config_file: PathBuf,
    // Output dir
    //#[structopt(short, long)]
    //input_dir: String,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let opt = Opt::from_args();
    let before = Instant::now();
    let config_file = opt.config_file;
    let config_file_display = config_file.display();
    // Open the path in read-only mode, returns `io::Result<File>`
    let mut file = match File::open(&config_file) {
        // The `description` method of `io::Error` returns a string that
        // describes the error
        Err(why) => panic!(
            "couldn't open {}: {}",
            config_file_display,
            why.description()
        ),
        Ok(file) => file,
    };
    // Read the file contents into a string, returns `io::Result<usize>`
    let mut config_file_string = String::new();
    match file.read_to_string(&mut config_file_string) {
        Err(why) => panic!(
            "couldn't read {}: {}",
            config_file_display,
            why.description()
        ),
        Ok(_) => (),
    }

    let config: Config = toml::from_str(&config_file_string).unwrap();

    let base_dir = config.base_dir;
    for experiment_dir in config.experiment_dirs {
        println!("Processing experiment {}", experiment_dir);
        for time_step in &config.pole_figures.time_steps {
            let lpo_dir = base_dir.clone() + &experiment_dir;

            let mut file_found: bool = false;
            let file_prefix = "particle_LPO/weighted_LPO";
            //let time = 100;
            let mut rank_id = 0;
            let particle_id = 0;

            let mut Pva = Vec::new();
            let mut Pvb = Vec::new();
            let mut Pvc = Vec::new();
            while !file_found {
                let angles_file =
                    format!("{}{}-{:05}.{:04}.dat", lpo_dir, file_prefix, time_step, rank_id);
                let angles_file = Path::new(&angles_file);
                let output_file = format!(
                    "{}{}_t{:05}.{:05}.png",
                    lpo_dir, file_prefix, time_step, particle_id
                );
                let output_file = Path::new(&output_file);

                println!("  trying file name: {}", angles_file.display());

                // check wheter file exists, if not it means that is reached the max rank, so stop.
                if !(fs::metadata(angles_file).is_ok()) {
                    println!(
                        "particle id {} not found for timestep {}.",
                        particle_id, time_step
                    );
                    file_found = false;
                    break;
                }

                // check wheter file is empty, if not continue to next rank
                if fs::metadata(angles_file)?.len() == 0 {
                    rank_id = rank_id + 1;
                    continue;
                }
                file_found = true;

                println!("  file:{}", angles_file.display());
                let file = File::open(angles_file)?;
                let mut rdr = csv::ReaderBuilder::new()
                    .has_headers(false)
                    .delimiter(b' ')
                    .from_reader(file);

                type Record = (u64, f64, f64, f64, f64);

                for result in rdr.deserialize() {
                    // We must tell Serde what type we want to deserialize into.
                    let record: Record = result?;
                    if record.0 == particle_id {
                        let deg_to_rad = std::f64::consts::PI / 180.;
                        let dcm = dir_cos_matrix2(
                            record.1 * deg_to_rad,
                            record.2 * deg_to_rad,
                            record.3 * deg_to_rad,
                        )?;

                        Pva.push(dcm.row(0).to_owned());
                        Pvb.push(dcm.row(1).to_owned());
                        Pvc.push(dcm.row(2).to_owned());
                    }
                }

                let sphere_points = 151;
                let n_grains = Pva.len();

                let mut Pa = Array2::zeros((n_grains, 3));
                let mut Pb = Array2::zeros((n_grains, 3));
                let mut Pc = Array2::zeros((n_grains, 3));
                for i in 0..n_grains {
                    for j in 0..3 {
                        Pa[[i, j]] = Pva[i][j];
                        Pb[[i, j]] = Pvb[i][j];
                        Pc[[i, j]] = Pvc[i][j];
                    }
                }

                let lambert = create_lambertEA_gridpoint(sphere_points, "upper".to_string())?;
                let mut S = Array2::zeros((3, sphere_points * sphere_points));

                for i in 0..sphere_points {
                    for j in 0..sphere_points {
                        S[[0, i * sphere_points + j]] = lambert.x[[i, j]];
                        S[[1, i * sphere_points + j]] = lambert.y[[i, j]];
                        S[[2, i * sphere_points + j]] = lambert.z[[i, j]];
                    }
                }

                let countsA = gaussian_orientation_counts(&Pa, &S, sphere_points)?;
                let countsB = gaussian_orientation_counts(&Pb, &S, sphere_points)?;
                let countsC = gaussian_orientation_counts(&Pc, &S, sphere_points)?;
                println!(
                    "  Before make_polefigures: Elapsed time: {:.2?}",
                    before.elapsed()
                );
                make_polefigures(
                    n_grains,
                    time_step,
                    0,
                    &countsA,
                    &countsB,
                    &countsC,
                    &lambert,
                    output_file,
                );

                println!(
                    "  After make_polefigures: Elapsed time: {:.2?}",
                    before.elapsed()
                );
            }
        }
    }
    Ok(())
}

fn dir_cos_matrix2(
    phi1: f64,
    theta: f64,
    phi2: f64,
) -> Result<Array2<f64>, Box<dyn std::error::Error>> {
    let mut dcm: Array2<f64> = Array::zeros((3, 3));

    dcm[[0, 0]] = ((phi2.cos()) * (phi1.cos())) - ((theta.cos()) * (phi1.sin()) * (phi2.sin()));
    dcm[[0, 1]] = phi2.cos() * phi1.sin() + theta.cos() * phi1.cos() * phi2.sin();
    dcm[[0, 2]] = phi2.sin() * theta.sin();

    dcm[[1, 0]] = -1.0 * phi2.sin() * phi1.cos() - theta.cos() * phi1.sin() * phi2.cos();
    dcm[[1, 1]] = -1.0 * phi2.sin() * phi1.sin() + theta.cos() * phi1.cos() * phi2.cos();
    dcm[[1, 2]] = phi2.cos() * theta.sin();

    dcm[[2, 0]] = theta.sin() * phi1.sin();
    dcm[[2, 1]] = -1.0 * theta.sin() * phi1.cos();
    dcm[[2, 2]] = theta.cos();

    Ok(dcm)
}

// Following method in Robin and Jowett, Tectonophysics, 1986
// Computerized contouring and statistical evaluation of orientation data
// using contouring circles and continuous weighting functions

fn gaussian_orientation_counts(
    P: &Array2<f64>,
    S: &Array2<f64>,
    sphere_points: usize,
) -> Result<Array2<f64>, Box<dyn std::error::Error>> {
    let npts = P.shape()[0];

    // Choose k, which defines width of spherical gaussian  (table 3)
    let k = 2. * (1. + npts as f64 / 9.);

    // Given k, calculate standard deviation (eq 13b)
    let std_dev = ((npts as f64 * (k as f64 / 2. - 1.) / (k * k)) as f64).sqrt();

    // Calculate dot product
    let mut cosalpha = P.dot(S);

    // Calculate the counts from the spherical gaussian
    //let counts = Array::zeros(cosalpha.shape());
    cosalpha.par_mapv_inplace(f64::abs);

    cosalpha = (k as f64) * (cosalpha - 1.);

    cosalpha.par_mapv_inplace(f64::exp);

    let counts = cosalpha.sum_axis(Axis(0));
    let counts = counts.into_shape((sphere_points, sphere_points))?;

    // normalize so each MUD is 3 sigma from that expected for a uniform
    // distribution
    let counts = counts / (3. * std_dev);

    Ok(counts)
}

fn make_polefigures(
    n_grains: usize,
    time_step: &u64,
    particle_id: u64,
    countsA: &Array2<f64>,
    countsB: &Array2<f64>,
    countsC: &Array2<f64>,
    Lambert: &Lambert,
    output_file: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let before = Instant::now();
    let one_eleventh = 1. / 11.;
    let grad1 = Gradient::with_domain(vec![
        (0.0, LinSrgb::new(0.164, 0.043, 0.85)),
        (1. * one_eleventh, LinSrgb::new(0.15, 0.306, 1.0)),
        (2. * one_eleventh, LinSrgb::new(0.25, 0.63, 1.0)),
        (3. * one_eleventh, LinSrgb::new(0.45, 0.853, 1.0)),
        (4. * one_eleventh, LinSrgb::new(0.67, 0.973, 1.0)),
        (5. * one_eleventh, LinSrgb::new(0.88, 1.0, 1.0)),
        (6. * one_eleventh, LinSrgb::new(1.0, 1.0, 0.75)),
        (7. * one_eleventh, LinSrgb::new(1.0, 0.88, 0.6)),
        (8. * one_eleventh, LinSrgb::new(1.0, 0.679, 0.45)),
        (9. * one_eleventh, LinSrgb::new(0.97, 0.430, 0.37)),
        (10. * one_eleventh, LinSrgb::new(0.85, 0.15, 0.196)),
        (1., LinSrgb::new(0.65, 0.0, 0.13)),
    ]);

    let mut counts = Vec::new();
    counts.push(countsA);
    counts.push(countsB);
    counts.push(countsC);
    // Grid of points is a square and it extends outside the pole figure circumference.
    // Create mask to only plot color and contours within the pole figure
    let mut mask = Lambert.X.clone();

    let mut circle_path: Vec<(f64, f64)> = Vec::new();
    Zip::from(&mut mask)
        .and(&Lambert.X)
        .and(&Lambert.Z)
        .par_apply(|a, x, z| {
            let radius = (x * x + z * z).sqrt();
            if radius >= Lambert.R + 0.001 {
                *a = std::f64::NAN
            } else {
                *a = 1.
            }
        });
    let npts = countsA.shape()[0];

    // Create a boundary circle for the Schmidt Net
    let bd_theta = Array::linspace(0., 2. * std::f64::consts::PI, 100);
    let bd_center = [0.0, 0.0];
    let bd_radius = 2.0 / 2.0_f64.sqrt();

    for i in 0..bd_theta.len() {
        circle_path.push((
            bd_theta[i].sin() * bd_radius + bd_center[0],
            bd_theta[i].cos() * bd_radius + bd_center[1],
        ));
    }

    let gam = 0.5; // exponent for power-law normalization of color-scale
                   //let f = 1.05; // factor to make plot limits slightly bigger than the circle

    // Determine which pole figures to plot based on size of countsX
    let mut have_aplot = 0;
    let mut have_bplot = 0;
    let mut have_cplot = 0;
    if countsA.len() > 1 {
        have_aplot = 1;
    }
    if countsB.len() > 1 {
        have_bplot = 1;
    }
    if countsC.len() > 1 {
        have_cplot = 1;
    }

    let figure_height = 800;
    let number_of_figures: usize = have_aplot + have_bplot + have_cplot;
    if number_of_figures < 1 {
        println!("No figures to make. Exit.");
        return Ok(());
    }

    let figure_width: u32 = number_of_figures as u32 * figure_height;

    println!("    Before drawing: Elapsed time: {:.2?}", before.elapsed());
    let path_string = output_file.to_string_lossy().into_owned();
    println!("    save file to {}", path_string);
    let root = BitMapBackend::new(&path_string, (figure_width + 10, figure_height + 100))
        .into_drawing_area();
    root.fill(&WHITE)?;

    println!("    made root: Elapsed time: {:.2?}", before.elapsed());
    let (left, right) = root.split_horizontally(figure_width);
    let drawing_areas = left.split_evenly((1, number_of_figures));

    println!("    number of figures = {}", number_of_figures);
    for figure_number in 0..number_of_figures {
        let (upper, lower) = drawing_areas[figure_number].split_vertically(100);
        let mut chart = ChartBuilder::on(&lower).build_ranged(
            -Lambert.R - 0.05..Lambert.R + 0.15,
            -Lambert.R - 0.05..Lambert.R + 0.15,
        )?;

        // get max valuein countsA
        let mut max_count_value = 0.0;
        for i in 0..npts - 1 {
            for j in 0..npts - 1 {
                if counts[figure_number][[i, j]] > max_count_value {
                    max_count_value = counts[figure_number][[i, j]];
                }
            }
        }

        let mut current: Vec<Vec<Vec<(f64, f64)>>> = Vec::new();
        for i in 0..npts - 1 {
            let mut current_i: Vec<Vec<(f64, f64)>> = Vec::new();
            for j in 0..npts - 1 {
                current_i.push(vec![
                    (Lambert.X[[i + 1, j]], Lambert.Z[[i + 1, j]]),
                    (Lambert.X[[i + 1, j + 1]], Lambert.Z[[i + 1, j + 1]]),
                    (Lambert.X[[i, j + 1]], Lambert.Z[[i, j + 1]]),
                    (Lambert.X[[i, j]], Lambert.Z[[i, j]]),
                ]);
            }
            current.push(current_i);
        }

        let mut total_mask = mask.clone();
        for i in 0..npts - 1 {
            for j in 0..npts - 1 {
                if !mask[[i, j]].is_nan()
                    && !mask[[i, j + 1]].is_nan()
                    && !mask[[i + 1, j + 1]].is_nan()
                    && !mask[[i + 1, j]].is_nan()
                {
                    total_mask[[i, j]] = 1.0;
                } else {
                    total_mask[[i, j]] = std::f64::NAN;
                }
            }
        }
        println!(
            "      before 1st drawing: Elapsed time: {:.2?}",
            before.elapsed()
        );
        for i in 0..npts - 1 {
            for j in 0..npts - 1 {
                if !mask[[i, j]].is_nan() {
                    chart.draw_series(std::iter::once(Polygon::new(
                        current[i][j].clone(),
                        &grad1.get(
                            (counts[figure_number][[i, j]]).powf(gam) / (max_count_value.powf(gam)),
                        ),
                    )))?;
                }
            }
        }

        chart.draw_series(std::iter::once(PathElement::new(
            circle_path.clone(),
            Into::<ShapeStyle>::into(&BLACK).stroke_width(5),
        )))?;

        //chart.draw_series(std::iter::once(PathElement::new(vec![(0.,-400.),(0.,400.)],&BLACK)));
        //chart.draw_series(std::iter::once(PathElement::new(vec![(-400.,0.),(400.,0.)],&BLACK)));

        let hp = Percentage {
            total: figure_height as f64,
        }; //(100/figure_height) as i32;
        let wp = Percentage {
            total: figure_width as f64 / number_of_figures as f64,
        };

        println!(
            "      before 2st drawing: Elapsed time: {:.2?}",
            before.elapsed()
        );
        let font_size = 50;
        let line_distance = 5.5;
        let top_margin = 0.2;
        let left_margin = 0.5;
        let font_type = "Inconsolata";//"consolas";//"sans-serif"
        match figure_number {
            0 => {
                drawing_areas[figure_number].draw(&Text::new(
                    format!("a-axis"),
                    (wp.calc(left_margin) as i32, hp.calc(top_margin) as i32),
                    (font_type, font_size).into_font(),
                )).unwrap();
                drawing_areas[figure_number].draw(&Text::new(
                    format!("max = {:.2}", max_count_value),
                    (wp.calc(left_margin) as i32, hp.calc(top_margin+line_distance) as i32),
                    (font_type, font_size).into_font(),
                ))?;
                drawing_areas[figure_number].draw(&Text::new(
                    format!("T = {}", n_grains),
                    (wp.calc(left_margin-0.4) as i32, hp.calc(top_margin+2.*line_distance) as i32),
                    (font_type, font_size).into_font(),
                ))?;
                drawing_areas[figure_number].draw(&Text::new(
                    format!("P = {}", n_grains),
                    (wp.calc(left_margin+0.2) as i32, hp.calc(top_margin+3.*line_distance) as i32),
                    (font_type, font_size).into_font(),
                ))?;
                drawing_areas[figure_number].draw(&Text::new(
                    format!("Z"),
                    (wp.calc(46.4) as i32, hp.calc(11.) as i32),
                    (font_type, font_size).into_font(),
                ))?;
                drawing_areas[figure_number].draw(&Text::new(
                    format!("X"),
                    (wp.calc(96.0) as i32, hp.calc(61.75) as i32),
                    (font_type, font_size).into_font(),
                ))?;
            }
            1 => {
                drawing_areas[figure_number].draw(&Text::new(
                    format!("b-axis"),
                    (wp.calc(left_margin) as i32, hp.calc(top_margin) as i32),
                    (font_type, font_size).into_font(),
                ))?;
                drawing_areas[figure_number].draw(&Text::new(
                    format!("max = {:.2}", max_count_value),
                    (wp.calc(left_margin) as i32, hp.calc(top_margin+line_distance) as i32),
                    (font_type, font_size).into_font(),
                ))?;
                drawing_areas[figure_number].draw(&Text::new(
                    format!("Z"),
                    (wp.calc(46.4) as i32, hp.calc(top_margin+2.*line_distance) as i32),
                    (font_type, font_size).into_font(),
                ))?;
                drawing_areas[figure_number].draw(&Text::new(
                    format!("X"),
                    (wp.calc(96.0) as i32, hp.calc(61.75) as i32),
                    (font_type, font_size).into_font(),
                ))?;
            }
            2 => {
                drawing_areas[figure_number].draw(&Text::new(
                    format!("c-axis"),
                    (wp.calc(left_margin) as i32, hp.calc(top_margin) as i32),
                    (font_type, font_size).into_font(),
                ))?;
                drawing_areas[figure_number].draw(&Text::new(
                    format!("max = {:.2}", max_count_value),
                    (wp.calc(left_margin) as i32, hp.calc(top_margin+line_distance) as i32),
                    (font_type, font_size).into_font(),
                ))?;
                drawing_areas[figure_number].draw(&Text::new(
                    format!("Z"),
                    (wp.calc(46.4) as i32, hp.calc(top_margin+2.*line_distance) as i32),
                    (font_type, font_size).into_font(),
                ))?;
                drawing_areas[figure_number].draw(&Text::new(
                    format!("X"),
                    (wp.calc(96.0) as i32, hp.calc(61.75) as i32),
                    (font_type, font_size).into_font(),
                ))?;
            }
            _ => println!("Error: wrong figure number!"),
        }

        println!(
            "      made one of the figures: Elapsed time: {:.2?}",
            before.elapsed()
        );
        //chart.draw_series(std::iter::once(Text::new(format!("A"), (10, 0), ("sans-serif", 10).into_font())))?;
    }

    println!(
        "    After make_polefigures: Elapsed time: {:.2?}",
        before.elapsed()
    );

    Ok(())
}

struct Percentage {
    total: f64,
}

impl Percentage {
    fn calc(&self, part: f64) -> f64 {
        (part / 100.) * self.total
    }
}

fn create_meshgrid(
    X: &Array<f64, Dim<[usize; 1]>>,
    Y: &Array<f64, Dim<[usize; 1]>>,
) -> Result<(Array<f64, Dim<[usize; 2]>>, Array<f64, Dim<[usize; 2]>>), Box<dyn std::error::Error>>
{
    let mut new_x: Array<f64, Dim<[usize; 2]>> = Array::zeros([X.len(), Y.len()]);
    let mut new_y: Array<f64, Dim<[usize; 2]>> = Array::zeros([X.len(), Y.len()]);
    let mut counter = 0;
    let max_count = X.len();
    for value in new_x.iter_mut() {
        *value = X[counter];
        counter += 1;
        if counter >= max_count {
            counter = 0;
        }
    }

    counter = 0;
    let mut counter_y = 0;
    for value in new_y.iter_mut() {
        *value = Y[counter];
        counter_y += 1;
        if counter_y >= max_count {
            counter_y = 0;
            counter += 1;
        }
    }

    Ok((new_x, new_y))
}

/*
* Create a grid of evenly spaced points for contouring pole figure
* matlab-version of drex uses 151 x 151 points.
*
* Note wikipedia has the transformation for the lambert equal area projection
* for both X,Y --> x,y,z and x,y,z --> X,Y
* so given the Pa directions (unit vectors, do R = 1 on a sphere) in x,y,z,
* you can get X,Y on the lambert projection to plot these as a scatter plot
* without ever converting the spherical coordinates
*/
fn create_lambertEA_gridpoint(
    sphere_points: usize,
    hemisphere: String,
) -> Result<Lambert, Box<dyn std::error::Error>> {
    // Create a grid of points at increasing radius in the X and Y direction
    // Use the coordinate X,Y,R to plot these points on the lambert projection
    let R: f64 = 2.0_f64.sqrt(); // need this to get full sphere in Lambert projection)
    let X = Array::linspace(-R, R, sphere_points);
    //println!("X={}", X);
    let Y = X.clone();
    let (X, mut Z) = create_meshgrid(&X, &Y)?;
    //println!("X={}", X);
    //println!("Z={}", Z);
    Z.invert_axis(Axis(0));
    //println!("Z={}", Z);
    //let mut Y = Y.into_raw_vec();
    //Y.reverse();
    //let Y: Array<f64, Dim<[usize; 2]>> = Array::from(Y); // Y increases up

    // map onto lambert projection, assumes r = 1?
    // added np.abs to avoid tiny negative numbers in sqrt
    // todo, turn into enum and matchs
    let mut x = Array::zeros([sphere_points, sphere_points]);
    let mut y = Array::zeros([sphere_points, sphere_points]);
    let mut z = Array::zeros([sphere_points, sphere_points]);
    let mut mag = Array::zeros([sphere_points, sphere_points]);

    Zip::from(&mut x).and(&X).and(&Z).par_apply(|a, &X, &Z| {
        *a = if 1. - (X * X + Z * Z) / 4. > std::f64::EPSILON {
            ((1. - (X * X + Z * Z) / 4.).abs()).sqrt() * X
        } else {
            0.
        };
    });

    if hemisphere == "lower" {
        Zip::from(&mut y).and(&X).and(&Z).par_apply(|a, &X, &Z| {
            *a = -(1. - (X * X + Z * Z) / 2.);
        });
        Zip::from(&mut z).and(&X).and(&Z).par_apply(|a, &X, &Z| {
            *a = ((1. - (X * X + Z * Z) / 4.).abs()).sqrt() * (-Z);
        });
    } else if hemisphere == "upper" {
        Zip::from(&mut y).and(&X).and(&Z).par_apply(|a, &X, &Z| {
            *a = 1. - (X * X + Z * Z) / 2.;
        });
        Zip::from(&mut z).and(&X).and(&Z).par_apply(|a, &X, &Z| {
            *a = ((1. - (X * X + Z * Z) / 4.).abs()).sqrt() * Z;
        });
    };

    /*if hemisphere == "lower" {
        Zip::from(&mut z).and(&X).and(&Y).apply(|a, &X, &Y| {
            *a = -(1. - (X * X + Y * X) / 2.);
        });
    } else if hemisphere == "upper" {
        Zip::from(&mut z).and(&X).and(&Y).apply(|a, &X, &Y| {
            *a = 1. - (X * X + Y * Y) / 2.;
        });
    };*/
    /*println!("");
    println!("===============================1====================================");
    println!("");
    println!("x={}", x);
    println!("");
    println!("===============================1====================================");
    println!("");*/

    //else {
    //    println!("Error: hemisphere should be lower or upper.");
    //}

    /* correct roundoff errors
    # Menno: this doesn't seem to do anything, because it only sets
    #        a value to zero when it is smaller than epsilon. Since
    #        this produces only positive values and increments of the
    #        values can't be smaller then epsilon (by defintion),
    #        it looks to me that this is never used.
    */

    //x[1-(X**2 + Y**2)/4 < np.finfo(float).eps] = 0;
    //y[1-(X**2 + Y**2)/4 < np.finfo(float).eps] = 0;

    // ensure unit vectors
    // Use these values of x,y,z to calculate the gaussian weighting function for contouring
    //let mag = (x*x + y*x + z*x).sqrt();
    Zip::from(&mut mag)
        .and(&x)
        .and(&y)
        .and(&z)
        .par_apply(|a, &x, &y, &z| {
            *a = (x * x + y * y + z * z).sqrt();
        });
    let mag = mag;
    x = x / &mag;
    y = y / &mag;
    z = z / &mag;
    /*println!("");
    println!("================================2===================================");
    println!("");
    println!("mag={}", mag);
    println!("x={}", x);
    println!("y={}", y);
    println!("z={}", z);
    println!("");
    println!("=================================2==================================");
    println!("");*/
    Ok(Lambert {
        X: X,
        Z: Z,
        R: R,
        x: x,
        y: y,
        z: z,
    })
}
