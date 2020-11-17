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
use rayon::prelude::*;
use std::io::prelude::*;
use std::path::PathBuf;
use std::time::Instant;
use structopt::StructOpt;

use serde_derive::{Deserialize, Serialize};
//use serde::{Serialize, Deserialize};

use std::io::BufReader;

#[derive(Serialize, Deserialize, Debug)]
enum LaticeAxes {
    AAxis,
    BAxis,
    CAxis,
}

#[derive(Debug)]
struct Lambert {
    x_plane: Array<f64, Dim<[usize; 2]>>,
    z_plane: Array<f64, Dim<[usize; 2]>>,
    r_plane: f64,
    x: Array<f64, Dim<[usize; 2]>>,
    y: Array<f64, Dim<[usize; 2]>>,
    z: Array<f64, Dim<[usize; 2]>>,
}

#[derive(Deserialize)]
struct Config {
    base_dir: String,
    experiment_dirs: Vec<String>,
    pole_figures: PoleFiguresConfiguration,
    compressed: bool,
}

#[derive(Debug, Deserialize)]
//#[serde(rename_all = "PascalCase")]
struct Record {
    id: usize,
    olivine_euler_angles_phi: Option<f64>,
    olivine_euler_angles_theta: Option<f64>,
    olivine_euler_angles_z: Option<f64>,
    enstatite_euler_angles_phi: Option<f64>,
    enstatite_euler_angles_theta: Option<f64>,
    enstatite_euler_angles_z: Option<f64>,
}

#[derive(Debug, Deserialize)]
//#[serde(rename_all = "PascalCase")]
struct ParticleRecord {
    id: usize,
    x: f64,
    y: f64,
    z: Option<f64>,
    olivine_deformation_type: f64,
    full_norm_square: Option<f64>,
    triclinic_norm_square_p1: Option<f64>,
    triclinic_norm_square_p2: Option<f64>,
    triclinic_norm_square_p3: Option<f64>,
    monoclinic_norm_square_p1: Option<f64>,
    monoclinic_norm_square_p2: Option<f64>,
    monoclinic_norm_square_p3: Option<f64>,
    orthohombic_norm_square_p1: Option<f64>,
    orthohombic_norm_square_p2: Option<f64>,
    orthohombic_norm_square_p3: Option<f64>,
    tetragonal_norm_square_p1: Option<f64>,
    tetragonal_norm_square_p2: Option<f64>,
    tetragonal_norm_square_p3: Option<f64>,
    hexagonal_norm_square_p1: Option<f64>,
    hexagonal_norm_square_p2: Option<f64>,
    hexagonal_norm_square_p3: Option<f64>,
    isotropic_norm_square: Option<f64>,
}

#[derive(Deserialize)]
enum Mineral {
    Olivine,
    Enstatite,
}

#[derive(Deserialize)]
struct PoleFiguresConfiguration {
    times: Vec<f64>,
    particle_ids: Vec<usize>,
    axes: Vec<LaticeAxes>,
    minerals: Vec<Mineral>,
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

struct PoleFigure {
    mineral: Mineral,
    latice_axis: LaticeAxes,
    counts: Array2<f64>,
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
        Err(why) => panic!("couldn't open {}: {}", config_file_display, why.to_string()),
        Ok(file) => file,
    };
    // Read the file contents into a string, returns `io::Result<usize>`
    let mut config_file_string = String::new();
    match file.read_to_string(&mut config_file_string) {
        Err(why) => panic!("couldn't read {}: {}", config_file_display, why.to_string()),
        Ok(_) => (),
    }

    let config: Config = toml::from_str(&config_file_string).unwrap();

    println!(
        "particle ids size {}",
        config.pole_figures.particle_ids.len()
    );

    let base_dir = config.base_dir.clone();

    // start the experiments
    let experiment_dirs = config.experiment_dirs.clone();
    experiment_dirs.par_iter().for_each(|experiment_dir| {
        //for experiment_dir in config.experiment_dirs {
        println!("Processing experiment {}", experiment_dir);

        let lpo_dir = base_dir.clone() + &experiment_dir;

        // get a vector with the time for all the timesteps
        let statistics_file = lpo_dir.to_owned() + "statistics";

        println!("  file:{}", statistics_file);
        let file = File::open(statistics_file).unwrap();
        let reader = BufReader::new(file);

        let mut data: String = "".to_string();
        for line in reader.lines() {
            let line = line.unwrap();
            let line = line.trim();
            let mut line = line.replace("  ", " ");
            while let Some(_) = line.find("  ") {
                line = line.replace("  ", " ");
            }

            if line.find('#') != Some(0) {
                if line.find("particle_LPO") != None {
                    //data.push_str(line);
                    data = data + &line + "\n";
                }
            }
        }

        //println!("{}",data);

        let mut timestep_to_time: Vec<f64> = vec![];
        let mut rdr = csv::ReaderBuilder::new()
            .trim(csv::Trim::All)
            .delimiter(b' ')
            .comment(Some(b'#'))
            .has_headers(false)
            .from_reader(data.as_bytes()); //csv::Reader::from_reader(file);
        for result in rdr.records() {
            // The iterator yields Result<StringRecord, Error>, so we check the
            // error here..
            let record = result.unwrap().clone();
            //println!("{:?}", record);
            //println!("{:?}", record.get(1));
            let time = record.get(1).clone();
            match time {
                Some(time) => timestep_to_time.push(time.to_string().parse::<f64>().unwrap()),
                None => assert!(false, "Time not found"),
            }
        }
        //std::process::exit(0);

        for output_time in &config.pole_figures.times {
            // find closest value in timestep_to_time
            // assume it always starts a zero
            //let before_time = timestep_to_time.iter().position(|x| x < &output_time);
            let after_time = timestep_to_time.iter().position(|x| x > &output_time);

            let after_timestep = match after_time {
                Some(timestep) => timestep,
                None => timestep_to_time.len() - 1,
            };

            // todo: oneline
            let mut before_timestep = after_timestep;
            if after_timestep > 0 {
                before_timestep = after_timestep - 1
            }

            // check wheter before_timestep or after_timestep is closer to output_time,
            // then use that one.
            let before_timestep_diff = (output_time - timestep_to_time[before_timestep]).abs();
            let after_timestep_diff = (output_time - timestep_to_time[after_timestep]).abs();

            let time_step = if before_timestep_diff < after_timestep_diff {
                before_timestep as u64
            } else {
                after_timestep as u64
            };

            let time = timestep_to_time[time_step as usize];

            println!(
                "flag 1 {}, {},-- {}, {}, -- time_step = {}, time = {}",
                before_timestep,
                after_timestep,
                timestep_to_time[before_timestep],
                timestep_to_time[after_timestep],
                time_step,
                time
            );
            //std::process::exit(0);

            let file_prefix = "particle_LPO/weighted_LPO";
            let file_particle_prefix = "particle_LPO/particles";
            let mut rank_id = 0;
            println!(
                "particle ids size {}",
                config.pole_figures.particle_ids.len()
            );
            for particle_id in &config.pole_figures.particle_ids {
                println!("processing particle_id {}", particle_id);
                let mut particle_olivine_a_axis_vectors = Vec::new();
                let mut particle_olivine_b_axis_vectors = Vec::new();
                let mut particle_olivine_c_axis_vectors = Vec::new();
                //let mut particle_enstatite_a_axis_vectors = Vec::new();
                //let mut particle_enstatite_b_axis_vectors = Vec::new();
                //let mut particle_enstatite_c_axis_vectors = Vec::new();

                let mut file_found: bool = false;
                while !file_found {
                    let angles_file = format!(
                        "{}{}-{:05}.{:04}.dat",
                        lpo_dir, file_prefix, time_step, rank_id
                    );
                    let angles_file = Path::new(&angles_file);
                    let output_file = format!(
                        "{}{}_t{:05}.{:05}.png",
                        lpo_dir, file_prefix, time_step, particle_id
                    );
                    let output_file = Path::new(&output_file);
                    let particle_file = format!(
                        "{}{}-{:05}.{:04}.dat",
                        lpo_dir, file_particle_prefix, time_step, rank_id
                    );
                    let particle_info_file = Path::new(&particle_file);

                    println!("  trying file name: {}", angles_file.display());

                    // check wheter file exists, if not it means that is reached the max rank, so stop.
                    if !(fs::metadata(angles_file).is_ok()) {
                        println!(
                            "particle id {} not found for timestep {}.",
                            particle_id, time_step
                        );
                        break;
                    }

                    // check wheter file is empty, if not continue to next rank
                    if fs::metadata(angles_file).unwrap().len() == 0 {
                        rank_id = rank_id + 1;
                        continue;
                    }

                    println!("  file:{}", angles_file.display());
                    let file = File::open(angles_file).unwrap();
                    let metadata = file.metadata().unwrap();

                    let mut buf_reader = BufReader::with_capacity(metadata.len() as usize, file);
                    //let mut buf_reader = BufReader::new(file);

                    let mut decoded_data = Vec::new();

                    let compressed = config.compressed;

                    let decoded_reader = if compressed {
                        let mut decoder = libflate::zlib::Decoder::new(buf_reader).unwrap();
                        decoder.read_to_end(&mut decoded_data).unwrap();
                        String::from_utf8_lossy(&decoded_data)
                    } else {
                        let data = buf_reader.fill_buf().unwrap();
                        String::from_utf8_lossy(&data)
                    };

                    let mut rdr = csv::ReaderBuilder::new()
                        .has_headers(true)
                        .delimiter(b' ')
                        .from_reader(decoded_reader.as_bytes());

                    for result in rdr.deserialize() {
                        let record: Record = result.unwrap();
                        if record.id == *particle_id {
                            let deg_to_rad = std::f64::consts::PI / 180.;
                            let dcm = dir_cos_matrix2(
                                record.olivine_euler_angles_phi.unwrap() * deg_to_rad,
                                record.olivine_euler_angles_theta.unwrap() * deg_to_rad,
                                record.olivine_euler_angles_z.unwrap() * deg_to_rad,
                            )
                            .unwrap();

                            particle_olivine_a_axis_vectors.push(dcm.row(0).to_owned());
                            particle_olivine_b_axis_vectors.push(dcm.row(1).to_owned());
                            particle_olivine_c_axis_vectors.push(dcm.row(2).to_owned());
                        }
                    }

                    // check if the particle id was found in this file, otherwise continue
                    if particle_olivine_a_axis_vectors.len() == 0 {
                        rank_id = rank_id + 1;
                        continue;
                    }
                    file_found = true;

                    // retrieve anisotropy info
                    let mut particle_record = ParticleRecord {
                        id: 0,
                        x: 0.0,
                        y: 0.0,
                        z: Some(0.0),
                        olivine_deformation_type: 0.0,
                        full_norm_square: None,
                        triclinic_norm_square_p1: None,
                        triclinic_norm_square_p2: None,
                        triclinic_norm_square_p3: None,
                        monoclinic_norm_square_p1: None,
                        monoclinic_norm_square_p2: None,
                        monoclinic_norm_square_p3: None,
                        orthohombic_norm_square_p1: None,
                        orthohombic_norm_square_p2: None,
                        orthohombic_norm_square_p3: None,
                        tetragonal_norm_square_p1: None,
                        tetragonal_norm_square_p2: None,
                        tetragonal_norm_square_p3: None,
                        hexagonal_norm_square_p1: None,
                        hexagonal_norm_square_p2: None,
                        hexagonal_norm_square_p3: None,
                        isotropic_norm_square: None,
                    };

                    let particle_info_file = File::open(particle_info_file).unwrap();
                    let buf_reader = BufReader::new(particle_info_file);

                    let mut rdr = csv::ReaderBuilder::new()
                        .has_headers(true)
                        .delimiter(b' ')
                        .from_reader(buf_reader);

                    for result in rdr.deserialize() {
                        // We must tell Serde what type we want to deserialize into.
                        let record: ParticleRecord = result.unwrap();
                        if record.id == *particle_id {
                            particle_record = record;
                        }
                    }
                    // end retrieve anisotropy info

                    let sphere_points = 151;
                    let n_grains = particle_olivine_a_axis_vectors.len();

                    let mut particle_olivine_a_axis_arrays = Array2::zeros((n_grains, 3)); // Pa
                    let mut particle_olivine_b_axis_arrays = Array2::zeros((n_grains, 3)); // Pb
                    let mut particle_olivine_c_axis_arrays = Array2::zeros((n_grains, 3)); // Pc
                    for i in 0..n_grains {
                        for j in 0..3 {
                            particle_olivine_a_axis_arrays[[i, j]] =
                                particle_olivine_a_axis_vectors[i][j];
                            particle_olivine_b_axis_arrays[[i, j]] =
                                particle_olivine_b_axis_vectors[i][j];
                            particle_olivine_c_axis_arrays[[i, j]] =
                                particle_olivine_c_axis_vectors[i][j];
                        }
                    }

                    let lambert =
                        create_lambert_equal_area_gridpoint(sphere_points, "upper".to_string())
                            .unwrap();
                    let mut sphere_point_grid = Array2::zeros((3, sphere_points * sphere_points));

                    for i in 0..sphere_points {
                        for j in 0..sphere_points {
                            sphere_point_grid[[0, i * sphere_points + j]] = lambert.x[[i, j]];
                            sphere_point_grid[[1, i * sphere_points + j]] = lambert.y[[i, j]];
                            sphere_point_grid[[2, i * sphere_points + j]] = lambert.z[[i, j]];
                        }
                    }

                    let counts_olivine_a_axis = gaussian_orientation_counts(
                        &particle_olivine_a_axis_arrays,
                        &sphere_point_grid,
                        sphere_points,
                    )
                    .unwrap();
                    let counts_olivine_b_axis = gaussian_orientation_counts(
                        &particle_olivine_b_axis_arrays,
                        &sphere_point_grid,
                        sphere_points,
                    )
                    .unwrap();
                    let counts_olivine_c_axis = gaussian_orientation_counts(
                        &particle_olivine_c_axis_arrays,
                        &sphere_point_grid,
                        sphere_points,
                    )
                    .unwrap();
                    println!(
                        "  Before make_polefigures: Elapsed time: {:.2?}",
                        before.elapsed()
                    );

                    make_polefigures(
                        n_grains,
                        &time_step,
                        0,
                        &counts_olivine_a_axis,
                        &counts_olivine_b_axis,
                        &counts_olivine_c_axis,
                        &lambert,
                        output_file,
                        &particle_record,
                        time,
                    )
                    .unwrap();

                    println!(
                        "  After make_polefigures: Elapsed time: {:.2?}",
                        before.elapsed()
                    );
                }
                println!("go to next id");
            }
        }
    }); //.collect();//}
    Ok(())
}

fn dir_cos_matrix2(
    phi1: f64,
    theta: f64,
    phi2: f64,
) -> Result<Array2<f64>, Box<dyn std::error::Error>> {
    let mut dcm: Array2<f64> = Array::zeros((3, 3));

    dcm[[0, 0]] = phi2.cos() * phi1.cos() - theta.cos() * phi1.sin() * phi2.sin();
    dcm[[0, 1]] = -phi2.cos() * phi1.sin() - theta.cos() * phi1.cos() * phi2.sin();
    dcm[[0, 2]] = -phi2.sin() * theta.sin();

    dcm[[1, 0]] = phi2.sin() * phi1.cos() + theta.cos() * phi1.sin() * phi2.cos();
    dcm[[1, 1]] = -phi2.sin() * phi1.sin() + theta.cos() * phi1.cos() * phi2.cos();
    dcm[[1, 2]] = phi2.cos() * theta.sin();

    dcm[[2, 0]] = -theta.sin() * phi1.sin();
    dcm[[2, 1]] = -theta.sin() * phi1.cos();
    dcm[[2, 2]] = theta.cos();

    Ok(dcm)
}

// Following method in Robin and Jowett, Tectonophysics, 1986
// Computerized contouring and statistical evaluation of orientation data
// using contouring circles and continuous weighting functions

fn gaussian_orientation_counts(
    particles: &Array2<f64>,
    sphere_point_grid: &Array2<f64>,
    sphere_points: usize,
) -> Result<Array2<f64>, Box<dyn std::error::Error>> {
    let npts = particles.shape()[0];

    // Choose k, which defines width of spherical gaussian  (table 3)
    let k = 2. * (1. + npts as f64 / 9.);

    // Given k, calculate standard deviation (eq 13b)
    let std_dev = ((npts as f64 * (k as f64 / 2. - 1.) / (k * k)) as f64).sqrt();

    // Calculate dot product
    let mut cosalpha = particles.dot(sphere_point_grid);

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
    _time_step: &u64,
    particle_id: u64,
    counts_a: &Array2<f64>,
    counts_b: &Array2<f64>,
    counts_c: &Array2<f64>,
    lambert: &Lambert,
    output_file: &Path,
    particle_record: &ParticleRecord,
    time: f64,
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
    counts.push(counts_a);
    counts.push(counts_b);
    counts.push(counts_c);
    // Grid of points is a square and it extends outside the pole figure circumference.
    // Create mask to only plot color and contours within the pole figure
    let mut mask = lambert.x_plane.clone();

    let mut circle_path: Vec<(f64, f64)> = Vec::new();
    Zip::from(&mut mask)
        .and(&lambert.x_plane)
        .and(&lambert.z_plane)
        .par_apply(|a, x, z| {
            let radius = (x * x + z * z).sqrt();
            if radius >= lambert.r_plane + 0.001 {
                *a = std::f64::NAN
            } else {
                *a = 1.
            }
        });
    let npts = counts_a.shape()[0];

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
    if counts_a.len() > 1 {
        have_aplot = 1;
    }
    if counts_b.len() > 1 {
        have_bplot = 1;
    }
    if counts_c.len() > 1 {
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
    let root = BitMapBackend::new(&path_string, (figure_width + 10, figure_height + 150))
        .into_drawing_area();
    root.fill(&WHITE)?;

    println!("    made root: Elapsed time: {:.2?}", before.elapsed());
    let (upper, lower) = root.split_vertically(150);

    // Do stuff in upper

    // preprocessing particle data:
    let pr = particle_record;
    let full_norm_square = particle_record.full_norm_square.unwrap();
    let isotropic = pr.isotropic_norm_square.unwrap();

    let tric_unsorted = [
        pr.triclinic_norm_square_p1.unwrap(),
        pr.triclinic_norm_square_p2.unwrap(),
        pr.triclinic_norm_square_p3.unwrap(),
    ];
    let mono_unsorted = [
        pr.monoclinic_norm_square_p1.unwrap(),
        pr.monoclinic_norm_square_p2.unwrap(),
        pr.monoclinic_norm_square_p3.unwrap(),
    ];
    let orth_unsorted = [
        pr.orthohombic_norm_square_p1.unwrap(),
        pr.orthohombic_norm_square_p2.unwrap(),
        pr.orthohombic_norm_square_p3.unwrap(),
    ];
    let tetr_unsorted = [
        pr.tetragonal_norm_square_p1.unwrap(),
        pr.tetragonal_norm_square_p2.unwrap(),
        pr.tetragonal_norm_square_p3.unwrap(),
    ];
    let hexa_unsorted = [
        pr.hexagonal_norm_square_p1.unwrap(),
        pr.hexagonal_norm_square_p2.unwrap(),
        pr.hexagonal_norm_square_p3.unwrap(),
    ];

    let mut tric_sorted = tric_unsorted.clone();
    let mut mono_sorted = mono_unsorted.clone();
    let mut orth_sorted = orth_unsorted.clone();
    let mut tetr_sorted = tetr_unsorted.clone();
    let mut hexa_sorted = hexa_unsorted.clone();

    let total_anisotropy =
        tric_sorted[0] + mono_sorted[0] + orth_sorted[0] + tetr_sorted[0] + hexa_sorted[0];

    tric_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    mono_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    orth_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    tetr_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    hexa_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let tric_perc_full = tric_unsorted
        .iter()
        .map(|v| (v / full_norm_square) * 100.)
        .collect::<Vec<f64>>();
    let mono_perc_full = mono_unsorted
        .iter()
        .map(|v| (v / full_norm_square) * 100.)
        .collect::<Vec<f64>>(); //.collect();
    let orth_perc_full = orth_unsorted
        .iter()
        .map(|v| (v / full_norm_square) * 100.)
        .collect::<Vec<f64>>(); //.collect();
    let tetr_perc_full = tetr_unsorted
        .iter()
        .map(|v| (v / full_norm_square) * 100.)
        .collect::<Vec<f64>>(); //.collect();
    let hexa_perc_full = hexa_unsorted
        .iter()
        .map(|v| (v / full_norm_square) * 100.)
        .collect::<Vec<f64>>(); //.collect();

    let hp = Percentage {
        total: figure_height as f64,
    }; //(100/figure_height) as i32;
    let wp = Percentage {
        total: figure_width as f64 / number_of_figures as f64,
    };

    let font_size = 35;
    let line_distance = 5.5; //6.;//5.5;
    let top_margin = 0.25;
    let left_margin = 0.5;
    //let halfway_margin = 52.5; //.;
    let font_type = "Inconsolata"; //"consolas";//"sans-serif"

    upper
        .draw(&Text::new(
            format!("id={},time={:.5e}, position=({:.3e}:{:.3e}:{:.3e}), ODT={:.4}, grains={}, anisotropic%={:.4}",particle_id,time,particle_record.x,particle_record.y,particle_record.z.unwrap(),particle_record.olivine_deformation_type, n_grains,((total_anisotropy)/full_norm_square)*100.),
            ((wp.calc(left_margin) ) as i32, hp.calc(top_margin) as i32),
            (font_type, font_size).into_font(),
        ))?;
    upper.draw(&Text::new(
        format!(
            "hex%={:.2},{:.2},{:.2}",
            hexa_perc_full[0], hexa_perc_full[1], hexa_perc_full[2]
        ),
        (
            (left_margin) as i32,
            hp.calc(top_margin + 1.0 * line_distance) as i32,
        ),
        (font_type, font_size).into_font(),
    ))?;
    upper.draw(&Text::new(
        format!(
            "h/a%={:.2},{:.2},{:.2}",
            ((hexa_unsorted[0]) / (total_anisotropy)) * 100.,
            ((hexa_unsorted[1]) / (total_anisotropy)) * 100.,
            ((hexa_unsorted[2]) / (total_anisotropy)) * 100.
        ),
        (
            (left_margin) as i32,
            hp.calc(top_margin + 2.0 * line_distance) as i32,
        ),
        (font_type, font_size).into_font(),
    ))?;

    upper.draw(&Text::new(
        format!(
            "tet%={:.2},{:.2},{:.2}",
            tetr_perc_full[0], tetr_perc_full[1], tetr_perc_full[2]
        ),
        (
            (left_margin + (figure_width as f64) * 0.2) as i32,
            hp.calc(top_margin + 1.0 * line_distance) as i32,
        ),
        (font_type, font_size).into_font(),
    ))?;
    upper.draw(&Text::new(
        format!(
            "t/a%={:.2},{:.2},{:.2}",
            ((tetr_unsorted[0]) / (total_anisotropy)) * 100.,
            ((tetr_unsorted[1]) / (total_anisotropy)) * 100.,
            ((tetr_unsorted[2]) / (total_anisotropy)) * 100.
        ),
        (
            (left_margin + (figure_width as f64) * 0.2) as i32,
            hp.calc(top_margin + 2.0 * line_distance) as i32,
        ),
        (font_type, font_size).into_font(),
    ))?;
    upper.draw(&Text::new(
        format!(
            "ort%={:.2},{:.2},{:.2}",
            orth_perc_full[0], orth_perc_full[1], orth_perc_full[2]
        ),
        (
            (left_margin + (figure_width as f64) * 0.4) as i32,
            hp.calc(top_margin + 1.0 * line_distance) as i32,
        ),
        (font_type, font_size).into_font(),
    ))?;
    upper.draw(&Text::new(
        format!(
            "o/a%={:.2},{:.2},{:.2}",
            ((orth_unsorted[0]) / (total_anisotropy)) * 100.,
            ((orth_unsorted[1]) / (full_norm_square - isotropic)) * 100.,
            ((orth_unsorted[2]) / (full_norm_square - isotropic)) * 100.
        ),
        (
            (left_margin + (figure_width as f64) * 0.4) as i32,
            hp.calc(top_margin + 2.0 * line_distance) as i32,
        ),
        (font_type, font_size).into_font(),
    ))?;
    upper.draw(&Text::new(
        format!(
            "mon%={:.2},{:.2},{:.2}",
            mono_perc_full[0], mono_perc_full[1], mono_perc_full[2]
        ),
        (
            (left_margin + (figure_width as f64) * 0.6) as i32,
            hp.calc(top_margin + 1.0 * line_distance) as i32,
        ),
        (font_type, font_size).into_font(),
    ))?;
    upper.draw(&Text::new(
        format!(
            "m/a%={:.2},{:.2},{:.2}",
            ((mono_unsorted[0]) / (total_anisotropy)) * 100.,
            ((mono_unsorted[1]) / (full_norm_square - isotropic)) * 100.,
            ((mono_unsorted[2]) / (full_norm_square - isotropic)) * 100.
        ),
        (
            (left_margin + (figure_width as f64) * 0.6) as i32,
            hp.calc(top_margin + 2.0 * line_distance) as i32,
        ),
        (font_type, font_size).into_font(),
    ))?;
    upper.draw(&Text::new(
        format!(
            "tri%={:.2},{:.2},{:.2}",
            tric_perc_full[0], tric_perc_full[1], tric_perc_full[2]
        ),
        (
            (left_margin + (figure_width as f64) * 0.8) as i32,
            hp.calc(top_margin + 1.0 * line_distance) as i32,
        ),
        (font_type, font_size).into_font(),
    ))?;
    upper.draw(&Text::new(
        format!(
            "t/a%={:.2},{:.2},{:.2}",
            ((tric_unsorted[0]) / (total_anisotropy)) * 100.,
            ((tric_unsorted[1]) / (full_norm_square - isotropic)) * 100.,
            ((tric_unsorted[2]) / (full_norm_square - isotropic)) * 100.
        ),
        (
            (left_margin + (figure_width as f64) * 0.8) as i32,
            hp.calc(top_margin + 2.0 * line_distance) as i32,
        ),
        (font_type, font_size).into_font(),
    ))?;

    // do stuff in lower:

    let font_size = 45;
    let (left, _right) = lower.split_horizontally(figure_width);
    let drawing_areas = left.split_evenly((1, number_of_figures));

    println!("    number of figures = {}", number_of_figures);
    for figure_number in 0..number_of_figures {
        let mut chart = ChartBuilder::on(&drawing_areas[figure_number]).build_ranged(
            -lambert.r_plane - 0.05..lambert.r_plane + 0.15,
            -lambert.r_plane - 0.05..lambert.r_plane + 0.15,
        )?;

        // get max valuein counts_a
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
                    (lambert.x_plane[[i + 1, j]], lambert.z_plane[[i + 1, j]]),
                    (
                        lambert.x_plane[[i + 1, j + 1]],
                        lambert.z_plane[[i + 1, j + 1]],
                    ),
                    (lambert.x_plane[[i, j + 1]], lambert.z_plane[[i, j + 1]]),
                    (lambert.x_plane[[i, j]], lambert.z_plane[[i, j]]),
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

        println!(
            "      before 2st drawing: Elapsed time: {:.2?}",
            before.elapsed()
        );

        match figure_number {
            0 => {
                //let x = -100e3;
                //let y = -100.125e3;
                drawing_areas[figure_number]
                    .draw(&Text::new(
                        format!("a-axis"),
                        (
                            wp.calc(left_margin) as i32,
                            hp.calc(top_margin + 1.0 * line_distance) as i32,
                        ),
                        (font_type, font_size, FontStyle::Bold).into_font(),
                    ))
                    .unwrap();
                drawing_areas[figure_number].draw(&Text::new(
                    format!("max={:.3}", max_count_value),
                    (
                        wp.calc(left_margin) as i32,
                        hp.calc(top_margin + 0.0 * line_distance) as i32,
                    ),
                    (font_type, font_size, FontStyle::Bold).into_font(),
                ))?;
                /*drawing_areas[figure_number]
                .draw(&Text::new(
                    format!("t={:.5}",time),
                    ((wp.calc(left_margin+halfway_margin) ) as i32, hp.calc(top_margin + 1.0 * line_distance) as i32),
                    (font_type, font_size).into_font(),
                ))
                .unwrap();*/
                /*drawing_areas[figure_number].draw(&Text::new(
                    format!("W={:.5}", particle_record.water),
                    (
                        wp.calc(left_margin+halfway_margin) as i32,
                        hp.calc(top_margin + 1. * line_distance) as i32,
                    ),
                    (font_type, font_size).into_font(),
                ))?;
                drawing_areas[figure_number].draw(&Text::new(
                    format!("G={}", n_grains),
                    (
                        wp.calc(left_margin+halfway_margin) as i32,
                        hp.calc(top_margin + 2. * line_distance) as i32,
                    ),
                    (font_type, font_size).into_font(),
                ))?;*/
                drawing_areas[figure_number].draw(&Text::new(
                    format!("Z"),
                    (wp.calc(46.4) as i32, (hp.calc(11.) - 85.) as i32),
                    (font_type, font_size).into_font(),
                ))?;
                drawing_areas[figure_number].draw(&Text::new(
                    format!("X"),
                    (wp.calc(96.0) as i32, (hp.calc(61.75) - 125.) as i32),
                    (font_type, font_size).into_font(),
                ))?;
            }
            1 => {
                drawing_areas[figure_number].draw(&Text::new(
                    format!("b-axis"),
                    (
                        wp.calc(left_margin) as i32,
                        hp.calc(top_margin + 1.0 * line_distance) as i32,
                    ),
                    (font_type, font_size, FontStyle::Bold).into_font(),
                ))?;
                drawing_areas[figure_number].draw(&Text::new(
                    format!("max={:.3}", max_count_value),
                    (
                        wp.calc(left_margin) as i32,
                        hp.calc(top_margin + 0.0 * line_distance) as i32,
                    ),
                    (font_type, font_size, FontStyle::Bold).into_font(),
                ))?;
                drawing_areas[figure_number].draw(&Text::new(
                    format!("Z"),
                    (wp.calc(46.4) as i32, (hp.calc(11.) - 85.) as i32),
                    (font_type, font_size).into_font(),
                ))?;
                drawing_areas[figure_number].draw(&Text::new(
                    format!("X"),
                    (wp.calc(96.0) as i32, (hp.calc(61.75) - 125.) as i32),
                    (font_type, font_size).into_font(),
                ))?;
            }
            2 => {
                drawing_areas[figure_number].draw(&Text::new(
                    format!("c-axis"),
                    (
                        wp.calc(left_margin) as i32,
                        hp.calc(top_margin + 1. * line_distance) as i32,
                    ),
                    (font_type, font_size, FontStyle::Bold).into_font(),
                ))?;
                drawing_areas[figure_number].draw(&Text::new(
                    format!("max={:.3}", max_count_value),
                    (
                        wp.calc(left_margin) as i32,
                        hp.calc(top_margin + 0.0 * line_distance) as i32,
                    ),
                    (font_type, font_size, FontStyle::Bold).into_font(),
                ))?;
                //drawing_areas[figure_number]
                //.draw(&Text::new(
                //    format!("ani%={:.5}",((full_norm_square-isotropic)/full_norm_square)*100.),
                //    ((wp.calc(left_margin+halfway_margin) ) as i32, hp.calc(top_margin + 0.0 * line_distance) as i32),
                //    (font_type, font_size).into_font(),
                //))?;
                //drawing_areas[figure_number]
                //.draw(&Text::new(
                //    format!("hex%={:.3}:{:.3}",orth_perc_full[0],orth_perc_full[2]),
                //    ((wp.calc(left_margin+halfway_margin) ) as i32, hp.calc(top_margin + 1.0 * line_distance) as i32),
                //    (font_type, font_size).into_font(),
                //))?;
                //drawing_areas[figure_number]
                //.draw(&Text::new(
                //    format!("h/a%={:.3}:{:.3}",((orth_sorted[0])/(total_anisotropy))*100.,((orth_sorted[2])/(full_norm_square-isotropic))*100.),
                //    ((wp.calc(left_margin+halfway_margin) ) as i32, hp.calc(top_margin + 2.0 * line_distance) as i32),
                //    (font_type, font_size).into_font(),
                //))?;
                drawing_areas[figure_number].draw(&Text::new(
                    format!("Z"),
                    (wp.calc(46.4) as i32, (hp.calc(11.) - 85.) as i32),
                    (font_type, font_size).into_font(),
                ))?;
                drawing_areas[figure_number].draw(&Text::new(
                    format!("X"),
                    (wp.calc(96.0) as i32, (hp.calc(61.75) - 125.) as i32),
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
    x_plane: &Array<f64, Dim<[usize; 1]>>,
    y_plane: &Array<f64, Dim<[usize; 1]>>,
) -> Result<(Array<f64, Dim<[usize; 2]>>, Array<f64, Dim<[usize; 2]>>), Box<dyn std::error::Error>>
{
    let mut new_x: Array<f64, Dim<[usize; 2]>> = Array::zeros([x_plane.len(), y_plane.len()]);
    let mut new_y: Array<f64, Dim<[usize; 2]>> = Array::zeros([x_plane.len(), y_plane.len()]);
    let mut counter = 0;
    let max_count = x_plane.len();
    for value in new_x.iter_mut() {
        *value = x_plane[counter];
        counter += 1;
        if counter >= max_count {
            counter = 0;
        }
    }

    counter = 0;
    let mut counter_y = 0;
    for value in new_y.iter_mut() {
        *value = y_plane[counter];
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
fn create_lambert_equal_area_gridpoint(
    sphere_points: usize,
    hemisphere: String,
) -> Result<Lambert, Box<dyn std::error::Error>> {
    // Create a grid of points at increasing radius in the X and Y direction
    // Use the coordinate X,Y,R to plot these points on the lambert projection
    let r_plane: f64 = 2.0_f64.sqrt(); // need this to get full sphere in Lambert projection)
    let x_plane = Array::linspace(-r_plane, r_plane, sphere_points);
    //println!("x_plane={}", x_plane);
    let y_plane = x_plane.clone();
    let (x_plane, mut z_plane) = create_meshgrid(&x_plane, &y_plane)?;
    //println!("x_plane={}", x_plane);
    //println!("z_plane={}", z_plane);
    z_plane.invert_axis(Axis(0));
    //println!("z_plane={}", z_plane);
    //let mut y_plane = y_plane.into_raw_vec();
    //y_plane.reverse();
    //let y_plane: Array<f64, Dim<[usize; 2]>> = Array::from(y_plane); // y_plane increases up

    // map onto lambert projection, assumes r = 1?
    // added np.abs to avoid tiny negative numbers in sqrt
    // todo, turn into enum and matchs
    let mut x = Array::zeros([sphere_points, sphere_points]);
    let mut y = Array::zeros([sphere_points, sphere_points]);
    let mut z = Array::zeros([sphere_points, sphere_points]);
    let mut mag = Array::zeros([sphere_points, sphere_points]);

    Zip::from(&mut x)
        .and(&x_plane)
        .and(&z_plane)
        .par_apply(|a, &x_plane, &z_plane| {
            *a = if 1. - (x_plane * x_plane + z_plane * z_plane) / 4. > std::f64::EPSILON {
                ((1. - (x_plane * x_plane + z_plane * z_plane) / 4.).abs()).sqrt() * x_plane
            } else {
                0.
            };
        });

    if hemisphere == "lower" {
        Zip::from(&mut y)
            .and(&x_plane)
            .and(&z_plane)
            .par_apply(|a, &x_plane, &z_plane| {
                *a = -(1. - (x_plane * x_plane + z_plane * z_plane) / 2.);
            });
        Zip::from(&mut z)
            .and(&x_plane)
            .and(&z_plane)
            .par_apply(|a, &x_plane, &z_plane| {
                *a =
                    ((1. - (x_plane * x_plane + z_plane * z_plane) / 4.).abs()).sqrt() * (-z_plane);
            });
    } else if hemisphere == "upper" {
        Zip::from(&mut y)
            .and(&x_plane)
            .and(&z_plane)
            .par_apply(|a, &x_plane, &z_plane| {
                *a = 1. - (x_plane * x_plane + z_plane * z_plane) / 2.;
            });
        Zip::from(&mut z)
            .and(&x_plane)
            .and(&z_plane)
            .par_apply(|a, &x_plane, &z_plane| {
                *a = ((1. - (x_plane * x_plane + z_plane * z_plane) / 4.).abs()).sqrt() * z_plane;
            });
    };

    /*if hemisphere == "lower" {
        Zip::from(&mut z).and(&x_plane).and(&y_plane).apply(|a, &x_plane, &y_plane| {
            *a = -(1. - (x_plane * x_plane + y_plane * x_plane) / 2.);
        });
    } else if hemisphere == "upper" {
        Zip::from(&mut z).and(&x_plane).and(&y_plane).apply(|a, &x_plane, &y_plane| {
            *a = 1. - (x_plane * x_plane + y_plane * y_plane) / 2.;
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

    //x[1-(x_plane**2 + y_plane**2)/4 < np.finfo(float).eps] = 0;
    //y[1-(x_plane**2 + y_plane**2)/4 < np.finfo(float).eps] = 0;

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
        x_plane: x_plane,
        z_plane: z_plane,
        r_plane: r_plane,
        x: x,
        y: y,
        z: z,
    })
}
