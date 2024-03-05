#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(unused_mut)]
#![allow(dead_code)]

// basics
use std::error::Error;
use std::path::{Path, PathBuf};
// arrays/vectors/tensors
use ndarray::{array, Array, Array1, Array2, Array3, Array4, ArrayBase, ArrayView, IxDynImpl};
use ndarray::{ArrayView2};
use ndarray::{s, Axis, Dim, IxDyn};
use ndarray::{ViewRepr, OwnedRepr};
// images
use image::io::Reader as ImageReader;
use image::{DynamicImage, GenericImageView, ImageBuffer, Rgb, Rgba, RgbImage};
use image::imageops::FilterType;
// use imageproc::drawing::draw_filled_rect_mut;
// use imageproc::rect::Rect;
// machine learning
use ort::{Session, GraphOptimizationLevel};

fn image_to_onnx_input(image: DynamicImage) -> Array4<f32> { 
    let mut img_arr = image.to_rgb8().into_vec();
    let (width, height) = image.dimensions();
    let channels = 3;
    let mut onnx_input = Array::zeros((1, channels, height as _, width as _));
    for (x, y, pixel) in image.into_rgb8().enumerate_pixels() {
        let (r, g, b) = (pixel[0], pixel[1], pixel[2]);
        // Set the RGB values in the array
        onnx_input[[0, 0, y as _, x as _]] = (r as f32) / 255.;
        onnx_input[[0, 1, y as _, x as _]] = (g as f32) / 255.;
        onnx_input[[0, 2, y as _, x as _]] = (b as f32) / 255.;
      };
    onnx_input
    //   x_d = np.array(img_d).transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)/256 // HWC -> NCHW
  }

// fn ndarray_to_dynamic_image(data: ArrayView2<f32>, width: u32, height: u32) -> DynamicImage {
fn ndarray_to_dynamic_image(data: Array<f32, Dim<IxDynImpl>>, width: u32, height: u32) -> DynamicImage {
    // Assuming `data` is of shape (height, width * 3) and holds RGB data
    let mut img_buf: RgbImage = ImageBuffer::new(width, height);

    // for (x, y, pixel) in img_buf.enumerate_pixels_mut() {
    //     let base_index = (y * width + x) as usize * 3; // 3 channels per pixel
    //     *pixel = image::Rgb([
    //         data[(base_index + 0 as usize)],
    //         data[(base_index + 1 as _)],
    //         data[(base_index + 2 as _)],
    //     ]);
    // }

    for y in 0..height {
        for x in 0..width {
            // let base_index = (y * width + x) as usize * 3; // 3 channels per pixel
            let r = (data[[0, 0, y as usize, x as usize]] * 1.0) as u8;
            let g = (data[[0, 0, y as usize, x as usize]] * 1.0) as u8;
            let b = (data[[0, 0, y as usize, x as usize]] * 1.0) as u8;
            img_buf.put_pixel(x, y, image::Rgb([r, g, b]));
        }
    }

    DynamicImage::ImageRgb8(img_buf)
}

// fn maximize_contrast(array: Array2<f32>) {
fn maximize_contrast(array: Array2<f32>) {

}

struct DepthAnything {
    onnx_file: PathBuf,
    model: Session,
}

impl DepthAnything {
    pub fn new(mode: &str) -> Result<Self, Box<dyn Error>> {
        let filename = match mode {
            "relative" => "./assets/depth_anything_small.onnx",
            "metric" => "./assets/depth_anything_small.onnx", // TODO
            _ => return Err("Invalid model".into())
        };
        let onnx_file = PathBuf::from(format!("./assets/{}.onnx", filename));
        let model = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(4)?
            .with_model_from_file(&onnx_file)?;        
        Ok(DepthAnything { onnx_file, model })
    }

    // pub fn process(self, image: DynamicImage) -> Result<(), Box<dyn Error>> {
    // pub fn process(self, image: DynamicImage) -> Result<Array2<f32>, Box<dyn Error>> {
    pub fn process(self, image: DynamicImage) -> Result<DynamicImage, Box<dyn Error>> {
        let (width, height) = image.dimensions();
        let image_array = image_to_onnx_input(image.clone());
        let inputs = ort::inputs!["image" => image_array.view()]?; 
        // 
        let outputs = self.model.run(inputs)?;
        
        // let depth_array = outputs[0].extract_tensor::<f32>()?.view().clone().into_owned();
        // let depth_image = ndarray_to_dynamic_image(depth_array, width, height);
        let pred = outputs["depth"].extract_tensor::<f32>()?.view().clone().into_owned(); // rename; good or bad?
        // println!("{:?}", pred.clone());
        let pred_image = ndarray_to_dynamic_image(pred, width, height); // rename; good or bad?
        println!("{:?}", pred_image.clone());
        
        // let depth_image = pred_image.resize_exact(width, height, FilterType::CatmullRom);
        // println!("{:?}", depth_image);
        // depth_image.save("depth_image.jpg")?;

        Ok( pred_image )
    }
        
}

fn test() {
    let mode: String = "relative".to_string();
    let depth_anything = DepthAnything::new(&mode).unwrap();
    let image = ImageReader::open("./test/test_images/bench.jpeg").unwrap().decode().unwrap();
    let depth_image = depth_anything.process(image).unwrap();
    depth_image.save("depth_image.jpg")?;
}

fn main() {
    test();
}
