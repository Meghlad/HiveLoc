//! perception_node: ONNX vision detector as a ROS 2 node.
//!
//! One instance per vehicle. Subscribes to that vehicle's grayscale camera
//! stream (sensor_msgs/Image, mono8, 320x240), runs the ONNX blob detector, and
//! publishes world-frame BearingObservation messages. The detector→bearing math
//! is the shared `swarm_perception` crate — identical to the file-driven CLI.
//!
//!   /camera/image_raw (sensor_msgs/Image) ─▶ [perception_node] ─▶ /bearings (BearingObservation)
//!
//! Node parameters (per vehicle): observer id, camera intrinsics (fx, cx),
//! current heading, model path, detector threshold/NMS radius. Heading would in
//! a real system be driven by a compass/IMU topic; kept a parameter here to keep
//! the node self-contained.

use std::sync::{Arc, Mutex};

use rclrs::*;
use swarm_perception::{centroid, column_to_world_bearing, find_peaks};

const W: usize = 320;
const H: usize = 240;

macro_rules! ort_try {
    ($e:expr) => {
        $e.map_err(|e| anyhow::anyhow!("ort: {e}"))?
    };
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let context = Context::default_from_env()?;
    let mut executor = context.create_basic_executor();
    let node = executor.create_node("swarm_perception")?;

    // ---- per-vehicle parameters (rclrs builder: declare -> default -> mandatory) ----
    let observer = node.declare_parameter("observer").default(0i64).mandatory()?;
    let fx = node.declare_parameter("fx").default(160.0f64).mandatory()?;
    let cx = node.declare_parameter("cx").default(160.0f64).mandatory()?;
    let heading = node.declare_parameter("heading").default(0.0f64).mandatory()?;
    let thresh = node.declare_parameter("thresh").default(0.05f64).mandatory()?;
    let nms = node.declare_parameter("nms_radius").default(6.0f64).mandatory()?;
    let model_path = node
        .declare_parameter::<Arc<str>>("model_path")
        .default(Arc::from("detector.onnx"))
        .mandatory()?;

    // Snapshot the parameter values into plain Copy locals so the subscription
    // closure captures values, not the parameter handles.
    let observer_id = observer.get() as u32;
    let (fx_v, cx_v, heading_v) = (fx.get(), cx.get(), heading.get());
    let (thresh_v, nms_v) = (thresh.get() as f32, nms.get() as f32);

    // ---- ONNX session (loaded once, reused per frame) ----
    let session = Arc::new(Mutex::new(ort_try!(ort_try!(ort_try!(
        ort::session::Session::builder()
    )
    .with_intra_threads(1))
    .commit_from_file(&*model_path.get()))));

    let bearing_pub = node.create_publisher::<swarm_msgs::msg::BearingObservation>("bearings")?;

    let session_cb = Arc::clone(&session);
    let logger = node.logger().clone();
    let _img_sub = node.create_subscription::<sensor_msgs::msg::Image, _>(
        "camera/image_raw",
        move |img: sensor_msgs::msg::Image| {
            if img.width as usize != W || img.height as usize != H {
                log_warn!(&logger, "unexpected image size {}x{}", img.width, img.height);
                return;
            }
            // mono8 → normalized f32
            let data: Vec<f32> = img.data.iter().map(|&p| p as f32 / 255.0).collect();

            let run = || -> anyhow::Result<()> {
                let input = ort_try!(ort::value::Tensor::from_array(([1usize, 1, H, W], data)));
                let mut sess = session_cb.lock().unwrap();
                let outputs = ort_try!(sess.run(ort::inputs!["image" => input]));
                let (_, resp) = ort_try!(outputs["response"].try_extract_tensor::<f32>());
                let resp: &[f32] = resp;

                for (px, py, conf) in find_peaks(resp, W, H, thresh_v, nms_v) {
                    let (u, v) = centroid(resp, W, H, px, py);
                    let mut obs = swarm_msgs::msg::BearingObservation::default();
                    obs.observer = observer_id;
                    obs.u = u;
                    obs.v = v;
                    obs.confidence = conf;
                    obs.bearing_world = column_to_world_bearing(u, cx_v, fx_v, heading_v);
                    let _ = bearing_pub.publish(obs);
                }
                Ok(())
            };
            if let Err(e) = run() {
                log_error!(&logger, "inference failed: {e}");
            }
        },
    )?;

    log_info!(
        node.logger(),
        "swarm_perception up (observer {}): /camera/image_raw -> /bearings",
        observer_id
    );
    executor.spin(SpinOptions::default());
    Ok(())
}
