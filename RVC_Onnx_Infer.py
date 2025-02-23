import os
import glob
import soundfile
from huggingface_hub import hf_hub_download

from infer.lib.infer_pack.onnx_inference import OnnxRVC


# PATHS 
root_path = os.path.dirname(os.path.abspath(__file__))  # setting the root dir for RVCp
input_path = os.path.join(root_path, "input")  # audio input
output_path = os.path.join(root_path, "output")  # inference output
onnx_models = os.path.join(root_path, "models") # ONNX models folder


# INFERENCE CONFIGURATION
sampling_rate = 40000  # Your model's sample rate;   32000, 40000, 48000
f0_up_key = 6  # transpose; in semitones either up or down.
f0_method = "dio"  # F0 pitch estimation method.  ( For now only dio works properly. PM is fixed and works, but Dio is better. Harvest is broken )
hop_size = 512 # hop size for inference. ( Currently, applies only to dio F0 ) Try: 32, 64, 128, 256, 512  or custom of your choice
sid = 0 # Speaker ID, unusable atm.

vec_path = "vec-768-layer-12.onnx"

if not os.path.exists(os.path.join(os.getcwd(), vec_path)):
        hf_hub_download(repo_id="NaruseMioShirakana/MoeSS-SUBModel", filename=vec_path, local_dir=os.getcwd(), token=False)
  # pretrained ONNX variant of vec

# DEVICE SETTINGS
device = "dml"  # options: dml, cuda, cpu

# Set your model's name                       / Here / 
model_path = os.path.join("onnx_models", "model1.onnx")  # Your .ONNX model
output_folder = "output"  # Output folder for inferences
output_filename = "infer_output_merged.wav"  # name for inference outputs

# Search for your .wav files in the input dir.
wav_files = [os.path.join(input_path, f) for f in os.listdir(input_path) if f.lower().endswith('.wav')]
if not wav_files:
    raise FileNotFoundError("No WAV files found in the 'input' dir.")
wav_path = wav_files[0] # input for inference ( First found .wav from input dir )
out_path = os.path.join("output", output_filename) # ( Inference output lands into output dir )

model = OnnxRVC(
    model_path,
    vec_path=vec_path,
    sr=sampling_rate,
    hop_size=hop_size,
    device=device
    )

audio = model.inference(wav_path, sid, f0_method=f0_method, f0_up_key=f0_up_key)
audio = model.inference(wav_path, sid, f0_method=f0_method, f0_up_key=f0_up_key)

try:
    soundfile.write(out_path, audio, sampling_rate)
    print("  INFERENCE SUCCESSFUL! CHECK 'output' FOLDER! ")
except Exception as e:
    print(f" AN ERROR OCCURRED: {e}")
