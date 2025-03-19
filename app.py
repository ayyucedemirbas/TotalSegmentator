import gradio as gr
import subprocess
import os
import shutil
import uuid
import zipfile
import nibabel as nib
import matplotlib.pyplot as plt

def run_segmentation(uploaded_file, modality):
    job_id = str(uuid.uuid4())
    input_filename = f"input_{job_id}.nii.gz"
    output_folder = f"segmentations_{job_id}"
    
    if isinstance(uploaded_file, str):
        shutil.copy(uploaded_file, input_filename)
    elif hasattr(uploaded_file, "read"):
        with open(input_filename, "wb") as f:
            f.write(uploaded_file.read())
    else:
        return None, None, "Invalid file input."
    
    command = ["TotalSegmentator", "-i", input_filename, "-o", output_folder]
    if modality == "MR":
        command.extend(["--task", "total_mr"])
    
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        error_message = f"Error during segmentation: {e}"
        if os.path.exists(input_filename): os.remove(input_filename)
        if os.path.exists(output_folder): shutil.rmtree(output_folder)
        return None, None, error_message
    
    zip_filename = f"segmentations_{job_id}.zip"
    with zipfile.ZipFile(zip_filename, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(output_folder):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, output_folder)
                zipf.write(file_path, arcname)
    
    seg_files = [os.path.join(output_folder, f) for f in os.listdir(output_folder) if f.endswith('.nii.gz')]
    image_filename = None
    if seg_files:
        seg_file = seg_files[0]
        try:
            seg_img = nib.load(seg_file)
            seg_data = seg_img.get_fdata()
            slice_idx = seg_data.shape[2] // 2
            seg_slice = seg_data[:, :, slice_idx]
            plt.figure(figsize=(6, 6))
            plt.imshow(seg_slice.T, cmap="gray", origin="lower")
            plt.axis('off')
            image_filename = f"segmentation_preview_{job_id}.png"
            plt.savefig(image_filename, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"Error creating preview: {e}")
            image_filename = None

    os.remove(input_filename)
    shutil.rmtree(output_folder)
    
    return zip_filename, image_filename, "Segmentation completed successfully."

with gr.Blocks() as demo:
    gr.Markdown("# TotalSegmentator Gradio App")
    gr.Markdown(
        "Upload a CT or MR image (in NIfTI format) and run segmentation using TotalSegmentator. "
        "For MR images, the task flag is set accordingly. A preview of one segmentation slice will be displayed."
    )
    
    with gr.Row():
        uploaded_file = gr.File(label="Upload NIfTI Image (.nii.gz)")
        modality = gr.Radio(choices=["CT", "MR"], label="Select Image Modality", value="CT")
    
    with gr.Row():
        zip_output = gr.File(label="Download Segmentation Output (zip)")
        preview_output = gr.Image(label="Segmentation Preview")
    
    status_output = gr.Textbox(label="Status", interactive=False)
    
    run_btn = gr.Button("Run Segmentation")
    run_btn.click(fn=run_segmentation, inputs=[uploaded_file, modality], outputs=[zip_output, preview_output, status_output])

demo.launch()
