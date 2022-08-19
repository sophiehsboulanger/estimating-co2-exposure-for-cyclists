import PySimpleGUI as sg
import counter

text_box_size = (17, 1)
input_line = [sg.Text("Input Video:", size=text_box_size), sg.In(size=(25, 1), key="-INPUT-", readonly=True),
              sg.FileBrowse()]
output_location_line = [sg.Text("Output Video Location:", size=text_box_size),
                        sg.In(size=(25, 1), key="-OUTPUT LOCATION-", readonly=True, default_text="C:/Users/sophi/OneDrive - University of East Anglia/Dissertation/dissertaion/outputs"), sg.FolderBrowse()]
output_name_line = [sg.Text("Output File Name:", size=text_box_size), sg.In(size=(25, 1), key="-OUTPUT NAME-")]
tracker_settings_column = [
    [sg.Text("Object Tracker Settings")],
    [sg.Text("Max Age:", size=text_box_size), sg.In(size=(5, 1), key="-MAX AGE-", default_text="5")],
    [sg.Text("Min Age:", size=text_box_size), sg.In(size=(5, 1), key="-MIN AGE-", default_text="5")],
    [sg.Text("Max Overlap:", size=text_box_size), sg.In(size=(5, 1), key="-MAX OVERLAP-", default_text="0.5")],
    [sg.Text("Frame Skip:", size=text_box_size), sg.In(size=(5, 1), key="-FRAME SKIP-", default_text="5")]
]
detector_settings_column = [
    [sg.Text("Object Detector Settings")],
    [sg.Text("Min Confidence:", size=text_box_size), sg.In(size=(5, 1), key="-MIN CONF-", default_text="0.75")],
    [sg.Text("NMS Threshold:", size=text_box_size), sg.In(size=(5, 1), key="-NMS-", default_text="0.4")]
]
process_button = [sg.Button("Process", key="-PROCESS-")]

layout = [input_line, output_location_line, output_name_line,
          [sg.Column(tracker_settings_column, pad=(0, 0), vertical_alignment="t"),
           sg.Column(detector_settings_column, pad=(0, 0), vertical_alignment="t")], process_button]
window = sg.Window("Vehicle Counter", layout)

while True:
    event, values = window.read()
    if event == sg.WIN_CLOSED:
        break
    if event == '-PROCESS-':
        window["-PROCESS-"].update(disabled=True)
        max_age = int(values["-MAX AGE-"])
        min_age = int(values["-MIN AGE-"])
        max_overlap = float(values["-MAX OVERLAP-"])
        frame_skip = int(values["-FRAME SKIP-"])
        input_video = values["-INPUT-"]
        output = values["-OUTPUT LOCATION-"] + "/" + values["-OUTPUT NAME-"] + ".mp4"
        conf_threshold = float(values["-MIN CONF-"])
        nms_threshold = float(values["-NMS-"])
        window.perform_long_operation(
            lambda: counter.main(max_age, min_age, max_overlap, frame_skip, input_video, output, conf_threshold,
                                 nms_threshold), "-FUNCTION COMPLETE-")
    if event == "-FUNCTION COMPLETE-":
        window["-PROCESS-"].update(disabled=False)

window.close()
