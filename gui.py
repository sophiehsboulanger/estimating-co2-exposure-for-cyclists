import PySimpleGUI as sg

text_box_size = (17, 1)
input_line = [sg.Text("Input Video:", size=text_box_size), sg.In(size=(25, 1), key="-INPUT-"), sg.FileBrowse()]
output_location_line = [sg.Text("Output Video Location:", size=text_box_size), sg.In(size=(25, 1), key="-OUTPUT_LOCATION-"), sg.FolderBrowse()]
output_name_line = [sg.Text("Output File Name:", size=text_box_size), sg.In(size=(25, 1), key="-OUTPUT_NAME-")]
tracker_settings_column = [
    [sg.Text("Object Tracker Settings")],
    [sg.Text("Max Age:", size=text_box_size), sg.In(size=(5, 1), key="-MAX_AGE-")],
    [sg.Text("Max Overlap:", size=text_box_size), sg.In(size=(5, 1), key="-MAX_OVERLAP-")],
    [sg.Text("Frame Skip:", size=text_box_size), sg.In(size=(5, 1), key="-FRAME_SKIP-")]
]
detector_settings_column = [
    [sg.Text("Object Detector Settings")],
    [sg.Text("Min Confidence:", size=text_box_size), sg.In(size=(5, 1), key="-MIN_CONF-")],
    [sg.Text("NMS Threshold:", size=text_box_size), sg.In(size=(5, 1), key="-NMS-")]
]
process_button = [sg.Button("Process", key="-PROCESS-")]

layout = [input_line, output_location_line, output_name_line, [sg.Column(tracker_settings_column, pad=(0, 0), vertical_alignment="t"), sg.Column(detector_settings_column, pad=(0, 0), vertical_alignment="t")], process_button]
window = sg.Window("Vehicle Counter", layout)

while True:
    event, values = window.read()
    if event == sg.WIN_CLOSED:
        break
    if event == '-PROCESS-':
        print("Button pressed")

window.close()
