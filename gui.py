import PySimpleGUI as sg
import counter

text_box_size = (17, 1)
input_line = [sg.Text("Input Video:", size=text_box_size), sg.In(size=(25, 1), key="-INPUT-", readonly=True),
              sg.FileBrowse()]
output_location_line = [sg.Text("Output Video Location:", size=text_box_size),
                        sg.In(size=(25, 1), key="-OUTPUT LOCATION-", readonly=True, default_text="C:/Users/sophi/OneDrive - University of East Anglia/Dissertation/dissertaion/outputs"), sg.FolderBrowse()]
output_name_line = [sg.Text("Output File Name:", size=text_box_size), sg.In(size=(25, 1), key="-OUTPUT NAME-")]
process_button = [sg.Button("Process", key="-PROCESS-")]
layout = [input_line, output_location_line, output_name_line, process_button]
window = sg.Window("Vehicle Counter", layout)

while True:
    event, values = window.read()
    if event == sg.WIN_CLOSED:
        break
    if event == '-PROCESS-':
        window["-PROCESS-"].update(disabled=True)
        input_video = values["-INPUT-"]
        output = values["-OUTPUT LOCATION-"] + "/" + values["-OUTPUT NAME-"] + ".mp4"
        window.perform_long_operation(
            lambda: counter.main(input_video, output), "-FUNCTION COMPLETE-")
    if event == "-FUNCTION COMPLETE-":
        window["-PROCESS-"].update(disabled=False)


window.close()
