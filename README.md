# Automated Source Classification of Marmoset Vocalizations

![Poster](<poster_pedro.png>)

## App

### Docker Usage

Clone the repository:
```bash
git clone https://github.com/pedro-varela1/Automated-Source-Classification-of-Marmoset-Vocalizations.git
```

```bash
cd Automated-Source-Classification-of-Marmoset-Vocalizations
```

Build the container by replacing it with the ```tag```, ```container name``` and ```version``` you want:
```bash
docker build -t <tag>/<container_name>:<version> .
```

Finally, run the app:
```bash
docker container run -p 5000:5000 <tag>/<container_name>:<version>
```

### App UI
![App UI](<app_ui.png>)

### Data Entry
- Audio in WAV format containing calls.
- CSV file containing at least three columns:
    - ```label```: Label indicating what type of call it is (only calls with the label ```p``` will be used);
    - ```onset_s```: Start time of the call in seconds, relative to the audio;
    - ```offset_s```: End time of the call in seconds, relative to the audio.

### Data Output
The data output will be a zip file containing images of the calls from each possible line that served as input for the marmoset classification model, as well as a CSV file with the model predictions added to each line.