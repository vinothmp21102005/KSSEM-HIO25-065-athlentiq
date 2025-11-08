
#  Athlentiq – Real-Time Jump Analysis Models

Athlentiq is a computer-vision-based jump analysis system that uses **MediaPipe Pose** and **OpenCV** to measure an athlete’s **Vertical Jump Height** and **Standing Broad Jump Distance** in real-time using only a regular camera.

Both models are lightweight, cross-platform, and can be integrated into a mobile or desktop application for sports performance analysis.

---

##  Features

###  Vertical Jump Model (`jump_height_webcam.py`)
- Measures vertical jump height using **airtime** and **pose landmarks**  
- Includes **dynamic ground tracking** to nullify camera shake  
- Real-time height visualization and CSV logging  
- Auto resets between jumps  
- Works on live camera feed (laptop or phone camera)

###  Standing Broad Jump Model (`broad_jump_webcam_calibrated.py`)
- Measures **horizontal jump distance** in real-world meters  
- Uses **optical flow stabilization** for mobile camera support  
- **Auto height calibration** (based on athlete’s height & torso ratio)  
- Displays real-time distance graph and logs results to CSV  
- Works from a **side-view camera setup**

---

##  Core Technologies

| Library | Purpose |
|----------|----------|
| **MediaPipe Pose** | Landmark detection for body joints |
| **OpenCV** | Video feed capture, drawing, and optical flow tracking |
| **NumPy** | Vectorized pose calculations |
| **Matplotlib** | Real-time plotting of jump graphs |
| **Pandas** | Logging and exporting results to CSV |

---

##  Installation

### 1️ Clone or download the repository:
```bash
git clone https://github.com/<yourusername>/<repo-name>.git
cd <repo-name>
```
### 2️ Create a virtual environment (optional but recommended)
```bash
python -m venv venv
venv\Scripts\activate     # On Windows
# or
source venv/bin/activate  # On macOS/Linux
```
### 3️ Install dependencies
```
pip install mediapipe opencv-python numpy matplotlib pandas
```

## Usage
### Vertical Jump

```
python vertical_jump.py
```
Instructions:

Stand still for 3 seconds — the system sets the ground reference.

Jump vertically — your jump height appears on screen.

All results are saved to jump_log.csv.

### Standing Broad Jump
```
python broad_jump.py
```
Instructions:

Enter your height (in meters) when prompted.

Stand still 3 seconds for calibration.

Perform a standing broad jump from a side-view camera angle.

Jump distance (in meters) is displayed and saved to broad_jump_log_meters.csv.

##  Recommended Camera Setup

| Parameter       | Recommendation                          |
| --------------- | --------------------------------------- |
| **Camera View** | Side view (90° to jump direction)       |
| **Distance**    | 3–4 meters away                         |
| **Height**      | ~1 meter (hip-level)                    |
| **Lighting**    | Bright, even illumination               |
| **Surface**     | Flat, textured floor (for optical flow) |
| **Frame Width** | Should capture ~3 meters of ground      |

##  Example Outputs

| Metric                | Description                                                           |
| --------------------- | --------------------------------------------------------------------- |
| **Vertical Jump (m)** | Calculated using airtime formula and gravity                          |
| **Broad Jump (m)**    | Calculated using horizontal ankle displacement × pixel-to-meter scale |
| **Airtime (s)**       | Total time feet were off the ground                                   |
| **Logs**              | Saved automatically as CSV files for review                           |

## System Requirements
Python 3.8 or newer

Webcam or phone camera (min. 720p recommended)

OpenCV + MediaPipe installed

Works on Windows, macOS, and Linux

## Project Structure
```
 athlentiq-jump-analysis
 ┣  jump_height_webcam.py              # Vertical Jump Model
 ┣  broad_jump_webcam_calibrated.py    # Standing Broad Jump Model
 ┣  jump_log.csv                       # Output logs for vertical jumps
 ┣  broad_jump_log_meters.csv          # Output logs for broad jumps
 ┣  README.md                          # Project documentation
 ┗  requirements.txt                   # Dependencies (optional)

```

### Contributors

| Name                 | Role                           |
| -------------------- | ------------------------------ |
| **Sanjit P**         | Research & developer           |
| **Sri vatsan**       | App Developer & Researcher     |
| **Vinoth M P**       | Model Developer & code tester  |
| **Karthick Raja**    | UI & UX  Designer              |


