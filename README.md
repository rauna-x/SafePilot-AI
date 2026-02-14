# 🚗 SafePilot – AI Driver Fatigue Detection System

SafePilot is a real-time AI-based Driver Monitoring System (DMS) designed to prevent road accidents caused by driver drowsiness and fatigue.

It uses computer vision and facial landmark analysis to detect:

- 👁 Eye closure (EAR)
- 📊 PERCLOS (Percentage of Eye Closure over time)
- 😮 Yawning detection (MAR)
- 🤕 Head nod detection
- 🚨 Multi-level fatigue alert system (Beep + High Alarm)

The system is optimized for offline edge deployment and can be integrated into vehicles such as trucks, buses, and cars.

---

## 🧠 Problem Statement

Drowsy driving is one of the leading causes of highway accidents, especially in long-haul trucking.

Existing advanced Driver Monitoring Systems are:
- Expensive
- Hardware dependent
- Not accessible in developing regions

SafePilot aims to provide a low-cost AI-based fatigue detection solution using standard cameras and edge computing.

---

## ⚙️ Tech Stack

- Python== 3.10.11
- OpenCV
- MediaPipe Face Mesh
- NumPy
- Real-time Video Processing
- Facial Landmark Geometry

---

## 🔬 Core Detection Logic

### 1️⃣ Eye Aspect Ratio (EAR)
Detects prolonged eye closure using landmark distance calculations.

### 2️⃣ PERCLOS
Tracks percentage of closed-eye frames over time window.

### 3️⃣ Mouth Aspect Ratio (MAR)
Detects yawning based on mouth openness ratio.

### 4️⃣ Head Nod Detection
Monitors vertical head displacement to detect micro-sleep head drops.

### 5️⃣ Multi-Level Alert System
- Mild fatigue → Warning Beep
- Severe fatigue → High-intensity alarm

---

## 📂 Project Structure

```
SafePilot/
│
├── main.py
├── utils.py
├── config.py
├── alarm.py
├── requirements.txt
├── Beep.mp3
└── README.md
```

---

## 🚀 Features

✔ Real-time face tracking  
✔ Multi-metric fatigue scoring  
✔ Offline working (No internet required)  
✔ Lightweight & Edge-ready  
✔ Modular architecture  
✔ Expandable for hardware integration  

---

## 🔧 Installation

```bash
git clone https://github.com/rauna-x/SafePilot-AI.git
cd SafePilot-AI-Driver-Monitoring
pip install -r requirements.txt
python main.py
```

---

## 🎯 Future Scope

- Android Edge App version
- Raspberry Pi deployment
- IR Camera support for night driving
- CAN bus integration for vehicle systems
- Cloud-based fleet monitoring dashboard

---

## 📊 Target Applications

- Truck fleets
- Commercial transport vehicles
- School buses
- Mining vehicles
- Long-distance drivers

---

## 🏆 Vision

To build an affordable, AI-powered driver safety system that can reduce fatigue-related accidents globally.

---

## 📬 Contact

For collaboration, research, or industry integration:

📧 Email: raunak3443@gmail.com  
🔗 LinkedIn: www.linkedin.com/in/raunak-chaturvedi-630488283 
🌍 Location: Prayagraj, India  

---

## ⚠ Disclaimer

This project is a prototype research implementation and not a certified automotive safety product.
