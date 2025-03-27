import { FaceLandmarker, FilesetResolver } from "@mediapipe/tasks-vision";
import { config } from "./config.js";

const webcamElement = document.getElementById("webcam");
const canvasElement = document.getElementById("overlay");
const canvasCtx = canvasElement.getContext("2d");
const statusElement = document.getElementById("status");

let faceLandmarker;
let runningMode = "VIDEO";
let webcamRunning = false;
let lastVideoTime = -1;
let gameState = "paused"; // paused, running
let lastHeadAngles = { yaw: 0, pitch: 0, roll: 0 };
let audioContext;
let gainNode;

// Initialize AudioContext on first user interaction (implicitly required by browsers)
function initAudio() {
    if (!audioContext) {
        audioContext = new (window.AudioContext || window.webkitAudioContext)();
        gainNode = audioContext.createGain();
        gainNode.gain.setValueAtTime(0.5, audioContext.currentTime); // Set volume
        gainNode.connect(audioContext.destination);
    }
}
document.body.addEventListener('click', initAudio, { once: true }); // Click anywhere to enable audio

// Function to play a pitch
function playPitch(frequency) {
    if (!audioContext || !gainNode) return; // Ensure audio context is initialized

    const oscillator = audioContext.createOscillator();
    oscillator.type = 'sine'; // sine, square, sawtooth, triangle
    oscillator.frequency.setValueAtTime(frequency, audioContext.currentTime); // value in hertz
    oscillator.connect(gainNode);
    oscillator.start();
    oscillator.stop(audioContext.currentTime + config.pitchDuration);
}


// Initialize the FaceLandmarker
async function createFaceLandmarker() {
    statusElement.textContent = "Loading models...";
    const filesetResolver = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm"
    );
    faceLandmarker = await FaceLandmarker.createFromOptions(filesetResolver, {
        baseOptions: {
            modelAssetPath: `https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task`,
            delegate: "GPU"
        },
        outputFaceBlendshapes: true,
        outputFacialTransformationMatrixes: true, // Needed for head pose
        runningMode,
        numFaces: 1
    });
    statusElement.textContent = "Models loaded. Enabling webcam...";
    enableCam();
}

// Enable the webcam stream
function enableCam() {
    if (!faceLandmarker) {
        console.log("Wait! faceLandmarker not loaded yet.");
        return;
    }

    if (webcamRunning) {
        webcamRunning = false;
        // Turn off webcam logic here if needed
    }

    const constraints = {
        video: true
    };

    // Activate the webcam stream.
    navigator.mediaDevices.getUserMedia(constraints).then(function (stream) {
        webcamElement.srcObject = stream;
        webcamElement.addEventListener("loadeddata", predictWebcam);
        webcamRunning = true;
        statusElement.textContent = "Webcam enabled. Waiting for closed eyes to start...";
    }).catch(err => {
        console.error("Error accessing webcam: ", err);
        statusElement.textContent = "Error accessing webcam. Please grant permission.";
    });
}

// Calculate Eye Aspect Ratio (EAR)
function getEAR(landmarks, eyeIndices) {
    // vertical distances
    const v1 = Math.hypot(landmarks[eyeIndices[1]].x - landmarks[eyeIndices[5]].x, landmarks[eyeIndices[1]].y - landmarks[eyeIndices[5]].y);
    const v2 = Math.hypot(landmarks[eyeIndices[2]].x - landmarks[eyeIndices[4]].x, landmarks[eyeIndices[2]].y - landmarks[eyeIndices[4]].y);
    // horizontal distance
    const h = Math.hypot(landmarks[eyeIndices[0]].x - landmarks[eyeIndices[3]].x, landmarks[eyeIndices[0]].y - landmarks[eyeIndices[3]].y);
    const ear = (v1 + v2) / (2.0 * h);
    return ear;
}

// Indices for left and right eyes landmarks
// Check MediaPipe documentation for exact landmark indices: https://developers.google.com/mediapipe/solutions/vision/face_landmarker#face_landmarks
const LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]; // Example indices, verify correct ones
const RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]; // Example indices, verify correct ones

// Function to estimate head rotation angles from the transformation matrix
// Source: Adapted from https://github.com/google/mediapipe/issues/1614#issuecomment-752534379 and OpenCV Rodrigues formula
function getHeadAngles(matrix) {
    const R = [
        [matrix[0].a, matrix[1].a, matrix[2].a],
        [matrix[0].b, matrix[1].b, matrix[2].b],
        [matrix[0].c, matrix[1].c, matrix[2].c]
    ];

    const sy = Math.sqrt(R[0][0] * R[0][0] + R[1][0] * R[1][0]);
    const singular = sy < 1e-6;

    let x, y, z;
    if (!singular) {
        x = Math.atan2(R[2][1], R[2][2]);
        y = Math.atan2(-R[2][0], sy);
        z = Math.atan2(R[1][0], R[0][0]);
    } else {
        x = Math.atan2(-R[1][2], R[1][1]);
        y = Math.atan2(-R[2][0], sy);
        z = 0;
    }

    // Convert radians to degrees
    const pitch = x * (180 / Math.PI); // Rotation around X-axis (Up/Down)
    const yaw = y * (180 / Math.PI);   // Rotation around Y-axis (Left/Right)
    const roll = z * (180 / Math.PI);  // Rotation around Z-axis (Tilt)

    return { pitch, yaw, roll };
}


async function predictWebcam() {
    if (!webcamRunning) return;

    // Set canvas size
    canvasElement.style.width = webcamElement.videoWidth + "px";
    canvasElement.style.height = webcamElement.videoHeight + "px";
    canvasElement.width = webcamElement.videoWidth;
    canvasElement.height = webcamElement.videoHeight;


    let startTimeMs = performance.now();
    if (lastVideoTime !== webcamElement.currentTime) {
        lastVideoTime = webcamElement.currentTime;
        const results = faceLandmarker.detectForVideo(webcamElement, startTimeMs);

        canvasCtx.save();
        canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);

        if (config.debugCanvas) {
             // Draw webcam image (mirrored)
            canvasCtx.save();
            canvasCtx.scale(-1, 1);
            canvasCtx.translate(-canvasElement.width, 0);
            canvasCtx.drawImage(webcamElement, 0, 0, canvasElement.width, canvasElement.height);
            canvasCtx.restore();
        }


        if (results.faceLandmarks && results.faceLandmarks.length > 0) {
            const landmarks = results.faceLandmarks[0];

            // Calculate Eye Aspect Ratios
            const leftEAR = getEAR(landmarks, LEFT_EYE_INDICES);
            const rightEAR = getEAR(landmarks, RIGHT_EYE_INDICES);
            const avgEAR = (leftEAR + rightEAR) / 2.0;

            const eyesClosed = avgEAR < config.eyeClosedThreshold;

             // Get Head Pose
             let headAngles = { yaw: 0, pitch: 0, roll: 0 };
             if (results.facialTransformationMatrixes && results.facialTransformationMatrixes.length > 0) {
                // Assuming the matrix data is structured correctly, otherwise adjust access
                // The matrix provided might need transformation or selection depending on the exact output format
                 const matrixData = results.facialTransformationMatrixes[0]?.data;
                 if (matrixData) {
                    // Convert the flat array into a more usable structure if needed or use directly
                    // Example: Assuming it's row-major 4x4 matrix
                    const matrix = [
                        { a: matrixData[0], b: matrixData[1], c: matrixData[2], d: matrixData[3] }, // Column 1
                        { a: matrixData[4], b: matrixData[5], c: matrixData[6], d: matrixData[7] }, // Column 2
                        { a: matrixData[8], b: matrixData[9], c: matrixData[10], d: matrixData[11] },// Column 3
                        { a: matrixData[12], b: matrixData[13], c: matrixData[14], d: matrixData[15] } // Column 4
                     ];

                     headAngles = getHeadAngles(matrix);

                    // Check for significant head rotation only when game is running
                    if (gameState === "running") {
                         const yawDiff = headAngles.yaw - lastHeadAngles.yaw;
                         const pitchDiff = headAngles.pitch - lastHeadAngles.pitch;

                         if (Math.abs(yawDiff) > config.headRotationThreshold || Math.abs(pitchDiff) > config.headRotationThreshold) {
                            if (yawDiff > config.headRotationThreshold) {
                                // Turned Right (Remember webcam is mirrored, so looking right moves features left on screen, results in positive yaw)
                                playPitch(config.pitchRight);
                                // console.log("Turned Right");
                            } else if (yawDiff < -config.headRotationThreshold) {
                                // Turned Left
                                playPitch(config.pitchLeft);
                                // console.log("Turned Left");
                            }

                            if (pitchDiff > config.headRotationThreshold) {
                                // Looked Down
                                playPitch(config.pitchDown);
                                // console.log("Looked Down");
                            } else if (pitchDiff < -config.headRotationThreshold) {
                                // Looked Up
                                playPitch(config.pitchUp);
                                // console.log("Looked Up");
                            }
                            lastHeadAngles = headAngles; // Update last angles after a sound is played
                         }
                    } else {
                         lastHeadAngles = headAngles; // Keep updating even when paused so first movement isn't huge
                    }

                 }
             }


            // Game State Logic
            if (eyesClosed) {
                if (gameState === "paused") {
                    gameState = "running";
                    statusElement.textContent = "Game Running! Keep eyes closed.";
                    console.log("Game Started");
                    // Reset last angles on game start to avoid immediate sound
                    lastHeadAngles = headAngles;
                }
            } else { // Eyes are open
                if (gameState === "running") {
                    gameState = "paused";
                    statusElement.textContent = "Game Paused (Eyes Open). Close eyes to resume.";
                    console.log("Game Paused - Eyes Open");
                } else {
                     // Still paused, waiting for eyes closed
                     statusElement.textContent = "Game Paused. Close eyes to start.";
                }
            }

            // Optional: Draw landmarks for debugging
            if (config.debugCanvas) {
                canvasCtx.fillStyle = eyesClosed ? 'red' : 'lime';
                // Draw eye landmarks
                 [...LEFT_EYE_INDICES, ...RIGHT_EYE_INDICES].forEach(index => {
                     const point = landmarks[index];
                     canvasCtx.beginPath();
                     canvasCtx.arc(point.x * canvasElement.width, point.y * canvasElement.height, 2, 0, 2 * Math.PI);
                     canvasCtx.fill();
                 });

                // Draw simple head direction indicator (optional)
                canvasCtx.strokeStyle = 'blue';
                canvasCtx.lineWidth = 3;
                const noseTip = landmarks[1]; // Example landmark (nose tip)
                const noseX = noseTip.x * canvasElement.width;
                const noseY = noseTip.y * canvasElement.height;
                const yawRad = headAngles.yaw * (Math.PI / 180);
                const pitchRad = headAngles.pitch * (Math.PI / 180);
                // Simple projection - adjust length/scaling as needed
                const length = 50;
                const endX = noseX + length * Math.sin(yawRad) * Math.cos(pitchRad);
                const endY = noseY - length * Math.sin(pitchRad); // Y is inverted in canvas coords
                canvasCtx.beginPath();
                canvasCtx.moveTo(noseX, noseY);
                canvasCtx.lineTo(endX, endY);
                canvasCtx.stroke();

            }

        } else { // No face detected
            if (gameState === "running") {
                gameState = "paused";
                statusElement.textContent = "Game Paused (Face lost). Look at the camera.";
                console.log("Game Paused - Face Lost");
            } else {
                 statusElement.textContent = "No face detected. Look at the camera.";
            }
        }
        canvasCtx.restore();
    }

    // Call this function again to keep predicting when the browser is ready.
    if (webcamRunning) {
        window.requestAnimationFrame(predictWebcam);
    }
}

// Start the process
createFaceLandmarker();

