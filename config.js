export const config = {
    debugCanvas: true, // Set to true to see the webcam feed and landmarks
    eyeClosedThreshold: 0.4, // Lower value means eyes need to be more closed to trigger
    headRotationThreshold: 10, // Degrees of rotation needed to trigger sound
    pitchUp: 440 * Math.pow(2, 4/12), // F#4
    pitchDown: 440 * Math.pow(2, -3/12), // F4
    pitchLeft: 440 * Math.pow(2, -8/12), // C4
    pitchRight: 440 * Math.pow(2, 9/12), // G#4/Ab4
    pitchDuration: 0.1, // seconds
};

