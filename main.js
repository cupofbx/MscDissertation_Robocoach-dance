import {
    PoseLandmarker,
    FilesetResolver,
    DrawingUtils
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0";

const demosSection = document.getElementById("demos");

let runningMode = "VIDEO"; // 毕设建议默认用 VIDEO 模式
let webcamRunning = false;
let lastVideoTime = -1;
let lastRefTime = -1;
let refPoints = null;
let livePoints = null;
let appState = "IDLE"; 
let isPredicting = false;// 这个变量用来防止 predictWebcam 被重复调用，导致多个循环同时运行
let stableFrames = 0;      // 关键：用于平滑检测全身
let evalFrameCounter = 0;  // 用于控制评分频率




// MediaPipe 关键点索引
const ALIGN_POINTS = {
  head: 0,
  leftHip: 23, rightHip: 24,
  leftFoot: 31, rightFoot: 32
};

// 1. 初始化两个独立的 PoseLandmarker 实例
// 一个管摄像头，一个管参考视频，彻底解决时间戳冲突问题
let poseLandmarker = undefined;    // 实时流专用
let refPoseLandmarker = undefined; // 参考视频专用

// 新增：基于时间戳的姿态缓存窗口###
class PoseTimeWindow {
    constructor(durationMs = 500) {
        this.duration = durationMs;
        this.buffer = []; // 存储 {points, timestamp}
    }

    // 压入归一化后的新坐标
    push(normPoints) {
        const now = performance.now();
        this.buffer.push({ points: normPoints, timestamp: now });
        // 自动剔除 1 秒前的数据
        this.buffer = this.buffer.filter(item => now - item.timestamp <= this.duration);
    }

    // 获取当前窗口内所有帧与参考帧的平均误差
    getAverageError(refNorm, keyIndices) {
        if (this.buffer.length === 0) return 999; // 没数据时给个大误差
        
        let totalWindowDist = 0;
        this.buffer.forEach(frame => {
            let frameDist = 0;
            keyIndices.forEach(idx => {
                frameDist += Math.sqrt(
                    Math.pow(refNorm[idx].x - frame.points[idx].x, 2) +
                    Math.pow(refNorm[idx].y - frame.points[idx].y, 2)
                );
            });
            totalWindowDist += (frameDist / keyIndices.length);
        });
        return totalWindowDist / this.buffer.length;
    }

    clear() { this.buffer = []; }
}

// 初始化全局窗口实例
const liveTimeWindow = new PoseTimeWindow(1000);

//节奏结束后，清空窗口###

const createPoseLandmarker = async () => {
    const vision = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm"
    );

    // 实例 A：用于实时摄像头
    poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
        baseOptions: {
            modelAssetPath: `https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task`,
            delegate: "GPU"
        },
        runningMode: "VIDEO",
        numPoses: 1 // 一个人跳，设为 1 性能更好
    });

    // 实例 B：用于参考视频
    refPoseLandmarker = await PoseLandmarker.createFromOptions(vision, {
        baseOptions: {
            modelAssetPath: `https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task`,
            delegate: "GPU"
        },
        runningMode: "VIDEO",
        numPoses: 1
    });

    demosSection.classList.remove("invisible");
    console.log("Double model instance initialization successful！");
};

// 执行初始化
createPoseLandmarker();

// 绑定关闭按钮
document.getElementById("close-modal").onclick = resetSession;

// 2. 获取 DOM 元素
const video = document.getElementById("webcam");
const canvasElement = document.getElementById("output_canvas");
const canvasCtx = canvasElement.getContext("2d");
const drawingUtils = new DrawingUtils(canvasCtx);

const referenceVideo = document.getElementById('referenceVideo');
const refCanvas = document.getElementById('ref_canvas');
const refCanvasCtx = refCanvas ? refCanvas.getContext("2d") : null;
const refDrawingUtils = refCanvasCtx ? new DrawingUtils(refCanvasCtx) : null;
const refUpload = document.getElementById('refUpload');

// 3. 上传参考视频逻辑 状态机未完成版

if (refUpload) {
    refUpload.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
            const url = URL.createObjectURL(file);
            referenceVideo.src = url;
            referenceVideo.load();

            // 只要写这一个回调就够了
            referenceVideo.onloadedmetadata = () => {
                // 1. 设置画布宽高
                refCanvas.width = referenceVideo.videoWidth;
                refCanvas.height = referenceVideo.videoHeight;

                // 2. 【核心】切状态并更新 UI
                appState = "READY_TO_WAKE";
                if (typeof updateUIState === "function") updateUIState(); 
                
                console.log("Video loading complete, status:READY_TO_WAKE");
            };
        }
    });
}
//监听
referenceVideo.addEventListener('ended', () => {
    // 1. 切换状态：停止评分逻辑
    appState = "FINISHED"; 

    // 2. 传入当前帧坐标进行“最后一次强制结算”
    // 因为我们重写了 updateGradeLogic，它现在接收的是坐标数组（refPoints）
    // 传入当前的 refPoints，确保视频最后一秒的动作也被计入总分
    updateGradeLogic(refPoints, true); 
    
    // 3. 核心：更新 UI 状态
    // 这行如果不加，你的上传按钮可能在视频结束后依然是禁用（disabled）状态
    if (typeof updateUIState === "function") {
        updateUIState();
    }
    
    // 4. 弹出结算小框
    showFinalResult();
    
    console.log("舞曲结束，进入结算状态。");
});

// 修改后的核心循环：串行执行防止死锁-7.6

let webcamFrameCount = 0;
let refFrameCount = 0;

async function predictWebcam() {
    if (isPredicting || appState === "FINISHED") {
        if (webcamRunning) window.requestAnimationFrame(predictWebcam);
        return;
    }
    isPredicting = true;

    try {
        const now = performance.now();

        // --- 1. 实时摄像头处理 ---
        if (webcamRunning && video.currentTime !== lastVideoTime) {
            webcamFrameCount++;
            // 【跳帧优化】：每 2 帧才让摄像头跑一次 AI 推理，算力直接省一半！
            if (webcamFrameCount % 2 === 0) {
                lastVideoTime = video.currentTime;
                
                if (canvasElement.width !== video.videoWidth) {
                    canvasElement.width = video.videoWidth;
                    canvasElement.height = video.videoHeight;
                }

                const liveResult = await poseLandmarker.detectForVideo(video, now);
                if (liveResult.landmarks && liveResult.landmarks[0]) {
                    livePoints = liveResult.landmarks[0];
                    
                    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
                    drawingUtils.drawConnectors(livePoints, PoseLandmarker.POSE_CONNECTIONS);
                    drawingUtils.drawLandmarks(livePoints, { radius: 2 });

                    if (appState === "ALIGNING") {
                        checkAlignment(livePoints);
                    }
                }
            }
        }

        // --- 2. 参考视频处理 ---
        if (appState === "SCANNING" && referenceVideo && !referenceVideo.paused) {
            if (referenceVideo.currentTime !== lastRefTime) {
                refFrameCount++;
                // 【跳帧优化】：视频帧也每 2 帧跑一次推理
                if (refFrameCount % 2 === 0) {
                    lastRefTime = referenceVideo.currentTime;
                    const refResult = await refPoseLandmarker.detectForVideo(referenceVideo, now);
                    
                    if (refResult.landmarks && refResult.landmarks[0]) {
                        refPoints = refResult.landmarks[0];
                        
                        refCanvasCtx.clearRect(0, 0, refCanvas.width, refCanvas.height);
                        refDrawingUtils.drawConnectors(refPoints, PoseLandmarker.POSE_CONNECTIONS);
                        refDrawingUtils.drawLandmarks(refPoints, { radius: 2 });

                        // --- 3. 评分同步触发 ---
                        if (livePoints) {
                            const liveNorm = normalizePoints(livePoints);
                            if (liveNorm) {
                                liveTimeWindow.push(liveNorm); 
                                // 这里不再疯狂堆积，里面有绝对时间拦截（500ms）
                                updateGradeLogic(refPoints); 
                            }
                        }
                    }
                }
            }
        }
    } catch (error) {
        console.error("Inference loop error:", error);
    } finally {
        isPredicting = false;
        if (webcamRunning) window.requestAnimationFrame(predictWebcam);
    }
}

function checkAlignment(points) {
    if (!points || points.length < 33) return;

    // 1. 安全取点
    const head = points[0];
    const footL = points[31];
    const footR = points[32];

    if (!head || !footL || !footR) return;

    // 2. 【核心修改】只看 Y 轴（高度），完全不管 X 轴（左右）
    // 这样镜像 rotateY(180deg) 产生的左右颠倒就不会干扰判定了
    
    // 头在画面最上方 35% 区域
    const isHeadTop = head.y < 0.35; 
    
    // 脚在画面下方（允许越界到 1.2），只要有一只脚在下面就算过
    // 门槛设为 0.75，对应你刚才 log 里的 1.10 绰绰有余
    const isFootBottom = (footL.y > 0.75 && footL.y < 1.3) || 
                         (footR.y > 0.75 && footR.y < 1.3);

    // 3. 判定与计数
    if (isHeadTop && isFootBottom) {
        stableFrames++;
        
        // 实时反馈：让你看到进度
        showUIFeedback(`[Locked] Hold on... ${Math.round(stableFrames/10*100)}%`, "lime");
        
        if (stableFrames > 10) {
            console.log(">>> Environment validation passed, switching to countdown!");
            appState = "COUNTDOWN";
            stableFrames = 0;
            startDanceSession(); 
        }
    } else {
        stableFrames = 0;
        // 精准提示
        let tip = "Please aim precisely:";
        if (!isHeadTop) tip += " The head is positioned too low ";
        if (!isFootBottom) tip += " The feet are not in the frame ";
        showUIFeedback(tip, "white");
    }
}

// 倒计时函数
function startDanceSession() {
    let count = 3;
    const overlay = document.getElementById("countdown-overlay");
    
    // 确保开始前清空旧坐标
    refPoints = null; 
    livePoints = null;

    const timer = setInterval(() => {
        if (count > 0) {
            overlay.innerText = count;
        } else if (count === 0) {
            overlay.innerText = "GO!";
            // 立即启动视频，不要等 setTimeout
            referenceVideo.play();
            appState = "SCANNING";
        } else {
            clearInterval(timer);
            overlay.innerText = "";
        }
        count--;
    }, 1000);
}

/**
 * 坐标归一化函数
 * @param {Array} landmarks - MediaPipe 返回的原始 33 个点
 * @returns {Array} 归一化后的点坐标
 */
function normalizePoints(landmarks) {
    if (!landmarks || landmarks.length < 25) return null; // 确保至少有点

    const lp = landmarks[23];
    const rp = landmarks[24];
    const ls = landmarks[11];
    const rs = landmarks[12];

    // 如果关键参考点不可见，直接返回 null 停止本次计算
    if (!lp || !rp || !ls || !rs) return null;

    const midHipX = (lp.x + rp.x) / 2;
    const midHipY = (lp.y + rp.y) / 2;
    const midShoulderX = (ls.x + rs.x) / 2;
    const midShoulderY = (ls.y + rs.y) / 2;
    
    const torsoSize = Math.sqrt(
        Math.pow(midShoulderX - midHipX, 2) + 
        Math.pow(midShoulderY - midHipY, 2)
    );

    const scale = torsoSize > 0.05 ? torsoSize : 1.0;

    return landmarks.map(point => ({
        x: (point.x - midHipX) / scale,
        y: (point.y - midHipY) / scale,
        z: point.z / scale,
        visibility: point.visibility
    }));
}

// 辅助 UI 函数（记得在 HTML 里加这两个 ID）
function showUIFeedback(text, color) {
    const infoBox = document.getElementById("info-display");
    if (infoBox) {
        infoBox.innerText = text;
        infoBox.style.color = color;
    }
}

// 5. 评分逻辑 修改后的函数，现在它与旧的 updateScoreUI 彻底脱钩了

function calculateAndDisplayScore(refRaw, liveRaw) {
    // 1. 依然要先做归一化
    const refNorm = normalizePoints(refRaw);
    const liveNorm = normalizePoints(liveRaw);
    if (!refNorm || !liveNorm) return;

    // 2. 将当前这一帧存入“时间桶”
    liveTimeWindow.push(liveNorm);

    // 3. 计算这一秒钟内的平均误差
    const keyIndices = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28];
    const avgWindowError = liveTimeWindow.getAverageError(refNorm, keyIndices);

    // 4. 将平滑后的误差传给等级逻辑
    updateGradeLogic(avgWindowError); 
}

//打分系统初设
const EVAL_INTERVAL = 8; // 采样周期：1秒 由于帧率太低了，改成 8 帧评一次分
let scoreBuffer = [];
let totalScorePoints = 0;   // 用于计算百分比的总分
let totalEvals = 0;        // 进行了多少次评价
let totalS = 0, totalA = 0, totalB = 0, totalMiss = 0;



let lastEvalTime = 0; // 确保在全局定义了这个变量

function updateGradeLogic(refRaw, forceFinal = false) {
    if (appState !== "SCANNING" && !forceFinal) return;

    // 核心改动：不用 evalFrameCounter < 8 这种肉眼不可控的帧计数
    // 改为基于真实时间戳：每 500 毫秒评一次分
    const now = performance.now();
    if (now - lastEvalTime < 500 && !forceFinal) return; 
    lastEvalTime = now;

    // 安全检查
    const refNorm = normalizePoints(refRaw);
    if (!refNorm) return;

    const keyIndices = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28];
    const avgWindowError = liveTimeWindow.getAverageError(refNorm, keyIndices);

    // 如果误差是 999 说明窗口没数据
    if (avgWindowError > 10) return;

    let grade;
    // 稍微放宽一点阈值，让人更容易得 S 和 A，增加游戏趣味性
    if (avgWindowError < 0.32) grade = "S";      
    else if (avgWindowError < 0.42) grade = "A"; 
    else if (avgWindowError < 0.52) grade = "B"; 
    else grade = "C"; 

    totalEvals++;
    if (grade === "S") { totalS++; totalScorePoints += 100; }
    else if (grade === "A") { totalA++; totalScorePoints += 80; }
    else if (grade === "B") { totalB++; totalScorePoints += 60; }
    else { totalMiss++; } 

    triggerGradeUI(grade);
    console.log(`[Grade Triggered] Real-time Error: ${avgWindowError.toFixed(3)} | Grade: ${grade}`);
}

// 触发 UI 显示的函数
function triggerGradeUI(grade) {
    const textEl = document.getElementById("grade-text");
    if (!textEl) return;

    // 先移除之前的动画类和颜色类，强制重置动画
    textEl.className = ""; 
    void textEl.offsetWidth; // 触发重绘（黑科技，必须写）

    // 设置文字和新颜色类
    textEl.innerText = grade === "C" ? "MISS" : grade; // C显示为MISS更专业
    textEl.classList.add(`color-${grade}`);
    textEl.classList.add("animate-grade");
}

// 辅助函数：计算众数
function getMode(array) {
    const counts = {};
    array.forEach(v => counts[v] = (counts[v] || 0) + 1);
    return Object.keys(counts).reduce((a, b) => counts[a] > counts[b] ? a : b);
}

// 游戏结束时计算最终百分比
function getFinalTotalScore() {
    // 如果一次评估都没有（比如用户刚开始就关了），直接回 0
    if (totalEvals === 0) return 0;

    // 计算原始平均分
    let avg = totalScorePoints / totalEvals;

    // 【可选】保留一位小数或者取整
    // 如果你想让用户觉得很精确，可以用 .toFixed(1)
    let finalScore = Math.round(avg);

    // 边界处理：确保分数在 0-100 之间
    return Math.max(0, Math.min(100, finalScore));
}

function resetSession() {//评分数据清零
    // 1. 清空统计数据
    totalS = 0; totalA = 0; totalB = 0; totalMiss = 0;
    totalScorePoints = 0; totalEvals = 0;
    scoreBuffer = [];

    if (typeof liveTimeWindow !== 'undefined') {
        liveTimeWindow.clear();
    }
    
    // 2. 视频回归起点但不播放
    referenceVideo.pause();
    referenceVideo.currentTime = 0;
    lastRefTime = -1; // 将上一次记录的时间设为负数，确保下一帧能通过检测
    
    // 3. 状态回退到对齐阶段
    appState = "ALIGNING"; 
    
    // 4. 清理画布
    refCanvasCtx.clearRect(0, 0, refCanvas.width, refCanvas.height);
    
    // 5. 隐藏结果框
    document.getElementById("result-modal").classList.add("invisible");
    
    console.log("Session reset, please realign the key points");
    // 强制隐藏打分字母
    const gradeText = document.getElementById("grade-text");
    if (gradeText) gradeText.style.opacity = "0";
    
    // 隐藏结算框
    document.getElementById("result-modal").classList.add("invisible");
    
    // 重置状态
    appState = "ALIGNING";
}

// 结算小框显示函数
function showFinalResult() {
    const modal = document.getElementById("result-modal");
    const scoreVal = document.getElementById("final-score-val");
    const statsVal = document.getElementById("stats-detail-val");

    // 计算总分
    const finalScore = getFinalTotalScore();

    scoreVal.innerText = finalScore;
    statsVal.innerHTML = `
        PERFECT (S): ${totalS} <br>
        GREAT (A): ${totalA} <br>
        GOOD (B): ${totalB} <br>
        MISS: ${totalMiss}
    `;

    modal.classList.remove("invisible");
}

// 6. 摄像头开关控制
const enableWebcamButton = document.getElementById("webcamButton");
// 修改后的摄像头按钮逻辑
if (enableWebcamButton) {
    enableWebcamButton.addEventListener("click", () => {
        // --- 新加的保险：没视频不给开 ---
        if (appState === "IDLE") {
            alert("请先上传参考视频！");
            return;
        }

        if (!poseLandmarker) return;

        if (webcamRunning) {
            webcamRunning = false;
            isPredicting = false; 
            enableWebcamButton.innerText = "ENABLE PREDICTIONS";
            // 如果关了摄像头，状态可以回退到 READY_TO_WAKE
            appState = "READY_TO_WAKE";
            updateUIState();
        } else {
            webcamRunning = true;
            enableWebcamButton.innerText = "DISABLE PREDICTIONS";
            navigator.mediaDevices.getUserMedia({ video: true }).then((stream) => {
                video.srcObject = stream;
                video.onloadeddata = () => {
                    // --- 开启成功，切到 ALIGNING ---
                    appState = "ALIGNING";
                    updateUIState();
                    predictWebcam();
                };
            });
        }
    });
}


function updateUIState() {
    const statusText = document.getElementById("status-text");
    const webcamBtn = document.getElementById("webcamButton");
    const webcamBtnLabel = document.getElementById("webcam-btn-text");
    const uploadInput = document.getElementById('refUpload');

    // --- 方案 A 的核心：全方位防守 ---
    // 只要有一个关键 UI 元素没加载好，就直接退出函数，不执行后面的 switch
    if (!statusText || !webcamBtn || !webcamBtnLabel) {
        console.warn("UI elements not fully loaded, waiting...");
        return; 
    }

    switch (appState) {
        case "IDLE":
            statusText.innerText = "Step 1: Upload the reference video";
            webcamBtn.disabled = true;
            webcamBtn.style.opacity = "0.5";
            webcamBtnLabel.innerText = "Waiting for upload...";
            break;

        case "READY_TO_WAKE":
            statusText.innerText = "Step 2: Enable Webcam";
            webcamBtn.disabled = false;
            webcamBtn.style.opacity = "1";
            webcamBtnLabel.innerText = "Enable Webcam";
            break;

        case "ALIGNING":
            statusText.innerText = "Please stand further away and ensure your whole body is in the frame";
            webcamBtnLabel.innerText = "Webcam is enabled";
            if (uploadInput) uploadInput.disabled = false; 
            break;

        case "SCANNING":
            statusText.innerText = "Practicing: Keep up with the rhythm!";
            if (uploadInput) uploadInput.disabled = true; 
            break;

        case "FINISHED":
            statusText.innerText = "Practice completed, please check the report below";
            if (uploadInput) uploadInput.disabled = false;
            webcamBtnLabel.innerText = "Enable Webcam";
            break;
    }
}

// 当脚本加载完成，立即同步一次 UI，把按钮锁死
window.addEventListener('load', () => {
    // 确保此时 appState 是 "IDLE"
    if (typeof updateUIState === "function") {
        updateUIState();
        console.log("Initialization successful: Webcam button locked, waiting for video upload...");
    }
});
//消灭 TypeScript 语法
