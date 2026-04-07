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
    console.log("双模型实例初始化成功！");
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
/**if (refUpload) {
  refUpload.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) {
      const url = URL.createObjectURL(file);
      referenceVideo.src = url;
      referenceVideo.load();
      referenceVideo.onloadedmetadata = () => {
        refCanvas.width = referenceVideo.videoWidth;
        refCanvas.height = referenceVideo.videoHeight;
      };
    }
    referenceVideo.onloadedmetadata = () => {
        // 关键：上传成功后，状态切到 READY
        appState = "READY_TO_WAKE";
        updateUIState(); // 更新按钮状态
        
        refCanvas.width = referenceVideo.videoWidth;
        refCanvas.height = referenceVideo.videoHeight;
    };
  });
}
  */

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
                
                console.log("视频加载完成，状态：READY_TO_WAKE");
            };
        }
    });
}
//监听
referenceVideo.addEventListener('ended', () => {
    // 1. 切换状态：停止评分逻辑
    appState = "FINISHED"; 

    // 2. 关键微调：传入当前帧坐标进行“最后一次强制结算”
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

// 修改后的核心循环：串行执行防止死锁
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
            lastVideoTime = video.currentTime;
            
            // 自动对齐画布尺寸
            if (canvasElement.width !== video.videoWidth) {
                canvasElement.width = video.videoWidth;
                canvasElement.height = video.videoHeight;
            }

            const liveResult = await poseLandmarker.detectForVideo(video, now);
            if (liveResult.landmarks && liveResult.landmarks[0]) {
                livePoints = liveResult.landmarks[0];
                
                // 绘制
                canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
                drawingUtils.drawConnectors(livePoints, PoseLandmarker.POSE_CONNECTIONS);
                drawingUtils.drawLandmarks(livePoints, { radius: 2 });

                // 只有在对齐阶段才运行检测
                if (appState === "ALIGNING") {
                    checkAlignment(livePoints);
                }
            }
        }

        // --- 2. 参考视频处理 ---
        if (appState === "SCANNING" && referenceVideo && !referenceVideo.paused) {
            if (referenceVideo.currentTime !== lastRefTime) {
                lastRefTime = referenceVideo.currentTime;
                // 使用 performance.now() 规避时间戳倒流报错
                const refResult = await refPoseLandmarker.detectForVideo(referenceVideo, now);
                
                if (refResult.landmarks && refResult.landmarks[0]) {
                    refPoints = refResult.landmarks[0];
                    
                    // 绘制参考骨架
                    refCanvasCtx.clearRect(0, 0, refCanvas.width, refCanvas.height);
                    refDrawingUtils.drawConnectors(refPoints, PoseLandmarker.POSE_CONNECTIONS);
                    refDrawingUtils.drawLandmarks(refPoints, { radius: 2 });

                    // --- 3. 评分同步触发 ---
                    // 确保两边都有点才评分
                    if (livePoints) {
                        const liveNorm = normalizePoints(livePoints);
                        if (liveNorm) {
                            liveTimeWindow.push(liveNorm); 
                            updateGradeLogic(refPoints); // 传入参考点
                        }
                    }
                }
            }
        }
    } catch (error) {
        console.error("推理循环出错:", error);
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
        showUIFeedback(`[已锁定] 保持住... ${Math.round(stableFrames/10*100)}%`, "lime");
        
        if (stableFrames > 10) {
            console.log(">>> 环境校验通过，切换至倒计时！");
            appState = "COUNTDOWN";
            stableFrames = 0;
            startDanceSession(); 
        }
    } else {
        stableFrames = 0;
        // 精准提示
        let tip = "请对准：";
        if (!isHeadTop) tip += " 头部太靠下 ";
        if (!isFootBottom) tip += " 脚部未入镜 ";
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

// 5. 评分逻辑 修改后的函数，现在它与旧的 updateScoreUI 彻底脱钩了 旧的
/** 
function calculateAndDisplayScore(refRaw, liveRaw) {
    // 1. 标准化两边的坐标
    const refNorm = normalizePoints(refRaw);
    const liveNorm = normalizePoints(liveRaw);

    if (!refNorm || !liveNorm) return;

    // 2. 选择核心点位进行误差计算
    // 选了肩膀、肘、腕、胯、膝、踝，覆盖全身主要动作
    const keyIndices = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28];
    let totalDist = 0;

    keyIndices.forEach(idx => {
        // 计算欧几里得距离
        const d = Math.sqrt(
            Math.pow(refNorm[idx].x - liveNorm[idx].x, 2) +
            Math.pow(refNorm[idx].y - liveNorm[idx].y, 2)
        );
        totalDist += d;
    });

    // 3. 计算平均误差 (avgDist)
    const avgDist = totalDist / keyIndices.length;

    // 4. 【关键小改动】直接把误差传给你的新系统
    updateGradeLogic(avgDist); 
}
*/

function calculateAndDisplayScore(refRaw, liveRaw) {
    // 1. 依然要先做归一化
    const refNorm = normalizePoints(refRaw);
    const liveNorm = normalizePoints(liveRaw);
    if (!refNorm || !liveNorm) return;

    // 2. 【核心改变】：将当前这一帧存入“时间桶”
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

/** 
function updateGradeLogic(currentSpatialError, forceFinal = false) { //得分敏感性在这里调整
    // 1. 只有非强制结算时才存入数据
    if (!forceFinal) {
        let grade;
        if (currentSpatialError < 0.15) grade = "S";
        else if (currentSpatialError < 0.25) grade = "A";
        else if (currentSpatialError < 0.35) grade = "B";
        else grade = "C";
        scoreBuffer.push(grade);
    }

    // 2. 达到周期 OR 强制结算（视频结束）
    if ((scoreBuffer.length >= EVAL_INTERVAL || forceFinal) && scoreBuffer.length > 0) {
        const finalGrade = getMode(scoreBuffer);
        
        totalEvals++; // 总评价次数加 1
        
        // 不仅加总分，还要加各个等级的计数器
        if (finalGrade === "S") {
            totalS++; 
            totalScorePoints += 100;
        } else if (finalGrade === "A") {
            totalA++; 
            totalScorePoints += 80;
        } else if (finalGrade === "B") {
            totalB++; 
            totalScorePoints += 60;
        } else {
            totalMiss++; // C 就是 MISS
        }

        triggerGradeUI(finalGrade);
        scoreBuffer = []; 
    }
}
    */

let lastEvalTime = 0; // 确保在全局定义了这个变量

function updateGradeLogic(refRaw, forceFinal = false) {
    if (appState !== "SCANNING" && !forceFinal) return;

    // 帧频率控制：每 8 帧处理一次，避免 UI 闪烁过快
    evalFrameCounter++;
    if (evalFrameCounter < 8 && !forceFinal) return;
    evalFrameCounter = 0;

    // 安全检查
    const refNorm = normalizePoints(refRaw);
    if (!refNorm) return;

    const keyIndices = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28];
    const avgWindowError = liveTimeWindow.getAverageError(refNorm, keyIndices);

    // 如果误差是 999 说明窗口没数据
    if (avgWindowError > 10) return;

    let grade;
    if (avgWindowError < 0.28) grade = "S";      
    else if (avgWindowError < 0.38) grade = "A"; 
    else if (avgWindowError < 0.48) grade = "B"; 
    else grade = "C"; 

    totalEvals++;
    if (grade === "S") { totalS++; totalScorePoints += 100; }
    else if (grade === "A") { totalA++; totalScorePoints += 80; }
    else if (grade === "B") { totalB++; totalScorePoints += 60; }
    else { totalMiss++; } 

    triggerGradeUI(grade);
    console.log(`Current Error: ${avgWindowError.toFixed(3)} | Grade: ${grade}`);
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
    
    console.log("会话已重置，请重新对齐关键点");
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
        console.warn("UI 元素尚未完全加载，等待中...");
        return; 
    }

    switch (appState) {
        case "IDLE":
            statusText.innerText = "Step 1: 上传参考视频";
            webcamBtn.disabled = true;
            webcamBtn.style.opacity = "0.5";
            webcamBtnLabel.innerText = "等待上传...";
            break;

        case "READY_TO_WAKE":
            statusText.innerText = "Step 2: 开启摄像头";
            webcamBtn.disabled = false;
            webcamBtn.style.opacity = "1";
            webcamBtnLabel.innerText = "开启摄像头";
            break;

        case "ALIGNING":
            statusText.innerText = "请站远一点，确保全身入镜";
            webcamBtnLabel.innerText = "摄像头已开启";
            if (uploadInput) uploadInput.disabled = false; 
            break;

        case "SCANNING":
            statusText.innerText = "正在练习：跟上节奏！";
            if (uploadInput) uploadInput.disabled = true; 
            break;

        case "FINISHED":
            statusText.innerText = "练习结束，查看下方报告";
            if (uploadInput) uploadInput.disabled = false;
            webcamBtnLabel.innerText = "再次开启";
            break;
    }
}

// 当脚本加载完成，立即同步一次 UI，把按钮锁死
window.addEventListener('load', () => {
    // 确保此时 appState 是 "IDLE"
    if (typeof updateUIState === "function") {
        updateUIState();
        console.log("初始化成功：已锁定摄像头按钮，等待上传视频...");
    }
});
//消灭 TypeScript 语法
