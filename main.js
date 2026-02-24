import {
  PoseLandmarker,
  FilesetResolver,
  DrawingUtils
} from "@mediapipe/tasks-vision";

const demosSection = document.getElementById("demos");

let runningMode = "VIDEO"; // 毕设建议默认用 VIDEO 模式
let webcamRunning = false;
let lastVideoTime = -1;
let lastRefTime = -1;
let refPoints = null;
let livePoints = null;
let appState = "ALIGNING"; 
let isPredicting = false;// 这个变量用来防止 predictWebcam 被重复调用，导致多个循环同时运行




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

// 3. 上传参考视频逻辑
if (refUpload) {
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
  });
}

//监听
referenceVideo.addEventListener('ended', () => {
    // 1. 停止摄像头运行标记（停止绘制和评分）
    appState = "FINISHED"; 

    //视频结束了，不管 buffer 里攒了多少，强制算一次分
    updateGradeLogic(0, true);
    
    // 2. 弹出结算小框
    showFinalResult();
});

// 修改后的核心循环：串行执行防止死锁
async function predictWebcam() {
    // 1. 唯一性锁
    if (isPredicting) return; 
    // 如果是 FINISHED 状态，不执行推理，直接跳下一帧等待状态改变
    if (appState === "FINISHED") {
        window.requestAnimationFrame(predictWebcam);
        return;
    }

    if (isPredicting) return;

    isPredicting = true;

    try {
        const now = performance.now();

        // --- 第一步：处理实时摄像头 (使用 poseLandmarker) ---
        if (webcamRunning && video.currentTime !== lastVideoTime) {
            lastVideoTime = video.currentTime;
            
            // 【关键点】这里用 poseLandmarker，配 performance.now()
            const liveResult = await poseLandmarker.detectForVideo(video, now);
            
            canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
            if (liveResult.landmarks && liveResult.landmarks.length > 0) {
                livePoints = liveResult.landmarks[0];
                
                if (canvasElement.width !== video.videoWidth) {
                    canvasElement.width = video.videoWidth;
                    canvasElement.height = video.videoHeight;
                }

                drawingUtils.drawConnectors(livePoints, PoseLandmarker.POSE_CONNECTIONS);
                drawingUtils.drawLandmarks(livePoints, { radius: 2 });

                if (appState === "ALIGNING") {
                    checkAlignment(livePoints);
                }
            }
        }

        // --- 第二步：处理参考视频 (使用 refPoseLandmarker) ---
        if (appState === "SCANNING" && referenceVideo && !referenceVideo.paused) {
            // 增加逻辑：只有时间戳大于 0 且真的在增加时才检测
            if (referenceVideo.currentTime > 0 && referenceVideo.currentTime !== lastRefTime) {
                lastRefTime = referenceVideo.currentTime;
                const refTimestamp = Math.floor(referenceVideo.currentTime * 1000); 
        
                try {
                    const refResult = await refPoseLandmarker.detectForVideo(referenceVideo, refTimestamp);
                    refCanvasCtx.clearRect(0, 0, refCanvas.width, refCanvas.height);
                    if (refResult.landmarks && refResult.landmarks.length > 0) {
                        refPoints = refResult.landmarks[0];
                        refDrawingUtils.drawConnectors(refPoints, PoseLandmarker.POSE_CONNECTIONS);
                        refDrawingUtils.drawLandmarks(refPoints, { radius: 2 });
                    }
                } catch (e) {
                    console.warn("参考视频检测跳过一帧:", e);
                }
            }
            
        }

        // --- 第三步：评分逻辑 ---
        if (appState === "SCANNING" && refPoints && livePoints) {
            calculateAndDisplayScore(refPoints, livePoints);
        }

    } catch (error) {
        console.error("MediaPipe Execution Error:", error);
    } finally {
        isPredicting = false; 
    }

    if (webcamRunning) {
        window.requestAnimationFrame(predictWebcam);
    }
}

function checkAlignment(points) {
    // 1. 定义核心点位
    const required = [0, 23, 24, 31, 32]; // 头, 左右胯, 左右脚
    
    // 2. 检查可见度
    const isReady = required.every(idx => 
        points[idx] && points[idx].visibility > 0.6
    );

    // 3. 根据检查结果执行动作
    if (isReady) {
        // 状态 A：全员进入画面
        // 注意：这里统一调用 showUIFeedback，保持 UI 样式一致
        showUIFeedback("Ready! Keep your pose...", "lime");

        // 只有在 ALIGNING 状态下检测成功，才触发倒计时
        // 这里的 appState 检查能防止 startDanceSession 被重复调用 100 次
        if (appState === "ALIGNING") {
            appState = "COUNTDOWN"; 
            startDanceSession(); 
        }
    } else {
        // 状态 B：有人出屏了
        showUIFeedback("Please show your full body (Head, Hips, Feet)", "white");
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
    if (!landmarks || landmarks.length === 0) return null;

    // 1. 找原点：左右胯的中心点
    const midHipX = (landmarks[23].x + landmarks[24].x) / 2;
    const midHipY = (landmarks[23].y + landmarks[24].y) / 2;

    // 2. 找缩放基准：计算躯干长度 (肩膀中心到胯部中心)
    const midShoulderX = (landmarks[11].x + landmarks[12].x) / 2;
    const midShoulderY = (landmarks[11].y + landmarks[12].y) / 2;
    
    // 躯干长度 (欧式距离)
    const torsoSize = Math.sqrt(
        Math.pow(midShoulderX - midHipX, 2) + 
        Math.pow(midShoulderY - midHipY, 2)
    );

    // 3. 归一化：所有点减去原点，再除以缩放基准
    // 如果 torsoSize 太小（比如没拍全），给个默认值 1 防止报错
    const scale = torsoSize > 0.1 ? torsoSize : 1.0;

    return landmarks.map(point => {
        return {
            x: (point.x - midHipX) / scale,
            y: (point.y - midHipY) / scale,
            z: point.z / scale, // Z 轴也同步缩放
            visibility: point.visibility
        };
    });
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

//打分系统初设
const EVAL_INTERVAL = 30; // 采样周期：1秒
let scoreBuffer = [];
let totalScorePoints = 0;   // 用于计算百分比的总分
let totalEvals = 0;        // 进行了多少次评价
let totalS = 0, totalA = 0, totalB = 0, totalMiss = 0;

function updateGradeLogic(currentSpatialError, forceFinal = false) {
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
        
        // ✨ 修复：不仅加总分，还要加各个等级的计数器！
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
    if (!poseLandmarker) return;
    if (webcamRunning) {
      webcamRunning = false;
      isPredicting = false; // 重置锁
      enableWebcamButton.innerText = "ENABLE PREDICTIONS";
    } else {
      webcamRunning = true;
      enableWebcamButton.innerText = "DISABLE PREDICTIONS";
      navigator.mediaDevices.getUserMedia({ video: true }).then((stream) => {
        video.srcObject = stream;
        // 关键：loadeddata 触发后，检查是否已经在运行，避免双重循环
        video.onloadeddata = () => {
            predictWebcam();
        };
      });
    }
  });
}

//消灭 TypeScript 语法
