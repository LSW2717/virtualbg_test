function ensureSlash(url) {
  return url.endsWith('/') ? url : url + '/';
}

function resolveBaseUrl() {
  if (typeof window !== 'undefined' && window.CXTOOLKIT_BASE_URL) {
    return ensureSlash(window.CXTOOLKIT_BASE_URL.toString());
  }
  if (typeof document !== 'undefined' && document.currentScript && document.currentScript.src) {
    try {
      const parsed = new URL(document.currentScript.src);
      const idx = parsed.pathname.lastIndexOf('/');
      const basePath = idx >= 0 ? parsed.pathname.substring(0, idx + 1) : parsed.pathname;
      return ensureSlash(parsed.origin + basePath);
    } catch (e) {
      // ignore, fallback below
    }
  }
  if (typeof window !== 'undefined' && window.location) {
    return ensureSlash(window.location.origin + '/assets/packages/cxtoolkit/assets/');
  }
  return '/assets/packages/cxtoolkit/assets/';
}

const CX_ASSETS_BASE_URL = resolveBaseUrl();

const SET_TIMEOUT = 1;
const CLEAR_TIMEOUT = 2;
const TIMEOUT_TICK = 3;

const code = `
    var timer;
    onmessage = function(request) {
        switch (request.data.id) {
        case ${SET_TIMEOUT}: {
            timer = setTimeout(() => {
                postMessage({ id: ${TIMEOUT_TICK} });
            }, request.data.timeMs);
            break;
        }
        case ${CLEAR_TIMEOUT}: {
            if (timer) {
                clearTimeout(timer);
            }
            break;
        }
        }
    };
  `;
const timerWorkerScript = URL.createObjectURL(
  new Blob([code], {
    type: 'application/javascript',
  })
);

// [수정 1] 256x256 해상도 규격으로 변경
const segmentationWidth = 256;
const segmentationHeight = 144;

const inputResolutions = {
  '256x144': [segmentationWidth, segmentationHeight],
};

const DEFAULT_SEGMENTATION_CONFIG = {
  inputResolution: '256x144',
  deferInputResizing: false,
  // [수정 2] 'meet' 모델(2채널 softmax)이 아닌 1채널 마스크 모델로 처리하도록 변경
  model: 'meet',
};

const DEFAULT_POST_PROCESSING_CONFIG = {
  jointBilateralFilter: {
    sigmaSpace: 1.5,
    sigmaColor: 0.15,
  },
  coverage: [0.15, 0.9],
  lightWrapping: 0.05,
  blendMode: 'screen',
};

var _model;
var _stream;
var _inputVideoElement;
var _maskFrameTimerWorker;
var _outputCanvasElement;
var _fallbackCtx;
var _preprocessCanvas;
var _preprocessCtx;
var _backgroundImage;
var _backgroundColor;
var _backgroundColorImage;
var _sourcePlayback = {
  width: 0,
  height: 0,
  htmlElement: null,
};
var _segmentationConfig = Object.assign({}, DEFAULT_SEGMENTATION_CONFIG);
var _postProcessingConfig = Object.assign({}, DEFAULT_POST_PROCESSING_CONFIG);
var _activeBackgroundConfig = { type: 'passthrough', blurStrength: 0 };
var _webgl2Pipeline;
var _pipelineDirty = true;
var _lastBackgroundToken = null;
var _lastVideoWidth = 0;
var _lastVideoHeight = 0;
var _config = {
  background: null,
  blur: 0,
  mirror: false,
  degree: 360,
  brightness: 100,
  grayscale: 0,
};
var _supportLogPrinted = false;
var _renderLogCount = 0;
var _identityMatrix = new Float32Array([1, 0, 0, 1]);
var _webgl2Supported = false;

// Workaround for FF issue https://bugzilla.mozilla.org/show_bug.cgi?id=1388974
_outputCanvasElement = document.createElement('canvas');
_webgl2Supported = (function detectWebGL2Support() {
  if (typeof window === 'undefined' || !window.WebGL2RenderingContext) {
    return false;
  }
  try {
    const testCanvas = document.createElement('canvas');
    const gl2 =
      testCanvas.getContext('webgl2', { preserveDrawingBuffer: false }) ||
      testCanvas.getContext('experimental-webgl2');
    return !!gl2;
  } catch (err) {
    console.warn('[Virtual BG] WebGL2 detection failed:', err);
    return false;
  }
})();

if (!_webgl2Supported) {
  _fallbackCtx = _outputCanvasElement.getContext('2d');
}
_inputVideoElement = document.createElement('video');
_inputVideoElement.autoplay = true;
_inputVideoElement.playsInline = true;

// Safari issue
_inputVideoElement.style.width = '1px';
_inputVideoElement.style.height = '1px';
document.body.append(_inputVideoElement);

async function _loadTFLiteModel() {
  if (_model != null) return;

  if (typeof WebAssembly === 'undefined') {
    console.warn('TFLite: WebAssembly not supported. Skipping model load.');
    return;
  }

  var createTFLiteModule = await import(CX_ASSETS_BASE_URL + 'tflite/tflite.js');
  var model = await createTFLiteModule.default();
  try {
    var createTFLiteSIMDModule = await import(CX_ASSETS_BASE_URL + 'tflite/tflite-simd.js');
    model = await createTFLiteSIMDModule.default();
  } catch (error) {
    console.warn('Failed to create TFLite SIMD WebAssembly module.', error);
  }

  // [확인 필요] 256x256 모델 파일 경로 (필요시 경로 수정)
  const modelUrl = CX_ASSETS_BASE_URL + 'models/segm_full_v679.tflite';

  var modelData = await fetch(modelUrl);
  const arrayBuffer = await modelData.arrayBuffer();

  const modelBufferOffset = model._getModelBufferMemoryOffset();
  model.HEAPU8.set(new Uint8Array(arrayBuffer), modelBufferOffset);
  model._loadModel(arrayBuffer.byteLength);
  _model = model;
  _pipelineDirty = true;
}

function _onMaskFrameTimer(response) {
  if (response.data.id === TIMEOUT_TICK) {
    _renderMask();
  }
}

function _videoProcessingSupported() {
  var support = {
    webAssembly: typeof WebAssembly !== 'undefined',
    createImageBitmap: typeof createImageBitmap === 'function',
    canvasElement: !!_outputCanvasElement,
    captureStream: _outputCanvasElement
      ? typeof _outputCanvasElement.captureStream === 'function'
      : false,
    webgl2: _webgl2Supported,
  };

  if (!_supportLogPrinted || !support.webAssembly || !support.captureStream || !support.webgl2) {
    console.log('[Virtual BG] Support detection ->', support);
    _supportLogPrinted = true;
  }

  return (
    support.webAssembly &&
    support.createImageBitmap &&
    support.canvasElement &&
    support.captureStream &&
    support.webgl2
  );
}

function _shouldActivatePipeline() {
  return (
    !!_config &&
    (_config.blur !== 0 ||
      _config.background !== null ||
      _config.brightness !== 100 ||
      _config.grayscale !== 0)
  );
}

function _scheduleNextFrame() {
  if (_maskFrameTimerWorker) {
    _maskFrameTimerWorker.postMessage({
      id: SET_TIMEOUT,
      timeMs: 0,
    });
  }
}

function _drawFallbackFrame(width, height) {
  if (!_fallbackCtx || !_inputVideoElement) {
    return;
  }
  _outputCanvasElement.width = width;
  _outputCanvasElement.height = height;
  _fallbackCtx.save();
  _fallbackCtx.clearRect(0, 0, width, height);
  _fallbackCtx.translate(width / 2, height / 2);
  const radians = (((_config.degree || 0) % 360) * Math.PI) / 180;
  if (_config.mirror) {
    _fallbackCtx.scale(-1, 1);
  }
  if (radians !== 0) {
    _fallbackCtx.rotate(radians);
  }
  _fallbackCtx.drawImage(_inputVideoElement, -width / 2, -height / 2, width, height);
  _fallbackCtx.restore();
}

function _getRotationRadians() {
  return (((_config.degree || 0) % 360) * Math.PI) / 180;
}

function _markPipelineDirty() {
  _pipelineDirty = true;
}

function _destroyPipeline() {
  if (_webgl2Pipeline) {
    try {
      _webgl2Pipeline.cleanUp();
    } catch (err) {
      console.warn('[Virtual BG] Failed to cleanup pipeline', err);
    }
  }
  _webgl2Pipeline = null;
}

function _updateBackgroundToken() {
  if (_backgroundImage) {
    _lastBackgroundToken = _backgroundImage;
  } else if (_backgroundColorImage) {
    _lastBackgroundToken = _backgroundColorImage;
  } else {
    _lastBackgroundToken = null;
  }
}

function _ensurePreprocessCanvas(width, height) {
  if (!_preprocessCanvas) {
    _preprocessCanvas = document.createElement('canvas');
    _preprocessCtx = _preprocessCanvas.getContext('2d');
  }
  if (_preprocessCanvas.width !== width || _preprocessCanvas.height !== height) {
    _preprocessCanvas.width = width;
    _preprocessCanvas.height = height;
  }
}

function _needsPreprocess(width, height) {
  const normalizedRotation = (_config.degree || 0) % 360;
  return _config.mirror || normalizedRotation !== 0;
}

function _prepareSourceElement(width, height) {
  if (!_needsPreprocess(width, height)) {
    _sourcePlayback.htmlElement = _inputVideoElement;
    return;
  }
  _ensurePreprocessCanvas(width, height);
  if (!_preprocessCtx) {
    _sourcePlayback.htmlElement = _inputVideoElement;
    return;
  }
  _preprocessCtx.save();
  _preprocessCtx.setTransform(1, 0, 0, 1, 0, 0);
  _preprocessCtx.clearRect(0, 0, width, height);
  _preprocessCtx.translate(width / 2, height / 2);
  const radians = _getRotationRadians();
  if (_config.mirror) {
    _preprocessCtx.scale(-1, 1);
  }
  if (radians !== 0) {
    _preprocessCtx.rotate(radians);
  }
  _preprocessCtx.drawImage(_inputVideoElement, -width / 2, -height / 2, width, height);
  _preprocessCtx.restore();
  _sourcePlayback.htmlElement = _preprocessCanvas;
}

function _getActiveBackgroundImageElement() {
  if (_backgroundImage) {
    return _backgroundImage;
  }
  return _backgroundColorImage || null;
}

function _computeBackgroundConfig() {
  const normalizedBlur = Math.max(0, Math.min(100, _config.blur || 0)) / 100;
  if (_config.background) {
    return { type: 'image', blurStrength: normalizedBlur };
  }
  if (normalizedBlur > 0) {
    return { type: 'blur', blurStrength: normalizedBlur };
  }
  return { type: 'passthrough', blurStrength: 0 };
}

function _computePostProcessingConfig() {
  const normalizedBlur = Math.max(0, Math.min(100, _config.blur || 0)) / 100;
  const imageBackground = !!_config.background;
  // 공간 필터를 조금 강하게 잡아 마스크의 픽셀 단위 떨림을 줄인다.
  const sigmaSpace = 1.6 + normalizedBlur * 3.2;
  const sigmaColor = 0.14 + normalizedBlur * 0.36;
  const feather = 0.15 - normalizedBlur * 0.05;
  return {
    blurStrength: normalizedBlur,
    jointBilateralFilter: {
      sigmaSpace,
      sigmaColor,
    },
    coverage: [Math.max(0.01, feather), Math.min(0.99, 0.9 + normalizedBlur * 0.05)],
    lightWrapping: 0.05 + normalizedBlur * 0.1,
    blendMode: 'screen',
    maskRefine: imageBackground
      ? {
          maskBlurPx: 1.2,
          edgeBlend: 0.4,
          edgeGamma: 0.98,
          edgeFeather: 0.58,
        }
      : {
          maskBlurPx: 1.1,
          edgeBlend: 0.36,
          edgeGamma: 0.98,
          edgeFeather: 0.54,
        },
    compositeEdge: imageBackground
      ? {
          spillSuppression: 0.18,
          edgeDarkening: 0.24,
        }
      : {
          spillSuppression: 0.14,
          edgeDarkening: 0.2,
        },
  };
}

async function _ensurePipelineReady(width, height) {
  if (!_shouldActivatePipeline()) {
    _destroyPipeline();
    return null;
  }
  if (!_model) {
    await _loadTFLiteModel();
    if (!_model) {
      return null;
    }
  }
  const backgroundToken = _getActiveBackgroundImageElement();
  const needsRebuild =
    !_webgl2Pipeline ||
    _pipelineDirty ||
    _lastVideoWidth !== width ||
    _lastVideoHeight !== height ||
    _lastBackgroundToken !== backgroundToken ||
    _activeBackgroundConfig.type !== _computeBackgroundConfig().type;

  if (!needsRebuild) {
    return _webgl2Pipeline;
  }

  _destroyPipeline();

  _outputCanvasElement.width = width;
  _outputCanvasElement.height = height;
  _sourcePlayback.width = width;
  _sourcePlayback.height = height;
  _sourcePlayback.htmlElement = _inputVideoElement;

  _activeBackgroundConfig = _computeBackgroundConfig();
  _postProcessingConfig = _computePostProcessingConfig();

  try {
    _webgl2Pipeline = buildWebGL2Pipeline(
      _sourcePlayback,
      backgroundToken,
      _activeBackgroundConfig,
      _segmentationConfig,
      _outputCanvasElement,
      _model,
      null,
      function noop() {}
    );
    _webgl2Pipeline.updatePostProcessingConfig(_postProcessingConfig);
    if (_webgl2Pipeline.updateOutputAdjustments) {
      _webgl2Pipeline.updateOutputAdjustments(_config.brightness / 100, _config.grayscale / 100);
    }
    if (_webgl2Pipeline.updateTransform) {
      _webgl2Pipeline.updateTransform({
        mirror: !!_config.mirror,
        rotation: _getRotationRadians(),
      });
    }
  } catch (error) {
    console.error('[Virtual BG] Failed to build WebGL2 pipeline.', error);
    if (!_fallbackCtx) {
      try {
        _fallbackCtx = _outputCanvasElement.getContext('2d');
      } catch (ctxError) {
        console.warn('[Virtual BG] Failed to create fallback 2D context.', ctxError);
      }
    }
    _webgl2Supported = false;
    _webgl2Pipeline = null;
    return null;
  }

  _lastVideoWidth = width;
  _lastVideoHeight = height;
  _lastBackgroundToken = backgroundToken;
  _pipelineDirty = false;
  return _webgl2Pipeline;
}

async function _renderMask() {
  try {
    const width = _inputVideoElement.videoWidth || 1;
    const height = _inputVideoElement.videoHeight || 1;
    const pipeline = await _ensurePipelineReady(width, height);

    if (pipeline) {
      _prepareSourceElement(width, height);
      const rotationRad = _getRotationRadians();
      if (pipeline.updateTransform) {
        pipeline.updateTransform({
          mirror: !!_config.mirror,
          rotation: rotationRad,
        });
      }
      if (typeof pipeline.render === 'function') {
        await pipeline.render();
      }
    } else if (_fallbackCtx) {
      _drawFallbackFrame(width, height);
    } else {
      // No WebGL2 pipeline yet and no 2D fallback (likely waiting for WebGL2 resources)
      // Leave the previous frame rendered.
    }

    if (_renderLogCount < 5) {
      console.log('[Virtual BG] render tick. pipeline active =', !!pipeline);
      _renderLogCount++;
    }
  } catch (error) {
    console.error('[Virtual BG] render failed:', error);
  } finally {
    _scheduleNextFrame();
  }
}

function _createColorImage(hexColor) {
  const canvas = document.createElement('canvas');
  canvas.width = 2;
  canvas.height = 2;
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = hexColor;
  ctx.fillRect(0, 0, 2, 2);
  const img = new Image();
  img.src = canvas.toDataURL('image/png');
  return img;
}

function _setBackground(base64String) {
  if (base64String) {
    if (base64String.startsWith('#')) {
      _backgroundImage = null;
      _backgroundColor = base64String;
      _backgroundColorImage = _createColorImage(base64String);
      console.log('[Virtual BG] background set to color:', _backgroundColor);
      _markPipelineDirty();
      _updateBackgroundToken();
      return;
    }
    var img = new Image();
    img.src = 'data:image/png;base64,' + base64String;
    img.crossOrigin = 'anonymous';
    img.onload = function () {
      if (typeof createImageBitmap === 'function') {
        createImageBitmap(img)
          .then((bitmap) => {
            _backgroundImage = bitmap;
            _backgroundColor = null;
            _backgroundColorImage = null;
            _markPipelineDirty();
            _updateBackgroundToken();
          })
          .catch(() => {
            _backgroundImage = img;
            _backgroundColor = null;
            _backgroundColorImage = null;
            _markPipelineDirty();
            _updateBackgroundToken();
          });
      } else {
        _backgroundImage = img;
        _backgroundColor = null;
        _backgroundColorImage = null;
        _markPipelineDirty();
        _updateBackgroundToken();
      }
    };
    img.onerror = function (err) {
      console.error('[Virtual BG] failed to load background image:', err);
    };
  } else {
    _backgroundImage = null;
    _backgroundColor = null;
    _backgroundColorImage = null;
    console.log('[Virtual BG] background cleared');
    _markPipelineDirty();
    _updateBackgroundToken();
  }
}

function createCustomVideoStream(stream) {
  if (!_videoProcessingSupported()) {
    console.warn('Virtual background: required APIs unavailable. Returning original video stream.');
    return stream;
  }

  _stream = stream;
  _maskFrameTimerWorker = new Worker(timerWorkerScript);
  _maskFrameTimerWorker.onmessage = _onMaskFrameTimer;

  const firstVideoTrack = _stream.getVideoTracks()[0];
  const frameRate = firstVideoTrack.getSettings
    ? firstVideoTrack.getSettings().frameRate
    : firstVideoTrack.getConstraints().frameRate;

  _inputVideoElement.srcObject = _stream;
  _inputVideoElement.onloadeddata = () => {
    if (_maskFrameTimerWorker) {
      _maskFrameTimerWorker.postMessage({
        id: SET_TIMEOUT,
        timeMs: 1000 / 30,
      });
    }
  };

  var capturedStream = _outputCanvasElement.captureStream(parseInt(frameRate || 30, 10));

  var tracks = capturedStream ? capturedStream.getVideoTracks() : [];
  if (!capturedStream || tracks.length === 0) {
    console.warn('[Virtual BG] captureStream produced no tracks. Falling back to original stream.');
    return stream;
  }
  console.log(
    '[Virtual BG] captureStream succeeded. readyState:',
    tracks[0].readyState,
    'frameRate:',
    frameRate
  );
  return capturedStream;
}

function destroyCustomVideoStream() {
  if (_maskFrameTimerWorker) {
    _maskFrameTimerWorker.postMessage({
      id: CLEAR_TIMEOUT,
    });
    _maskFrameTimerWorker.terminate();
    _maskFrameTimerWorker = null;
  }
  _destroyPipeline();
}

function configCustomVideoStream(config) {
  console.log('[Virtual BG] config received:', config);
  if (_config.background !== config.background) {
    _setBackground(config.background);
  }
  _config = config;
  _renderLogCount = 0;
  if (_videoProcessingSupported() && _shouldActivatePipeline()) {
    _loadTFLiteModel();
    _markPipelineDirty();
  }
  if (_webgl2Pipeline) {
    _postProcessingConfig = _computePostProcessingConfig();
    _webgl2Pipeline.updatePostProcessingConfig(_postProcessingConfig);
    if (_webgl2Pipeline.updateOutputAdjustments) {
      _webgl2Pipeline.updateOutputAdjustments(_config.brightness / 100, _config.grayscale / 100);
    }
  }
}

function _glsl(strings) {
  var out = '';
  for (var i = 0; i < strings.length; i++) {
    out += strings[i];
    if (i < arguments.length - 1) {
      out += arguments[i + 1];
    }
  }
  return out;
}

function compileShader(gl, type, source) {
  const shader = gl.createShader(type);
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    console.error('WebGL shader compile failed:', gl.getShaderInfoLog(shader));
    gl.deleteShader(shader);
    return null;
  }
  return shader;
}

function createPiplelineStageProgram(
  gl,
  vertexShader,
  fragmentShader,
  positionBuffer,
  texCoordBuffer
) {
  const program = gl.createProgram();
  gl.attachShader(program, vertexShader);
  gl.attachShader(program, fragmentShader);
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    console.error('WebGL program link failed:', gl.getProgramInfoLog(program));
    gl.deleteProgram(program);
    return null;
  }

  const positionLocation = gl.getAttribLocation(program, 'a_position');
  const texCoordLocation = gl.getAttribLocation(program, 'a_texCoord');

  gl.useProgram(program);

  gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
  gl.enableVertexAttribArray(positionLocation);
  gl.vertexAttribPointer(positionLocation, 2, gl.FLOAT, false, 0, 0);

  gl.bindBuffer(gl.ARRAY_BUFFER, texCoordBuffer);
  gl.enableVertexAttribArray(texCoordLocation);
  gl.vertexAttribPointer(texCoordLocation, 2, gl.FLOAT, false, 0, 0);

  return program;
}

function createTexture(gl, internalFormat, width, height, minFilter, magFilter) {
  const texture = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, minFilter || gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, magFilter || minFilter || gl.NEAREST);

  let format = gl.RGBA;
  let type = gl.UNSIGNED_BYTE;
  if (internalFormat === gl.R32F) {
    format = gl.RED;
    type = gl.FLOAT;
  } else if (internalFormat === gl.RG32F) {
    format = gl.RG;
    type = gl.FLOAT;
  }

  gl.texImage2D(gl.TEXTURE_2D, 0, internalFormat, width, height, 0, format, type, null);
  return texture;
}

function readPixelsAsync(timerWorker, gl, x, y, width, height, format, type, pixels) {
  return new Promise((resolve, reject) => {
    try {
      gl.readPixels(x, y, width, height, format, type, pixels);
      resolve(pixels);
    } catch (error) {
      reject(error);
    }
  });
}

function buildBackgroundBlurStage(
  gl,
  vertexShader,
  positionBuffer,
  texCoordBuffer,
  personMaskTexture,
  canvas
) {
  const blurPass = buildBlurPass(
    gl,
    vertexShader,
    positionBuffer,
    texCoordBuffer,
    personMaskTexture,
    canvas
  );
  const blendPass = buildBlendPass(
    gl,
    positionBuffer,
    texCoordBuffer,
    canvas,
    personMaskTexture,
    blurPass.getOutputTexture()
  );

  function render() {
    blurPass.render();
    blendPass.render();
  }

  function updateCoverage(coverage) {
    blendPass.updateCoverage(coverage);
  }

  function cleanUp() {
    blendPass.cleanUp();
    blurPass.cleanUp();
  }

  function updateOutputAdjustments(brightness, grayscale) {
    blendPass.updateOutputAdjustments(brightness, grayscale);
  }

  function updateBlurAmount(amount) {
    blendPass.updateBlurAmount(amount);
  }

  function updateEdgeConfig(edgeConfig) {
    blendPass.updateEdgeConfig(edgeConfig);
  }

  return {
    render,
    updateCoverage,
    updateOutputAdjustments,
    updateBlurAmount,
    updateEdgeConfig,
    cleanUp,
  };
}

function buildBlurPass(
  gl,
  vertexShader,
  positionBuffer,
  texCoordBuffer,
  personMaskTexture,
  canvas
) {
  const fragmentShaderSource = _glsl`#version 300 es
  
      precision highp float;
  
      uniform sampler2D u_inputFrame;
      uniform sampler2D u_personMask;
      uniform vec2 u_texelSize;
  
      in vec2 v_texCoord;
  
      out vec4 outColor;
  
      const float offset[7] = float[](0.0, 1.5, 3.0, 4.5, 6.0, 7.5, 9.0);
      const float weight[7] = float[](0.25, 0.20, 0.15, 0.10, 0.07, 0.04, 0.02);
  
      void main() {
        vec4 centerColor = texture(u_inputFrame, v_texCoord);
        float personMask = texture(u_personMask, v_texCoord).a;
  
        vec4 frameColor = centerColor * weight[0] * (1.0 - personMask);
  
        for (int i = 1; i < 7; i++) {
          vec2 offset = vec2(offset[i]) * u_texelSize;
  
          vec2 texCoord = v_texCoord + offset;
          frameColor += texture(u_inputFrame, texCoord) * weight[i] *
            (1.0 - texture(u_personMask, texCoord).a);
  
          texCoord = v_texCoord - offset;
          frameColor += texture(u_inputFrame, texCoord) * weight[i] *
            (1.0 - texture(u_personMask, texCoord).a);
        }
        outColor = vec4(frameColor.rgb + (1.0 - frameColor.a) * centerColor.rgb, 1.0);
      }
    `;

  const scale = 0.2;
  const outputWidth = canvas.width * scale;
  const outputHeight = canvas.height * scale;
  const texelWidth = 1 / outputWidth;
  const texelHeight = 1 / outputHeight;

  const fragmentShader = compileShader(gl, gl.FRAGMENT_SHADER, fragmentShaderSource);
  const program = createPiplelineStageProgram(
    gl,
    vertexShader,
    fragmentShader,
    positionBuffer,
    texCoordBuffer
  );
  const inputFrameLocation = gl.getUniformLocation(program, 'u_inputFrame');
  const personMaskLocation = gl.getUniformLocation(program, 'u_personMask');
  const texelSizeLocation = gl.getUniformLocation(program, 'u_texelSize');
  const texture1 = createTexture(gl, gl.RGBA8, outputWidth, outputHeight, gl.NEAREST, gl.LINEAR);
  const texture2 = createTexture(gl, gl.RGBA8, outputWidth, outputHeight, gl.NEAREST, gl.LINEAR);

  const frameBuffer1 = gl.createFramebuffer();
  gl.bindFramebuffer(gl.FRAMEBUFFER, frameBuffer1);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture1, 0);

  const frameBuffer2 = gl.createFramebuffer();
  gl.bindFramebuffer(gl.FRAMEBUFFER, frameBuffer2);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture2, 0);

  gl.useProgram(program);
  gl.uniform1i(personMaskLocation, 1);

  function render() {
    gl.viewport(0, 0, outputWidth, outputHeight);
    gl.useProgram(program);
    gl.uniform1i(inputFrameLocation, 0);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, personMaskTexture);

    for (let i = 0; i < 6; i++) {
      gl.uniform2f(texelSizeLocation, 0, texelHeight);
      gl.bindFramebuffer(gl.FRAMEBUFFER, frameBuffer1);
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

      gl.activeTexture(gl.TEXTURE2);
      gl.bindTexture(gl.TEXTURE_2D, texture1);
      gl.uniform1i(inputFrameLocation, 2);

      gl.uniform2f(texelSizeLocation, texelWidth, 0);
      gl.bindFramebuffer(gl.FRAMEBUFFER, frameBuffer2);
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

      gl.bindTexture(gl.TEXTURE_2D, texture2);
    }
  }

  function getOutputTexture() {
    return texture2;
  }

  function cleanUp() {
    gl.deleteFramebuffer(frameBuffer2);
    gl.deleteFramebuffer(frameBuffer1);
    gl.deleteTexture(texture2);
    gl.deleteTexture(texture1);
    gl.deleteProgram(program);
    gl.deleteShader(fragmentShader);
  }

  return {
    render,
    getOutputTexture,
    cleanUp,
  };
}

function buildBlendPass(
  gl,
  positionBuffer,
  texCoordBuffer,
  canvas,
  personMaskTexture,
  blurredInputTexture
) {
  const vertexShaderSource = _glsl`#version 300 es
  
      in vec2 a_position;
      in vec2 a_texCoord;
  
      out vec2 v_texCoord;
  
      void main() {
        gl_Position = vec4(a_position * vec2(1.0, -1.0), 0.0, 1.0);
        v_texCoord = a_texCoord;
      }
    `;

  const fragmentShaderSource = _glsl`#version 300 es
  
      precision highp float;
  
      uniform sampler2D u_inputFrame;
      uniform sampler2D u_personMask;
      uniform sampler2D u_blurredInputFrame;
      uniform vec2 u_coverage;
      uniform float u_blurStrength;
      uniform float u_backgroundBrightness;
      uniform float u_backgroundGrayscale;
      uniform float u_spillSuppression;
      uniform float u_edgeDarkening;
  
      in vec2 v_texCoord;
  
      out vec4 outColor;
  
      void main() {
        vec3 color = texture(u_inputFrame, v_texCoord).rgb;
        vec3 blurredColor = texture(u_blurredInputFrame, v_texCoord).rgb;
        float rawMask = texture(u_personMask, v_texCoord).a;
        float personMask = smoothstep(u_coverage.x, u_coverage.y, rawMask);
        vec3 backgroundColor = mix(color, blurredColor, clamp(u_blurStrength, 0.0, 1.0));
        float backgroundGray = dot(backgroundColor, vec3(0.299, 0.587, 0.114));
        backgroundColor = mix(backgroundColor, vec3(backgroundGray), clamp(u_backgroundGrayscale, 0.0, 1.0));
        backgroundColor *= u_backgroundBrightness;
        float edge = clamp(1.0 - abs(personMask * 2.0 - 1.0), 0.0, 1.0);
        vec3 foreground = color;
        float fgGray = dot(foreground, vec3(0.299, 0.587, 0.114));
        foreground = mix(
          foreground,
          vec3(fgGray),
          edge * clamp(u_spillSuppression, 0.0, 1.0)
        );
        vec3 finalColor = mix(backgroundColor, foreground, personMask);
        finalColor *= 1.0 - edge * clamp(u_edgeDarkening, 0.0, 1.0) * 0.06;
        outColor = vec4(finalColor, 1.0);
      }
    `;

  const { width: outputWidth, height: outputHeight } = canvas;

  const vertexShader = compileShader(gl, gl.VERTEX_SHADER, vertexShaderSource);
  const fragmentShader = compileShader(gl, gl.FRAGMENT_SHADER, fragmentShaderSource);
  const program = createPiplelineStageProgram(
    gl,
    vertexShader,
    fragmentShader,
    positionBuffer,
    texCoordBuffer
  );
  const inputFrameLocation = gl.getUniformLocation(program, 'u_inputFrame');
  const personMaskLocation = gl.getUniformLocation(program, 'u_personMask');
  const blurredInputFrame = gl.getUniformLocation(program, 'u_blurredInputFrame');
  const coverageLocation = gl.getUniformLocation(program, 'u_coverage');
  const blurStrengthLocation = gl.getUniformLocation(program, 'u_blurStrength');
  const backgroundBrightnessLocation = gl.getUniformLocation(program, 'u_backgroundBrightness');
  const backgroundGrayscaleLocation = gl.getUniformLocation(program, 'u_backgroundGrayscale');
  const spillSuppressionLocation = gl.getUniformLocation(program, 'u_spillSuppression');
  const edgeDarkeningLocation = gl.getUniformLocation(program, 'u_edgeDarkening');

  gl.useProgram(program);
  gl.uniform1i(inputFrameLocation, 0);
  gl.uniform1i(personMaskLocation, 1);
  gl.uniform1i(blurredInputFrame, 2);
  gl.uniform2f(coverageLocation, 0, 1);
  gl.uniform1f(blurStrengthLocation, 1.0);
  gl.uniform1f(backgroundBrightnessLocation, 1.0);
  gl.uniform1f(backgroundGrayscaleLocation, 0.0);
  gl.uniform1f(spillSuppressionLocation, 0.14);
  gl.uniform1f(edgeDarkeningLocation, 0.2);

  let blurStrength = 1.0;
  let backgroundBrightness = 1.0;
  let backgroundGrayscale = 0.0;
  let spillSuppression = 0.14;
  let edgeDarkening = 0.2;

  function render() {
    gl.viewport(0, 0, outputWidth, outputHeight);
    gl.useProgram(program);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, personMaskTexture);
    gl.activeTexture(gl.TEXTURE2);
    gl.bindTexture(gl.TEXTURE_2D, blurredInputTexture);
    gl.uniform1f(blurStrengthLocation, blurStrength);
    gl.uniform1f(backgroundBrightnessLocation, backgroundBrightness);
    gl.uniform1f(backgroundGrayscaleLocation, backgroundGrayscale);
    gl.uniform1f(spillSuppressionLocation, spillSuppression);
    gl.uniform1f(edgeDarkeningLocation, edgeDarkening);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  }

  function updateCoverage(coverage) {
    gl.useProgram(program);
    gl.uniform2f(coverageLocation, coverage[0], coverage[1]);
  }

  function updateBlurAmount(value) {
    blurStrength = Math.max(0, Math.min(1, value));
  }

  function updateOutputAdjustments(brightness, grayscale) {
    backgroundBrightness = brightness;
    backgroundGrayscale = Math.max(0, Math.min(1, grayscale));
  }

  function updateEdgeConfig(edgeConfig) {
    if (!edgeConfig) return;
    if (typeof edgeConfig.spillSuppression === 'number') {
      spillSuppression = Math.max(0, Math.min(1, edgeConfig.spillSuppression));
    }
    if (typeof edgeConfig.edgeDarkening === 'number') {
      edgeDarkening = Math.max(0, Math.min(1, edgeConfig.edgeDarkening));
    }
  }

  function cleanUp() {
    gl.deleteProgram(program);
    gl.deleteShader(fragmentShader);
    gl.deleteShader(vertexShader);
  }

  return {
    render,
    updateCoverage,
    updateBlurAmount,
    updateOutputAdjustments,
    updateEdgeConfig,
    cleanUp,
  };
}

function buildPassthroughStage(gl, positionBuffer, texCoordBuffer, personMaskTexture, canvas) {
  const vertexShaderSource = _glsl`#version 300 es
  
      in vec2 a_position;
      in vec2 a_texCoord;
  
      out vec2 v_texCoord;
  
      void main() {
        gl_Position = vec4(a_position * vec2(1.0, -1.0), 0.0, 1.0);
        v_texCoord = a_texCoord;
      }
    `;

  const fragmentShaderSource = _glsl`#version 300 es
  
      precision highp float;
  
      uniform sampler2D u_inputFrame;
      uniform sampler2D u_personMask;
      uniform vec2 u_coverage;
      uniform float u_outputBrightness;
      uniform float u_outputGrayscale;
  
      in vec2 v_texCoord;
  
      out vec4 outColor;
  
      void main() {
        vec3 color = texture(u_inputFrame, v_texCoord).rgb;
        float personMask = texture(u_personMask, v_texCoord).a;
        
        personMask = smoothstep(u_coverage.x, u_coverage.y, personMask);
        
        vec3 backgroundColor = color;
        float backgroundGray = dot(backgroundColor, vec3(0.299, 0.587, 0.114));
        backgroundColor = mix(backgroundColor, vec3(backgroundGray), clamp(u_outputGrayscale, 0.0, 1.0));
        backgroundColor *= u_outputBrightness;
        vec3 finalColor = mix(backgroundColor, color, personMask);
        outColor = vec4(finalColor, 1.0);
      }
    `;

  const vertexShader = compileShader(gl, gl.VERTEX_SHADER, vertexShaderSource);
  const fragmentShader = compileShader(gl, gl.FRAGMENT_SHADER, fragmentShaderSource);
  const program = createPiplelineStageProgram(
    gl,
    vertexShader,
    fragmentShader,
    positionBuffer,
    texCoordBuffer
  );

  const inputFrameLocation = gl.getUniformLocation(program, 'u_inputFrame');
  const personMaskLocation = gl.getUniformLocation(program, 'u_personMask');
  const coverageLocation = gl.getUniformLocation(program, 'u_coverage');
  const outputBrightnessLocation = gl.getUniformLocation(program, 'u_outputBrightness');
  const outputGrayscaleLocation = gl.getUniformLocation(program, 'u_outputGrayscale');

  gl.useProgram(program);
  gl.uniform1i(inputFrameLocation, 0);
  gl.uniform1i(personMaskLocation, 1);
  gl.uniform2f(coverageLocation, 0.0, 1.0);
  gl.uniform1f(outputBrightnessLocation, 1.0);
  gl.uniform1f(outputGrayscaleLocation, 0.0);

  let outputBrightness = 1.0;
  let outputGrayscale = 0.0;
  let coverage = [0.0, 1.0];

  function render() {
    gl.viewport(0, 0, canvas.width, canvas.height);
    gl.useProgram(program);
    gl.uniform2f(coverageLocation, coverage[0], coverage[1]);
    gl.uniform1f(outputBrightnessLocation, outputBrightness);
    gl.uniform1f(outputGrayscaleLocation, outputGrayscale);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, personMaskTexture);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  }

  function updateCoverage(value) {
    coverage = value;
    gl.useProgram(program);
    gl.uniform2f(coverageLocation, coverage[0], coverage[1]);
  }

  function updateOutputAdjustments(brightness, grayscale) {
    outputBrightness = brightness;
    outputGrayscale = Math.max(0, Math.min(1, grayscale));
  }

  function cleanUp() {
    gl.deleteProgram(program);
    gl.deleteShader(fragmentShader);
    gl.deleteShader(vertexShader);
  }

  return {
    render,
    updateCoverage,
    updateOutputAdjustments,
    cleanUp,
  };
}

function buildBackgroundImageStage(
  gl,
  positionBuffer,
  texCoordBuffer,
  personMaskTexture,
  backgroundImage,
  canvas
) {
  const vertexShaderSource = _glsl`#version 300 es
  
      uniform vec2 u_backgroundScale;
      uniform vec2 u_backgroundOffset;
      uniform mat2 u_backgroundTransform;
      uniform vec2 u_backgroundAspect;
  
      in vec2 a_position;
      in vec2 a_texCoord;
  
      out vec2 v_texCoord;
      out vec2 v_backgroundCoord;
  
      void main() {
        gl_Position = vec4(a_position * vec2(1.0, -1.0), 0.0, 1.0);
        v_texCoord = a_texCoord;
        vec2 centered = a_texCoord * 2.0 - 1.0;
        centered *= u_backgroundAspect;
        centered = u_backgroundTransform * centered;
        centered /= u_backgroundAspect;
        centered = centered * 0.5 + 0.5;
        v_backgroundCoord = centered * u_backgroundScale + u_backgroundOffset;
      }
    `;

  const fragmentShaderSource = _glsl`#version 300 es
  
      precision highp float;
  
      uniform sampler2D u_inputFrame;
      uniform sampler2D u_personMask;
      uniform sampler2D u_background;
      uniform vec2 u_coverage;
      uniform float u_lightWrapping;
      uniform float u_blendMode;
      uniform float u_outputBrightness;
      uniform float u_outputGrayscale;
      uniform float u_blurStrength;
      uniform vec2 u_texelSize;
      uniform float u_spillSuppression;
      uniform float u_edgeDarkening;
  
      in vec2 v_texCoord;
      in vec2 v_backgroundCoord;
  
      out vec4 outColor;
  
      vec3 screen(vec3 a, vec3 b) {
        return 1.0 - (1.0 - a) * (1.0 - b);
      }
  
      vec3 linearDodge(vec3 a, vec3 b) {
        return a + b;
      }
  
      vec3 sampleBackground(vec2 coord) {
        return texture(u_background, coord).rgb;
      }
  
      vec3 blurBackground(vec2 coord) {
        float s = clamp(u_blurStrength, 0.0, 1.0);
        if (s <= 0.001) {
          return sampleBackground(coord);
        }
        const int maxRadius = 12;
        float radiusFloat = mix(1.0, float(maxRadius), s);
        int radius = int(radiusFloat);
        vec3 sum = vec3(0.0);
        float totalWeight = 0.0;
        for (int x = -maxRadius; x <= maxRadius; x++) {
          for (int y = -maxRadius; y <= maxRadius; y++) {
            if (abs(x) > radius || abs(y) > radius) {
              continue;
            }
            vec2 offset = vec2(float(x), float(y)) * u_texelSize * 2.0;
            float dist = float(x * x + y * y);
            float weight = exp(-dist / (2.0 * radiusFloat * radiusFloat + 1.0e-4));
            sum += sampleBackground(coord + offset) * weight;
            totalWeight += weight;
          }
        }
        return sum / max(totalWeight, 1.0e-4);
      }
  
      void main() {
        vec3 frameColor = texture(u_inputFrame, v_texCoord).rgb;
        vec3 backgroundColor = blurBackground(v_backgroundCoord);
        float rawMask = texture(u_personMask, v_texCoord).a;
        float lightWrapMask = 1.0 - max(0.0, rawMask - u_coverage.y) / (1.0 - u_coverage.y);
        vec3 lightWrap = u_lightWrapping * lightWrapMask * backgroundColor;
        frameColor = u_blendMode * linearDodge(frameColor, lightWrap) +
          (1.0 - u_blendMode) * screen(frameColor, lightWrap);
        float personMask = smoothstep(u_coverage.x, u_coverage.y, rawMask);
        float edge = clamp(1.0 - abs(personMask * 2.0 - 1.0), 0.0, 1.0);
        float frameGray = dot(frameColor, vec3(0.299, 0.587, 0.114));
        frameColor = mix(frameColor, vec3(frameGray), edge * clamp(u_spillSuppression, 0.0, 1.0));
        vec3 adjustedBackground = backgroundColor;
        float backgroundGray = dot(adjustedBackground, vec3(0.299, 0.587, 0.114));
        adjustedBackground = mix(adjustedBackground, vec3(backgroundGray), clamp(u_outputGrayscale, 0.0, 1.0));
        adjustedBackground *= u_outputBrightness;
        // [원본 유지] backgroundStage의 mix(backgroundColor, foreground, personMask) 그대로 사용
        vec3 finalColor = frameColor * personMask + adjustedBackground * (1.0 - personMask);
        finalColor *= 1.0 - edge * clamp(u_edgeDarkening, 0.0, 1.0) * 0.06;
        outColor = vec4(finalColor, 1.0);
      }
    `;

  const { width: outputWidth, height: outputHeight } = canvas;
  const outputRatio = outputWidth / outputHeight;

  const vertexShader = compileShader(gl, gl.VERTEX_SHADER, vertexShaderSource);
  const fragmentShader = compileShader(gl, gl.FRAGMENT_SHADER, fragmentShaderSource);
  const program = createPiplelineStageProgram(
    gl,
    vertexShader,
    fragmentShader,
    positionBuffer,
    texCoordBuffer
  );
  const backgroundScaleLocation = gl.getUniformLocation(program, 'u_backgroundScale');
  const backgroundOffsetLocation = gl.getUniformLocation(program, 'u_backgroundOffset');
  const backgroundTransformLocation = gl.getUniformLocation(program, 'u_backgroundTransform');
  const backgroundAspectLocation = gl.getUniformLocation(program, 'u_backgroundAspect');
  const inputFrameLocation = gl.getUniformLocation(program, 'u_inputFrame');
  const personMaskLocation = gl.getUniformLocation(program, 'u_personMask');
  const backgroundLocation = gl.getUniformLocation(program, 'u_background');
  const coverageLocation = gl.getUniformLocation(program, 'u_coverage');
  const lightWrappingLocation = gl.getUniformLocation(program, 'u_lightWrapping');
  const blendModeLocation = gl.getUniformLocation(program, 'u_blendMode');
  const outputBrightnessLocation = gl.getUniformLocation(program, 'u_outputBrightness');
  const outputGrayscaleLocation = gl.getUniformLocation(program, 'u_outputGrayscale');
  const blurStrengthLocation = gl.getUniformLocation(program, 'u_blurStrength');
  const texelSizeLocation = gl.getUniformLocation(program, 'u_texelSize');
  const spillSuppressionLocation = gl.getUniformLocation(program, 'u_spillSuppression');
  const edgeDarkeningLocation = gl.getUniformLocation(program, 'u_edgeDarkening');

  gl.useProgram(program);
  gl.uniform2f(backgroundScaleLocation, 1, 1);
  gl.uniform2f(backgroundOffsetLocation, 0, 0);
  gl.uniform1i(inputFrameLocation, 0);
  gl.uniform1i(personMaskLocation, 1);
  gl.uniform2f(coverageLocation, 0, 1);
  gl.uniform1f(lightWrappingLocation, 0);
  gl.uniform1f(blendModeLocation, 0);
  gl.uniform1f(outputBrightnessLocation, 1.0);
  gl.uniform1f(outputGrayscaleLocation, 0.0);
  gl.uniform1f(blurStrengthLocation, 0.0);
  gl.uniform2f(texelSizeLocation, 1 / outputWidth, 1 / outputHeight);
  gl.uniform1f(spillSuppressionLocation, 0.18);
  gl.uniform1f(edgeDarkeningLocation, 0.24);
  gl.uniform2f(backgroundAspectLocation, 1, 1);
  gl.uniformMatrix2fv(backgroundTransformLocation, false, _identityMatrix);

  let backgroundTexture = null;
  let blurStrength = 0.0;
  let baseScale = [1, 1];
  let baseOffset = [0, 0];
  let transformScaleMultiplier = 1.0;
  let mirrorSign = 1;
  let spillSuppression = 0.18;
  let edgeDarkening = 0.24;
  if (backgroundImage && backgroundImage.complete !== false) {
    updateBackgroundImage(backgroundImage);
  } else if (backgroundImage) {
    backgroundImage.onload = () => {
      updateBackgroundImage(backgroundImage);
    };
  } else {
    backgroundTexture = createTexture(gl, gl.RGBA8, 2, 2, gl.LINEAR, gl.LINEAR);
    gl.texSubImage2D(
      gl.TEXTURE_2D,
      0,
      0,
      0,
      2,
      2,
      gl.RGBA,
      gl.UNSIGNED_BYTE,
      new Uint8Array([0, 0, 0, 255, 0, 0, 0, 255, 0, 0, 0, 255, 0, 0, 0, 255])
    );
    baseScale = [1, 1];
    baseOffset = [0, 0];
  }
  applyScaleAndOffset();

  function applyScaleAndOffset() {
    const effectiveScaleX = baseScale[0] * transformScaleMultiplier;
    const effectiveScaleY = baseScale[1] * transformScaleMultiplier;
    const offsetX = baseOffset[0] + (baseScale[0] - effectiveScaleX) * 0.5;
    const offsetY = baseOffset[1] + (baseScale[1] - effectiveScaleY) * 0.5;
    gl.useProgram(program);
    gl.uniform2f(backgroundScaleLocation, effectiveScaleX, effectiveScaleY);
    gl.uniform2f(backgroundOffsetLocation, offsetX, offsetY);
    gl.uniform2f(texelSizeLocation, effectiveScaleX / outputWidth, effectiveScaleY / outputHeight);
    updateAspectComp();
  }

  function updateAspectComp() {
    const width = baseScale[0] || 0.0001;
    const height = baseScale[1] || 0.0001;
    let aspectX = 1;
    let aspectY = 1;
    if (width > height) {
      aspectX = width / height;
    } else if (height > width) {
      aspectY = height / width;
    }
    gl.useProgram(program);
    gl.uniform2f(backgroundAspectLocation, aspectX, aspectY);
  }

  function render() {
    gl.viewport(0, 0, outputWidth, outputHeight);
    gl.useProgram(program);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, personMaskTexture);
    gl.uniform1f(blurStrengthLocation, blurStrength);
    gl.uniform1f(spillSuppressionLocation, spillSuppression);
    gl.uniform1f(edgeDarkeningLocation, edgeDarkening);
    if (backgroundTexture !== null) {
      gl.activeTexture(gl.TEXTURE2);
      gl.bindTexture(gl.TEXTURE_2D, backgroundTexture);
      gl.uniform1i(backgroundLocation, 2);
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  }

  function updateBackgroundImage(backgroundImageElement) {
    backgroundTexture = createTexture(
      gl,
      gl.RGBA8,
      backgroundImageElement.naturalWidth || backgroundImageElement.width,
      backgroundImageElement.naturalHeight || backgroundImageElement.height,
      gl.LINEAR,
      gl.LINEAR
    );
    gl.texSubImage2D(
      gl.TEXTURE_2D,
      0,
      0,
      0,
      backgroundImageElement.naturalWidth || backgroundImageElement.width,
      backgroundImageElement.naturalHeight || backgroundImageElement.height,
      gl.RGBA,
      gl.UNSIGNED_BYTE,
      backgroundImageElement
    );

    const imageWidth = backgroundImageElement.naturalWidth || backgroundImageElement.width;
    const imageHeight = backgroundImageElement.naturalHeight || backgroundImageElement.height;
    const imageRatio = imageWidth / imageHeight;
    const canvasRatio = outputRatio;

    let scaleX = 1;
    let scaleY = 1;
    if (imageRatio > canvasRatio) {
      // Image is wider than canvas -> letterbox top/bottom.
      scaleY = canvasRatio / imageRatio;
    } else {
      // Image is taller than canvas -> pillarbox left/right.
      scaleX = imageRatio / canvasRatio;
    }

    baseScale = [scaleX, scaleY];
    baseOffset = [(1 - scaleX) * 0.5, (1 - scaleY) * 0.5];
    applyScaleAndOffset();
  }

  function updateCoverage(coverage) {
    gl.useProgram(program);
    gl.uniform2f(coverageLocation, coverage[0], coverage[1]);
  }

  function updateLightWrapping(lightWrapping) {
    gl.useProgram(program);
    gl.uniform1f(lightWrappingLocation, lightWrapping);
  }

  function updateBlendMode(blendMode) {
    gl.useProgram(program);
    gl.uniform1f(blendModeLocation, blendMode === 'screen' ? 0 : 1);
  }

  function updateOutputAdjustments(brightness, grayscale) {
    gl.useProgram(program);
    gl.uniform1f(outputBrightnessLocation, brightness);
    gl.uniform1f(outputGrayscaleLocation, Math.max(0, Math.min(1, grayscale)));
  }

  function updateBlurAmount(value) {
    blurStrength = Math.max(0, Math.min(1, value));
  }

  function updateEdgeConfig(edgeConfig) {
    if (!edgeConfig) return;
    if (typeof edgeConfig.spillSuppression === 'number') {
      spillSuppression = Math.max(0, Math.min(1, edgeConfig.spillSuppression));
    }
    if (typeof edgeConfig.edgeDarkening === 'number') {
      edgeDarkening = Math.max(0, Math.min(1, edgeConfig.edgeDarkening));
    }
  }

  function updateTransform(transform) {
    mirrorSign = transform.mirror ? -1 : 1;
    const rotation = transform.rotation || 0;
    const cosR = Math.cos(rotation);
    const sinR = Math.sin(rotation);
    const absCos = Math.abs(cosR);
    const absSin = Math.abs(sinR);
    let matrix = new Float32Array([cosR * mirrorSign, -sinR, sinR * mirrorSign, cosR]);

    const normalizedRotation = ((rotation % (2 * Math.PI)) + 2 * Math.PI) % (2 * Math.PI);
    const isQuarterTurn =
      Math.abs(normalizedRotation - Math.PI / 2) < 0.001 ||
      Math.abs(normalizedRotation - (3 * Math.PI) / 2) < 0.001;
    if (transform.mirror && isQuarterTurn) {
      matrix = new Float32Array([-matrix[0], -matrix[1], -matrix[2], -matrix[3]]);
    }

    const contentWidth = baseScale[0] * outputWidth;
    const contentHeight = baseScale[1] * outputHeight;
    const rotatedWidth = absCos * contentWidth + absSin * contentHeight;
    const rotatedHeight = absSin * contentWidth + absCos * contentHeight;
    let containScale = 1;
    if (rotatedWidth > 0 && rotatedHeight > 0) {
      const scaleToWidth = outputWidth / rotatedWidth;
      const scaleToHeight = outputHeight / rotatedHeight;
      containScale = Math.min(scaleToWidth, scaleToHeight);
    }
    transformScaleMultiplier = Math.min(containScale, 1);
    gl.useProgram(program);
    gl.uniformMatrix2fv(backgroundTransformLocation, false, matrix);
    applyScaleAndOffset();
  }

  function cleanUp() {
    if (backgroundTexture) {
      gl.deleteTexture(backgroundTexture);
    }
    gl.deleteProgram(program);
    gl.deleteShader(fragmentShader);
    gl.deleteShader(vertexShader);
  }

  return {
    render,
    updateCoverage,
    updateLightWrapping,
    updateBlendMode,
    updateOutputAdjustments,
    updateBlurAmount,
    updateEdgeConfig,
    updateTransform,
    cleanUp,
  };
}

function buildJointBilateralFilterStage(
  gl,
  vertexShader,
  positionBuffer,
  texCoordBuffer,
  inputTexture,
  segmentationConfig,
  outputTexture,
  canvas
) {
  const fragmentShaderSource = _glsl`#version 300 es
  
      precision highp float;
  
      uniform sampler2D u_inputFrame;
      uniform sampler2D u_segmentationMask;
      uniform vec2 u_texelSize;
      uniform float u_step;
      uniform float u_radius;
      uniform float u_offset;
      uniform float u_sigmaTexel;
      uniform float u_sigmaColor;
  
      in vec2 v_texCoord;
  
      out vec4 outColor;
  
      float gaussian(float x, float sigma) {
        float coeff = -0.5 / (sigma * sigma * 4.0 + 1.0e-6);
        return exp((x * x) * coeff);
      }
  
      void main() {
        vec2 centerCoord = v_texCoord;
        vec3 centerColor = texture(u_inputFrame, centerCoord).rgb;
        float newVal = 0.0;
  
        float spaceWeight = 0.0;
        float colorWeight = 0.0;
        float totalWeight = 0.0;
  
        for (float i = -u_radius + u_offset; i <= u_radius; i += u_step) {
          for (float j = -u_radius + u_offset; j <= u_radius; j += u_step) {
            vec2 shift = vec2(j, i) * u_texelSize;
            vec2 coord = vec2(centerCoord + shift);
            vec3 frameColor = texture(u_inputFrame, coord).rgb;
            float outVal = texture(u_segmentationMask, coord).a;
  
            spaceWeight = gaussian(distance(centerCoord, coord), u_sigmaTexel);
            colorWeight = gaussian(distance(centerColor, frameColor), u_sigmaColor);
            totalWeight += spaceWeight * colorWeight;
  
            newVal += spaceWeight * colorWeight * outVal;
          }
        }
        newVal /= totalWeight;
  
        outColor = vec4(vec3(0.0), newVal);
      }
    `;

  const resolution = inputResolutions[segmentationConfig.inputResolution];
  const segmentationWidth = resolution[0];
  const segmentationHeight = resolution[1];
  const { width: outputWidth, height: outputHeight } = canvas;
  const texelWidth = 1 / outputWidth;
  const texelHeight = 1 / outputHeight;

  const fragmentShader = compileShader(gl, gl.FRAGMENT_SHADER, fragmentShaderSource);
  const program = createPiplelineStageProgram(
    gl,
    vertexShader,
    fragmentShader,
    positionBuffer,
    texCoordBuffer
  );
  const inputFrameLocation = gl.getUniformLocation(program, 'u_inputFrame');
  const segmentationMaskLocation = gl.getUniformLocation(program, 'u_segmentationMask');
  const texelSizeLocation = gl.getUniformLocation(program, 'u_texelSize');
  const stepLocation = gl.getUniformLocation(program, 'u_step');
  const radiusLocation = gl.getUniformLocation(program, 'u_radius');
  const offsetLocation = gl.getUniformLocation(program, 'u_offset');
  const sigmaTexelLocation = gl.getUniformLocation(program, 'u_sigmaTexel');
  const sigmaColorLocation = gl.getUniformLocation(program, 'u_sigmaColor');

  const frameBuffer = gl.createFramebuffer();
  gl.bindFramebuffer(gl.FRAMEBUFFER, frameBuffer);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, outputTexture, 0);

  gl.useProgram(program);
  gl.uniform1i(inputFrameLocation, 0);
  gl.uniform1i(segmentationMaskLocation, 1);
  gl.uniform2f(texelSizeLocation, texelWidth, texelHeight);

  updateSigmaSpace(0);
  updateSigmaColor(0);

  function render() {
    gl.viewport(0, 0, outputWidth, outputHeight);
    gl.useProgram(program);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, inputTexture);
    gl.bindFramebuffer(gl.FRAMEBUFFER, frameBuffer);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  }

  function updateSigmaSpace(sigmaSpace) {
    sigmaSpace *= Math.max(outputWidth / segmentationWidth, outputHeight / segmentationHeight);

    const kSparsityFactor = 0.66;
    const sparsity = Math.max(1, Math.sqrt(sigmaSpace) * kSparsityFactor);
    const step = sparsity;
    const radius = sigmaSpace;
    const offset = step > 1 ? step * 0.5 : 0;
    const sigmaTexel = Math.max(texelWidth, texelHeight) * sigmaSpace;

    gl.useProgram(program);
    gl.uniform1f(stepLocation, step);
    gl.uniform1f(radiusLocation, radius);
    gl.uniform1f(offsetLocation, offset);
    gl.uniform1f(sigmaTexelLocation, sigmaTexel);
  }

  function updateSigmaColor(sigmaColor) {
    gl.useProgram(program);
    gl.uniform1f(sigmaColorLocation, sigmaColor);
  }

  function cleanUp() {
    gl.deleteFramebuffer(frameBuffer);
    gl.deleteProgram(program);
    gl.deleteShader(fragmentShader);
  }

  return {
    render,
    cleanUp,
    updateSigmaSpace, // 이 줄이 있어야 외부에서 호출 가능
    updateSigmaColor,
  };
}

function buildMaskPostProcessStage(
  gl,
  vertexShader,
  positionBuffer,
  texCoordBuffer,
  inputMaskTexture,
  canvas
) {
  const blurFragmentShaderSource = _glsl`#version 300 es
  
      precision highp float;
  
      uniform sampler2D u_inputMask;
      uniform vec2 u_direction;
  
      in vec2 v_texCoord;
  
      out vec4 outColor;
  
      float sampledMask(sampler2D tex, vec2 uv) {
        vec4 m = texture(tex, uv);
        float colorMask = max(max(m.r, m.g), m.b);
        return max(colorMask, m.a);
      }
  
      void main() {
        float c = sampledMask(u_inputMask, v_texCoord) * 0.227027;
        c += sampledMask(u_inputMask, v_texCoord + u_direction * 1.384615) * 0.316216;
        c += sampledMask(u_inputMask, v_texCoord - u_direction * 1.384615) * 0.316216;
        c += sampledMask(u_inputMask, v_texCoord + u_direction * 3.230769) * 0.070270;
        c += sampledMask(u_inputMask, v_texCoord - u_direction * 3.230769) * 0.070270;
        outColor = vec4(0.0, 0.0, 0.0, clamp(c, 0.0, 1.0));
      }
    `;

  const refineFragmentShaderSource = _glsl`#version 300 es
  
      precision highp float;
  
      uniform sampler2D u_blurredMask;
      uniform sampler2D u_rawMask;
      uniform vec2 u_texelSize;
      uniform float u_edgeBlend;
      uniform float u_edgeGamma;
      uniform float u_edgeFeather;
  
      in vec2 v_texCoord;
  
      out vec4 outColor;
  
      float sampledMask(sampler2D tex, vec2 uv) {
        vec4 m = texture(tex, uv);
        float colorMask = max(max(m.r, m.g), m.b);
        return max(colorMask, m.a);
      }
  
      void main() {
        float blurred = sampledMask(u_blurredMask, v_texCoord);
        float raw = sampledMask(u_rawMask, v_texCoord);
        float px = sampledMask(u_blurredMask, v_texCoord + vec2(u_texelSize.x, 0.0));
        float nx = sampledMask(u_blurredMask, v_texCoord - vec2(u_texelSize.x, 0.0));
        float py = sampledMask(u_blurredMask, v_texCoord + vec2(0.0, u_texelSize.y));
        float ny = sampledMask(u_blurredMask, v_texCoord - vec2(0.0, u_texelSize.y));
        float edge = clamp(abs(px - nx) + abs(py - ny), 0.0, 1.0);
        float edgeMix = clamp(u_edgeBlend * edge, 0.0, 1.0);
        float refined = mix(blurred, raw, edgeMix);
        refined = mix(refined, smoothstep(0.0, 1.0, refined), clamp(u_edgeFeather, 0.0, 1.0));
        refined = pow(clamp(refined, 0.0, 1.0), max(0.5, u_edgeGamma));
        outColor = vec4(0.0, 0.0, 0.0, refined);
      }
    `;

  const outputWidth = canvas.width;
  const outputHeight = canvas.height;
  const blurFragmentShader = compileShader(gl, gl.FRAGMENT_SHADER, blurFragmentShaderSource);
  const refineFragmentShader = compileShader(gl, gl.FRAGMENT_SHADER, refineFragmentShaderSource);
  const blurProgram = createPiplelineStageProgram(
    gl,
    vertexShader,
    blurFragmentShader,
    positionBuffer,
    texCoordBuffer
  );
  const refineProgram = createPiplelineStageProgram(
    gl,
    vertexShader,
    refineFragmentShader,
    positionBuffer,
    texCoordBuffer
  );

  const blurInputLocation = gl.getUniformLocation(blurProgram, 'u_inputMask');
  const blurDirectionLocation = gl.getUniformLocation(blurProgram, 'u_direction');
  const refineBlurredLocation = gl.getUniformLocation(refineProgram, 'u_blurredMask');
  const refineRawLocation = gl.getUniformLocation(refineProgram, 'u_rawMask');
  const refineTexelSizeLocation = gl.getUniformLocation(refineProgram, 'u_texelSize');
  const refineEdgeBlendLocation = gl.getUniformLocation(refineProgram, 'u_edgeBlend');
  const refineEdgeGammaLocation = gl.getUniformLocation(refineProgram, 'u_edgeGamma');
  const refineEdgeFeatherLocation = gl.getUniformLocation(refineProgram, 'u_edgeFeather');

  const blurTempTexture = createTexture(
    gl,
    gl.RGBA8,
    outputWidth,
    outputHeight,
    gl.LINEAR,
    gl.LINEAR
  );
  const blurredTexture = createTexture(
    gl,
    gl.RGBA8,
    outputWidth,
    outputHeight,
    gl.LINEAR,
    gl.LINEAR
  );
  const outputTexture = createTexture(
    gl,
    gl.RGBA8,
    outputWidth,
    outputHeight,
    gl.LINEAR,
    gl.LINEAR
  );

  const blurTempFbo = gl.createFramebuffer();
  gl.bindFramebuffer(gl.FRAMEBUFFER, blurTempFbo);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, blurTempTexture, 0);

  const blurredFbo = gl.createFramebuffer();
  gl.bindFramebuffer(gl.FRAMEBUFFER, blurredFbo);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, blurredTexture, 0);

  const outputFbo = gl.createFramebuffer();
  gl.bindFramebuffer(gl.FRAMEBUFFER, outputFbo);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, outputTexture, 0);

  gl.useProgram(blurProgram);
  gl.uniform1i(blurInputLocation, 1);

  gl.useProgram(refineProgram);
  gl.uniform1i(refineBlurredLocation, 1);
  gl.uniform1i(refineRawLocation, 2);
  gl.uniform2f(refineTexelSizeLocation, 1 / outputWidth, 1 / outputHeight);

  let maskBlurPx = 1.1;
  let edgeBlend = 0.36;
  let edgeGamma = 0.98;
  let edgeFeather = 0.54;

  function render() {
    const strength = Math.max(1.0, maskBlurPx);
    const dirX = strength / Math.max(1, outputWidth);
    const dirY = strength / Math.max(1, outputHeight);

    gl.viewport(0, 0, outputWidth, outputHeight);

    gl.useProgram(blurProgram);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, inputMaskTexture);
    gl.uniform2f(blurDirectionLocation, dirX, 0);
    gl.bindFramebuffer(gl.FRAMEBUFFER, blurTempFbo);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, blurTempTexture);
    gl.uniform2f(blurDirectionLocation, 0, dirY);
    gl.bindFramebuffer(gl.FRAMEBUFFER, blurredFbo);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

    gl.useProgram(refineProgram);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, blurredTexture);
    gl.activeTexture(gl.TEXTURE2);
    gl.bindTexture(gl.TEXTURE_2D, inputMaskTexture);
    gl.uniform1f(refineEdgeBlendLocation, edgeBlend);
    gl.uniform1f(refineEdgeGammaLocation, edgeGamma);
    gl.uniform1f(refineEdgeFeatherLocation, edgeFeather);
    gl.bindFramebuffer(gl.FRAMEBUFFER, outputFbo);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  }

  function updateMaskRefineConfig(maskRefineConfig) {
    if (!maskRefineConfig) return;
    if (typeof maskRefineConfig.maskBlurPx === 'number') {
      maskBlurPx = Math.max(0.5, Math.min(4.0, maskRefineConfig.maskBlurPx));
    }
    if (typeof maskRefineConfig.edgeBlend === 'number') {
      edgeBlend = Math.max(0.0, Math.min(1.0, maskRefineConfig.edgeBlend));
    }
    if (typeof maskRefineConfig.edgeGamma === 'number') {
      edgeGamma = Math.max(0.5, Math.min(1.5, maskRefineConfig.edgeGamma));
    }
    if (typeof maskRefineConfig.edgeFeather === 'number') {
      edgeFeather = Math.max(0.0, Math.min(1.0, maskRefineConfig.edgeFeather));
    }
  }

  function getOutputTexture() {
    return outputTexture;
  }

  function cleanUp() {
    gl.deleteFramebuffer(outputFbo);
    gl.deleteFramebuffer(blurredFbo);
    gl.deleteFramebuffer(blurTempFbo);
    gl.deleteTexture(outputTexture);
    gl.deleteTexture(blurredTexture);
    gl.deleteTexture(blurTempTexture);
    gl.deleteProgram(refineProgram);
    gl.deleteProgram(blurProgram);
    gl.deleteShader(refineFragmentShader);
    gl.deleteShader(blurFragmentShader);
  }

  return {
    render,
    updateMaskRefineConfig,
    getOutputTexture,
    cleanUp,
  };
}

function buildLoadSegmentationStage(
  gl,
  vertexShader,
  positionBuffer,
  texCoordBuffer,
  segmentationConfig,
  tflite,
  outputTexture
) {
  const fragmentShaderSource = _glsl`#version 300 es
      precision highp float;
      uniform sampler2D u_inputSegmentation;
      in vec2 v_texCoord;
      out vec4 outColor;
      void main() {
        // Y축 뒤집어서 마스크 읽기
        float segmentation = texture(u_inputSegmentation, vec2(v_texCoord.x, v_texCoord.y)).r;
        outColor = vec4(vec3(0.0), segmentation);
      }
    `;

  const tfliteOutputMemoryOffset = tflite._getOutputMemoryOffset() / 4;
  const resolution = inputResolutions[segmentationConfig.inputResolution];
  const segmentationWidth = resolution[0];
  const segmentationHeight = resolution[1];

  const fragmentShader = compileShader(gl, gl.FRAGMENT_SHADER, fragmentShaderSource);
  const program = createPiplelineStageProgram(
    gl,
    vertexShader,
    fragmentShader,
    positionBuffer,
    texCoordBuffer
  );
  const inputLocation = gl.getUniformLocation(program, 'u_inputSegmentation');
  const inputTexture = createTexture(gl, gl.R32F, segmentationWidth, segmentationHeight);

  const frameBuffer = gl.createFramebuffer();
  gl.bindFramebuffer(gl.FRAMEBUFFER, frameBuffer);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, outputTexture, 0);

  gl.useProgram(program);
  gl.uniform1i(inputLocation, 1);

  // 이전 프레임 저장용 버퍼
  let previousMask = new Float32Array(segmentationWidth * segmentationHeight);

  function render() {
    gl.viewport(0, 0, segmentationWidth, segmentationHeight);
    gl.useProgram(program);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, inputTexture);

    const size = segmentationWidth * segmentationHeight;
    for (let i = 0; i < size; i++) {
      let curr = tflite.HEAPF32[tfliteOutputMemoryOffset + i];
      let prev = previousMask[i];

      // 현재 프레임과 이전 프레임의 픽셀 값 차이 계산
      let diff = Math.abs(curr - prev);

      // [핵심 로직] 동적 알파(Alpha) 값 계산
      // 차이가 0.15보다 크면 움직임으로 간주 -> 알파 0.8 (잔상 억제, 즉각 반응)
      // 차이가 0.15 이하면 노이즈로 간주 -> 알파 0.15 (강한 스무딩, 깜빡임 억제)
      let alpha = diff > 0.3 ? 0.7 : 0.3;

      let smoothed = curr * alpha + prev * (1.0 - alpha);

      tflite.HEAPF32[tfliteOutputMemoryOffset + i] = smoothed;
      previousMask[i] = smoothed;
    }

    gl.texSubImage2D(
      gl.TEXTURE_2D,
      0,
      0,
      0,
      segmentationWidth,
      segmentationHeight,
      gl.RED,
      gl.FLOAT,
      tflite.HEAPF32,
      tfliteOutputMemoryOffset
    );
    gl.bindFramebuffer(gl.FRAMEBUFFER, frameBuffer);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  }

  function cleanUp() {
    gl.deleteFramebuffer(frameBuffer);
    gl.deleteTexture(inputTexture);
    gl.deleteProgram(program);
    gl.deleteShader(fragmentShader);
  }

  return { render, cleanUp };
}

function buildSoftmaxStage(
  gl,
  vertexShader,
  positionBuffer,
  texCoordBuffer,
  segmentationConfig,
  tflite,
  outputTexture
) {
  const fragmentShaderSource = _glsl`#version 300 es
        precision highp float;
        uniform sampler2D u_inputSegmentation;
        in vec2 v_texCoord;
        out vec4 outColor;
        void main() {
          // 맨 처음 대화하셨던 Y축 반전이 들어있는 순정 상태입니다.
          // 마스크가 뒤집혀 나온다면 '1.0 - v_texCoord.y'를 'v_texCoord.y'로 바꾸세요.
          vec2 segmentation = texture(u_inputSegmentation, vec2(v_texCoord.x, 1.0 - v_texCoord.y)).rg;
          
          // Softmax 공식 계산 부분
          float shift = max(segmentation.r, segmentation.g);
          float backgroundExp = exp(segmentation.r - shift);
          float personExp = exp(segmentation.g - shift);
          
          // 최종적으로 사람일 확률을 계산해서 Alpha 채널에 넣음
          outColor = vec4(vec3(0.0), personExp / (backgroundExp + personExp));
        }
      `;

  const tfliteOutputMemoryOffset = tflite._getOutputMemoryOffset() / 4;
  const resolution = inputResolutions[segmentationConfig.inputResolution];
  const segmentationWidth = resolution[0];
  const segmentationHeight = resolution[1];

  const fragmentShader = compileShader(gl, gl.FRAGMENT_SHADER, fragmentShaderSource);
  const program = createPiplelineStageProgram(
    gl,
    vertexShader,
    fragmentShader,
    positionBuffer,
    texCoordBuffer
  );

  const inputLocation = gl.getUniformLocation(program, 'u_inputSegmentation');
  const inputTexture = createTexture(gl, gl.RG32F, segmentationWidth, segmentationHeight);

  const frameBuffer = gl.createFramebuffer();
  gl.bindFramebuffer(gl.FRAMEBUFFER, frameBuffer);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, outputTexture, 0);

  gl.useProgram(program);
  gl.uniform1i(inputLocation, 1);

  function render() {
    gl.viewport(0, 0, segmentationWidth, segmentationHeight);
    gl.useProgram(program);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, inputTexture);

    // TFLite 메모리에서 배열 값을 그대로 텍스처로 밀어넣음 (for문 없음)
    gl.texSubImage2D(
      gl.TEXTURE_2D,
      0,
      0,
      0,
      segmentationWidth,
      segmentationHeight,
      gl.RG,
      gl.FLOAT,
      tflite.HEAPF32,
      tfliteOutputMemoryOffset
    );

    gl.bindFramebuffer(gl.FRAMEBUFFER, frameBuffer);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  }

  function cleanUp() {
    gl.deleteFramebuffer(frameBuffer);
    gl.deleteTexture(inputTexture);
    gl.deleteProgram(program);
    gl.deleteShader(fragmentShader);
  }

  return { render, cleanUp };
}

function buildSoftmaxStage(
  gl,
  vertexShader,
  positionBuffer,
  texCoordBuffer,
  segmentationConfig,
  tflite,
  outputTexture
) {
  const fragmentShaderSource = _glsl`#version 300 es
          precision highp float;
          uniform sampler2D u_inputSegmentation;
          uniform sampler2D u_historyMask;
          in vec2 v_texCoord;
          out vec4 outColor;
  
          uniform float u_diffThreshold; 
          uniform float u_minAlpha;      
          uniform float u_maxAlpha;      
  
          void main() {
            vec2 segmentation = texture(u_inputSegmentation, vec2(v_texCoord.x, v_texCoord.y)).rg;
            float shift = max(segmentation.r, segmentation.g);
            float backgroundExp = exp(segmentation.r - shift);
            float personExp = exp(segmentation.g - shift);
            float currProb = personExp / (backgroundExp + personExp);
  
            float prevProb = texture(u_historyMask, v_texCoord).a;
  
            float diff = abs(currProb - prevProb);
            // 여기서 값이 0이면 alpha가 0이 되어 마스크가 투명해짐
            float alpha = diff > u_diffThreshold ? u_maxAlpha : u_minAlpha;
          
            float smoothedProb = mix(prevProb, currProb, alpha); 
  
            outColor = vec4(vec3(0.0), smoothedProb);
          }
        `;

  const tfliteOutputMemoryOffset = tflite._getOutputMemoryOffset() / 4;
  const resolution = inputResolutions[segmentationConfig.inputResolution];
  const segmentationWidth = resolution[0];
  const segmentationHeight = resolution[1];

  const fragmentShader = compileShader(gl, gl.FRAGMENT_SHADER, fragmentShaderSource);
  const program = createPiplelineStageProgram(
    gl,
    vertexShader,
    fragmentShader,
    positionBuffer,
    texCoordBuffer
  );

  const inputLocation = gl.getUniformLocation(program, 'u_inputSegmentation');
  const historyLocation = gl.getUniformLocation(program, 'u_historyMask');

  // 유니폼 위치 찾기
  const diffLoc = gl.getUniformLocation(program, 'u_diffThreshold');
  const minAlphaLoc = gl.getUniformLocation(program, 'u_minAlpha');
  const maxAlphaLoc = gl.getUniformLocation(program, 'u_maxAlpha');

  const inputTexture = createTexture(gl, gl.RG32F, segmentationWidth, segmentationHeight);
  const historyTexture = createTexture(gl, gl.RGBA8, segmentationWidth, segmentationHeight);

  const frameBuffer = gl.createFramebuffer();
  gl.bindFramebuffer(gl.FRAMEBUFFER, frameBuffer);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, outputTexture, 0);

  // [중요] 초기값 설정 - 슬라이더 움직이기 전에도 마스크가 보여야 함
  gl.useProgram(program);
  gl.uniform1i(inputLocation, 1);
  gl.uniform1i(historyLocation, 2);
  gl.uniform1f(diffLoc, 0.4); // 초기값 0.4
  gl.uniform1f(minAlphaLoc, 0.1); // 초기값 0.1
  gl.uniform1f(maxAlphaLoc, 0.9); // 초기값 0.9

  function render() {
    gl.viewport(0, 0, segmentationWidth, segmentationHeight);
    gl.useProgram(program);

    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, inputTexture);
    gl.texSubImage2D(
      gl.TEXTURE_2D,
      0,
      0,
      0,
      segmentationWidth,
      segmentationHeight,
      gl.RG,
      gl.FLOAT,
      tflite.HEAPF32,
      tfliteOutputMemoryOffset
    );

    gl.activeTexture(gl.TEXTURE2);
    gl.bindTexture(gl.TEXTURE_2D, historyTexture);

    gl.bindFramebuffer(gl.FRAMEBUFFER, frameBuffer);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

    gl.bindTexture(gl.TEXTURE_2D, historyTexture);
    gl.copyTexSubImage2D(gl.TEXTURE_2D, 0, 0, 0, 0, 0, segmentationWidth, segmentationHeight);
  }

  function cleanUp() {
    gl.deleteFramebuffer(frameBuffer);
    gl.deleteTexture(inputTexture);
    gl.deleteTexture(historyTexture);
    gl.deleteProgram(program);
    gl.deleteShader(fragmentShader);
  }

  return {
    render,
    cleanUp,
    updateAlphaParams: (diff, minA, maxA) => {
      gl.useProgram(program);
      gl.uniform1f(diffLoc, diff);
      gl.uniform1f(minAlphaLoc, minA);
      gl.uniform1f(maxAlphaLoc, maxA);
    },
  };
}

function buildResizingStage(
  timerWorker,
  gl,
  vertexShader,
  positionBuffer,
  texCoordBuffer,
  segmentationConfig,
  tflite
) {
  const fragmentShaderSource = _glsl`#version 300 es
  
      precision highp float;
  
      uniform sampler2D u_inputFrame;
  
      in vec2 v_texCoord;
  
      out vec4 outColor;
  
      void main() {
        outColor = texture(u_inputFrame, v_texCoord);
      }
    `;

  const tfliteInputMemoryOffset = tflite._getInputMemoryOffset() / 4;

  const resolution = inputResolutions[segmentationConfig.inputResolution];
  const outputWidth = resolution[0];
  const outputHeight = resolution[1];
  const outputPixelCount = outputWidth * outputHeight;

  const fragmentShader = compileShader(gl, gl.FRAGMENT_SHADER, fragmentShaderSource);
  const program = createPiplelineStageProgram(
    gl,
    vertexShader,
    fragmentShader,
    positionBuffer,
    texCoordBuffer
  );
  const inputFrameLocation = gl.getUniformLocation(program, 'u_inputFrame');
  const outputTexture = createTexture(gl, gl.RGBA8, outputWidth, outputHeight);

  const frameBuffer = gl.createFramebuffer();
  gl.bindFramebuffer(gl.FRAMEBUFFER, frameBuffer);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, outputTexture, 0);
  const outputPixels = new Uint8Array(outputPixelCount * 4);

  gl.useProgram(program);
  gl.uniform1i(inputFrameLocation, 0);

  async function render() {
    gl.viewport(0, 0, outputWidth, outputHeight);
    gl.useProgram(program);
    gl.bindFramebuffer(gl.FRAMEBUFFER, frameBuffer);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

    const readPixelsPromise = readPixelsAsync(
      timerWorker,
      gl,
      0,
      0,
      outputWidth,
      outputHeight,
      gl.RGBA,
      gl.UNSIGNED_BYTE,
      outputPixels
    );

    if (!segmentationConfig.deferInputResizing) {
      await readPixelsPromise;
    }

    for (let i = 0; i < outputPixelCount; i++) {
      const tfliteIndex = tfliteInputMemoryOffset + i * 3;
      const outputIndex = i * 4;
      tflite.HEAPF32[tfliteIndex] = outputPixels[outputIndex] / 255;
      tflite.HEAPF32[tfliteIndex + 1] = outputPixels[outputIndex + 1] / 255;
      tflite.HEAPF32[tfliteIndex + 2] = outputPixels[outputIndex + 2] / 255;
    }
  }

  function cleanUp() {
    gl.deleteFramebuffer(frameBuffer);
    gl.deleteTexture(outputTexture);
    gl.deleteProgram(program);
    gl.deleteShader(fragmentShader);
  }

  return { render, cleanUp };
}

function buildWebGL2Pipeline(
  sourcePlayback,
  backgroundImage,
  backgroundConfig,
  segmentationConfig,
  canvas,
  tflite,
  timerWorker,
  addFrameEvent
) {
  const vertexShaderSource = _glsl`#version 300 es
  
      in vec2 a_position;
      in vec2 a_texCoord;
  
      out vec2 v_texCoord;
  
      void main() {
        gl_Position = vec4(a_position, 0.0, 1.0);
        v_texCoord = a_texCoord;
      }
    `;

  const frameWidth = sourcePlayback.width;
  const frameHeight = sourcePlayback.height;
  const resolution = inputResolutions[segmentationConfig.inputResolution];
  const segmentationWidth = resolution[0];
  const segmentationHeight = resolution[1];

  const gl =
    canvas.getContext('webgl2', {
      preserveDrawingBuffer: false,
      powerPreference: 'high-performance',
    }) || canvas.getContext('experimental-webgl2');
  if (!gl) {
    throw new Error('WebGL2 not available');
  }

  const vertexShader = compileShader(gl, gl.VERTEX_SHADER, vertexShaderSource);

  const vertexArray = gl.createVertexArray();
  gl.bindVertexArray(vertexArray);

  const positionBuffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
  gl.bufferData(
    gl.ARRAY_BUFFER,
    new Float32Array([-1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0]),
    gl.STATIC_DRAW
  );

  const texCoordBuffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, texCoordBuffer);
  gl.bufferData(
    gl.ARRAY_BUFFER,
    new Float32Array([0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0]),
    gl.STATIC_DRAW
  );

  const inputFrameTexture = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, inputFrameTexture);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);

  const segmentationTexture = createTexture(gl, gl.RGBA8, segmentationWidth, segmentationHeight);
  const personMaskTexture = createTexture(gl, gl.RGBA8, frameWidth, frameHeight);

  const resizingStage = buildResizingStage(
    timerWorker,
    gl,
    vertexShader,
    positionBuffer,
    texCoordBuffer,
    segmentationConfig,
    tflite
  );
  const loadSegmentationStage =
    segmentationConfig.model === 'meet'
      ? buildSoftmaxStage(
          gl,
          vertexShader,
          positionBuffer,
          texCoordBuffer,
          segmentationConfig,
          tflite,
          segmentationTexture
        )
      : buildLoadSegmentationStage(
          gl,
          vertexShader,
          positionBuffer,
          texCoordBuffer,
          segmentationConfig,
          tflite,
          segmentationTexture
        );
  const jointBilateralFilterStage = buildJointBilateralFilterStage(
    gl,
    vertexShader,
    positionBuffer,
    texCoordBuffer,
    segmentationTexture,
    segmentationConfig,
    personMaskTexture,
    canvas
  );
  const maskPostProcessStage = buildMaskPostProcessStage(
    gl,
    vertexShader,
    positionBuffer,
    texCoordBuffer,
    personMaskTexture,
    canvas
  );
  const refinedMaskTexture = maskPostProcessStage.getOutputTexture();
  const backgroundStage =
    backgroundConfig.type === 'blur'
      ? buildBackgroundBlurStage(
          gl,
          vertexShader,
          positionBuffer,
          texCoordBuffer,
          refinedMaskTexture,
          canvas
        )
      : backgroundConfig.type === 'image'
        ? buildBackgroundImageStage(
            gl,
            positionBuffer,
            texCoordBuffer,
            refinedMaskTexture,
            backgroundImage,
            canvas
          )
        : buildPassthroughStage(gl, positionBuffer, texCoordBuffer, refinedMaskTexture, canvas);

  async function render() {
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, inputFrameTexture);

    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, sourcePlayback.htmlElement);

    gl.bindVertexArray(vertexArray);

    await resizingStage.render();

    if (addFrameEvent) addFrameEvent();

    tflite._runInference();

    if (addFrameEvent) addFrameEvent();

    loadSegmentationStage.render();
    jointBilateralFilterStage.render();
    maskPostProcessStage.render();
    backgroundStage.render();
  }

  function updatePostProcessingConfig(postProcessingConfig) {
    jointBilateralFilterStage.updateSigmaSpace(
      postProcessingConfig.jointBilateralFilter.sigmaSpace
    );
    jointBilateralFilterStage.updateSigmaColor(
      postProcessingConfig.jointBilateralFilter.sigmaColor
    );
    if (maskPostProcessStage.updateMaskRefineConfig) {
      maskPostProcessStage.updateMaskRefineConfig(postProcessingConfig.maskRefine);
    }

    if (backgroundStage.updateCoverage) {
      backgroundStage.updateCoverage(postProcessingConfig.coverage);
    }
    if (backgroundStage.updateLightWrapping) {
      backgroundStage.updateLightWrapping(postProcessingConfig.lightWrapping);
    }
    if (backgroundStage.updateBlendMode) {
      backgroundStage.updateBlendMode(postProcessingConfig.blendMode);
    }
    if (backgroundStage.updateBlurAmount) {
      backgroundStage.updateBlurAmount(postProcessingConfig.blurStrength);
    }
    if (backgroundStage.updateEdgeConfig) {
      backgroundStage.updateEdgeConfig(postProcessingConfig.compositeEdge);
    }
  }

  function updateOutputAdjustments(brightness, grayscale) {
    if (backgroundStage.updateOutputAdjustments) {
      backgroundStage.updateOutputAdjustments(brightness, grayscale);
    }
  }

  function updateAlphaParams(diff, minA, maxA) {
    // segmentationConfig.model === 'meet'일 때 생성된 loadSegmentationStage가 buildSoftmaxStage임
    if (loadSegmentationStage && loadSegmentationStage.updateAlphaParams) {
      loadSegmentationStage.updateAlphaParams(diff, minA, maxA);
    }
  }

  function updateBilateralParams(sigmaSpace, sigmaColor) {
    if (jointBilateralFilterStage) {
      // buildJointBilateralFilterStage가 리턴한 내부 함수를 직접 호출
      jointBilateralFilterStage.updateSigmaSpace(sigmaSpace);
      jointBilateralFilterStage.updateSigmaColor(sigmaColor);
    }
  }

  function updateTransform(transform) {
    if (backgroundStage.updateTransform) {
      backgroundStage.updateTransform(transform);
    }
  }

  function cleanUp() {
    if (backgroundStage && backgroundStage.cleanUp) backgroundStage.cleanUp();
    maskPostProcessStage.cleanUp();
    jointBilateralFilterStage.cleanUp();
    loadSegmentationStage.cleanUp();
    resizingStage.cleanUp();
    gl.deleteTexture(personMaskTexture);
    gl.deleteTexture(segmentationTexture);
    gl.deleteTexture(inputFrameTexture);
    gl.deleteBuffer(texCoordBuffer);
    gl.deleteBuffer(positionBuffer);
    gl.deleteVertexArray(vertexArray);
    gl.deleteShader(vertexShader);
  }

  return {
    render,
    updatePostProcessingConfig,
    updateOutputAdjustments,
    updateTransform,
    updateAlphaParams,
    updateBilateralParams,
    cleanUp,
  };
}

window.createCustomVideoStream = createCustomVideoStream;
window.destroyCustomVideoStream = destroyCustomVideoStream;
window.configCustomVideoStream = configCustomVideoStream;
