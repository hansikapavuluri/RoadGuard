
import { useEffect, useRef, useState } from "react";
import "./App.css";

const API_URL = "http://127.0.0.1:5000";

function getVideoUrl(data) {
  const rawUrl =
    data?.video_url ||
    data?.processed_video_url ||
    data?.result_video_url ||
    data?.results?.video_url ||
    data?.results?.processed_video_url ||
    data?.results?.result_video_url ||
    "";

  if (!rawUrl) return "";

  try {
    return new URL(rawUrl, `${API_URL}/`).href;
  } catch {
    return "";
  }
}

function getCsvUrl(data) {
  const rawUrl =
    data?.csv_url ||
    data?.detections_csv_url ||
    data?.output_csv_url ||
    data?.results?.csv_url ||
    data?.results?.detections_csv_url ||
    data?.results?.output_csv_url ||
    "";

  if (!rawUrl) return "";

  try {
    return new URL(rawUrl, `${API_URL}/`).href;
  } catch {
    return "";
  }
}

function parseCSV(text) {
  const lines = text.replace(/^\uFEFF/, "").split(/\r?\n/).filter((line) => line.trim());
  if (lines.length < 2) return [];

  const parseLine = (line) => {
    const values = [];
    let value = "";
    let quoted = false;

    for (let i = 0; i < line.length; i += 1) {
      const char = line[i];
      if (char === '"') {
        if (quoted && line[i + 1] === '"') {
          value += '"';
          i += 1;
        } else {
          quoted = !quoted;
        }
      } else if (char === "," && !quoted) {
        values.push(value);
        value = "";
      } else {
        value += char;
      }
    }
    values.push(value);
    return values;
  };

  const headers = parseLine(lines[0]).map((header) => header.trim());
  return lines.slice(1).map((line) => {
    const values = parseLine(line);
    const row = {};
    headers.forEach((header, index) => {
      row[header] = (values[index] ?? "").trim();
    });
    return row;
  });
}

function numericValue(value) {
  if (value === undefined || value === null || value === "") return null;
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

function truthyCSV(value) {
  return ["true", "1", "yes", "y"].includes(String(value ?? "").trim().toLowerCase());
}

function App() {
  const [page, setPage] = useState("drive");
  const [backendReady, setBackendReady] = useState(false);
  const [selectedFile, setSelectedFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [videoError, setVideoError] = useState("");
  const [result, setResult] = useState(null);
  const [videoUrl, setVideoUrl] = useState("");
  const [history, setHistory] = useState([]);
  const videoRef = useRef(null);
  const [frameMetrics, setFrameMetrics] = useState(null);
  const [frameData, setFrameData] = useState([]);
  const [csvLoading, setCsvLoading] = useState(false);

  useEffect(() => {
    let active = true;

    const checkBackend = async () => {
      try {
        const response = await fetch(`${API_URL}/api/health`);
        if (active) setBackendReady(response.ok);
      } catch {
        if (active) setBackendReady(false);
      }
    };

    checkBackend();

    const interval = window.setInterval(checkBackend, 10000);

    return () => {
      active = false;
      window.clearInterval(interval);
    };
  }, []);

  const handleUpload = async (event) => {
    event.preventDefault();

    if (!selectedFile) {
      setError("Please select a road video first.");
      return;
    }

    if (!backendReady) {
      setError(
        "The detection service is unavailable. Start the Flask backend and try again."
      );
      return;
    }

    setLoading(true);
    setError("");
    setVideoError("");
    setResult(null);
    setVideoUrl("");
    setFrameMetrics(null);
    setFrameData([]);
    setCsvLoading(false);
    setPage("drive");

    try {
      const formData = new FormData();
      formData.append("video", selectedFile);

      const response = await fetch(`${API_URL}/api/detect/video`, {
        method: "POST",
        body: formData,
      });

      const contentType = response.headers.get("content-type") || "";

      if (!contentType.includes("application/json")) {
        throw new Error(
          "The backend did not return JSON. Check the Flask terminal for errors."
        );
      }

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          data.error || data.message || "Video processing failed."
        );
      }

      const processedUrl = getVideoUrl(data);

      if (!processedUrl) {
        throw new Error(
          "Processing finished, but the backend did not return a video URL. Check backend/app.py."
        );
      }

      setResult(data);
      setVideoUrl(processedUrl);

      let loadedFrameData = [];
      const csvUrl = getCsvUrl(data);
      if (csvUrl) {
        setCsvLoading(true);
        try {
          const csvResponse = await fetch(csvUrl);
          if (!csvResponse.ok) {
            throw new Error(`CSV request failed (${csvResponse.status}).`);
          }

          const rows = parseCSV(await csvResponse.text());
          loadedFrameData = rows
            .filter((row) => row.object_type === "frame_summary")
            .map((row) => ({
              timestamp: numericValue(row.timestamp_seconds) ?? 0,
              potholes: numericValue(row.potholes_in_frame) ?? 0,
              vehicles: numericValue(row.vehicles_in_frame) ?? 0,
              distance: numericValue(row.frame_distance_m),
              timeToReach: numericValue(row.frame_ttc_seconds),
              impactScore: numericValue(row.frame_impact_score),
              impactLevel: row.frame_impact_level || "",
              warning: truthyCSV(row.upcoming_pothole_detected),
            }))
            .sort((a, b) => a.timestamp - b.timestamp);

          if (loadedFrameData.length === 0) {
            console.warn(
              "No frame_summary rows were found in the CSV. Ensure combined.py writes frame_summary records."
            );
          }
          setFrameData(loadedFrameData);
        } catch (csvError) {
          console.error("Detection CSV loading failed:", csvError);
          setError(
            "Video processed, but live dashboard data could not be loaded. Check the CSV URL and ensure combined.py writes frame_summary rows."
          );
        } finally {
          setCsvLoading(false);
        }
      } else {
        console.warn("The backend response did not include a CSV URL.", data);
      }

      setHistory((previous) => [
        {
          name: selectedFile.name,
          date: new Date().toLocaleString(),
          data,
          videoUrl: processedUrl,
          frameData: loadedFrameData,
        },
        ...previous,
      ]);
    } catch (err) {
      setError(err.message || "Unable to process the video.");
    } finally {
      setLoading(false);
    }
  };

  const retryVideo = () => {
    if (!videoUrl) return;

    setVideoError("");

    // Changing the key forces the browser to create a fresh video element.
    setVideoRetry((previous) => previous + 1);
  };

  const [videoRetry, setVideoRetry] = useState(0);

  const getValue = (keys, fallback = null) => {
    if (!result) return fallback;

    for (const key of keys) {
      if (result[key] !== undefined && result[key] !== null) {
        return result[key];
      }

      if (
        result.results &&
        result.results[key] !== undefined &&
        result.results[key] !== null
      ) {
        return result.results[key];
      }
    }

    return fallback;
  };

  const potholes = frameMetrics
    ? frameMetrics.potholes
    : getValue([
        "unique_potholes",
        "potholes_detected",
        "pothole_count",
        "total_potholes",
      ]);

  const vehicles = frameMetrics
    ? frameMetrics.vehicles
    : getValue(["vehicles_detected", "vehicle_count", "total_vehicles"]);

  const distance = frameMetrics
    ? frameMetrics.distance
    : getValue([
        "distance_meters",
        "distance",
        "upcoming_pothole_distance",
      ]);

  const timeToReach = frameMetrics
    ? frameMetrics.timeToReach
    : getValue([
        "time_to_reach",
        "time_to_reach_seconds",
        "ttc",
      ]);

  const impact = frameMetrics
    ? frameMetrics.impactLevel ||
      (frameMetrics.impactScore !== null
        ? `Estimated (${frameMetrics.impactScore}/100)`
        : null)
    : getValue(["impact_level", "impact_assessment", "impact"]);

  const warning = frameMetrics
    ? frameMetrics.warning
      ? "UPCOMING POTHOLE DETECTED"
      : "NO UPCOMING POTHOLE IN CURRENT FRAME"
    : result
      ? "ANALYSIS COMPLETED"
      : "READY FOR ROAD ANALYSIS";

  const displayValue = (value, suffix = "") => {
    if (value === null || value === undefined || value === "") {
      return "—";
    }

    return String(value) + suffix;
  };

  const renderNavigation = () => (
    <nav className="drive-nav">
      <button
        className={page === "drive" ? "nav-button active" : "nav-button"}
        onClick={() => setPage("drive")}
      >
        <span>◉</span> Driving Display
      </button>

      <button
        className={page === "analysis" ? "nav-button active" : "nav-button"}
        onClick={() => setPage("analysis")}
      >
        <span>▣</span> Video Analysis
      </button>

      <button
        className={page === "summary" ? "nav-button active" : "nav-button"}
        onClick={() => setPage("summary")}
      >
        <span>▤</span> Trip Summary
      </button>
    </nav>
  );

  const renderVideo = () => (
    <div className="road-screen">
      {videoUrl ? (
        <div className="video-container">
          <video
            ref={videoRef}
            key={`${videoUrl}-${videoRetry}`}
            className="road-video"
            controls
            playsInline
            preload="metadata"
            onTimeUpdate={(event) => {
              const currentTime = event.currentTarget.currentTime;
              let currentFrame = null;
              for (const frame of frameData) {
                if (frame.timestamp <= currentTime + 0.05) {
                  currentFrame = frame;
                } else {
                  break;
                }
              }
              setFrameMetrics(currentFrame);
            }}
            onSeeked={(event) => {
              const currentTime = event.currentTarget.currentTime;
              let currentFrame = null;
              for (const frame of frameData) {
                if (frame.timestamp <= currentTime + 0.05) {
                  currentFrame = frame;
                } else {
                  break;
                }
              }
              setFrameMetrics(currentFrame);
            }}
            onLoadedMetadata={() => {
              setVideoError("");
              console.log("Video metadata loaded:", videoUrl);
            }}
            onCanPlay={() => {
              setVideoError("");
              console.log("Processed video is ready to play.");
            }}
            onError={(event) => {
              const mediaError = event.currentTarget.error;

              const messages = {
                1: "Video loading was interrupted. Try reloading the video.",
                2: "A network error occurred while loading the video. Check the Flask server.",
                3: "The browser could not decode this video. The MP4 may use an unsupported codec.",
                4: "The video URL or format is unsupported. Check the video-serving route.",
              };

              const message =
                messages[mediaError?.code] ||
                "The browser could not load the processed video.";

              setVideoError(message);

              console.error("Processed video playback failed:", {
                url: videoUrl,
                code: mediaError?.code,
                message: mediaError?.message,
              });
            }}
          >
            <source src={videoUrl} type="video/mp4" />
            Your browser does not support HTML5 video.
          </video>

          {videoError && (
            <div className="video-error-message" role="alert">
              <p>{videoError}</p>
              <button
                type="button"
                className="secondary-action"
                onClick={retryVideo}
              >
                Reload Video
              </button>
              <a href={videoUrl} target="_blank" rel="noreferrer">
                Open Video Directly
              </a>
            </div>
          )}
        </div>
      ) : (
        <div className="road-placeholder">
          <div className="road-horizon">
            <div className="road-sky" />
            <div className="road-surface">
              <div className="road-markings" />
            </div>
          </div>

          <div className="camera-message">
            <div className="camera-icon">▣</div>
            <h3>
              {loading ? "PROCESSING ROAD VIDEO" : "ROAD CAMERA DISPLAY"}
            </h3>
            <p>
              {loading
                ? "Please wait while pothole detection is running."
                : "Upload a recorded road video to begin pothole detection."}
            </p>
          </div>
        </div>
      )}

      <div className="camera-label">
        <span className="live-indicator" />
        {videoUrl ?"" : "ROAD VIEW"}
      </div>
    </div>
  );

  const renderDrivePage = () => (
    <>
      <section className="driving-heading">
        <div>
          <div className="eyebrow">INTELLIGENT DRIVER ASSISTANCE</div>
          <h1>Driving Display</h1>
          <p>Road hazard monitoring and pothole detection</p>
        </div>

        <div className="drive-status">
          <span
            className={
              backendReady ? "status-light ready" : "status-light"
            }
          />
          {backendReady ? "SYSTEM READY" : "SYSTEM UNAVAILABLE"}
        </div>
      </section>

      <section className="driving-layout">
        <div className="road-column">
          {renderVideo()}

          <div className="warning-panel">
            <div className="warning-symbol">
              {(frameMetrics ? frameMetrics.warning : potholes !== null && Number(potholes) > 0) ? "⚠" : "◉"}
            </div>

            <div className="warning-copy">
              <div className="eyebrow">ROAD SAFETY ALERT</div>
              <h2>
                {loading ? "ANALYSING ROAD..." : warning}
              </h2>

              <p>
                {loading
                  ? "Please wait while the road video is processed."
                  : error
                    ? error
                    : frameMetrics
                      ? frameMetrics.warning
                        ? "An upcoming pothole is flagged in the current video frame."
                        : "No upcoming pothole is flagged in the current video frame."
                      : result
                        ? "Analysis finished. Play the processed video to sync the panel with each frame."
                        : "Upload a road video to check for potholes."}
              </p>
            </div>
          </div>

          {error && <div className="error-banner">{error}</div>}
        </div>

        <aside className="instrument-panel">
          <div className="instrument-title">
            <span>ROAD INTELLIGENCE</span>
            <span className="instrument-dot" />
          </div>

          <div className="distance-gauge">
            <div className="gauge-label">POTHOLE DISTANCE</div>

            <div className="distance-value">
              {distance !== null ? displayValue(distance) : "—"}
              <span>{distance !== null ? " m" : ""}</span>
            </div>

            <p>
              {distance !== null
                ? "Estimated distance ahead"
                : "Distance estimate unavailable"}
            </p>
          </div>

          <div className="instrument-divider" />

          <div className="instrument-reading">
            <span className="reading-icon">◷</span>
            <div>
              <p>Estimated time to reach</p>
              <strong>
                {timeToReach !== null
                  ? displayValue(timeToReach, " s")
                  : "Unavailable"}
              </strong>
            </div>
          </div>

          <div className="instrument-reading">
            <span className="reading-icon">⚠</span>
            <div>
              <p>Vehicle impact assessment</p>
              <strong className="impact-reading">
                {impact !== null ? String(impact) : "Not available"}
              </strong>
            </div>
          </div>

          <div className="instrument-reading">
            <span className="reading-icon">◇</span>
            <div>
              <p>Potholes detected</p>
              <strong>{displayValue(potholes)}</strong>
            </div>
          </div>

          <div className="instrument-reading">
            <span className="reading-icon">🚘</span>
            <div>
              <p>Vehicles detected</p>
              <strong>{displayValue(vehicles)}</strong>
            </div>
          </div>

          <div className="instrument-note">
            {csvLoading
              ? "Loading frame-by-frame road intelligence…"
              : frameData.length
                ? "Road Intelligence follows the processed video's playback position. Distance and impact are model estimates; speed may be assumed."
                : "Frame-synced values require frame_summary rows in the detection CSV. Distance and impact are estimates, not verified measurements."}
          </div>
        </aside>
      </section>

      <section className="bottom-controls">
        <div className="control-copy">
          <div className="eyebrow">ROAD VIDEO</div>
          <h3>{selectedFile ? selectedFile.name : "No video selected"}</h3>
          <p>
            Use a recorded forward-facing road or dashcam video for analysis.
          </p>
        </div>

        <button
          className="primary-action"
          onClick={() => setPage("analysis")}
        >
          {result ? "Analyse Another Video →" : "Select Road Video →"}
        </button>
      </section>
    </>
  );

  const renderAnalysisPage = () => (
    <>
      <section className="driving-heading">
        <div>
          <div className="eyebrow">RECORDED VIDEO PROCESSING</div>
          <h1>Video Analysis</h1>
          <p>Upload a road recording for pothole and vehicle detection.</p>
        </div>
      </section>

      <section className="upload-card">
        <div className="upload-symbol">⇧</div>
        <h2>Import Road Video</h2>

        <p className="upload-description">
          Choose a recorded dashcam or forward-facing road video from your
          computer.
        </p>

        <form onSubmit={handleUpload}>
          <label className="file-picker">
            <span>Choose video file</span>
            <input
              type="file"
              accept="video/mp4,video/x-msvideo,video/quicktime,video/x-matroska,video/*"
              onChange={(event) => {
                setSelectedFile(event.target.files?.[0] || null);
                setError("");
                setVideoError("");
              }}
            />
          </label>

          {selectedFile && (
            <div className="selected-video">
              <span>▣</span>
              <div>
                <strong>{selectedFile.name}</strong>
                <p>
                  {(selectedFile.size / (1024 * 1024)).toFixed(2)} MB
                </p>
              </div>
            </div>
          )}

          {error && <div className="error-banner">{error}</div>}

          <button
            type="submit"
            className="primary-action process-button"
            disabled={loading || !selectedFile}
          >
            {loading
              ? "Processing video..."
              : "Start Pothole Detection →"}
          </button>
        </form>

        {loading && (
          <div className="processing-message">
            <span className="loading-spinner" />
            <p>
              Processing the video. Processing time depends on its length and
              your computer's performance.
            </p>
          </div>
        )}
      </section>

      <section className="analysis-info-grid">
        <div className="info-tile">
          <span>01</span>
          <h3>Pothole Detection</h3>
          <p>
            Identify potholes in the road footage using your trained model.
          </p>
        </div>

        <div className="info-tile">
          <span>02</span>
          <h3>Vehicle Detection</h3>
          <p>
            Display vehicle detection results when provided by the backend.
          </p>
        </div>

        <div className="info-tile">
          <span>03</span>
          <h3>Hazard Assessment</h3>
          <p>
            Present distance and impact estimates when available.
          </p>
        </div>
      </section>
    </>
  );

  const renderSummaryPage = () => (
    <>
      <section className="driving-heading">
        <div>
          <div className="eyebrow">DETECTION HISTORY</div>
          <h1>Trip Summary</h1>
          <p>Review road-video analysis completed during this session.</p>
        </div>
      </section>

      {!history.length ? (
        <div className="empty-history">
          <div>▤</div>
          <h2>No analysis yet</h2>

          <p>
            Process a road video to see its results and summary here. History
            is currently stored only for this browser session.
          </p>

          <button
            className="primary-action"
            onClick={() => setPage("analysis")}
          >
            Analyse a Road Video →
          </button>
        </div>
      ) : (
        <div className="history-list">
          {history.map((item, index) => {
            const data = item.data || {};

            const itemPotholes =
              data.unique_potholes ??
              data.potholes_detected ??
              data.pothole_count ??
              data.results?.unique_potholes ??
              data.results?.total_frame_detections ??
              "—";

            const itemVehicles =
              data.vehicles_detected ??
              data.vehicle_count ??
              data.results?.vehicles_detected ??
              "—";

            return (
              <article className="history-card" key={item.job_id || index}>
                <div className="history-icon">▣</div>

                <div className="history-main">
                  <h3>{item.name}</h3>
                  <p>{item.date}</p>

                  <div className="history-stats">
                    <span>Pothole detections: {itemPotholes}</span>
                    <span>Vehicles: {itemVehicles}</span>
                  </div>
                </div>

                <button
                  className="secondary-action"
                  onClick={() => {
                    setResult(data);
                    setVideoUrl(item.videoUrl || "");
                    setFrameData(item.frameData || []);
                    setFrameMetrics(null);
                    setVideoError("");
                    setError("");
                    setPage("drive");
                  }}
                >
                  View Results
                </button>
              </article>
            );
          })}
        </div>
      )}
    </>
  );

  return (
    <div className="automotive-app">
      <header className="app-header">
        <div className="brand">
          <div className="brand-mark">⌁</div>

          <div>
            <div className="brand-name">
              ROAD<span>GUARD</span>
            </div>
            <div className="brand-caption">
              INTELLIGENT ROAD SAFETY
            </div>
          </div>
        </div>

        {renderNavigation()}

        <div className="header-indicator">
          <span
            className={
              backendReady ? "status-light ready" : "status-light"
            }
          />
          <span>{backendReady ? "READY" : "STANDBY"}</span>
        </div>
      </header>

      <main className="main-dashboard">
        {page === "drive" && renderDrivePage()}
        {page === "analysis" && renderAnalysisPage()}
        {page === "summary" && renderSummaryPage()}
      </main>

      <footer className="app-footer">
        <span>ROADGUARD</span>
        <span>DRIVER ASSISTANCE PROTOTYPE</span>
        <span>Recorded-video analysis mode</span>
      </footer>
    </div>
  );
}

export default App;

