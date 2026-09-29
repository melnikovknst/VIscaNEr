import { useEffect, useRef, useState } from "react";
import type { CSSProperties, ChangeEvent, ReactNode } from "react";
import {
  ArrowLeft,
  ArrowRight,
  ArrowUpRight,
  BookOpen,
  Camera,
  Check,
  CheckCheck,
  ChevronRight,
  CircleHelp,
  Clock3,
  Grape,
  Heart,
  ImagePlus,
  Leaf,
  Loader2,
  MapPin,
  ScanLine,
  Search,
  ShieldCheck,
  SlidersHorizontal,
  Sparkles,
  Upload,
  Utensils,
  Wine as WineIcon,
  X,
  Fish,
  Beef,
  Salad,
  CakeSlice,
  Milk,
  RotateCcw,
  Zap,
} from "lucide-react";
import { api, useSavedInitial } from "./api";
import type {
  CatalogPage,
  Health,
  Meta,
  Pairing,
  ScanResult,
  SommelierAnswer,
  SommelierStatus,
  Wine,
} from "./api";

type Page = "scanner" | "catalog" | "sommelier" | "saved" | "history";
const pages: { id: Page; name: string; icon: typeof ScanLine }[] = [
  { id: "scanner", name: "Сканер вина", icon: ScanLine },
  { id: "catalog", name: "Каталог вин", icon: WineIcon },
  { id: "sommelier", name: "К столу", icon: Utensils },
  { id: "saved", name: "Моя коллекция", icon: Heart },
  { id: "history", name: "История", icon: Clock3 },
];
const initialPage = (): Page =>
  pages.find((p) => p.id === location.hash.slice(1))?.id || "scanner";
const formatCount = (n: number) => n.toLocaleString("ru-RU");
const errorText = (e: unknown) =>
  e instanceof Error ? e.message : "Что-то пошло не так. Попробуйте ещё раз.";

function ErrorBox({
  children,
  retry,
}: {
  children: ReactNode;
  retry?: () => void;
}) {
  return (
    <div className="error-box" role="alert">
      <CircleHelp size={19} />
      <span>{children}</span>
      {retry && <button onClick={retry}>Повторить</button>}
    </div>
  );
}
function Loading() {
  return (
    <div className="loading" role="status">
      <Loader2 className="spin" size={25} />
      <span>Загружаем…</span>
    </div>
  );
}
function Empty({
  icon: Icon,
  title,
  text,
  action,
}: {
  icon: typeof Heart;
  title: string;
  text: string;
  action?: ReactNode;
}) {
  return (
    <div className="empty-state">
      <div className="empty-icon">
        <Icon size={30} />
      </div>
      <h2>{title}</h2>
      <p>{text}</p>
      {action}
    </div>
  );
}

function WineImage({
  wine,
  className = "",
}: {
  wine: Wine;
  className?: string;
}) {
  const [failed, setFailed] = useState(false);
  return failed ? (
    <WineIcon
      className={`image-fallback ${className}`}
      aria-label="Фото отсутствует"
    />
  ) : (
    <img
      className={className}
      src={wine.image_url}
      alt={wine.name}
      loading="lazy"
      onError={() => setFailed(true)}
    />
  );
}
function WineCard({
  wine,
  saved,
  onSave,
  onOpen,
}: {
  wine: Wine;
  saved: boolean;
  onSave: () => void;
  onOpen: () => void;
}) {
  return (
    <article className="wine-card">
      <div className={`wine-card-image tone-${wine.category}`}>
        <span className="wine-tag">
          <i />
          {wine.category}
        </span>
        <button
          className={`icon-button save-button ${saved ? "is-saved" : ""}`}
          aria-label={
            saved
              ? `Убрать ${wine.name} из коллекции`
              : `Сохранить ${wine.name}`
          }
          aria-pressed={saved}
          onClick={onSave}
        >
          <Heart size={18} fill={saved ? "currentColor" : "none"} />
        </button>
        <button
          className="wine-image-button"
          onClick={onOpen}
          aria-label={`Открыть ${wine.name}`}
        >
          <WineImage wine={wine} />
        </button>
      </div>
      <button className="wine-card-copy" onClick={onOpen}>
        <span className="eyebrow">{wine.winery || "Российское вино"}</span>
        <h3>{wine.name}</h3>
        <span className="wine-region">
          <MapPin size={13} />
          {wine.region || "Россия"}
          <ArrowUpRight size={17} />
        </span>
      </button>
    </article>
  );
}

/** Tap-to-pick rows of catalog wines, used wherever the scanner offers a choice. */
function CandidateList({
  wines,
  onPick,
  compact = false,
}: {
  wines: Wine[];
  onPick: (wine: Wine) => void;
  compact?: boolean;
}) {
  return (
    <ul className={`candidate-list ${compact ? "is-compact" : ""}`}>
      {wines.map((wine) => (
        <li key={wine.slug}>
          <button className="candidate-row" onClick={() => onPick(wine)}>
            <span className={`candidate-thumb tone-${wine.category}`}>
              <WineImage wine={wine} />
            </span>
            <span className="candidate-copy">
              <small>{wine.winery || "Российское вино"}</small>
              <strong>{wine.name}</strong>
              <span>
                {[wine.category, wine.region].filter(Boolean).join(" · ")}
              </span>
            </span>
            <ChevronRight size={18} className="candidate-chevron" />
          </button>
        </li>
      ))}
    </ul>
  );
}

/**
 * An uncertain answer: the model's top guess first, one tap to confirm it,
 * the other candidates only on request. The list is always rendered so it can
 * animate open; while closed it is inert (no focus, no screen reader).
 */
function BestGuess({
  wines,
  onPick,
}: {
  wines: Wine[];
  onPick: (wine: Wine) => void;
}) {
  const [open, setOpen] = useState(false);
  const [top, ...others] = wines;
  if (!top) return null;
  return (
    <div className="best-guess">
      <article className="guess-card">
        <span className={`guess-thumb tone-${top.category}`}>
          <WineImage wine={top} />
        </span>
        <div className="guess-copy">
          <small>{top.winery || "Российское вино"}</small>
          <h2>{top.name}</h2>
          <span>
            {[top.category, top.region, top.grapes].filter(Boolean).join(" · ")}
          </span>
        </div>
        <div className="guess-actions">
          <button className="primary-button" onClick={() => onPick(top)}>
            <Check size={18} />
            Да, это оно
          </button>
          {others.length > 0 && (
            <button
              className={`secondary-button more-toggle ${open ? "is-open" : ""}`}
              aria-expanded={open}
              aria-controls="other-guesses"
              onClick={() => setOpen((value) => !value)}
            >
              {open ? "Скрыть варианты" : "Другие варианты"}
              <ChevronRight size={17} className="more-chevron" />
            </button>
          )}
        </div>
      </article>
      {others.length > 0 && (
        <div
          id="other-guesses"
          className={`other-guesses ${open ? "is-open" : ""}`}
          inert={!open}
        >
          <div className="other-guesses-inner">
            <span className="other-guesses-title">
              Может быть, одно из этих:
            </span>
            <ul className="candidate-list">
              {others.map((wine, index) => (
                <li key={wine.slug} style={{ "--i": index } as CSSProperties}>
                  <button
                    className="candidate-row"
                    onClick={() => onPick(wine)}
                  >
                    <span className={`candidate-thumb tone-${wine.category}`}>
                      <WineImage wine={wine} />
                    </span>
                    <span className="candidate-copy">
                      <small>{wine.winery || "Российское вино"}</small>
                      <strong>{wine.name}</strong>
                      <span>
                        {[wine.category, wine.region]
                          .filter(Boolean)
                          .join(" · ")}
                      </span>
                    </span>
                    <ChevronRight size={18} className="candidate-chevron" />
                  </button>
                </li>
              ))}
            </ul>
          </div>
        </div>
      )}
    </div>
  );
}

/** The visitor's own photo, so a match can be checked by eye. */
function ScanPhoto({
  src,
  className = "",
}: {
  src: string;
  className?: string;
}) {
  return (
    <figure className={`scan-photo ${className}`}>
      <img src={src} alt="Ваше фото этикетки" />
      <figcaption>Ваше фото</figcaption>
    </figure>
  );
}

/** Uncertain and not-found scans: let the visitor finish the job. */
function ResultChooser({
  result,
  photo,
  onPick,
  onRetake,
  navigate,
}: {
  result: ScanResult;
  photo: string;
  onPick: (wine: Wine) => void;
  onRetake: () => void;
  navigate: (page: Page) => void;
}) {
  // After an "uncertain" answer the right wine is usually among the first five,
  // after "not found" much less often. So the first is a real choice, the
  // second a low-key fallback.
  const uncertain = result.status === "uncertain";
  const wines = result.candidates.slice(0, 5).map((c) => c.wine);
  return (
    <section className={`result-chooser ${photo ? "" : "no-photo"}`}>
      <button className="text-button back-button" onClick={onRetake}>
        <ArrowLeft size={17} />К сканеру
      </button>
      <div className="chooser-grid">
        {photo && <ScanPhoto src={photo} className="chooser-photo" />}
        <div className="chooser-body">
          <span className="eyebrow">
            {uncertain ? "СКОРЕЕ ВСЕГО" : "НЕ УДАЛОСЬ УЗНАТЬ ВИНО"}
          </span>
          <h1>{uncertain ? "Похоже, это оно" : "Пока не узнали это вино"}</h1>
          <p>
            {uncertain
              ? "Сверьте с этикеткой: уверенного совпадения нет, поэтому решать вам."
              : "Снимите этикетку крупнее и ровнее, без бликов и соседних бутылок."}
          </p>
          {uncertain ? (
            <BestGuess wines={wines} onPick={onPick} />
          ) : (
            <div className="chooser-actions">
              <button className="primary-button" onClick={onRetake}>
                <Camera size={18} />
                Сделать другое фото
              </button>
              <button
                className="secondary-button"
                onClick={() => navigate("catalog")}
              >
                <Search size={17} />
                Найти в каталоге
              </button>
            </div>
          )}
          {uncertain ? (
            <div className="chooser-footer">
              <span>Нет вашего вина?</span>
              <div className="chooser-actions">
                <button className="secondary-button" onClick={onRetake}>
                  <Camera size={17} />
                  Сделать другое фото
                </button>
                <button
                  className="text-button"
                  onClick={() => navigate("catalog")}
                >
                  Найти в каталоге <ArrowRight size={16} />
                </button>
              </div>
            </div>
          ) : (
            wines.length > 0 && (
              <details className="chooser-maybe">
                <summary>Вдруг оно среди похожих?</summary>
                <CandidateList wines={wines} onPick={onPick} compact />
              </details>
            )
          )}
        </div>
      </div>
    </section>
  );
}

export default function App() {
  const [page, setPage] = useState<Page>(initialPage);
  const [health, setHealth] = useState<Health | null>(null);
  const [meta, setMeta] = useState<Meta | null>(null);
  const [bootError, setBootError] = useState("");
  const [saved, setSaved] = useState<string[]>(useSavedInitial);
  const [detail, setDetail] = useState<Wine | null>(null);
  const [result, setResult] = useState<ScanResult | null>(null);
  const [help, setHelp] = useState(false);
  const [toast, setToast] = useState("");
  const [refresh, setRefresh] = useState(0);
  // The photo behind the current result. Kept only in memory for this tab:
  // the server never stores photos, and history entries have none.
  const [photoFile, setPhotoFile] = useState<File | null>(null);
  const [photo, setPhoto] = useState("");
  // True when the visitor chose the wine from candidates rather than the
  // model deciding it - the card must not claim "found" in that case.
  const [picked, setPicked] = useState(false);
  // A retake from a result screen opens the camera directly; the new photo
  // then waits in the scanner for the visitor to confirm.
  const retakeInput = useRef<HTMLInputElement>(null);
  const retakeGallery = useRef<HTMLInputElement>(null);
  const [pendingFile, setPendingFile] = useState<File | null>(null);
  useEffect(() => {
    if (!photoFile) {
      setPhoto("");
      return;
    }
    const url = URL.createObjectURL(photoFile);
    setPhoto(url);
    return () => URL.revokeObjectURL(url);
  }, [photoFile]);

  useEffect(() => {
    let current = true;
    setBootError("");
    Promise.all([api<Health>("/api/health"), api<Meta>("/api/catalog/meta")])
      .then(([h, m]) => {
        if (current) {
          setHealth(h);
          setMeta(m);
        }
      })
      .catch((e) => {
        if (current) setBootError(errorText(e));
      });
    return () => {
      current = false;
    };
  }, [refresh]);
  useEffect(() => {
    const handler = () => {
      setPage(initialPage());
      setDetail(null);
      setResult(null);
      setPhotoFile(null);
    };
    window.addEventListener("hashchange", handler);
    return () => window.removeEventListener("hashchange", handler);
  }, []);
  useEffect(() => {
    if (!toast) return;
    const timer = window.setTimeout(() => setToast(""), 3000);
    return () => clearTimeout(timer);
  }, [toast]);
  const navigate = (next: Page) => {
    setDetail(null);
    setResult(null);
    setPhotoFile(null);
    setPage(next);
    if (location.hash !== `#${next}`) location.hash = next;
    window.scrollTo({ top: 0, behavior: "smooth" });
  };
  const openWine = (wine: Wine) => {
    setDetail(wine);
    setResult(null);
    window.scrollTo({ top: 0, behavior: "smooth" });
  };
  const toggleSave = (wine: Wine) => {
    const next = saved.includes(wine.slug)
      ? saved.filter((s) => s !== wine.slug)
      : [...saved, wine.slug];
    try {
      localStorage.setItem("viscaner-saved", JSON.stringify(next));
      setSaved(next);
      setToast(
        next.includes(wine.slug)
          ? "Вино добавлено в коллекцию"
          : "Вино убрано из коллекции",
      );
    } catch {
      setToast("Браузер не разрешил сохранить коллекцию");
    }
  };
  const showResult = (value: ScanResult, file: File | null = null) => {
    setResult(value);
    setDetail(value.wine);
    setPicked(false);
    setPhotoFile(file);
    window.scrollTo({ top: 0, behavior: "smooth" });
  };
  // Choosing among candidates opens that card but keeps the scan context
  // (photo, alternatives), so a wrong pick can be corrected in one tap.
  const pickCandidate = (wine: Wine) => {
    setDetail(wine);
    setPicked(true);
    window.scrollTo({ top: 0, behavior: "smooth" });
  };
  const backToScanner = () => {
    setDetail(null);
    setResult(null);
    setPhotoFile(null);
  };
  // Must run inside the click handler: browsers only open a file or camera
  // picker in direct response to a user gesture.
  const [liveRetake, setLiveRetake] = useState(false);
  const retake = () =>
    liveCameraAvailable() ? setLiveRetake(true) : retakeInput.current?.click();
  const retaken = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (file) takeRetake(file);
  };
  const takeRetake = (file: File) => {
    backToScanner();
    setPendingFile(file);
    if (page !== "scanner") navigate("scanner");
    window.scrollTo({ top: 0, behavior: "smooth" });
  };
  const card = (wine: Wine) => (
    <WineCard
      key={wine.slug}
      wine={wine}
      saved={saved.includes(wine.slug)}
      onSave={() => toggleSave(wine)}
      onOpen={() => openWine(wine)}
    />
  );

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <a
          className="brand"
          href="#scanner"
          onClick={() => navigate("scanner")}
          aria-label="winescanner — сканер вина"
        >
          <ScanLine size={30} strokeWidth={1.4} />
          <div>
            <strong>
              wine<span>scanner</span>
            </strong>
          </div>
        </a>
        <div className="sidebar-divider" />
        <p className="nav-label">ОТКРЫВАЙТЕ СВОЁ</p>
        <nav aria-label="Основная навигация">
          {pages.map(({ id, name, icon: Icon }) => (
            <button
              key={id}
              className={`nav-item ${page === id ? "active" : ""}`}
              onClick={() => navigate(id)}
              aria-current={page === id ? "page" : undefined}
            >
              <Icon size={20} strokeWidth={1.7} />
              <span>{name}</span>
              {id === "saved" && saved.length > 0 && (
                <small>{saved.length}</small>
              )}
              {id === "scanner" && <span className="nav-dot" />}
            </button>
          ))}
        </nav>
        <div className="sidebar-note">
          <span className="note-grape">
            <Grape size={31} strokeWidth={1.2} />
          </span>
          <p>
            Большая страна.
            <br />
            Удивительные вина.
          </p>
          <span>
            Откройте для себя
            <br />
            российское виноделие.
          </span>
          <a href="#catalog" onClick={() => navigate("catalog")}>
            Открыть каталог <ArrowUpRight size={15} />
          </a>
        </div>
        {/* Phones hide the breadcrumb row; the age mark moves up here. */}
        <span className="age-tag mobile-age">18+</span>
        <button className="sidebar-help" onClick={() => setHelp(true)}>
          <CircleHelp size={18} />
          Как это работает
        </button>
        <div className="sidebar-bottom">
          <span>winescanner</span>
          <small>С заботой о вашем выборе</small>
        </div>
      </aside>
      <div className="main-shell">
        <header className="topbar">
          <div className="breadcrumb">
            winescanner <ChevronRight size={14} />
            <span>
              {detail
                ? "Карточка вина"
                : pages.find((p) => p.id === page)?.name}
            </span>
          </div>
          <a
            href="#catalog"
            onClick={() => navigate("catalog")}
            className="portal-link"
          >
            Исследовать каталог <ArrowUpRight size={15} />
          </a>
          <span className="age-tag">18+</span>
        </header>
        <main id="main-content">
          {bootError && (
            <ErrorBox retry={() => setRefresh((x) => x + 1)}>
              {bootError}
            </ErrorBox>
          )}
          {detail ? (
            <WineDetail
              wine={detail}
              result={result}
              photo={photo}
              picked={picked}
              saved={saved.includes(detail.slug)}
              onSave={() => toggleSave(detail)}
              onBack={backToScanner}
              onPick={pickCandidate}
              onRetake={retake}
              onPair={() => navigate("sommelier")}
            />
          ) : result && result.status !== "demo" ? (
            <ResultChooser
              result={result}
              photo={photo}
              onPick={pickCandidate}
              onRetake={retake}
              navigate={navigate}
            />
          ) : page === "scanner" ? (
            <Scanner
              key={
                pendingFile
                  ? `retake-${pendingFile.name}-${pendingFile.lastModified}`
                  : "scanner"
              }
              health={health}
              meta={meta}
              initialFile={pendingFile}
              onResult={(value, file) => {
                setPendingFile(null);
                showResult(value, file);
              }}
              navigate={navigate}
              card={card}
            />
          ) : page === "catalog" ? (
            <Catalog meta={meta} card={card} />
          ) : page === "sommelier" ? (
            <Sommelier card={card} />
          ) : page === "saved" ? (
            <Saved slugs={saved} card={card} navigate={navigate} />
          ) : (
            <History onResult={showResult} navigate={navigate} />
          )}
        </main>
        <footer>
          <span>
            <ScanLine size={16} /> winescanner
          </span>
          <span>Открывайте. Узнавайте. Сохраняйте.</span>
        </footer>
      </div>
      {liveRetake && (
        <CameraCapture
          onCapture={(shot) => {
            setLiveRetake(false);
            takeRetake(shot);
          }}
          onClose={() => setLiveRetake(false)}
          onFallback={() => {
            setLiveRetake(false);
            retakeInput.current?.click();
          }}
          onGallery={() => {
            setLiveRetake(false);
            retakeGallery.current?.click();
          }}
        />
      )}
      <input
        type="file"
        accept="image/jpeg,image/png,image/webp"
        className="visually-hidden"
        ref={retakeGallery}
        onChange={retaken}
        aria-label="Новое фото из галереи"
        tabIndex={-1}
      />
      <input
        type="file"
        accept="image/*"
        capture="environment"
        className="visually-hidden"
        ref={retakeInput}
        onChange={retaken}
        aria-label="Сделать другое фото"
        tabIndex={-1}
      />
      <nav className="mobile-nav" aria-label="Мобильная навигация">
        {pages.map(({ id, name, icon: Icon }) => (
          <button
            key={id}
            className={page === id ? "active" : ""}
            onClick={() => navigate(id)}
            aria-label={name}
            aria-current={page === id ? "page" : undefined}
          >
            <Icon size={21} />
            <span>
              {id === "scanner"
                ? "Сканер"
                : id === "catalog"
                  ? "Каталог"
                  : id === "saved"
                    ? "Коллекция"
                    : name}
            </span>
          </button>
        ))}
      </nav>
      {toast && (
        <div className="toast" role="status">
          <Check size={18} />
          {toast}
        </div>
      )}
      {help && <Help onClose={() => setHelp(false)} />}
    </div>
  );
}

/** True when the browser can show a live camera inside the page (HTTPS or localhost). */
const liveCameraAvailable = () =>
  typeof window !== "undefined" &&
  window.isSecureContext &&
  !!navigator.mediaDevices?.getUserMedia;

/**
 * Full-screen viewfinder with a crosshair. The crosshair sits exactly at the
 * frame centre - where the recogniser looks for the target bottle - and the
 * whole frame is captured, so what is under the crosshair is what gets
 * recognised. Falls back to the system camera if the stream cannot start.
 */
function CameraCapture({
  onCapture,
  onClose,
  onFallback,
  onGallery,
}: {
  onCapture: (file: File) => void;
  onClose: () => void;
  onFallback: () => void;
  onGallery: () => void;
}) {
  const video = useRef<HTMLVideoElement>(null);
  const stream = useRef<MediaStream | null>(null);
  const [ready, setReady] = useState(false);
  const [flash, setFlash] = useState(false);
  // null = the camera has no torch; otherwise whether it is on.
  const [torch, setTorch] = useState<boolean | null>(null);
  useEffect(() => {
    let cancelled = false;
    navigator.mediaDevices
      .getUserMedia({
        audio: false,
        video: {
          facingMode: { ideal: "environment" },
          width: { ideal: 3840 },
          height: { ideal: 2160 },
        },
      })
      .then((media) => {
        if (cancelled) {
          media.getTracks().forEach((track) => track.stop());
          return;
        }
        stream.current = media;
        const track = media.getVideoTracks()[0];
        const caps = (track.getCapabilities?.() ?? {}) as { torch?: boolean };
        if (caps.torch) setTorch(false);
        if (video.current) {
          video.current.srcObject = media;
          video.current.play().catch(() => undefined);
        }
      })
      .catch(() => {
        if (!cancelled) onFallback();
      });
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    document.body.classList.add("camera-open");
    return () => {
      cancelled = true;
      window.removeEventListener("keydown", onKey);
      document.body.classList.remove("camera-open");
      stream.current?.getTracks().forEach((track) => track.stop());
    };
    // The stream is opened once per mount; the callbacks only close it.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  const toggleTorch = async () => {
    const track = stream.current?.getVideoTracks()[0];
    if (!track || torch === null) return;
    try {
      await track.applyConstraints({
        advanced: [{ torch: !torch } as MediaTrackConstraintSet],
      });
      setTorch(!torch);
    } catch {
      setTorch(null);
    }
  };
  const shoot = () => {
    const element = video.current;
    if (!element || !element.videoWidth) return;
    const canvas = document.createElement("canvas");
    canvas.width = element.videoWidth;
    canvas.height = element.videoHeight;
    canvas.getContext("2d")?.drawImage(element, 0, 0);
    setFlash(true);
    canvas.toBlob(
      (blob) => {
        if (blob)
          onCapture(
            new File([blob], `label-${Date.now()}.jpg`, { type: "image/jpeg" }),
          );
      },
      "image/jpeg",
      0.92,
    );
  };
  return (
    <div className="camera" role="dialog" aria-modal="true" aria-label="Камера">
      <video
        ref={video}
        className="camera-video"
        playsInline
        muted
        autoPlay
        onLoadedData={() => setReady(true)}
      />
      <div
        className={`camera-overlay ${ready ? "is-ready" : ""}`}
        aria-hidden="true"
      >
        <div className="camera-window">
          <span className="corner tl" />
          <span className="corner tr" />
          <span className="corner bl" />
          <span className="corner br" />
          <span className="crosshair" />
        </div>
      </div>
      {flash && (
        <div className="camera-flash" onAnimationEnd={() => setFlash(false)} />
      )}
      <div className="camera-top">
        <button
          className="camera-round"
          onClick={onClose}
          aria-label="Закрыть камеру"
        >
          <X size={22} />
        </button>
        <span className="camera-hint">
          {ready ? "Наведите перекрестие на этикетку" : "Включаем камеру…"}
        </span>
        {torch !== null ? (
          <button
            className={`camera-round ${torch ? "is-on" : ""}`}
            onClick={toggleTorch}
            aria-label={torch ? "Выключить фонарик" : "Включить фонарик"}
            aria-pressed={torch}
          >
            <Zap size={20} />
          </button>
        ) : (
          <span className="camera-round placeholder" />
        )}
      </div>
      <div className="camera-bottom">
        <button
          className="camera-round"
          onClick={onGallery}
          aria-label="Выбрать из галереи"
        >
          <ImagePlus size={22} />
        </button>
        <button
          className="camera-shutter"
          onClick={shoot}
          disabled={!ready}
          aria-label="Сделать снимок"
        />
        <span className="camera-round placeholder" />
      </div>
    </div>
  );
}

function Scanner({
  health,
  meta,
  initialFile = null,
  onResult,
  navigate,
  card,
}: {
  health: Health | null;
  meta: Meta | null;
  initialFile?: File | null;
  onResult: (r: ScanResult, photo: File) => void;
  navigate: (p: Page) => void;
  card: (w: Wine) => ReactNode;
}) {
  const input = useRef<HTMLInputElement>(null);
  const camera = useRef<HTMLInputElement>(null);
  const [file, setFile] = useState<File | null>(initialFile);
  // Where the current photo came from, so "retake" reopens the same picker:
  // the camera for a snapshot, the gallery for a chosen file.
  const [source, setSource] = useState<"camera" | "file">(
    initialFile ? "camera" : "file",
  );
  const [preview, setPreview] = useState("");
  const [dragging, setDragging] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const abort = useRef<AbortController | null>(null);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const active = useRef(true);
  useEffect(() => {
    active.current = true;
    return () => {
      active.current = false;
      abort.current?.abort();
      if (timer.current) clearTimeout(timer.current);
    };
  }, []);
  useEffect(() => {
    if (!file) {
      setPreview("");
      return;
    }
    const url = URL.createObjectURL(file);
    setPreview(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);
  // On a short phone the hero pushes the photo and its two buttons below the
  // fold; bring them up once a photo is chosen so "recognise" and "retake"
  // are both in reach without hunting.
  const previewActions = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!preview || !window.matchMedia("(max-width: 920px)").matches) return;
    const frame = requestAnimationFrame(() =>
      previewActions.current?.scrollIntoView({
        block: "end",
        behavior: "smooth",
      }),
    );
    return () => cancelAnimationFrame(frame);
  }, [preview]);
  const selectFile = (value?: File) => {
    setError("");
    if (!value) return;
    if (!["image/jpeg", "image/png", "image/webp"].includes(value.type)) {
      setError("Выберите изображение в формате JPG, PNG или WebP.");
      return;
    }
    if (value.size > (health?.max_upload_mb || 12) * 1024 * 1024) {
      setError(
        `Файл слишком большой. Максимум — ${health?.max_upload_mb || 12} МБ.`,
      );
      return;
    }
    setFile(value);
  };
  const changed =
    (from: "camera" | "file") => (event: ChangeEvent<HTMLInputElement>) => {
      const chosen = event.target.files?.[0];
      event.target.value = "";
      if (!chosen) return;
      setSource(from);
      selectFile(chosen);
    };
  const [live, setLive] = useState(false);
  const openCamera = () =>
    liveCameraAvailable() ? setLive(true) : camera.current?.click();
  const openFiles = () => input.current?.click();
  const reopen = () => (source === "camera" ? openCamera() : openFiles());
  const scan = async () => {
    if (busy || !file) return;
    setBusy(true);
    setError("");
    const controller = new AbortController();
    abort.current = controller;
    timer.current = setTimeout(() => controller.abort(), 120000);
    try {
      const body = new FormData();
      body.append("file", file);
      const response = await api<ScanResult>("/api/scan", {
        method: "POST",
        body,
        signal: controller.signal,
      });
      if (active.current) onResult(response, file);
    } catch (e) {
      if (active.current)
        setError(
          controller.signal.aborted
            ? "Запрос остановлен. Можно попробовать ещё раз."
            : errorText(e),
        );
    } finally {
      if (timer.current) clearTimeout(timer.current);
      if (active.current) setBusy(false);
    }
  };
  return (
    <>
      <section className="page-heading scanner-heading">
        <div>
          <div className="eyebrow heading-eyebrow">
            <span />
            ЗНАКОМСТВО НАЧИНАЕТСЯ С ЭТИКЕТКИ
          </div>
          <h1>
            Ваше вино.
            <br />
            <em>С первого взгляда.</em>
          </h1>
          <p>
            Сфотографируйте этикетку — и узнайте, что скрывается
            <br className="desktop-break" /> за ней: винодельня, характер и
            история вашего вина.
          </p>
        </div>
        <div className="catalog-count">
          <div>
            <Grape size={20} />
            <strong>{meta ? formatCount(meta.total) : "—"}</strong>
          </div>
          <span>
            российских вин
            <br />в одном сканере
          </span>
        </div>
      </section>
      <section className="scanner-grid" aria-label="Загрузка фотографии">
        <div className="upload-panel">
          <div className="panel-top">
            <span>
              <ScanLine size={18} />
              Сканер этикетки
            </span>
            <span className="step-label">01 / 02</span>
          </div>
          <input
            type="file"
            accept="image/jpeg,image/png,image/webp"
            className="visually-hidden"
            ref={input}
            onChange={changed("file")}
            aria-label="Выбрать фото этикетки"
          />
          <input
            type="file"
            accept="image/*"
            capture="environment"
            className="visually-hidden"
            ref={camera}
            onChange={changed("camera")}
            aria-label="Сфотографировать этикетку"
          />
          {live && (
            <CameraCapture
              onCapture={(shot) => {
                setLive(false);
                setSource("camera");
                selectFile(shot);
              }}
              onClose={() => setLive(false)}
              onFallback={() => {
                setLive(false);
                camera.current?.click();
              }}
              onGallery={() => {
                setLive(false);
                openFiles();
              }}
            />
          )}
          <div
            className={`drop-zone ${dragging ? "dragging" : ""} ${preview ? "has-preview" : ""}`}
            onDragOver={(e) => {
              e.preventDefault();
              if (!busy) setDragging(true);
            }}
            onDragLeave={() => setDragging(false)}
            onDrop={(e) => {
              e.preventDefault();
              setDragging(false);
              if (!busy) selectFile(e.dataTransfer.files[0]);
            }}
          >
            {preview ? (
              <>
                <img
                  className="upload-preview"
                  src={preview}
                  alt="Выбранная фотография этикетки"
                />
                <button
                  className="icon-button remove-preview"
                  aria-label="Убрать фото"
                  disabled={busy}
                  onClick={() => setFile(null)}
                >
                  <X size={18} />
                </button>
                <span className="preview-name">{file?.name}</span>
              </>
            ) : (
              <>
                <div className="upload-symbol">
                  <ScanLine size={36} strokeWidth={1.25} />
                  <span>
                    <ImagePlus size={14} />
                  </span>
                </div>
                <h2>
                  Давайте познакомимся
                  <br />с вашим вином
                </h2>
                <p>
                  Перетащите фото этикетки сюда
                  <br />
                  или выберите его на устройстве
                </p>
                <button
                  className="primary-button desktop-upload"
                  onClick={openFiles}
                  disabled={busy}
                >
                  <Upload size={17} />
                  Загрузить фото
                  <ArrowUpRight size={17} />
                </button>
                {/* On a phone the camera is the point: it leads, the
                    gallery follows, and drag-and-drop copy is hidden. */}
                <div className="mobile-scan-actions">
                  <button
                    className="primary-button"
                    onClick={openCamera}
                    disabled={busy}
                  >
                    <Camera size={19} />
                    Сфотографировать
                  </button>
                  <button
                    className="secondary-button"
                    onClick={openFiles}
                    disabled={busy}
                  >
                    <ImagePlus size={18} />
                    Выбрать из галереи
                  </button>
                </div>
                <span className="file-hint">
                  JPG, PNG, WEBP · до {health?.max_upload_mb || 12} МБ
                </span>
              </>
            )}
            {busy && (
              <div className="scanning-overlay" role="status">
                <div className="scan-animation" />
                <Loader2 className="spin" size={30} />
                <strong>Изучаем этикетку…</strong>
                <span>Ищем ваше вино в каталоге</span>
                <button
                  className="text-button"
                  onClick={() => abort.current?.abort()}
                >
                  Отменить
                </button>
              </div>
            )}
          </div>
          {preview ? (
            <div className="preview-actions" ref={previewActions}>
              <button
                className="primary-button scan-submit"
                disabled={busy}
                onClick={() => scan()}
              >
                <ScanLine size={18} />
                Распознать вино
                <ArrowRight size={18} />
              </button>
              <button
                className="secondary-button retake-button"
                disabled={busy}
                onClick={reopen}
              >
                {source === "camera" ? (
                  <Camera size={17} />
                ) : (
                  <ImagePlus size={17} />
                )}
                {source === "camera" ? "Переснять" : "Выбрать другое"}
              </button>
            </div>
          ) : (
            <button
              className="camera-button"
              onClick={openCamera}
              disabled={busy}
            >
              <Camera size={18} />
              Сделать фото с камеры
              <ArrowRight size={16} />
            </button>
          )}
          {error && (
            <div className="scan-error">
              <ErrorBox>{error}</ErrorBox>
              <button
                className="text-button"
                onClick={() => navigate("catalog")}
              >
                Найти в каталоге <ArrowRight size={17} />
              </button>
            </div>
          )}
          <div className="upload-privacy">
            <ShieldCheck size={14} />
            <span>Ваши фотографии не сохраняются</span>
          </div>
        </div>
        <div className="wine-visual">
          <div className="visual-grid" />
          <div className="visual-caption">
            <span className="eyebrow">РОДОМ ИЗ РОССИИ</span>
            <span>Ближе, чем кажется.</span>
          </div>
          <span className="visual-word">вино</span>
          <div className="bottle-stage">
            <div className="bottle-halo" />
            {meta?.featured[0] && (
              <WineImage wine={meta.featured[0]} className="hero-bottle" />
            )}
            <div className="scan-corners">
              <i />
              <i />
              <i />
              <i />
            </div>
            <span className="label-pointer">
              <span />
              <span>
                У каждой этикетки
                <br />
                есть своя история
              </span>
            </span>
          </div>
          <div className="visual-card">
            <span className="visual-check">
              <Check size={17} />
            </span>
            <div>
              <span>ОТ ЭТИКЕТКИ К ОТКРЫТИЮ</span>
              <strong>
                {meta?.featured[0]?.winery || "Знакомое вино. Новая история."}
              </strong>
              <small>
                {meta?.featured[0]?.region || "Российское виноделие"}
              </small>
            </div>
            <Grape size={24} strokeWidth={1.2} />
          </div>
        </div>
      </section>
      <section className="how-it-works" aria-label="Как это работает">
        {[
          {
            icon: Camera,
            title: "Один снимок",
            text: "Этикетка целиком, без бликов",
            n: "01",
          },
          {
            icon: ScanLine,
            title: "Точное знакомство",
            text: "Находим вино в нашем каталоге",
            n: "02",
          },
          {
            icon: BookOpen,
            title: "Больше, чем название",
            text: "Происхождение, сорта и описание",
            n: "03",
          },
        ].map(({ icon: Icon, title, text, n }) => (
          <div key={n}>
            <span className="how-icon">
              <Icon size={22} strokeWidth={1.5} />
            </span>
            <div>
              <h3>{title}</h3>
              <p>{text}</p>
            </div>
            <small>{n}</small>
          </div>
        ))}
      </section>
      <section className="discovery">
        <div className="section-heading">
          <div>
            <span className="eyebrow">ЕСТЬ ПОВОД ПОЗНАКОМИТЬСЯ</span>
            <h2>Откройте своё вино</h2>
          </div>
          <button className="text-button" onClick={() => navigate("catalog")}>
            Весь каталог <ArrowRight size={17} />
          </button>
        </div>
        <div className="wine-grid">
          {meta?.featured.slice(0, 4).map(card) || <Loading />}
        </div>
      </section>
      <div className="pairing-banner">
        <div className="pairing-banner-icon">
          <Utensils size={30} strokeWidth={1.2} />
        </div>
        <div>
          <span className="eyebrow">ВКУСНОЕ ПРОДОЛЖЕНИЕ</span>
          <h2>Хорошее вино любит компанию.</h2>
          <p>Подберите вино к ужину — или ужин к настроению.</p>
        </div>
        <button onClick={() => navigate("sommelier")}>
          Подобрать к столу <ArrowUpRight size={18} />
        </button>
      </div>
    </>
  );
}

function Catalog({
  meta,
  card,
}: {
  meta: Meta | null;
  card: (wine: Wine) => ReactNode;
}) {
  const [query, setQuery] = useState("");
  const [category, setCategory] = useState("");
  const [offset, setOffset] = useState(0);
  const [data, setData] = useState<CatalogPage | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [refresh, setRefresh] = useState(0);
  useEffect(() => {
    const controller = new AbortController();
    setLoading(true);
    setError("");
    const timer = window.setTimeout(() => {
      api<CatalogPage>(
        `/api/catalog?${new URLSearchParams({ q: query, category, offset: String(offset), limit: "24" })}`,
        { signal: controller.signal },
      )
        .then(setData)
        .catch((e) => {
          if (!controller.signal.aborted) setError(errorText(e));
        })
        .finally(() => {
          if (!controller.signal.aborted) setLoading(false);
        });
    }, 250);
    return () => {
      clearTimeout(timer);
      controller.abort();
    };
  }, [query, category, offset, refresh]);
  return (
    <>
      <PageHeading
        eyebrow="ВИНОДЕЛИЕ, КОТОРЫМ МЫ ГОРДИМСЯ"
        title="Откройте своё вино"
        text="Знакомые имена и новые открытия со всей России."
      />
      <div className="catalog-toolbar">
        <label className="search-field">
          <Search size={19} />
          <input
            value={query}
            onChange={(e) => {
              setQuery(e.target.value);
              setOffset(0);
            }}
            placeholder="Вино, винодельня, сорт или регион"
            aria-label="Поиск по каталогу"
          />
          {query && (
            <button
              className="icon-button"
              onClick={() => {
                setQuery("");
                setOffset(0);
              }}
              aria-label="Очистить поиск"
            >
              <X size={17} />
            </button>
          )}
        </label>
        <span className="catalog-total">
          {data ? formatCount(data.total) : "—"} вин в каталоге
        </span>
      </div>
      <div className="filter-row">
        <SlidersHorizontal size={17} />
        {["", ...(meta?.categories || [])].map((c) => (
          <button
            className={`chip ${category === c ? "selected" : ""}`}
            key={c}
            onClick={() => {
              setCategory(c);
              setOffset(0);
            }}
          >
            {c || "Все вина"}
          </button>
        ))}
      </div>
      {error ? (
        <ErrorBox retry={() => setRefresh((x) => x + 1)}>{error}</ErrorBox>
      ) : loading ? (
        <Loading />
      ) : data?.items.length ? (
        <>
          <div className="wine-grid catalog-grid">{data.items.map(card)}</div>
          <div className="pagination">
            <button
              className="secondary-button"
              disabled={offset === 0}
              onClick={() => {
                setOffset((x) => Math.max(0, x - 24));
                window.scrollTo(0, 0);
              }}
            >
              <ArrowLeft size={16} />
              Назад
            </button>
            <span>
              {offset + 1}–{Math.min(offset + 24, data.total)} из{" "}
              {formatCount(data.total)}
            </span>
            <button
              className="secondary-button"
              disabled={offset + 24 >= data.total}
              onClick={() => {
                setOffset((x) => x + 24);
                window.scrollTo(0, 0);
              }}
            >
              Далее
              <ArrowRight size={16} />
            </button>
          </div>
        </>
      ) : (
        <Empty
          icon={Search}
          title="Пока ничего не нашли"
          text="Попробуйте другое название, сорт винограда или уберите фильтр."
        />
      )}
    </>
  );
}

function PageHeading({
  eyebrow,
  title,
  text,
}: {
  eyebrow: string;
  title: string;
  text: string;
}) {
  return (
    <section className="page-heading">
      <span className="eyebrow heading-eyebrow">{eyebrow}</span>
      <h1>{title}</h1>
      <p>{text}</p>
    </section>
  );
}

/** Whether the local LLM sommelier is switched on, and once it has warmed up. */
function useSommelierStatus() {
  const [status, setStatus] = useState<SommelierStatus | null>(null);
  useEffect(() => {
    let alive = true;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const poll = () =>
      api<SommelierStatus>("/api/sommelier/status")
        .then((value) => {
          if (!alive) return;
          setStatus(value);
          // The models load in the background after startup; check back until ready.
          if (value.enabled && !value.ready && !value.error)
            timer = setTimeout(poll, 3000);
        })
        .catch(() => alive && setStatus({ enabled: false, ready: false, error: null }));
    poll();
    return () => {
      alive = false;
      if (timer) clearTimeout(timer);
    };
  }, []);
  return status;
}

/** The answer text with [N] citations turned into numbered badges. */
function SommelierText({
  text,
  numbers,
}: {
  text: string;
  numbers: Map<number, number>;
}) {
  const parts = text.split(/(\[\d+\])/g);
  return (
    <p className="sommelier-text">
      {parts.map((part, index) => {
        const match = /^\[(\d+)\]$/.exec(part);
        if (!match) return <span key={index}>{part}</span>;
        const shown = numbers.get(Number(match[1]));
        return shown ? (
          <sup key={index} className="sommelier-cite">
            {shown}
          </sup>
        ) : null;
      })}
    </p>
  );
}

/**
 * Ask the sommelier. On a wine card it answers about that wine and may suggest
 * alternatives; on the pairing page it picks wines from the whole catalog.
 * Renders `fallback` when the local model is not switched on.
 */
function SommelierChat({
  wine,
  suggestions,
  onPick,
  card,
  fallback = null,
}: {
  wine?: Wine;
  suggestions: string[];
  onPick?: (wine: Wine) => void;
  card?: (wine: Wine) => ReactNode;
  fallback?: ReactNode;
}) {
  const status = useSommelierStatus();
  const [question, setQuestion] = useState("");
  const [asked, setAsked] = useState("");
  const [answer, setAnswer] = useState<SommelierAnswer | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const abort = useRef<AbortController | null>(null);
  useEffect(() => () => abort.current?.abort(), []);
  if (!status) return null;
  if (!status.enabled || status.error) return <>{fallback}</>;
  const ask = async (text: string) => {
    const value = text.trim();
    if (!value || busy) return;
    setBusy(true);
    setError("");
    setAnswer(null);
    setAsked(value);
    setQuestion("");
    const controller = new AbortController();
    abort.current = controller;
    const timer = setTimeout(() => controller.abort(), 90000);
    try {
      setAnswer(
        await api<SommelierAnswer>("/api/sommelier", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ question: value, wine_slug: wine?.slug ?? null }),
          signal: controller.signal,
        }),
      );
    } catch (e) {
      setError(
        controller.signal.aborted
          ? "Сомелье не успел ответить. Попробуйте спросить ещё раз."
          : errorText(e),
      );
    } finally {
      clearTimeout(timer);
      setBusy(false);
    }
  };
  // On a wine card [1] is the wine itself: it needs no badge, and the
  // suggested alternatives are numbered from 1.
  const shownWines = answer
    ? answer.wines.filter((w) => !wine || w.slug !== wine.slug)
    : [];
  const numbers = new Map<number, number>();
  answer?.cited.forEach((n) => {
    const target = answer.wines[answer.cited.indexOf(n)];
    const position = shownWines.findIndex((w) => w.slug === target?.slug);
    if (position >= 0) numbers.set(n, position + 1);
  });
  return (
    <section className={`sommelier-chat ${wine ? "is-wine" : ""}`}>
      <div className="sommelier-head">
        <span className="sommelier-avatar">
          <Sparkles size={20} />
        </span>
        <div>
          <span className="eyebrow">
            {wine ? "СПРОСИТЕ ОБ ЭТОМ ВИНЕ" : "СПРОСИТЕ СОМЕЛЬЕ"}
          </span>
          <h2>
            {wine ? "Сомелье подскажет, как его подать" : "Что будем пить сегодня?"}
          </h2>
          <small>
            {!status.ready
              ? "Сомелье просыпается — это займёт около минуты…"
              : status.remote
                ? "Языковая модель через OpenRouter. Отвечает только по данным каталога."
                : "Локальная модель YandexGPT-5 Lite. Отвечает только по данным каталога."}
          </small>
        </div>
      </div>
      <div className="sommelier-suggestions">
        {suggestions.map((text) => (
          <button
            key={text}
            className="chip"
            disabled={busy || !status.ready}
            onClick={() => ask(text)}
          >
            {text}
          </button>
        ))}
      </div>
      <form
        className="sommelier-form"
        onSubmit={(event) => {
          event.preventDefault();
          ask(question);
        }}
      >
        <input
          value={question}
          onChange={(event) => setQuestion(event.target.value)}
          maxLength={500}
          placeholder={
            wine
              ? "Например: подойдёт ли к утке?"
              : "Например: что взять к пасте с морепродуктами?"
          }
          aria-label="Вопрос сомелье"
          disabled={busy || !status.ready}
        />
        <button
          className="primary-button"
          type="submit"
          disabled={busy || !status.ready || !question.trim()}
          aria-label="Спросить"
        >
          {busy ? <Loader2 className="spin" size={18} /> : <ArrowRight size={18} />}
        </button>
      </form>
      {(busy || answer || error) && (
        <div className="sommelier-dialog" aria-live="polite">
          <div className="sommelier-bubble is-guest">{asked}</div>
          {busy && (
            <div className="sommelier-bubble is-thinking">
              <span className="dots">
                <i />
                <i />
                <i />
              </span>
              Сомелье подбирает ответ…
            </div>
          )}
          {error && <ErrorBox>{error}</ErrorBox>}
          {answer && (
            <div className="sommelier-bubble is-answer">
              <SommelierText text={answer.answer} numbers={numbers} />
              <small>
                Рекомендация модели · {(answer.elapsed_ms / 1000).toFixed(1)} сек.
              </small>
            </div>
          )}
        </div>
      )}
      {shownWines.length > 0 &&
        (card ? (
          <div className="wine-grid sommelier-wines">
            {shownWines.map((w, index) => (
              <div key={w.slug} className="sommelier-pick">
                <span className="sommelier-cite">{index + 1}</span>
                {card(w)}
              </div>
            ))}
          </div>
        ) : (
          onPick && (
            <div className="sommelier-alternatives">
              <span className="alternatives-label">Сомелье советует</span>
              <CandidateList wines={shownWines} onPick={onPick} compact />
            </div>
          )
        ))}
    </section>
  );
}

function WineDetail({
  wine,
  result,
  photo,
  picked,
  saved,
  onSave,
  onBack,
  onPick,
  onRetake,
  onPair,
}: {
  wine: Wine;
  result: ScanResult | null;
  photo: string;
  picked: boolean;
  saved: boolean;
  onSave: () => void;
  onBack: () => void;
  onPick: (wine: Wine) => void;
  onRetake: () => void;
  onPair: () => void;
}) {
  const scanned = result !== null && result.status !== "demo";
  // The right wine is often among the first few candidates even when the
  // first answer misses - so the alternatives are part of the answer, not an
  // afterthought.
  const alternatives = scanned
    ? result.candidates
        .map((c) => c.wine)
        .filter((w) => w.slug !== wine.slug)
        .slice(0, 3)
    : [];
  return (
    <section className="wine-detail">
      <div className="detail-toolbar">
        <button className="text-button back-button" onClick={onBack}>
          <ArrowLeft size={17} />
          {result ? "К сканеру" : "Назад к винам"}
        </button>
        {scanned && (
          <button className="text-button retake-link" onClick={onRetake}>
            <Camera size={17} />
            Сделать другое фото
          </button>
        )}
      </div>
      {scanned && (picked || result.status === "matched") && (
        <div className={`result-notice ${picked ? "is-picked" : ""}`}>
          <span>
            {picked ? <Check size={18} /> : <CheckCheck size={18} />}
            {picked ? "Вы выбрали это вино из похожих" : "Ваше вино найдено"}
          </span>
          {!picked && (
            <small>{(result.elapsed_ms / 1000).toFixed(2)} сек.</small>
          )}
        </div>
      )}
      {alternatives.length > 0 && (
        <div className="alternatives">
          <span className="alternatives-label">
            {picked ? "Другие похожие вина" : "Не то вино? Возможно, это:"}
          </span>
          <CandidateList wines={alternatives} onPick={onPick} compact />
        </div>
      )}
      <div className="detail-grid">
        <div className={`detail-photo tone-${wine.category}`}>
          <span className="wine-tag">
            <i />
            {wine.category}
          </span>
          <span className="detail-photo-word">wine</span>
          <WineImage wine={wine} />
          {photo ? (
            <ScanPhoto src={photo} className="detail-scan-photo" />
          ) : (
            <span className="detail-photo-caption">РОССИЙСКОЕ ВИНОДЕЛИЕ</span>
          )}
        </div>
        <div className="detail-copy">
          <div className="eyebrow">{wine.winery}</div>
          <h1>{wine.name}</h1>
          <div className="detail-location">
            <MapPin size={17} />
            {wine.region || "Россия"}
          </div>
          <div className="detail-tags">
            <span>{wine.category}</span>
            {wine.grapes && <span>{wine.grapes}</span>}
          </div>
          <div className="detail-description">
            <h2>Характер вина</h2>
            <p>
              {wine.description ||
                "В каталоге пока нет подробного описания этого вина."}
            </p>
          </div>
          <dl className="wine-facts">
            <div>
              <dt>Винодельня</dt>
              <dd>{wine.winery || "Не указана"}</dd>
            </div>
            <div>
              <dt>Регион</dt>
              <dd>{wine.region || "Не указан"}</dd>
            </div>
            <div>
              <dt>Сорт винограда</dt>
              <dd>{wine.grapes || "Не указан"}</dd>
            </div>
            <div>
              <dt>Цвет</dt>
              <dd>{wine.color || wine.category}</dd>
            </div>
            <div>
              <dt>Рейтинг Роскачества</dt>
              <dd>{wine.rosquality_rating ?? "Нет данных в каталоге"}</dd>
            </div>
          </dl>
          <div className="detail-actions">
            <button
              className={`primary-button ${saved ? "saved-primary" : ""}`}
              onClick={onSave}
            >
              <Heart size={18} fill={saved ? "currentColor" : "none"} />
              {saved ? "В вашей коллекции" : "Сохранить в коллекцию"}
            </button>
            <a
              className="secondary-button"
              href={wine.source_url}
              target="_blank"
              rel="noreferrer"
            >
              Карточка источника
              <ArrowUpRight size={17} />
            </a>
          </div>
        </div>
      </div>
      <SommelierChat
        key={wine.slug}
        wine={wine}
        onPick={onPick}
        suggestions={[
          "К каким блюдам подать?",
          "Как подать и при какой температуре?",
          "Посоветуйте похожее вино",
        ]}
        fallback={
          <div className="detail-pairing">
            <span className="empty-icon">
              <Utensils size={24} />
            </span>
            <div>
              <span className="eyebrow">ПРОДОЛЖИМ ЗНАКОМСТВО?</span>
              <h2>Идеальная пара для вашего стола</h2>
              <p>Расскажите, что на ужин. Подберём подходящие стили вина.</p>
            </div>
            <button className="primary-button" onClick={onPair}>
              Подобрать пару
              <ArrowRight size={17} />
            </button>
          </div>
        }
      />
    </section>
  );
}

function Sommelier({ card }: { card: (wine: Wine) => ReactNode }) {
  const [dish, setDish] = useState("fish");
  const [preference, setPreference] = useState("any");
  const [result, setResult] = useState<Pairing | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const options = [
    { id: "meat", name: "Мясо и гриль", icon: Beef },
    { id: "fish", name: "Рыба и море", icon: Fish },
    { id: "cheese", name: "Сырная тарелка", icon: Milk },
    { id: "vegetables", name: "Овощи", icon: Salad },
    { id: "dessert", name: "Десерт", icon: CakeSlice },
  ];
  const submit = async () => {
    setBusy(true);
    setError("");
    setResult(null);
    try {
      setResult(
        await api<Pairing>("/api/pairing", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ dish, preference }),
        }),
      );
    } catch (e) {
      setError(errorText(e));
    } finally {
      setBusy(false);
    }
  };
  return (
    <>
      <PageHeading
        eyebrow="ВКУС СКЛАДЫВАЕТСЯ ИЗ ДЕТАЛЕЙ"
        title="Вино к вашему столу"
        text="Хорошая пара делает вечер особенным. Начнём с того, что вы готовите."
      />
      <SommelierChat
        card={card}
        suggestions={[
          "Что взять к утке с вишнёвым соусом?",
          "Лёгкое белое к рыбе на гриле",
          "Сладкое вино к шоколадному десерту",
          "Игристое для праздничного вечера",
        ]}
      />
      <section className="sommelier-panel">
        <span className="eyebrow">01 · ЧТО СЕГОДНЯ В МЕНЮ?</span>
        <div className="dish-options">
          {options.map(({ id, name, icon: Icon }) => (
            <button
              disabled={busy}
              key={id}
              className={dish === id ? "selected" : ""}
              onClick={() => {
                setDish(id);
                setResult(null);
              }}
              aria-pressed={dish === id}
            >
              <Icon size={34} strokeWidth={1.25} />
              <span>{name}</span>
              {dish === id && <Check size={14} />}
            </button>
          ))}
        </div>
        <span className="eyebrow">02 · КАКОЕ ВИНО ПРЕДПОЧИТАЕТЕ?</span>
        <div className="filter-row">
          {[
            ["any", "Доверюсь подбору"],
            ["red", "Красное"],
            ["white", "Белое"],
            ["rose", "Розовое"],
          ].map(([id, label]) => (
            <button
              disabled={busy}
              key={id}
              className={`chip ${preference === id ? "selected" : ""}`}
              onClick={() => {
                setPreference(id);
                setResult(null);
              }}
            >
              {label}
            </button>
          ))}
        </div>
        <div className="sommelier-submit">
          <span>
            <Leaf size={16} />
            Подбор по стилю вина и типу блюда
          </span>
          <button className="primary-button" disabled={busy} onClick={submit}>
            {busy ? (
              <Loader2 className="spin" size={18} />
            ) : (
              <Sparkles size={18} />
            )}
            Подобрать вино
            <ArrowRight size={18} />
          </button>
        </div>
      </section>
      {error && <ErrorBox>{error}</ErrorBox>}
      {result && (
        <section className="pairing-results">
          <div className="pairing-advice">
            <span className="empty-icon">
              <Utensils size={27} />
            </span>
            <div>
              <h2>{result.title}</h2>
              {result.cited ? (
                <SommelierText
                  text={result.explanation}
                  numbers={new Map(result.cited.map((n, index) => [n, index + 1]))}
                />
              ) : (
                <p>{result.explanation}</p>
              )}
              <small>{result.note}</small>
            </div>
          </div>
          {result.wines.length ? (
            <div className={`wine-grid ${result.cited ? "sommelier-wines" : ""}`}>
              {result.cited
                ? result.wines.map((w, index) => (
                    <div key={w.slug} className="sommelier-pick">
                      <span className="sommelier-cite">{index + 1}</span>
                      {card(w)}
                    </div>
                  ))
                : result.wines.map(card)}
            </div>
          ) : (
            <Empty
              icon={WineIcon}
              title="Такой пары пока нет"
              text="Попробуйте другой стиль вина — подходящего сочетания в текущем каталоге не нашлось."
            />
          )}
        </section>
      )}
    </>
  );
}

function Saved({
  slugs,
  card,
  navigate,
}: {
  slugs: string[];
  card: (wine: Wine) => ReactNode;
  navigate: (page: Page) => void;
}) {
  const [wines, setWines] = useState<Wine[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  useEffect(() => {
    let current = true;
    setBusy(true);
    setError("");
    Promise.all(
      slugs.map((slug) =>
        api<Wine>(`/api/catalog/${encodeURIComponent(slug)}`),
      ),
    )
      .then((data) => {
        if (current) setWines(data);
      })
      .catch((e) => {
        if (current) setError(errorText(e));
      })
      .finally(() => {
        if (current) setBusy(false);
      });
    return () => {
      current = false;
    };
  }, [slugs]);
  return (
    <>
      <PageHeading
        eyebrow="ТО, ЧТО ХОЧЕТСЯ ЗАПОМНИТЬ"
        title="Моя коллекция"
        text="Вина для особого случая, следующего ужина или нового открытия. Сохранены в этом браузере."
      />
      {error ? (
        <ErrorBox>{error}</ErrorBox>
      ) : busy ? (
        <Loading />
      ) : wines.length ? (
        <div className="wine-grid">{wines.map(card)}</div>
      ) : (
        <Empty
          icon={Heart}
          title="Ваша история только начинается"
          text="Нажмите на сердечко у понравившегося вина — оно будет ждать вас здесь."
          action={
            <button
              className="primary-button"
              onClick={() => navigate("catalog")}
            >
              Открыть каталог
              <ArrowRight size={17} />
            </button>
          }
        />
      )}
    </>
  );
}

function History({
  onResult,
  navigate,
}: {
  onResult: (result: ScanResult) => void;
  navigate: (page: Page) => void;
}) {
  const [items, setItems] = useState<ScanResult[]>([]);
  const [busy, setBusy] = useState(true);
  const [error, setError] = useState("");
  const [confirm, setConfirm] = useState(false);
  const [refresh, setRefresh] = useState(0);
  useEffect(() => {
    let current = true;
    setError("");
    setBusy(true);
    api<{ items: ScanResult[] }>("/api/history")
      .then((data) => {
        if (current)
          setItems(data.items.filter((item) => item.status !== "demo"));
      })
      .catch((e) => {
        if (current) setError(errorText(e));
      })
      .finally(() => {
        if (current) setBusy(false);
      });
    return () => {
      current = false;
    };
  }, [refresh]);
  const clear = async () => {
    try {
      await api("/api/history", { method: "DELETE" });
      setItems([]);
      setConfirm(false);
    } catch (e) {
      setError(errorText(e));
    }
  };
  return (
    <>
      <PageHeading
        eyebrow="ВАШИ НЕДАВНИЕ ОТКРЫТИЯ"
        title="История знакомства"
        text="Последние сканирования в этом браузере. История хранится 30 дней, сами фотографии — нет."
      />
      {error && (
        <ErrorBox retry={() => setRefresh((x) => x + 1)}>{error}</ErrorBox>
      )}
      {busy ? (
        <Loading />
      ) : items.length ? (
        <>
          <div className="history-toolbar">
            <span>{items.length} записей</span>
            {confirm ? (
              <div className="confirm-clear">
                <span>Очистить историю?</span>
                <button onClick={clear}>Да, очистить</button>
                <button onClick={() => setConfirm(false)}>Отмена</button>
              </div>
            ) : (
              <button className="text-button" onClick={() => setConfirm(true)}>
                <RotateCcw size={15} />
                Очистить историю
              </button>
            )}
          </div>
          <div className="history-list">
            {items.map((item) => (
              <button
                key={item.id}
                onClick={() => onResult(item)}
                className="history-row"
              >
                <div className="history-image">
                  {item.wine ? (
                    <WineImage wine={item.wine} />
                  ) : (
                    <ScanLine size={28} />
                  )}
                </div>
                <div>
                  <span className="eyebrow">
                    {item.status === "matched"
                      ? "ВИНО НАЙДЕНО"
                      : "НУЖНО ДРУГОЕ ФОТО"}
                  </span>
                  <h3>{item.wine?.name || "Без точного совпадения"}</h3>
                  <p>{item.wine?.winery || item.message}</p>
                </div>
                <time dateTime={item.created_at}>
                  {new Date(item.created_at).toLocaleString("ru-RU", {
                    day: "numeric",
                    month: "short",
                    hour: "2-digit",
                    minute: "2-digit",
                  })}
                </time>
                <ChevronRight size={20} />
              </button>
            ))}
          </div>
        </>
      ) : (
        !error && (
          <Empty
            icon={Clock3}
            title="Здесь появятся ваши открытия"
            text="Сфотографируйте этикетку — найденные вина появятся здесь."
            action={
              <button
                className="primary-button"
                onClick={() => navigate("scanner")}
              >
                К сканеру
                <ArrowRight size={17} />
              </button>
            }
          />
        )
      )}
    </>
  );
}

function Help({ onClose }: { onClose: () => void }) {
  const dialog = useRef<HTMLDialogElement>(null);
  useEffect(() => {
    dialog.current?.showModal();
  }, []);
  return (
    <dialog
      ref={dialog}
      className="help-dialog"
      onCancel={onClose}
      onClick={(e) => {
        if (e.target === dialog.current) onClose();
      }}
    >
      <div>
        <button
          className="icon-button modal-close"
          onClick={onClose}
          aria-label="Закрыть"
        >
          <X size={22} />
        </button>
        <div className="empty-icon">
          <ScanLine size={30} />
        </div>
        <span className="eyebrow">НЕСКОЛЬКО ПРОСТЫХ СОВЕТОВ</span>
        <h2>
          Хороший снимок —<br />
          точное знакомство
        </h2>
        <ol>
          <li>
            <strong>Этикетка в центре.</strong> Снимите одну бутылку крупно,
            чтобы название и год были читаемы.
          </li>
          <li>
            <strong>Мягкий свет.</strong> Избегайте бликов, вспышки и сильного
            наклона.
          </li>
          <li>
            <strong>Честный результат.</strong> Если этикетки слишком похожи,
            сканер попросит другой снимок.
          </li>
        </ol>
        <p>
          Не нашли совпадение? Попробуйте поиск по названию, винодельне или
          сорту винограда в каталоге.
        </p>
        <button className="primary-button" onClick={onClose}>
          Всё понятно
          <Check size={17} />
        </button>
      </div>
    </dialog>
  );
}
