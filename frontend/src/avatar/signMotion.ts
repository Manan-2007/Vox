/**
 * The avatar's motion format, loader and player.
 *
 * Mirrors the layout written by ml/build_motion.py. One file per sign, fetched
 * the first time that sign is needed and cached for the session — the whole
 * library is several megabytes, but a sentence only ever needs a handful of
 * words, so nothing is paid for up front.
 *
 * Frame layout, 140 floats:
 *
 *     [  0 :  2 ]  left  wrist, image space (x, y); 0,0 means absent
 *     [  2 : 65 ]  left  hand, 21 world landmarks (x, y, z) in metres
 *     [ 65 : 67 ]  right wrist, image space
 *     [ 67 :130 ]  right hand, 21 world landmarks
 *     [130 :140 ]  pose, image space (x, y): nose, L/R shoulder, L/R elbow
 *
 * Image-space values place a hand in the signing space around the body; the
 * world landmarks carry its actual 3D shape and orientation. See the header of
 * ml/build_motion.py for why both are needed.
 */

export const FRAME_FLOATS = 140;
export const LEFT_WRIST = 0;
export const LEFT_WORLD = 2;
export const RIGHT_WRIST = 65;
export const RIGHT_WORLD = 67;
export const POSE_BLOCK = 130;
export const POSE_NOSE = 0;
export const POSE_L_SHOULDER = 1;
export const POSE_R_SHOULDER = 2;
export const POSE_L_ELBOW = 3;
export const POSE_R_ELBOW = 4;

export interface SignMotion {
  gloss: string;
  english: string;
  pos: string;
  fps: number;
  frames: number[][];
}

export interface ManifestEntry {
  english: string;
  pos: string;
  frames: number;
  seconds: number;
  twoHanded: boolean;
  quality: number;
}

export interface SignManifest {
  fps: number;
  signs: Record<string, ManifestEntry>;
}

/* ----------------------------------------------------------------- loading -- */

const cache = new Map<string, SignMotion>();
const inFlight = new Map<string, Promise<SignMotion | null>>();

let manifestPromise: Promise<SignManifest> | null = null;

export function loadManifest(): Promise<SignManifest> {
  manifestPromise ??= fetch("/signs/manifest.json")
    .then((response) => {
      if (!response.ok) throw new Error(`manifest ${response.status}`);
      return response.json() as Promise<SignManifest>;
    })
    .catch(() => ({ fps: 15, signs: {} }));
  return manifestPromise;
}

export function loadSign(gloss: string): Promise<SignMotion | null> {
  const cached = cache.get(gloss);
  if (cached) return Promise.resolve(cached);

  let pending = inFlight.get(gloss);
  if (!pending) {
    pending = fetch(`/signs/${encodeURIComponent(gloss)}.json`)
      .then(async (response) => {
        if (!response.ok) return null;
        const motion = (await response.json()) as SignMotion;
        if (!Array.isArray(motion.frames) || motion.frames.length === 0) return null;
        cache.set(gloss, motion);
        return motion;
      })
      .catch(() => null)
      .finally(() => inFlight.delete(gloss));
    inFlight.set(gloss, pending);
  }
  return pending;
}

/** Warm the cache without waiting — used the moment a phrase is glossed. */
export function prefetchSigns(glosses: readonly string[]): void {
  for (const gloss of glosses) void loadSign(gloss);
}

/* ----------------------------------------------------------- interpolation -- */

/**
 * Read a sign at an arbitrary (fractional) frame position.
 *
 * Blending two frames component-wise is only valid where both frames have the
 * same hands present: interpolating between "hand here" and "no hand" would
 * produce a hand at half strength in the wrong place. So presence is taken from
 * the nearer frame, and a hand present in only one of the two is used as-is.
 */
export function sampleFrame(
  motion: SignMotion,
  position: number,
  out: Float32Array,
): Float32Array {
  const frames = motion.frames;
  const clamped = Math.max(0, Math.min(frames.length - 1, position));
  const low = Math.floor(clamped);
  const high = Math.min(frames.length - 1, low + 1);
  const t = clamped - low;
  const a = frames[low];
  const b = frames[high];

  blendBlock(out, a, b, t, LEFT_WRIST, LEFT_WORLD);
  blendBlock(out, a, b, t, RIGHT_WRIST, RIGHT_WORLD);
  for (let i = POSE_BLOCK; i < FRAME_FLOATS; i += 1) {
    out[i] = a[i] === 0 || b[i] === 0 ? (t < 0.5 ? a[i] : b[i]) : a[i] * (1 - t) + b[i] * t;
  }
  return out;
}

function blendBlock(
  out: Float32Array,
  a: number[],
  b: number[],
  t: number,
  wrist: number,
  world: number,
): void {
  const inA = a[wrist] !== 0 || a[wrist + 1] !== 0;
  const inB = b[wrist] !== 0 || b[wrist + 1] !== 0;
  const copy = (source: number[]) => {
    out[wrist] = source[wrist];
    out[wrist + 1] = source[wrist + 1];
    for (let i = 0; i < 63; i += 1) out[world + i] = source[world + i];
  };
  const clear = () => {
    out[wrist] = 0;
    out[wrist + 1] = 0;
    for (let i = 0; i < 63; i += 1) out[world + i] = 0;
  };

  if (!inA && !inB) return clear();
  if (inA && !inB) return t < 0.5 ? copy(a) : clear();
  if (!inA && inB) return t < 0.5 ? clear() : copy(b);

  out[wrist] = a[wrist] * (1 - t) + b[wrist] * t;
  out[wrist + 1] = a[wrist + 1] * (1 - t) + b[wrist + 1] * t;
  for (let i = 0; i < 63; i += 1) {
    out[world + i] = a[world + i] * (1 - t) + b[world + i] * t;
  }
}

/* ------------------------------------------------------------------ player -- */

export interface QueueItem {
  gloss: string;
  /** What to show as the caption; may differ from the gloss ("thank you"). */
  label: string;
  motion: SignMotion | null;
  /** True when there is no sign for this word and it can only be named. */
  missing: boolean;
}

export interface PlayerState {
  index: number;
  /** 0-1 through the current sign. */
  progress: number;
  playing: boolean;
  finished: boolean;
}

/**
 * Plays a queue of signs as one continuous animation.
 *
 * Between two signs it holds briefly and then cross-fades, because that is what
 * a signer does: signs are separated by a short transition, not cut together.
 * Without the hold, a sentence reads as one long twitch; without the fade, the
 * hands teleport between the end of one sign and the start of the next.
 */
export class SignPlayer {
  private queue: QueueItem[] = [];
  private index = 0;
  private position = 0;
  private holding = 0;
  private playing = false;
  /** Holds the next sign's opening frame during a cross-fade. */
  private readonly scratchB = new Float32Array(FRAME_FLOATS);

  /** Frames held at the end of a sign before moving on, at the sign's own fps. */
  static readonly HOLD_FRAMES = 3;
  /** Frames of cross-fade into the next sign. */
  static readonly BLEND_FRAMES = 3;

  speed = 1;
  loop = false;

  setQueue(items: QueueItem[]): void {
    this.queue = items;
    this.index = 0;
    this.position = 0;
    this.holding = 0;
    this.playing = items.length > 0;
  }

  clear(): void {
    this.queue = [];
    this.playing = false;
    this.position = 0;
    this.index = 0;
  }

  get items(): readonly QueueItem[] {
    return this.queue;
  }

  get current(): QueueItem | null {
    return this.queue[this.index] ?? null;
  }

  get state(): PlayerState {
    const motion = this.current?.motion;
    const length = motion ? motion.frames.length : 1;
    return {
      index: this.index,
      progress: Math.min(1, this.position / Math.max(1, length - 1)),
      playing: this.playing,
      finished: !this.playing && this.index >= this.queue.length - 1,
    };
  }

  play(): void {
    if (this.queue.length === 0) return;
    // Replay from the top once the queue has run out.
    if (this.index >= this.queue.length - 1 && !this.playing) {
      const motion = this.current?.motion;
      if (motion && this.position >= motion.frames.length - 1) {
        this.index = 0;
        this.position = 0;
      }
    }
    this.playing = true;
  }

  pause(): void {
    this.playing = false;
  }

  toggle(): void {
    this.playing ? this.pause() : this.play();
  }

  restart(): void {
    this.index = 0;
    this.position = 0;
    this.holding = 0;
    this.playing = this.queue.length > 0;
  }

  seek(fraction: number): void {
    const motion = this.current?.motion;
    if (!motion) return;
    this.position = Math.max(0, Math.min(1, fraction)) * (motion.frames.length - 1);
  }

  step(index: number): void {
    this.index = Math.max(0, Math.min(this.queue.length - 1, index));
    this.position = 0;
    this.holding = 0;
  }

  /**
   * Advance by `dtSeconds` and write the frame to render into `out`.
   * Returns false when there is nothing to draw.
   */
  advance(dtSeconds: number, out: Float32Array): boolean {
    const item = this.current;
    if (!item) return false;
    const motion = item.motion;

    if (!motion) {
      // A word with no sign still occupies time, so the caption can be read.
      if (this.playing) {
        this.position += dtSeconds * 15 * this.speed;
        if (this.position > 12) this.next();
      }
      return false;
    }

    const last = motion.frames.length - 1;
    if (this.playing) {
      if (this.holding > 0) {
        this.holding -= dtSeconds * motion.fps * this.speed;
        if (this.holding <= 0) this.next();
      } else {
        this.position += dtSeconds * motion.fps * this.speed;
        if (this.position >= last) {
          this.position = last;
          this.holding = SignPlayer.HOLD_FRAMES;
        }
      }
    }

    sampleFrame(motion, this.position, out);

    // Cross-fade into the next sign over the final frames of this one.
    const next = this.queue[this.index + 1];
    const remaining = last - this.position;
    if (next?.motion && remaining < SignPlayer.BLEND_FRAMES && this.holding <= 0) {
      const t = 1 - remaining / SignPlayer.BLEND_FRAMES;
      sampleFrame(next.motion, 0, this.scratchB);
      crossFade(out, this.scratchB, t * 0.5);
    }
    return true;
  }

  private next(): void {
    this.holding = 0;
    if (this.index < this.queue.length - 1) {
      this.index += 1;
      this.position = 0;
      return;
    }
    if (this.loop) {
      this.index = 0;
      this.position = 0;
      return;
    }
    this.playing = false;
  }
}

/** Blend `into` toward `other` by t, respecting hand presence. */
function crossFade(into: Float32Array, other: Float32Array, t: number): void {
  for (const [wrist, world] of [
    [LEFT_WRIST, LEFT_WORLD],
    [RIGHT_WRIST, RIGHT_WORLD],
  ]) {
    const hasInto = into[wrist] !== 0 || into[wrist + 1] !== 0;
    const hasOther = other[wrist] !== 0 || other[wrist + 1] !== 0;
    if (!hasInto || !hasOther) continue; // presence changes are not blendable
    into[wrist] += (other[wrist] - into[wrist]) * t;
    into[wrist + 1] += (other[wrist + 1] - into[wrist + 1]) * t;
    for (let i = 0; i < 63; i += 1) {
      into[world + i] += (other[world + i] - into[world + i]) * t;
    }
  }
}

/* ---------------------------------------------- live frames -> this format -- */

/**
 * Convert the recogniser's 141-float live frame plus its world landmarks into
 * the avatar's 140-float layout, so the same rig can show your own signing.
 *
 * The two formats disagree about x by exactly one factor, and getting it wrong
 * makes the live figure a different shape from the reference one. The
 * recogniser's frames are normalized to a 16:9 reference aspect (see
 * REFERENCE_ASPECT in tracking.ts, and ml/collect.py), because that is the
 * geometry every training recording was in. The motion library stores SQUARE
 * coordinates, where one unit of x and one unit of y are the same distance,
 * because the rig has to compute real angles and lengths from them. Multiplying
 * by the reference aspect converts the first into the second.
 */
const X_TO_SQUARE = 16 / 9;

export function fromLiveFrame(
  frame: Float32Array,
  world: Float32Array,
  out: Float32Array,
): Float32Array {
  out.fill(0);
  for (let block = 0; block < 2; block += 1) {
    const imageBase = block * 63;
    const present =
      frame[imageBase] !== 0 || frame[imageBase + 1] !== 0;
    const wrist = block === 0 ? LEFT_WRIST : RIGHT_WRIST;
    const worldOut = block === 0 ? LEFT_WORLD : RIGHT_WORLD;
    if (!present) continue;
    out[wrist] = frame[imageBase] * X_TO_SQUARE;
    out[wrist + 1] = frame[imageBase + 1];
    for (let i = 0; i < 63; i += 1) out[worldOut + i] = world[block * 63 + i];
  }
  // Pose: the live frame carries x, y, z per point; the avatar format keeps x, y.
  for (let slot = 0; slot < 5; slot += 1) {
    out[POSE_BLOCK + slot * 2] = frame[126 + slot * 3] * X_TO_SQUARE;
    out[POSE_BLOCK + slot * 2 + 1] = frame[126 + slot * 3 + 1];
  }
  return out;
}
