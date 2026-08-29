/** NAM (.nam) playback — classic standard WaveNet, dependency-free. */

export interface NamLayerConfig {
  input_size: number
  condition_size: number
  head_size: number
  channels: number
  kernel_size: number
  dilations: number[]
  activation: string
  gated: boolean
  head_bias: boolean
}

export interface NamModel {
  arrays: object[]
  headScale: number
  /** samples of history the net looks back on (warmup transient length) */
  receptiveField: number
  sampleRate: number | null
  version: string
}

/** Parse a .nam file (JSON string or object). Throws on unsupported variants. */
export function parse(source: string | object): NamModel

/** Run the model over a buffer in place (offline, zero left-padded warmup). */
export function process(model: NamModel, data: Float32Array): Float32Array

export interface NamAmp {
  (data: Float32Array): Float32Array
  model: NamModel
  receptiveField: number
  sampleRate: number | null
}

/** nam(json) → callable amp: amp(data) processes in place and returns it. */
export default function nam(source: string | object): NamAmp
