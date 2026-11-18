// the existing ops on the validation pairs' inputs: deepfilter() at its default limit, deepfilter(0), auto('speech');
// run from audio's repository, whose audio.js and bench/rx/lib.mjs it imports
import { readdirSync, existsSync } from 'node:fs'
import { pathToFileURL } from 'node:url'
const from = file => import(pathToFileURL(`${process.cwd()}/${file}`).href)
const { default: audio } = await from('audio.js'), { readWav, writeWav } = await from('bench/rx/lib.mjs')
const E = process.argv[2]
const SYSTEMS = { dfn18: a => a.deepfilter(), dfn0: a => a.deepfilter(0), auto: a => a.auto('speech') }
for (let f of readdirSync(E).filter(f => f.endsWith('-in.wav')).sort()) {
  let x = readWav(`${E}/${f}`)
  for (let [name, fn] of Object.entries(SYSTEMS)) {
    let out = `${E}/${f.replace('-in.wav', '-' + name + '.wav')}`
    if (existsSync(out)) continue
    try {
      let y = (await fn(audio.from(x.ch, { sampleRate: x.sr })).read()).map(c => Float32Array.from(c))
      writeWav(out, y, x.sr); console.log(out.split('/').pop())
    } catch (e) { console.log('fail', name, f, e.message.slice(0, 120)) }
  }
}
process.exit(0)
