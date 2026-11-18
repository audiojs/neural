// stat manifest: DNSMOS P.835 of a whole signal, { sig, bak, ovrl, segments } (quality.js). The host reads the PCM and
// calls compute once, the (channels, { sampleRate }) shape dnsmos() takes; compute returns a Promise, which the host's
// stat method awaits. Options pass through: model (a load() handle), windows, weights, backend. As a stat it hears the
// whole clip (windows 'all'): the reference script skips windows 7 to 23, so it never hears 15 to 24 s of a clip of
// 17 s or more, and a verdict on an edit must not rest on a part of it; `windows: 'upstream'` gives the script's numbers.

import score from './quality.js'

export const dnsmos = {
	stat: 'dnsmos',
	compute: (channels, { sampleRate, ...opts }) => score(channels, { sampleRate, windows: 'all', ...opts }),
}
