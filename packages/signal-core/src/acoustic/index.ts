// SP-7: Acoustic Simulation
export {
  OCTAVE_BANDS,
  MATERIAL_DB,
  getMaterial,
  weightedAbsorption,
} from './material-database.js';

export {
  calculateRT60,
  furnitureAcousticImpact,
} from './rt60.js';

export {
  estimateSTI,
  estimateSTIFromRT60,
} from './sti.js';

export {
  generateShoeboxRIR,
  convolveWithRIR,
} from './impulse-response.js';
