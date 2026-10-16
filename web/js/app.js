// Entry point of the standalone bundle: every page is bundled into one script and the
// page to run is chosen by <body data-page="...">. Each page module only runs when loaded.
const pages = {
  index: () => import('./pages/index.js'),
  core: () => import('./pages/core.js'),
  behaviorism: () => import('./pages/behaviorism.js'),
  humanism: () => import('./pages/humanism.js'),
  cognitivism: () => import('./pages/cognitivism.js'),
  constructivism: () => import('./pages/constructivism.js'),
  social: () => import('./pages/social.js'),
  development: () => import('./pages/development.js'),
  andragogy: () => import('./pages/andragogy.js'),
  motivation: () => import('./pages/motivation.js'),
  classroom: () => import('./pages/classroom.js'),
  theories: () => import('./pages/theories.js'),
  allocation: () => import('./pages/allocation.js'),
  sensitivity: () => import('./pages/sensitivity.js'),
};

const id = document.body.dataset.page || 'index';
(pages[id] || pages.index)();
