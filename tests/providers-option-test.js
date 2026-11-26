import { Session } from 'ort';

console.log('Testing Session with providers option...');

// Test 1: Session with providers option
try {
  const session = new Session('/home/roman/Sources/quickjs/qjs-opencv/models/u2netp.onnx', { providers: ['cpu'] });
  console.log('Session with providers option: OK');
  console.log('inputNames:', session.inputNames);
  console.log('outputNames:', session.outputNames);
  session.release();
} catch(e) {
  console.log('Session with providers option: FAIL -', e.message);
}

// Test 2: Session without providers option (should still work)
try {
  const session = new Session('/home/roman/Sources/quickjs/qjs-opencv/models/u2netp.onnx');
  console.log('Session without providers: OK');
  console.log('inputNames:', session.inputNames);
  session.release();
} catch(e) {
  console.log('Session without providers: FAIL -', e.message);
}

// Test 3: Session with invalid provider
try {
  const session = new Session('/home/roman/Sources/quickjs/qjs-opencv/models/u2netp.onnx', { providers: ['nonexistent'] });
  console.log('Session with invalid provider: FAIL - should have thrown');
} catch(e) {
  console.log('Session with invalid provider: OK -', e.name, e.message);
}

// Test 4: Session with empty providers array
try {
  const session = new Session('/home/roman/Sources/quickjs/qjs-opencv/models/u2netp.onnx', { providers: [] });
  console.log('Session with empty providers: OK');
  session.release();
} catch(e) {
  console.log('Session with empty providers: FAIL -', e.message);
}

console.log('All tests completed');