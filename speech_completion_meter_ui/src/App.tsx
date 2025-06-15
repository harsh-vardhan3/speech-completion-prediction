// src/App.tsx
import React, { useState } from 'react';
import TextInput from './components/TextInput';
import FileDrop from './components/FileDrop';
import CompletionMeter from './components/CompletionMeter';
import DummyGraph from './components/DummyGraph';

const App: React.FC = () => {
  const [text, setText] = useState('');
  const [progress, setProgress] = useState(0);
  const [startTime, setStartTime] = useState<string | null>(null);
  const [endTime, setEndTime] = useState<string | null>(null);
  const [done, setDone] = useState(false);

  const handlePredict = async () => {
    setProgress(0);
    setDone(false);
    const start = new Date();
    setStartTime(start.toLocaleTimeString());
    const interval = setInterval(() => {
      setProgress((prev) => {
        const next = prev + 10;
        if (next >= 100) {
          clearInterval(interval);
          setDone(true);
          const end = new Date();
          setEndTime(end.toLocaleTimeString());
        }
        return next;
      });
    }, 300);
  };

  return (
    <div className="p-6 max-w-2xl mx-auto space-y-4">
      <h1 className="text-xl font-bold text-center">Speech Completion Meter</h1>
      <TextInput onChange={setText} />
      <FileDrop onFileRead={setText} />
      <button
        onClick={handlePredict}
        className="px-4 py-2 bg-blue-500 text-white rounded"
      >
        Predict
      </button>
      <CompletionMeter percentage={progress} />
      {startTime && <p className="text-center">Speech started at: {startTime}</p>}
      {endTime && <p className="text-center">Speech ended at: {endTime}</p>}
      {done && <>
        <p className="text-green-600 text-center font-semibold">Analysis complete!</p>
        <DummyGraph />
      </>}
    </div>
  );
};
export default App;

// src/components/CompletionMeter.tsx




// src/components/DummyGraph.tsx
