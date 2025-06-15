// TimestampDisplay.tsx
import React from 'react';

interface Props {
  timestamp: string;
}

const TimestampDisplay: React.FC<Props> = ({ timestamp }) => {
  return <p className="text-center">Timestamp: {timestamp}</p>;
};

export default TimestampDisplay;
