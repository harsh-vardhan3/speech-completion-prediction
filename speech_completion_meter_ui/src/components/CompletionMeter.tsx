import React from 'react';

interface Props {
  percentage: number;
}

const CompletionMeter: React.FC<Props> = ({ percentage }) => {
  return (
    <div className="w-full h-4 bg-gray-200 rounded-full overflow-hidden">
      <div
        className="h-full bg-blue-500 transition-all duration-300"
        style={{ width: `${percentage}%` }}
      ></div>
    </div>
  );
};
export default CompletionMeter;