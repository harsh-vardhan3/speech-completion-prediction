// File: src/components/DragAndDrop.tsx
import React, { useState } from 'react';

interface DragAndDropProps {
  onTextExtracted: (text: string) => void;
}

const DragAndDrop: React.FC<DragAndDropProps> = ({ onTextExtracted }) => {
  const [dragging, setDragging] = useState(false);

  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setDragging(true);
  };

  const handleDragLeave = () => {
    setDragging(false);
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setDragging(false);
    const file = e.dataTransfer.files[0];

    if (file && (file.type === 'text/plain' || file.type === 'application/pdf')) {
      const reader = new FileReader();
      reader.onload = (event) => {
        const text = event.target?.result as string;
        onTextExtracted(text);
      };
      reader.readAsText(file);
    } else {
      alert('Please drop a .txt or .pdf file.');
    }
  };

  return (
    <div
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
      style={{
        border: dragging ? '2px dashed #4caf50' : '2px dashed #ccc',
        borderRadius: '8px',
        padding: '20px',
        textAlign: 'center',
        marginBottom: '20px'
      }}
    >
      {dragging ? 'Drop the file here...' : 'Drag and drop a .txt or .pdf file here'}
    </div>
  );
};

export default DragAndDrop;
