import React, { useState } from 'react';
import { useDropzone } from 'react-dropzone';

interface Props {
  onTextReady: (text: string) => void;
}

const InputArea: React.FC<Props> = ({ onTextReady }) => {
  const [text, setText] = useState('');

  const handleDrop = (acceptedFiles: File[]) => {
    const reader = new FileReader();
    reader.onload = () => {
      const fileText = reader.result as string;
      setText(fileText);
      onTextReady(fileText);
    };
    reader.readAsText(acceptedFiles[0]);
  };

  const { getRootProps, getInputProps } = useDropzone({ onDrop: handleDrop });

  return (
    <div className="input-area">
      <textarea
        value={text}
        onChange={(e) => {
          setText(e.target.value);
          onTextReady(e.target.value);
        }}
        placeholder="Paste or type your speech text here..."
      />
      <div {...getRootProps()} className="dropzone">
        <input {...getInputProps()} />
        <p>Drag and drop a .txt file here</p>
      </div>
    </div>
  );
};

export default InputArea;
