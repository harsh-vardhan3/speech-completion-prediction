import React from 'react';
import { useDropzone } from 'react-dropzone';

interface Props {
  onFileRead: (content: string) => void;
}

const FileDrop: React.FC<Props> = ({ onFileRead }) => {
  const onDrop = (acceptedFiles: File[]) => {
    const reader = new FileReader();
    reader.onload = () => {
      onFileRead(reader.result as string);
    };
    reader.readAsText(acceptedFiles[0]);
  };

  const { getRootProps, getInputProps } = useDropzone({ onDrop });

  return (
    <div {...getRootProps()} className="border-dashed border-2 p-4 rounded text-center cursor-pointer">
      <input {...getInputProps()} />
      <p>Drag & drop a .txt/.pdf file here, or click to select</p>
    </div>
  );
};

export default FileDrop;
