import React from 'react';

interface Props {
  onChange: (text: string) => void;
}

const TextInput: React.FC<Props> = ({ onChange }) => {
  return (
    <textarea
      placeholder="Paste your speech text here..."
      onChange={(e) => onChange(e.target.value)}
      className="w-full p-2 border rounded"
      rows={8}
    />
  );
};

export default TextInput;
