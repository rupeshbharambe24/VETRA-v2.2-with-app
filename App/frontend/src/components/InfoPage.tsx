import React from 'react';
import { Link } from 'react-router-dom';
import { ArrowLeft } from 'lucide-react';
import { Button } from '@/components/ui/button';

interface InfoPageProps {
  title: string;
  children: React.ReactNode;
}

const InfoPage: React.FC<InfoPageProps> = ({ title, children }) => {
  return (
    <div className="vetra-container">
      {/* Header */}
      <header className="flex items-center gap-4 p-4 border-b border-border bg-card">
        <Link to="/">
          <Button variant="ghost" size="icon">
            <ArrowLeft className="w-5 h-5" />
          </Button>
        </Link>
        <h1 className="text-xl font-bold">{title}</h1>
      </header>

      {/* Content */}
      <div className="p-6">
        {children}
      </div>
    </div>
  );
};

export default InfoPage;