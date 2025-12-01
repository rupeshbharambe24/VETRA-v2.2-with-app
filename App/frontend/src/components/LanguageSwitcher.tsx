import React from 'react';
import { useTranslation } from 'react-i18next';
import { Languages, Globe } from 'lucide-react';
import { Button } from '@/components/ui/button';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import { languages } from '@/lib/i18n';
import { useToast } from '@/hooks/use-toast';

const LanguageSwitcher = () => {
  const { i18n, t } = useTranslation();
  const { toast } = useToast();

  const handleLanguageChange = (languageCode: string) => {
    i18n.changeLanguage(languageCode);
    const selectedLanguage = languages.find(lang => lang.code === languageCode);
    
    toast({
      title: t('language.languageChanged', { language: selectedLanguage?.nativeName }),
      duration: 2000,
    });
  };

  const currentLanguage = languages.find(lang => lang.code === i18n.language) || languages[0];

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button variant="ghost" size="sm" className="h-9 px-2">
          <Globe className="w-4 h-4 mr-1" />
          <span className="hidden sm:inline">{currentLanguage.nativeName}</span>
          <Languages className="w-3 h-3 ml-1" />
        </Button>
      </DropdownMenuTrigger>
      
      <DropdownMenuContent 
        align="end" 
        className="w-48 max-h-96 overflow-y-auto bg-card border border-border shadow-lg z-50"
      >
        {languages.map((language) => (
          <DropdownMenuItem
            key={language.code}
            onClick={() => handleLanguageChange(language.code)}
            className={`cursor-pointer py-2 px-3 hover:bg-accent ${
              i18n.language === language.code ? 'bg-accent font-medium' : ''
            }`}
          >
            <div className="flex flex-col">
              <span className="font-medium">{language.nativeName}</span>
              <span className="text-xs text-muted-foreground">{language.name}</span>
            </div>
          </DropdownMenuItem>
        ))}
      </DropdownMenuContent>
    </DropdownMenu>
  );
};

export default LanguageSwitcher;