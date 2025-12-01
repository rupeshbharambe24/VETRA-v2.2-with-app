import i18n from 'i18next';
import { initReactI18next } from 'react-i18next';
import LanguageDetector from 'i18next-browser-languagedetector';

// Import translation files
import enTranslations from '../locales/en.json';
import hiTranslations from '../locales/hi.json';
import mrTranslations from '../locales/mr.json';
import guTranslations from '../locales/gu.json';
import paTranslations from '../locales/pa.json';
import taTranslations from '../locales/ta.json';
import teTranslations from '../locales/te.json';
import knTranslations from '../locales/kn.json';
import mlTranslations from '../locales/ml.json';
import bnTranslations from '../locales/bn.json';
import asTranslations from '../locales/as.json';
import orTranslations from '../locales/or.json';
import urTranslations from '../locales/ur.json';

// Language configuration
export const languages = [
  { code: 'en', name: 'English', nativeName: 'English' },
  { code: 'hi', name: 'Hindi', nativeName: 'हिंदी' },
  { code: 'mr', name: 'Marathi', nativeName: 'मराठी' },
  { code: 'gu', name: 'Gujarati', nativeName: 'ગુજરાતી' },
  { code: 'pa', name: 'Punjabi', nativeName: 'ਪੰਜਾਬੀ' },
  { code: 'ta', name: 'Tamil', nativeName: 'தமிழ்' },
  { code: 'te', name: 'Telugu', nativeName: 'తెలుగు' },
  { code: 'kn', name: 'Kannada', nativeName: 'ಕನ್ನಡ' },
  { code: 'ml', name: 'Malayalam', nativeName: 'മലയാളം' },
  { code: 'bn', name: 'Bengali', nativeName: 'বাংলা' },
  { code: 'as', name: 'Assamese', nativeName: 'অসমীয়া' },
  { code: 'or', name: 'Odia', nativeName: 'ଓଡ଼ିଆ' },
  { code: 'ur', name: 'Urdu', nativeName: 'اردو' }
];

// Translation resources
const resources = {
  en: { translation: enTranslations },
  hi: { translation: hiTranslations },
  mr: { translation: mrTranslations },
  gu: { translation: guTranslations },
  pa: { translation: paTranslations },
  ta: { translation: taTranslations },
  te: { translation: teTranslations },
  kn: { translation: knTranslations },
  ml: { translation: mlTranslations },
  bn: { translation: bnTranslations },
  as: { translation: asTranslations },
  or: { translation: orTranslations },
  ur: { translation: urTranslations }
};

// Initialize i18n
i18n
  .use(LanguageDetector)
  .use(initReactI18next)
  .init({
    resources,
    fallbackLng: 'en',
    debug: false,
    
    detection: {
      order: ['localStorage', 'navigator', 'htmlTag'],
      lookupLocalStorage: 'vetra-language',
      caches: ['localStorage']
    },
    
    interpolation: {
      escapeValue: false, // React already escapes values
    },
    
    // Support for pluralization in different languages
    compatibilityJSON: 'v4',
  });

export default i18n;