import React from 'react';
import { Link } from 'react-router-dom';
import { Camera, Database, FileText, Shield, Phone, Info, Users, ArrowRight, Sparkles, BarChart3, Download, CheckCircle, Zap, Clock, TrendingUp, Heart, Scale } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import LanguageSwitcher from './LanguageSwitcher';
import vetraLogo from '@/assets/vetra-logo.png';
import traitIcon from '@/assets/trait-icon.png';
import dataIcon from '@/assets/data-icon.png';

const VetraHome = () => {
  const { t } = useTranslation();

  const coreFeatures = [
    {
      icon: <Scale className="w-6 h-6" />,
      title: "Body Measurements",
      description: "Precise body length, height, and girth measurements using AI"
    },
    {
      icon: <Heart className="w-6 h-6" />,
      title: "Health Scoring",
      description: "1-9 scale body condition scoring for health assessment"
    },
    {
      icon: <TrendingUp className="w-6 h-6" />,
      title: "Weight Estimation",
      description: "Accurate weight estimation based on body measurements"
    },
    {
      icon: <Download className="w-6 h-6" />,
      title: "PDF Reports",
      description: "Download detailed analysis reports for record keeping"
    }
  ];

  const benefits = [
    { icon: <Zap className="w-5 h-5" />, text: "Instant AI Analysis" },
    { icon: <CheckCircle className="w-5 h-5" />, text: "High Accuracy" },
    { icon: <Clock className="w-5 h-5" />, text: "Save Time" },
    { icon: <Database className="w-5 h-5" />, text: "Track History" }
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-background via-background to-muted/30">
      {/* Hero Section */}
      <header className="relative overflow-hidden">
        {/* Background Pattern */}
        <div className="absolute inset-0 hero-gradient"></div>
        <div className="absolute inset-0 opacity-10">
          <div className="absolute top-20 left-[10%] w-64 h-64 rounded-full bg-white/20 blur-3xl"></div>
          <div className="absolute bottom-10 right-[15%] w-48 h-48 rounded-full bg-white/15 blur-2xl"></div>
          <div className="absolute top-40 right-[30%] w-32 h-32 rounded-full bg-white/10 blur-xl"></div>
        </div>
        
        {/* Language Switcher */}
        <div className="absolute top-6 right-6 md:right-12 z-20">
          <LanguageSwitcher />
        </div>

        <div className="relative z-10 max-w-7xl mx-auto px-6 md:px-12 py-12 md:py-20">
          <div className="grid md:grid-cols-2 gap-10 items-center">
            {/* Left Content */}
            <div className="text-center md:text-left">
              <div className="flex items-center justify-center md:justify-start gap-4 mb-5">
                <img 
                  src={vetraLogo} 
                  alt="VETRA Logo" 
                  className="w-14 h-14 md:w-18 md:h-18 rounded-full shadow-2xl ring-4 ring-white/20"
                />
                <h1 className="text-3xl md:text-5xl font-bold text-white tracking-tight">
                  {t('app.name')}
                </h1>
              </div>
              <p className="text-white/90 text-lg md:text-xl max-w-lg mx-auto md:mx-0 leading-relaxed mb-4">
                {t('app.subtitle')}
              </p>
              <p className="text-white/70 text-base max-w-md mx-auto md:mx-0 mb-8">
                AI-powered livestock analysis for precise body measurements, health scoring, and weight estimation.
              </p>
              
              {/* Benefits Pills */}
              <div className="flex flex-wrap gap-2 justify-center md:justify-start mb-8">
                {benefits.map((benefit, index) => (
                  <div 
                    key={index}
                    className="flex items-center gap-2 bg-white/10 backdrop-blur-sm px-3 py-1.5 rounded-full text-white/90 text-sm"
                  >
                    {benefit.icon}
                    {benefit.text}
                  </div>
                ))}
              </div>
              
              {/* CTA Buttons */}
              <div className="flex flex-col sm:flex-row gap-4 justify-center md:justify-start">
                <Link to="/trait-extraction">
                  <Button 
                    size="lg"
                    className="w-full sm:w-auto bg-white text-primary hover:bg-white/90 font-bold text-lg px-8 py-6 rounded-xl shadow-xl hover:shadow-2xl transition-all duration-300"
                  >
                    <Sparkles className="w-5 h-5 mr-2" />
                    Start Analysis
                    <ArrowRight className="w-5 h-5 ml-2" />
                  </Button>
                </Link>
                <Link to="/data-records">
                  <Button 
                    variant="outline"
                    size="lg"
                    className="w-full sm:w-auto bg-white/10 border-white/30 text-white hover:bg-white/20 font-semibold text-lg px-8 py-6 rounded-xl"
                  >
                    <Database className="w-5 h-5 mr-2" />
                    View Records
                  </Button>
                </Link>
              </div>
            </div>

            {/* Right Content - Feature Cards */}
            <div className="hidden md:grid grid-cols-2 gap-4">
              <div className="bg-white/10 backdrop-blur-sm rounded-2xl p-5 border border-white/20">
                <Camera className="w-9 h-9 text-white mb-3" />
                <h3 className="text-white font-bold text-base mb-1">Image Capture</h3>
                <p className="text-white/70 text-sm">Capture or upload livestock images for instant analysis</p>
              </div>
              <div className="bg-white/10 backdrop-blur-sm rounded-2xl p-5 border border-white/20">
                <BarChart3 className="w-9 h-9 text-white mb-3" />
                <h3 className="text-white font-bold text-base mb-1">Trait Analysis</h3>
                <p className="text-white/70 text-sm">Get precise measurements and health scores</p>
              </div>
              <div className="bg-white/10 backdrop-blur-sm rounded-2xl p-5 border border-white/20">
                <Database className="w-9 h-9 text-white mb-3" />
                <h3 className="text-white font-bold text-base mb-1">Data Storage</h3>
                <p className="text-white/70 text-sm">Store and track all analysis history</p>
              </div>
              <div className="bg-white/10 backdrop-blur-sm rounded-2xl p-5 border border-white/20">
                <Download className="w-9 h-9 text-white mb-3" />
                <h3 className="text-white font-bold text-base mb-1">PDF Reports</h3>
                <p className="text-white/70 text-sm">Download detailed analysis reports</p>
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* Core Features Grid */}
      <section className="py-12 md:py-16 px-6 md:px-12 max-w-7xl mx-auto">
        <div className="text-center mb-10">
          <h2 className="text-2xl md:text-3xl font-bold text-foreground mb-3">What VETRA Offers</h2>
          <p className="text-muted-foreground text-base max-w-2xl mx-auto">
            Comprehensive AI-powered tools for livestock health monitoring and management
          </p>
        </div>

        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
          {coreFeatures.map((feature, index) => (
            <Card key={index} className="border-border hover:border-primary/30 transition-all duration-300 hover:-translate-y-1">
              <CardContent className="p-5 text-center">
                <div className="w-12 h-12 mx-auto mb-3 bg-primary/10 rounded-xl flex items-center justify-center text-primary">
                  {feature.icon}
                </div>
                <h3 className="font-semibold text-foreground mb-1 text-sm md:text-base">{feature.title}</h3>
                <p className="text-muted-foreground text-xs md:text-sm">{feature.description}</p>
              </CardContent>
            </Card>
          ))}
        </div>
      </section>

      {/* Main Features Section */}
      <section className="py-8 md:py-12 px-6 md:px-12 bg-muted/20">
        <div className="max-w-4xl mx-auto">
          <div className="grid md:grid-cols-2 gap-5">
            <Link to="/trait-extraction" className="group">
              <div className="bg-card border border-border rounded-2xl p-6 hover:shadow-xl transition-all duration-300 hover:border-primary/30 hover:-translate-y-1 h-full">
                <div className="flex items-start gap-4">
                  <div className="bg-gradient-to-br from-primary to-primary/70 rounded-xl p-3 shadow-lg flex-shrink-0">
                    <img src={traitIcon} alt="Trait" className="w-8 h-8" />
                  </div>
                  <div className="flex-1">
                    <h3 className="font-bold text-lg text-foreground mb-2 group-hover:text-primary transition-colors">
                      {t('navigation.vetra')}
                    </h3>
                    <p className="text-muted-foreground text-sm">
                      AI-powered trait extraction with body measurements, health scoring on 1-9 scale, and weight estimation.
                    </p>
                  </div>
                  <ArrowRight className="w-5 h-5 text-muted-foreground group-hover:text-primary group-hover:translate-x-1 transition-all flex-shrink-0" />
                </div>
              </div>
            </Link>

            <Link to="/data-records" className="group">
              <div className="bg-card border border-border rounded-2xl p-6 hover:shadow-xl transition-all duration-300 hover:border-secondary/30 hover:-translate-y-1 h-full">
                <div className="flex items-start gap-4">
                  <div className="bg-gradient-to-br from-secondary to-secondary/70 rounded-xl p-3 shadow-lg flex-shrink-0">
                    <img src={dataIcon} alt="Data" className="w-8 h-8" />
                  </div>
                  <div className="flex-1">
                    <h3 className="font-bold text-lg text-foreground mb-2 group-hover:text-secondary transition-colors">
                      {t('navigation.data')}
                    </h3>
                    <p className="text-muted-foreground text-sm">
                      Access past analysis records, track livestock health trends, and download reports.
                    </p>
                  </div>
                  <ArrowRight className="w-5 h-5 text-muted-foreground group-hover:text-secondary group-hover:translate-x-1 transition-all flex-shrink-0" />
                </div>
              </div>
            </Link>
          </div>
        </div>
      </section>

      {/* Quick Access Section */}
      <section className="py-10 md:py-14 px-6 md:px-12 bg-muted/30">
        <div className="max-w-6xl mx-auto">
          <h3 className="text-xl md:text-2xl font-bold mb-6 text-foreground text-center">
            {t('home.quickAccess')}
          </h3>
          <div className="grid grid-cols-2 md:grid-cols-5 gap-3 md:gap-4">
            <Link to="/schemes" className="group">
              <div className="bg-card border border-border rounded-xl p-5 text-center hover:shadow-lg transition-all duration-300 hover:border-primary/30 hover:-translate-y-1">
                <FileText className="w-8 h-8 mx-auto mb-2 text-primary" />
                <span className="font-semibold text-foreground group-hover:text-primary transition-colors text-sm">{t('navigation.schemes')}</span>
              </div>
            </Link>

            <Link to="/insurance" className="group">
              <div className="bg-card border border-border rounded-xl p-5 text-center hover:shadow-lg transition-all duration-300 hover:border-secondary/30 hover:-translate-y-1">
                <Shield className="w-8 h-8 mx-auto mb-2 text-secondary" />
                <span className="font-semibold text-foreground group-hover:text-secondary transition-colors text-sm">{t('navigation.insurance')}</span>
              </div>
            </Link>

            <Link to="/helpline" className="group">
              <div className="bg-card border border-border rounded-xl p-5 text-center hover:shadow-lg transition-all duration-300 hover:border-primary/30 hover:-translate-y-1">
                <Phone className="w-8 h-8 mx-auto mb-2 text-primary" />
                <span className="font-semibold text-foreground group-hover:text-primary transition-colors text-sm">{t('navigation.helpline')}</span>
              </div>
            </Link>

            <Link to="/about" className="group">
              <div className="bg-card border border-border rounded-xl p-5 text-center hover:shadow-lg transition-all duration-300 hover:border-secondary/30 hover:-translate-y-1">
                <Info className="w-8 h-8 mx-auto mb-2 text-secondary" />
                <span className="font-semibold text-foreground group-hover:text-secondary transition-colors text-sm">{t('navigation.aboutUs')}</span>
              </div>
            </Link>

            <Link to="/contact" className="group col-span-2 md:col-span-1">
              <div className="bg-card border border-border rounded-xl p-5 text-center hover:shadow-lg transition-all duration-300 hover:border-primary/30 hover:-translate-y-1">
                <Users className="w-8 h-8 mx-auto mb-2 text-primary" />
                <span className="font-semibold text-foreground group-hover:text-primary transition-colors text-sm">{t('navigation.contact')}</span>
              </div>
            </Link>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="text-center py-8 px-8 border-t border-border/50 bg-card">
        <div className="flex items-center justify-center gap-3 mb-3">
          <img src={vetraLogo} alt="VETRA" className="w-7 h-7 rounded-full" />
          <span className="font-bold text-base text-foreground">VETRA</span>
        </div>
        <p className="text-muted-foreground text-sm">
          {t('app.poweredBy')}
        </p>
      </footer>
    </div>
  );
};

export default VetraHome;
