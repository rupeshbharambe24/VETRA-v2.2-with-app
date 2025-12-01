import React from 'react';
import { Link } from 'react-router-dom';
import { Phone, MessageCircle, Mail, Clock, ArrowLeft, AlertTriangle } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { useTranslation } from 'react-i18next';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';

const Helpline = () => {
  const { t } = useTranslation();

  const helplines = [
    {
      title: t('helpline.animalHelpline'),
      number: "1962",
      description: "24x7 Animal Disease Emergency Helpline",
      availability: t('helpline.available247')
    },
    {
      title: t('helpline.kisanCallCentre'),
      number: "1800-180-1551",
      description: "Agricultural guidance and livestock support",
      availability: "6:00 AM to 10:00 PM"
    },
    {
      title: t('helpline.nationalEmergency'),
      number: "112",
      description: "Emergency services for critical situations",
      availability: t('helpline.available247')
    },
    {
      title: t('helpline.veterinaryServices'),
      number: "1800-11-2004",
      description: "Veterinary consultation and guidance",
      availability: "9:00 AM to 6:00 PM"
    }
  ];

  const guidelines = [
    t('helpline.guidelines.g1'),
    t('helpline.guidelines.g2'),
    t('helpline.guidelines.g3'),
    t('helpline.guidelines.g4'),
    t('helpline.guidelines.g5')
  ];

  const handleCall = (number: string) => {
    window.open(`tel:${number}`);
  };

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="flex items-center gap-4 p-4 md:p-6 border-b border-border bg-card">
        <Link to="/">
          <Button variant="ghost" size="icon">
            <ArrowLeft className="w-5 h-5" />
          </Button>
        </Link>
        <h1 className="text-xl md:text-2xl font-bold text-foreground">{t('helpline.title')}</h1>
      </header>

      {/* Main Content */}
      <main className="max-w-6xl mx-auto p-6 md:p-12">
        {/* Introduction Card */}
        <Card className="mb-8">
          <CardHeader>
            <CardTitle className="flex items-center gap-3">
              <Phone className="w-6 h-6 text-primary" />
              {t('helpline.emergencySupport')}
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-muted-foreground text-lg">
              {t('helpline.description')}
            </p>
          </CardContent>
        </Card>

        {/* Helpline Numbers Grid */}
        <div className="grid md:grid-cols-2 gap-6 mb-8">
          {helplines.map((helpline, index) => (
            <Card key={index} className="hover:shadow-lg transition-shadow">
              <CardContent className="pt-6">
                <div className="flex items-start justify-between mb-4">
                  <div className="flex-1">
                    <h3 className="font-bold text-primary text-lg mb-2">{helpline.title}</h3>
                    <p className="text-muted-foreground mb-3">{helpline.description}</p>
                    <div className="flex items-center gap-2 text-sm text-muted-foreground">
                      <Clock className="w-4 h-4" />
                      {helpline.availability}
                    </div>
                  </div>
                </div>
                
                <Button 
                  variant="vetra" 
                  size="lg"
                  className="w-full"
                  onClick={() => handleCall(helpline.number)}
                >
                  <Phone className="w-4 h-4 mr-2" />
                  {t('helpline.call')} {helpline.number}
                </Button>
              </CardContent>
            </Card>
          ))}
        </div>

        {/* Additional Support */}
        <Card className="mb-8 bg-secondary/5 border-secondary/20">
          <CardHeader>
            <CardTitle className="flex items-center gap-3">
              <MessageCircle className="w-6 h-6 text-secondary" />
              {t('helpline.additionalSupport')}
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid md:grid-cols-2 gap-4">
              <Button variant="outline" size="lg" className="justify-start">
                <Mail className="w-4 h-4 mr-2" />
                {t('helpline.email')}: support@vetra.gov.in
              </Button>
              
              <Button variant="outline" size="lg" className="justify-start">
                <MessageCircle className="w-4 h-4 mr-2" />
                {t('helpline.whatsapp')}: +91-98765-43210
              </Button>
            </div>
          </CardContent>
        </Card>

        {/* Emergency Guidelines */}
        <Card className="mb-8 border-destructive/30">
          <CardHeader>
            <CardTitle className="flex items-center gap-3 text-destructive">
              <AlertTriangle className="w-6 h-6" />
              {t('helpline.emergencyGuidelines')}
            </CardTitle>
          </CardHeader>
          <CardContent>
            <ul className="space-y-3">
              {guidelines.map((guideline, index) => (
                <li key={index} className="flex items-start gap-3">
                  <div className="w-2 h-2 bg-destructive rounded-full mt-2 flex-shrink-0"></div>
                  <span className="text-foreground">{guideline}</span>
                </li>
              ))}
            </ul>
          </CardContent>
        </Card>

        {/* Quick Tips */}
        <Card className="bg-primary/5 border-primary/20">
          <CardHeader>
            <CardTitle className="text-primary">{t('helpline.quickTips')}</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-muted-foreground">
              {t('helpline.tipDescription')}
            </p>
          </CardContent>
        </Card>
      </main>
    </div>
  );
};

export default Helpline;
