import React from 'react';
import { Link } from 'react-router-dom';
import { Cpu, Target, Users, Award, ExternalLink, ArrowLeft } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { useTranslation } from 'react-i18next';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import vetraLogo from '@/assets/vetra-logo.png';

const About = () => {
  const { t } = useTranslation();

  const features = [
    {
      title: t('about.features.instantExtraction.title'),
      description: t('about.features.instantExtraction.description'),
      color: 'primary'
    },
    {
      title: t('about.features.dataManagement.title'),
      description: t('about.features.dataManagement.description'),
      color: 'secondary'
    },
    {
      title: t('about.features.schemeIntegration.title'),
      description: t('about.features.schemeIntegration.description'),
      color: 'primary'
    },
    {
      title: t('about.features.expertSupport.title'),
      description: t('about.features.expertSupport.description'),
      color: 'secondary'
    }
  ];

  const techFeatures = [
    t('about.techAI'),
    t('about.techMobile'),
    t('about.techOffline'),
    t('about.techCloud'),
    t('about.techSecurity')
  ];

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="flex items-center gap-4 p-4 md:p-6 border-b border-border bg-card">
        <Link to="/">
          <Button variant="ghost" size="icon">
            <ArrowLeft className="w-5 h-5" />
          </Button>
        </Link>
        <h1 className="text-xl md:text-2xl font-bold text-foreground">{t('about.title')}</h1>
      </header>

      {/* Main Content */}
      <main className="max-w-6xl mx-auto p-6 md:p-12">
        {/* Hero Card */}
        <Card className="mb-8 text-center">
          <CardContent className="pt-8 pb-8">
            <img 
              src={vetraLogo} 
              alt="VETRA Logo" 
              className="w-24 h-24 mx-auto mb-6 rounded-full shadow-lg"
            />
            <h2 className="text-3xl font-bold gradient-text mb-3">{t('app.name')}</h2>
            <p className="text-muted-foreground text-lg mb-2">
              {t('about.subtitle')}
            </p>
            <p className="text-sm text-muted-foreground">
              {t('about.version')}
            </p>
          </CardContent>
        </Card>

        {/* Mission and Technology */}
        <div className="grid md:grid-cols-2 gap-6 mb-8">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-3">
                <Target className="w-6 h-6 text-primary" />
                {t('about.mission')}
              </CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-muted-foreground leading-relaxed">
                {t('about.missionDescription')}
              </p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-3">
                <Cpu className="w-6 h-6 text-secondary" />
                {t('about.technology')}
              </CardTitle>
            </CardHeader>
            <CardContent>
              <ul className="space-y-2">
                {techFeatures.map((feature, index) => (
                  <li key={index} className="text-muted-foreground text-sm">
                    • {feature}
                  </li>
                ))}
              </ul>
            </CardContent>
          </Card>
        </div>

        {/* Key Features */}
        <Card className="mb-8">
          <CardHeader>
            <CardTitle className="flex items-center gap-3">
              <Award className="w-6 h-6 text-primary" />
              {t('about.keyFeatures')}
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid md:grid-cols-2 gap-6">
              {features.map((feature, index) => (
                <div key={index} className="flex items-start gap-4">
                  <div className={`w-3 h-3 rounded-full mt-1.5 flex-shrink-0 ${feature.color === 'primary' ? 'bg-primary' : 'bg-secondary'}`}></div>
                  <div>
                    <p className="font-semibold text-foreground">{feature.title}</p>
                    <p className="text-sm text-muted-foreground">{feature.description}</p>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* Development Team */}
        <Card className="mb-8">
          <CardHeader>
            <CardTitle className="flex items-center gap-3">
              <Users className="w-6 h-6 text-secondary" />
              {t('about.developmentTeam')}
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-muted-foreground mb-6">
              {t('about.teamDescription')}
            </p>
            
            <div className="grid md:grid-cols-3 gap-4">
              <div className="bg-muted/50 rounded-lg p-4">
                <p className="font-semibold text-foreground">{t('about.teamLead')}</p>
                <p className="text-sm text-muted-foreground">Dr. Rajesh Kumar, IVRI</p>
              </div>
              <div className="bg-muted/50 rounded-lg p-4">
                <p className="font-semibold text-foreground">{t('about.teamAI')}</p>
                <p className="text-sm text-muted-foreground">IIT Kharagpur Team</p>
              </div>
              <div className="bg-muted/50 rounded-lg p-4">
                <p className="font-semibold text-foreground">{t('about.teamApp')}</p>
                <p className="text-sm text-muted-foreground">NIC Development Center</p>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Acknowledgments */}
        <Card className="mb-8 bg-primary/5 border-primary/20">
          <CardHeader>
            <CardTitle className="text-primary">{t('about.acknowledgments')}</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-muted-foreground mb-6">
              {t('about.acknowledgmentDescription')}
            </p>
            
            <Button 
              variant="vetra" 
              size="lg"
              onClick={() => window.open('https://dahd.nic.in/', '_blank')}
            >
              <ExternalLink className="w-4 h-4 mr-2" />
              {t('about.visitWebsite')}
            </Button>
          </CardContent>
        </Card>

        {/* Footer */}
        <div className="text-center text-sm text-muted-foreground">
          <p>{t('about.copyright')}</p>
          <p>Government of India. All rights reserved.</p>
        </div>
      </main>
    </div>
  );
};

export default About;
