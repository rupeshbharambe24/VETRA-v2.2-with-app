import React from 'react';
import { Link } from 'react-router-dom';
import { ExternalLink, FileText, IndianRupee, ArrowLeft } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { useTranslation } from 'react-i18next';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';

const Schemes = () => {
  const { t } = useTranslation();

  const schemes = [
    {
      name: t('schemes.rashtriyaGokul.name'),
      description: t('schemes.rashtriyaGokul.description'),
      eligibility: t('schemes.rashtriyaGokul.eligibility'),
      benefits: t('schemes.rashtriyaGokul.benefits'),
      link: "https://dahd.nic.in/schemes/rashtriya-gokul-mission"
    },
    {
      name: t('schemes.dairyDevelopment.name'),
      description: t('schemes.dairyDevelopment.description'),
      eligibility: t('schemes.dairyDevelopment.eligibility'),
      benefits: t('schemes.dairyDevelopment.benefits'),
      link: "https://dahd.nic.in/schemes/programmes-implementation/national-programme-dairy-development"
    },
    {
      name: t('schemes.healthControl.name'),
      description: t('schemes.healthControl.description'),
      eligibility: t('schemes.healthControl.eligibility'),
      benefits: t('schemes.healthControl.benefits'),
      link: "https://dahd.nic.in/schemes/livestock-health-disease-control"
    }
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
        <h1 className="text-xl md:text-2xl font-bold text-foreground">{t('schemes.title')}</h1>
      </header>

      {/* Main Content */}
      <main className="max-w-6xl mx-auto p-6 md:p-12">
        {/* Introduction Card */}
        <Card className="mb-8">
          <CardHeader>
            <CardTitle className="flex items-center gap-3">
              <FileText className="w-6 h-6 text-primary" />
              {t('schemes.availableSchemes')}
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-muted-foreground text-lg">
              {t('schemes.description')}
            </p>
          </CardContent>
        </Card>

        {/* Schemes Grid */}
        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6 mb-8">
          {schemes.map((scheme, index) => (
            <Card key={index} className="hover:shadow-lg transition-shadow">
              <CardHeader>
                <CardTitle className="text-primary text-lg">{scheme.name}</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <p className="text-muted-foreground">{scheme.description}</p>
                
                <div className="space-y-3">
                  <div>
                    <span className="font-semibold text-foreground">{t('schemes.eligibility')}: </span>
                    <span className="text-muted-foreground">{scheme.eligibility}</span>
                  </div>
                  <div>
                    <span className="font-semibold text-foreground">{t('schemes.benefits')}: </span>
                    <span className="text-muted-foreground">{scheme.benefits}</span>
                  </div>
                </div>

                <Button 
                  variant="outline" 
                  className="w-full mt-4"
                  onClick={() => window.open(scheme.link, '_blank')}
                >
                  <ExternalLink className="w-4 h-4 mr-2" />
                  {t('schemes.learnMore')}
                </Button>
              </CardContent>
            </Card>
          ))}
        </div>

        {/* Apply Section */}
        <Card className="bg-primary/5 border-primary/20">
          <CardHeader>
            <CardTitle className="flex items-center gap-3">
              <IndianRupee className="w-6 h-6 text-primary" />
              {t('schemes.applyForSchemes')}
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-muted-foreground mb-6">
              {t('schemes.applyDescription')}
            </p>
            <Button 
              variant="vetra" 
              size="lg"
              onClick={() => window.open('https://dahd.nic.in/', '_blank')}
            >
              <ExternalLink className="w-4 h-4 mr-2" />
              {t('schemes.visitOfficialWebsite')}
            </Button>
          </CardContent>
        </Card>
      </main>
    </div>
  );
};

export default Schemes;
