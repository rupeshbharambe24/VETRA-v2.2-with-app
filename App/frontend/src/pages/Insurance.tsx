import React from 'react';
import { Link } from 'react-router-dom';
import { Shield, Phone, IndianRupee, ArrowLeft } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { useTranslation } from 'react-i18next';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';

const Insurance = () => {
  const { t } = useTranslation();

  const insuranceSchemes = [
    {
      name: t('insurance.livestockScheme.name'),
      description: t('insurance.livestockScheme.description'),
      coverage: t('insurance.livestockScheme.coverage'),
      premium: t('insurance.livestockScheme.premium'),
      subsidy: t('insurance.livestockScheme.subsidy')
    },
    {
      name: t('insurance.pmfby.name'),
      description: t('insurance.pmfby.description'),
      coverage: t('insurance.pmfby.coverage'),
      premium: t('insurance.pmfby.premium'),
      subsidy: t('insurance.pmfby.subsidy')
    }
  ];

  const applySteps = [
    t('insurance.applySteps.step1'),
    t('insurance.applySteps.step2'),
    t('insurance.applySteps.step3'),
    t('insurance.applySteps.step4')
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
        <h1 className="text-xl md:text-2xl font-bold text-foreground">{t('insurance.title')}</h1>
      </header>

      {/* Main Content */}
      <main className="max-w-6xl mx-auto p-6 md:p-12">
        {/* Introduction Card */}
        <Card className="mb-8">
          <CardHeader>
            <CardTitle className="flex items-center gap-3">
              <Shield className="w-6 h-6 text-secondary" />
              {t('insurance.insuranceCoverage')}
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-muted-foreground text-lg">
              {t('insurance.description')}
            </p>
          </CardContent>
        </Card>

        {/* Insurance Schemes Grid */}
        <div className="grid md:grid-cols-2 gap-6 mb-8">
          {insuranceSchemes.map((scheme, index) => (
            <Card key={index} className="hover:shadow-lg transition-shadow">
              <CardHeader>
                <CardTitle className="text-secondary text-lg">{scheme.name}</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <p className="text-muted-foreground">{scheme.description}</p>
                
                <div className="space-y-3">
                  <div>
                    <span className="font-semibold text-foreground">{t('insurance.coverage')}: </span>
                    <span className="text-muted-foreground">{scheme.coverage}</span>
                  </div>
                  <div>
                    <span className="font-semibold text-foreground">{t('insurance.premium')}: </span>
                    <span className="text-muted-foreground">{scheme.premium}</span>
                  </div>
                  <div>
                    <span className="font-semibold text-foreground">{t('insurance.subsidy')}: </span>
                    <span className="text-muted-foreground">{scheme.subsidy}</span>
                  </div>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>

        {/* Insurance Helpline */}
        <Card className="mb-8 bg-secondary/5 border-secondary/20">
          <CardHeader>
            <CardTitle className="flex items-center gap-3">
              <Phone className="w-6 h-6 text-secondary" />
              {t('insurance.insuranceHelpline')}
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-muted-foreground mb-6">
              {t('insurance.contactProviders')}
            </p>
            <div className="grid md:grid-cols-2 gap-4">
              <Button variant="outline" size="lg" className="justify-start">
                <Phone className="w-4 h-4 mr-2" />
                National Insurance: 1800-200-5030
              </Button>
              <Button variant="outline" size="lg" className="justify-start">
                <Phone className="w-4 h-4 mr-2" />
                Agriculture Insurance: 1800-180-1551
              </Button>
            </div>
          </CardContent>
        </Card>

        {/* How to Apply */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-3">
              <IndianRupee className="w-6 h-6 text-primary" />
              {t('insurance.howToApply')}
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid md:grid-cols-2 gap-6">
              {applySteps.map((step, index) => (
                <div key={index} className="flex items-start gap-4">
                  <div className="w-10 h-10 bg-primary text-primary-foreground rounded-full flex items-center justify-center text-lg font-bold flex-shrink-0">
                    {index + 1}
                  </div>
                  <p className="text-foreground pt-2">{step}</p>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </main>
    </div>
  );
};

export default Insurance;
