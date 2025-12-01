import React from 'react';
import { Link } from 'react-router-dom';
import { MapPin, Phone, Mail, Globe, Clock, Building, ArrowLeft } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { useTranslation } from 'react-i18next';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';

const Contact = () => {
  const { t } = useTranslation();

  const offices = [
    {
      name: t('contact.offices.headOffice'),
      address: "Department of Animal Husbandry & Dairying\nKrishi Bhawan, New Delhi - 110001",
      phone: "+91-11-2338-2651",
      email: "secy-dahd@nic.in"
    },
    {
      name: t('contact.offices.technicalSupport'),
      address: "National Informatics Centre\nCGO Complex, Lodhi Road\nNew Delhi - 110003",
      phone: "+91-11-2430-5000",
      email: "support-vetra@nic.in"
    },
    {
      name: t('contact.offices.northRegion'),
      address: "Punjab Agricultural University\nLudhiana, Punjab - 141004",
      phone: "+91-161-240-1960",
      email: "north-vetra@pau.edu"
    },
    {
      name: t('contact.offices.southRegion'),
      address: "Tamil Nadu Veterinary University\nChennai, Tamil Nadu - 600051",
      phone: "+91-44-2536-1100",
      email: "south-vetra@tanuvas.ac.in"
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
        <h1 className="text-xl md:text-2xl font-bold text-foreground">{t('contact.title')}</h1>
      </header>

      {/* Main Content */}
      <main className="max-w-6xl mx-auto p-6 md:p-12">
        {/* Introduction Card */}
        <Card className="mb-8">
          <CardHeader>
            <CardTitle className="flex items-center gap-3">
              <Building className="w-6 h-6 text-primary" />
              {t('contact.getInTouch')}
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-muted-foreground text-lg">
              {t('contact.description')}
            </p>
          </CardContent>
        </Card>

        {/* Offices Grid */}
        <div className="grid md:grid-cols-2 gap-6 mb-8">
          {offices.map((office, index) => (
            <Card key={index} className="hover:shadow-lg transition-shadow">
              <CardHeader>
                <CardTitle className="text-primary text-lg">{office.name}</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex items-start gap-3">
                  <MapPin className="w-5 h-5 mt-0.5 text-muted-foreground flex-shrink-0" />
                  <div className="text-muted-foreground">
                    {office.address.split('\n').map((line, i) => (
                      <div key={i}>{line}</div>
                    ))}
                  </div>
                </div>
                
                <div className="flex items-center gap-3">
                  <Phone className="w-5 h-5 text-muted-foreground" />
                  <a 
                    href={`tel:${office.phone}`} 
                    className="text-primary hover:underline"
                  >
                    {office.phone}
                  </a>
                </div>
                
                <div className="flex items-center gap-3">
                  <Mail className="w-5 h-5 text-muted-foreground" />
                  <a 
                    href={`mailto:${office.email}`} 
                    className="text-primary hover:underline"
                  >
                    {office.email}
                  </a>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>

        {/* Office Hours */}
        <Card className="mb-8 bg-secondary/5 border-secondary/20">
          <CardHeader>
            <CardTitle className="flex items-center gap-3">
              <Clock className="w-6 h-6 text-secondary" />
              {t('contact.officeHours')}
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid md:grid-cols-2 gap-4">
              <div className="flex justify-between">
                <span className="text-foreground">{t('contact.hours.weekdays')}</span>
                <span className="text-muted-foreground">{t('contact.hours.weekdaysTime')}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-foreground">{t('contact.hours.saturday')}</span>
                <span className="text-muted-foreground">{t('contact.hours.saturdayTime')}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-foreground">{t('contact.hours.sunday')}</span>
                <span className="text-muted-foreground">{t('contact.hours.sundayTime')}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-foreground">{t('contact.hours.emergency')}</span>
                <span className="text-destructive font-semibold">{t('contact.hours.emergencyTime')}</span>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Quick Contact */}
        <Card className="mb-8">
          <CardHeader>
            <CardTitle>{t('contact.quickContact')}</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid md:grid-cols-3 gap-4">
              <Button 
                variant="vetra" 
                size="lg"
                className="justify-start"
                onClick={() => window.open('tel:1962')}
              >
                <Phone className="w-4 h-4 mr-2" />
                {t('contact.emergencyHelpline')}: 1962
              </Button>
              
              <Button 
                variant="outline" 
                size="lg"
                className="justify-start"
                onClick={() => window.open('mailto:support-vetra@nic.in')}
              >
                <Mail className="w-4 h-4 mr-2" />
                {t('contact.emailSupport')}
              </Button>
              
              <Button 
                variant="outline" 
                size="lg"
                className="justify-start"
                onClick={() => window.open('https://dahd.nic.in/', '_blank')}
              >
                <Globe className="w-4 h-4 mr-2" />
                {t('contact.visitOfficialWebsite')}
              </Button>
            </div>
          </CardContent>
        </Card>

        {/* Feedback Section */}
        <Card className="mb-8 bg-primary/5 border-primary/20">
          <CardHeader>
            <CardTitle>{t('contact.feedbackSuggestions')}</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-muted-foreground mb-6">
              {t('contact.feedbackDescription')}
            </p>
            <Button 
              variant="vetra-secondary"
              size="lg"
              onClick={() => window.open('mailto:feedback-vetra@nic.in?subject=VETRA App Feedback')}
            >
              <Mail className="w-4 h-4 mr-2" />
              {t('contact.sendFeedback')}
            </Button>
          </CardContent>
        </Card>

        {/* Footer Note */}
        <div className="text-center text-sm text-muted-foreground">
          <p>{t('contact.technicalNote')}</p>
        </div>
      </main>
    </div>
  );
};

export default Contact;
