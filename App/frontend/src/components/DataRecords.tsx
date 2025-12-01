import React, { useState, useEffect, useCallback } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { ArrowLeft, Calendar, Ruler, Activity, Search, RefreshCw, AlertCircle, Plus, TrendingUp } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { useToast } from '@/hooks/use-toast';
import { getResults, type HistoricalRecord } from '@/lib/api';
import { z } from 'zod';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';

// Input validation schema
const searchSchema = z.object({
  search: z.string().max(100, 'Search term must be less than 100 characters'),
});

const DataRecords = () => {
  const [records, setRecords] = useState<HistoricalRecord[]>([]);
  const [searchTerm, setSearchTerm] = useState('');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [hasMore, setHasMore] = useState(false);
  const [totalCount, setTotalCount] = useState(0);
  const { toast } = useToast();
  const navigate = useNavigate();

  const fetchRecords = useCallback(async (search: string = '', offset: number = 0, showToast: boolean = false) => {
    try {
      setError(null);
      
      if (search) {
        const validatedSearch = searchSchema.parse({ search });
        search = validatedSearch.search;
      }

      const response = await getResults(50, offset, search);
      
      if (response.success && response.data) {
        // Sort records by date - most recent first
        const sortedRecords = [...response.data].sort((a, b) => 
          new Date(b.date).getTime() - new Date(a.date).getTime()
        );
        
        setRecords(sortedRecords);
        setTotalCount(response.totalCount || 0);
        setHasMore(response.hasMore || false);
        
        if (showToast) {
          toast({
            title: "Records Updated",
            description: `Found ${response.data.length} records`,
          });
        }
      } else {
        throw new Error(response.error || 'Failed to fetch records');
      }
    } catch (error) {
      console.error('Error fetching records:', error);
      const errorMessage = error instanceof Error ? error.message : 'Failed to load records';
      setError(errorMessage);
      
      if (showToast) {
        toast({
          title: "Error Loading Records",
          description: errorMessage,
          variant: "destructive",
        });
      }
    } finally {
      setLoading(false);
    }
  }, [toast]);

  // Initial fetch and auto-refresh every 30 seconds
  useEffect(() => {
    fetchRecords();
    
    // Auto-refresh interval
    const refreshInterval = setInterval(() => {
      fetchRecords(searchTerm, 0, false);
    }, 30000);

    return () => clearInterval(refreshInterval);
  }, [fetchRecords, searchTerm]);

  // Debounced search
  useEffect(() => {
    const timer = setTimeout(() => {
      if (searchTerm !== undefined) {
        setLoading(true);
        fetchRecords(searchTerm);
      }
    }, 500);

    return () => clearTimeout(timer);
  }, [searchTerm, fetchRecords]);

  const formatDate = (dateString: string) => {
    try {
      return new Date(dateString).toLocaleDateString('en-US', {
        year: 'numeric',
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
      });
    } catch {
      return dateString;
    }
  };

  const getBodyScoreLabel = (score: number): string => {
    if (score <= 3) return 'Weak';
    if (score <= 6) return 'Average';
    return 'Strong';
  };

  const getBodyScoreColor = (score: number): string => {
    if (score <= 3) return 'bg-red-100 text-red-700';
    if (score <= 6) return 'bg-yellow-100 text-yellow-700';
    return 'bg-green-100 text-green-700';
  };

  const handleNewAnalysis = () => {
    navigate('/trait-extraction');
  };

  return (
    <div className="min-h-screen bg-background">
      <header className="flex items-center justify-between p-4 md:p-6 border-b border-border bg-card">
        <div className="flex items-center gap-4">
          <Link to="/">
            <Button variant="ghost" size="icon">
              <ArrowLeft className="w-5 h-5" />
            </Button>
          </Link>
          <div>
            <h1 className="text-xl md:text-2xl font-bold text-foreground">Past Records</h1>
            <p className="text-sm text-muted-foreground">
              {totalCount > 0 ? `${totalCount} analysis records` : 'View your analysis history'}
            </p>
          </div>
        </div>
        <Button variant="vetra" onClick={handleNewAnalysis} className="gap-2">
          <Plus className="w-4 h-4" />
          <span className="hidden sm:inline">New Analysis</span>
        </Button>
      </header>

      <div className="p-4 md:p-8 lg:p-12 max-w-7xl mx-auto">
        {/* Search and Actions Bar */}
        <Card className="mb-6">
          <CardContent className="p-4">
            <div className="flex flex-col sm:flex-row gap-4">
              <div className="relative flex-1">
                <Search className="absolute left-3 top-3 w-4 h-4 text-muted-foreground" />
                <Input
                  placeholder="Search by animal type or notes..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  className="pl-10"
                />
              </div>
              <Button 
                variant="outline" 
                onClick={() => fetchRecords(searchTerm, 0, true)}
                className="gap-2"
              >
                <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
                Refresh
              </Button>
            </div>
          </CardContent>
        </Card>

        {/* Loading State */}
        {loading && (
          <Card>
            <CardContent className="py-12 text-center">
              <div className="animate-spin w-10 h-10 border-3 border-primary border-t-transparent rounded-full mx-auto mb-4"></div>
              <p className="text-muted-foreground">Loading records...</p>
            </CardContent>
          </Card>
        )}

        {/* Empty State */}
        {!loading && records.length === 0 && (
          <Card>
            <CardContent className="py-16 text-center">
              <Activity className="w-16 h-16 mx-auto mb-6 text-muted-foreground/50" />
              <h3 className="text-xl font-semibold mb-2 text-foreground">No Records Found</h3>
              <p className="text-muted-foreground mb-6">
                {searchTerm ? 'Try adjusting your search terms' : 'Start your first analysis to see records here'}
              </p>
              <Button variant="vetra" size="lg" onClick={handleNewAnalysis} className="gap-2">
                <Plus className="w-5 h-5" />
                Start New Analysis
              </Button>
            </CardContent>
          </Card>
        )}

        {/* Records Table - Desktop */}
        {!loading && records.length > 0 && (
          <>
            <Card className="hidden md:block">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <TrendingUp className="w-5 h-5 text-primary" />
                  Analysis History
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="rounded-lg border border-border overflow-hidden">
                  <Table>
                    <TableHeader>
                      <TableRow className="bg-muted/50">
                        <TableHead className="font-semibold">Date</TableHead>
                        <TableHead className="font-semibold">Animal</TableHead>
                        <TableHead className="font-semibold text-center">Body Length</TableHead>
                        <TableHead className="font-semibold text-center">Height</TableHead>
                        <TableHead className="font-semibold text-center">Girth</TableHead>
                        <TableHead className="font-semibold text-center">Confidence</TableHead>
                        <TableHead className="font-semibold">Notes</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {records.map((record) => (
                        <TableRow key={record.id} className="hover:bg-muted/30 transition-colors">
                          <TableCell className="font-medium">
                            <div className="flex items-center gap-2">
                              <Calendar className="w-4 h-4 text-muted-foreground" />
                              {formatDate(record.date)}
                            </div>
                          </TableCell>
                          <TableCell className="capitalize">{record.animalType}</TableCell>
                          <TableCell className="text-center font-semibold text-primary">{record.bodyLength} cm</TableCell>
                          <TableCell className="text-center font-semibold text-blue-600">{record.withersHeight} cm</TableCell>
                          <TableCell className="text-center font-semibold text-purple-600">{record.girth} cm</TableCell>
                          <TableCell className="text-center">
                            <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                              record.confidence >= 90 ? 'bg-green-100 text-green-700' : 
                              record.confidence >= 80 ? 'bg-yellow-100 text-yellow-700' : 
                              'bg-red-100 text-red-700'
                            }`}>
                              {record.confidence}%
                            </span>
                          </TableCell>
                          <TableCell className="text-muted-foreground text-sm max-w-[200px] truncate">
                            {record.notes || '-'}
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </div>
              </CardContent>
            </Card>

            {/* Records Cards - Mobile */}
            <div className="md:hidden space-y-4">
              {records.map((record) => (
                <Card key={record.id} className="overflow-hidden">
                  <CardContent className="p-4">
                    <div className="flex items-center justify-between mb-4">
                      <div className="flex items-center gap-2">
                        <span className="px-2 py-1 bg-primary/10 text-primary rounded-full text-xs font-medium capitalize">
                          {record.animalType}
                        </span>
                        <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                          record.confidence >= 90 ? 'bg-green-100 text-green-700' : 
                          record.confidence >= 80 ? 'bg-yellow-100 text-yellow-700' : 
                          'bg-red-100 text-red-700'
                        }`}>
                          {record.confidence}%
                        </span>
                      </div>
                      <div className="flex items-center gap-1 text-xs text-muted-foreground">
                        <Calendar className="w-3 h-3" />
                        {formatDate(record.date)}
                      </div>
                    </div>
                    
                    <div className="grid grid-cols-3 gap-3">
                      <div className="text-center p-3 bg-primary/5 rounded-lg">
                        <div className="text-lg font-bold text-primary">{record.bodyLength}</div>
                        <div className="text-xs text-muted-foreground">Length (cm)</div>
                      </div>
                      <div className="text-center p-3 bg-blue-500/5 rounded-lg">
                        <div className="text-lg font-bold text-blue-600">{record.withersHeight}</div>
                        <div className="text-xs text-muted-foreground">Height (cm)</div>
                      </div>
                      <div className="text-center p-3 bg-purple-500/5 rounded-lg">
                        <div className="text-lg font-bold text-purple-600">{record.girth}</div>
                        <div className="text-xs text-muted-foreground">Girth (cm)</div>
                      </div>
                    </div>
                    
                    {record.notes && (
                      <p className="mt-3 text-sm text-muted-foreground truncate">
                        {record.notes}
                      </p>
                    )}
                  </CardContent>
                </Card>
              ))}
            </div>

            {/* Load More / Summary */}
            <div className="mt-6 text-center">
              <p className="text-sm text-muted-foreground">
                Showing {records.length} of {totalCount} records
                {hasMore && ' • Scroll for more'}
              </p>
            </div>
          </>
        )}
      </div>
    </div>
  );
};

export default DataRecords;
