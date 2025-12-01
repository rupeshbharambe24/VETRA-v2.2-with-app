import { createClient } from '@supabase/supabase-js'

// Initialize Supabase client with proper environment variables for Lovable
const supabaseUrl = import.meta.env.VITE_SUPABASE_URL || ''
const supabaseAnonKey = import.meta.env.VITE_SUPABASE_ANON_KEY || ''
const apiBaseUrl = import.meta.env.VITE_API_BASE_URL || ''

// Create Supabase client only if both URL and key are available
export const supabase = supabaseUrl && supabaseAnonKey 
  ? createClient(supabaseUrl, supabaseAnonKey)
  : null

// Types for API responses
export interface TraitExtractionResult {
  id: string;
  bodyLength: number;
  withersHeight: number;
  girth: number;
  bodyScore: number;
  confidence: number;
  processingStatus: 'processing' | 'completed' | 'failed';
  timestamp: string;
  animalType?: string;
  estimatedWeight?: number;
}

export interface HistoricalRecord {
  id: string;
  date: string;
  imageThumbnail: string;
  bodyLength: number;
  withersHeight: number;
  girth: number;
  confidence: number;
  animalType: string;
  notes: string;
}

export interface ApiResponse<T> {
  success: boolean;
  data?: T;
  error?: string;
}

// Process image for trait extraction
export const processImage = async (
  imageData: string,
  animalType: string = 'cattle',
  notes: string = ''
): Promise<ApiResponse<TraitExtractionResult>> => {
  try {
    if (apiBaseUrl) {
      const response = await fetch(`${apiBaseUrl}/api/v1/process-image`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          imageData,
          animalType,
          notes,
          viewType: 'side',
        }),
      })

      if (!response.ok) {
        const errorText = await response.text()
        throw new Error(errorText || `Request failed with status ${response.status}`)
      }

      const data = (await response.json()) as TraitExtractionResult
      return { success: true, data }
    }

    // Supabase path (if configured)
    if (supabase) {
      const { data, error } = await supabase.functions.invoke('process-image', {
        body: {
          imageData,
          animalType,
          notes,
        },
      })

      if (error) {
        console.error('API Error:', error)
        return {
          success: false,
          error: error.message || 'Failed to process image'
        }
      }

      return data
    }

    // Fallback: Return mock data if no API configured
    console.warn('API base URL not set, using mock data')
    await new Promise(resolve => setTimeout(resolve, 2000)) // Simulate processing delay
    
    // Generate more varied and realistic body scores using weighted distribution
    // This simulates real-world distribution where most animals are average, fewer are weak/strong
    const generateBodyScore = (): number => {
      const rand = Math.random();
      // 20% chance weak (1-3), 50% chance average (4-6), 30% chance strong (7-9)
      if (rand < 0.20) {
        return Math.floor(Math.random() * 3) + 1; // 1-3 (Weak)
      } else if (rand < 0.70) {
        return Math.floor(Math.random() * 3) + 4; // 4-6 (Average)
      } else {
        return Math.floor(Math.random() * 3) + 7; // 7-9 (Strong)
      }
    };

    const bodyScore = generateBodyScore();
    
    // Correlate measurements with body score for more realistic results
    const baseLength = 120 + (bodyScore * 5); // Higher score = larger animal
    const baseHeight = 110 + (bodyScore * 3);
    const baseGirth = 150 + (bodyScore * 5);
    const baseWeight = 300 + (bodyScore * 25);

    const mockResult: TraitExtractionResult = {
      id: `mock_${Date.now()}`,
      bodyLength: Math.round((baseLength + Math.random() * 15 - 7.5) * 100) / 100,
      withersHeight: Math.round((baseHeight + Math.random() * 10 - 5) * 100) / 100,
      girth: Math.round((baseGirth + Math.random() * 15 - 7.5) * 100) / 100,
      bodyScore: bodyScore,
      confidence: Math.round((Math.random() * 12 + 82) * 100) / 100, // 82-94%
      processingStatus: 'completed',
      timestamp: new Date().toISOString(),
      animalType: animalType,
      estimatedWeight: Math.round(baseWeight + Math.random() * 50 - 25),
    }
    
    return {
      success: true,
      data: mockResult
    }
  } catch (error) {
    console.error('Network Error:', error)
    return {
      success: false,
      error: error instanceof Error ? error.message : 'Network error. Please check your connection.'
    }
  }
}

// Get historical results
export const getResults = async (
  limit: number = 50,
  offset: number = 0,
  search: string = ''
): Promise<ApiResponse<HistoricalRecord[]> & { totalCount?: number; hasMore?: boolean }> => {
  try {
    if (apiBaseUrl) {
      const url = new URL(`${apiBaseUrl}/api/v1/results`)
      url.searchParams.set('limit', String(limit))
      const response = await fetch(url.toString(), { method: 'GET' })

      if (!response.ok) {
        const errorText = await response.text()
        throw new Error(errorText || `Request failed with status ${response.status}`)
      }

      const data = (await response.json()) as HistoricalRecord[]
      return {
        success: true,
        data,
        totalCount: data.length,
        hasMore: data.length >= limit
      }
    }

    // Supabase path (if configured)
    if (supabase) {
      const { data, error } = await supabase.functions.invoke('get-results', {
        body: {
          limit,
          offset,
          search,
        },
      })

      if (error) {
        console.error('API Error:', error)
        return {
          success: false,
          error: error.message || 'Failed to fetch results'
        }
      }

      return data
    }

    // Fallback: Return mock data if no API configured
    console.warn('API base URL not set, using mock data')
    await new Promise(resolve => setTimeout(resolve, 1000)) // Simulate network delay
    
    const mockRecords: HistoricalRecord[] = [
      {
        id: '1',
        date: '2024-01-15',
        imageThumbnail: 'data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjAwIiBoZWlnaHQ9IjIwMCIgdmlld0JveD0iMCAwIDIwMCAyMDAiIGZpbGw9Im5vbmUiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+CjxyZWN0IHdpZHRoPSIyMDAiIGhlaWdodD0iMjAwIiBmaWxsPSIjRjNGNEY2Ii8+Cjx0ZXh0IHg9IjEwMCIgeT0iMTEwIiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBmaWxsPSIjOUI5QkEzIiBmb250LXNpemU9IjE0cHgiPkNvdyAxPC90ZXh0Pgo8L3N2Zz4K',
        bodyLength: 145.5,
        withersHeight: 128.2,
        girth: 165.8,
        confidence: 92.5,
        animalType: 'cattle',
        notes: 'Sample analysis 1'
      },
      {
        id: '2',
        date: '2024-01-12',
        imageThumbnail: 'data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjAwIiBoZWlnaHQ9IjIwMCIgdmlld0JveD0iMCAwIDIwMCAyMDAiIGZpbGw9Im5vbmUiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+CjxyZWN0IHdpZHRoPSIyMDAiIGhlaWdodD0iMjAwIiBmaWxsPSIjRjNGNEY2Ii8+Cjx0ZXh0IHg9IjEwMCIgeT0iMTEwIiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBmaWxsPSIjOUI5QkEzIiBmb250LXNpemU9IjE0cHgiPkNvdyAyPC90ZXh0Pgo8L3N2Zz4K',
        bodyLength: 152.3,
        withersHeight: 132.1,
        girth: 171.2,
        confidence: 88.7,
        animalType: 'cattle',
        notes: 'Sample analysis 2'
      },
      {
        id: '3',
        date: '2024-01-10',
        imageThumbnail: 'data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjAwIiBoZWlnaHQ9IjIwMCIgdmlld0JveD0iMCAwIDIwMCAyMDAiIGZpbGw9Im5vbmUiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+CjxyZWN0IHdpZHRoPSIyMDAiIGhlaWdodD0iMjAwIiBmaWxsPSIjRjNGNEY2Ii8+Cjx0ZXh0IHg9IjEwMCIgeT0iMTEwIiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBmaWxsPSIjOUI5QkEzIiBmb250LXNpemU9IjE0cHgiPkJ1ZmZhbG88L3RleHQ+Cjwvc3ZnPgo=',
        bodyLength: 168.9,
        withersHeight: 142.5,
        girth: 185.3,
        confidence: 94.2,
        animalType: 'buffalo',
        notes: 'Sample analysis 3'
      }
    ].filter(record => 
      !search || 
      record.animalType.toLowerCase().includes(search.toLowerCase()) ||
      record.notes.toLowerCase().includes(search.toLowerCase())
    );
    
    return {
      success: true,
      data: mockRecords.slice(offset, offset + limit),
      totalCount: mockRecords.length,
      hasMore: (offset + limit) < mockRecords.length
    }
  } catch (error) {
    console.error('Network Error:', error)
    return {
      success: false,
      error: error instanceof Error ? error.message : 'Network error. Please check your connection.'
    }
  }
}
