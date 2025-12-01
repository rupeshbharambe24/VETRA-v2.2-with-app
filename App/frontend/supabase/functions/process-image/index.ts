import { serve } from "https://deno.land/std@0.168.0/http/server.ts"
import { createClient } from 'https://esm.sh/@supabase/supabase-js@2'

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
}

interface TraitExtractionRequest {
  imageData: string; // base64 encoded image
  imageUrl?: string;
  animalType?: string;
  notes?: string;
}

interface TraitExtractionResult {
  id: string;
  bodyLength: number;
  withersHeight: number; 
  girth: number;
  confidence: number;
  processingStatus: 'completed' | 'failed';
  timestamp: string;
}

serve(async (req) => {
  // Handle CORS preflight requests
  if (req.method === 'OPTIONS') {
    return new Response('ok', { headers: corsHeaders })
  }

  try {
    // Initialize Supabase client
    const supabaseClient = createClient(
      Deno.env.get('SUPABASE_URL') ?? '',
      Deno.env.get('SUPABASE_ANON_KEY') ?? '',
      {
        global: {
          headers: { Authorization: req.headers.get('Authorization')! },
        },
      }
    )

    // Get user from JWT token
    const {
      data: { user },
    } = await supabaseClient.auth.getUser()

    if (req.method === 'POST') {
      const { imageData, imageUrl, animalType = 'cattle', notes = '' }: TraitExtractionRequest = await req.json()

      // Input validation
      if (!imageData && !imageUrl) {
        throw new Error('Image data or URL is required')
      }

      // Simulate AI processing (replace with actual AI model integration)
      console.log('Processing image for trait extraction...')
      
      // Mock processing delay
      await new Promise(resolve => setTimeout(resolve, 2000))
      
      // Generate mock measurements (replace with actual AI results)
      const mockResults = {
        bodyLength: Math.round((Math.random() * 50 + 120) * 100) / 100, // 120-170 cm
        withersHeight: Math.round((Math.random() * 30 + 110) * 100) / 100, // 110-140 cm  
        girth: Math.round((Math.random() * 40 + 150) * 100) / 100, // 150-190 cm
        confidence: Math.round((Math.random() * 15 + 85) * 100) / 100, // 85-100%
      }

      // Create thumbnail (in real implementation, would resize the image)
      const thumbnail = imageData || 'data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjAwIiBoZWlnaHQ9IjIwMCIgdmlld0JveD0iMCAwIDIwMCAyMDAiIGZpbGw9Im5vbmUiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+CjxyZWN0IHdpZHRoPSIyMDAiIGhlaWdodD0iMjAwIiBmaWxsPSIjRjNGNEY2Ii8+Cjx0ZXh0IHg9IjEwMCIgeT0iMTEwIiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBmaWxsPSIjOUI5QkEzIiBmb250LXNpemU9IjE0cHgiPkFuYWx5c2lzPC90ZXh0Pgo8L3N2Zz4K'

      // Store result in database
      const { data: extraction, error } = await supabaseClient
        .from('trait_extractions')
        .insert([
          {
            image_url: imageUrl,
            image_thumbnail: thumbnail,
            body_length: mockResults.bodyLength,
            withers_height: mockResults.withersHeight,
            girth: mockResults.girth,
            confidence: mockResults.confidence,
            processing_status: 'completed',
            animal_type: animalType,
            notes: notes,
            user_id: user?.id || null,
          }
        ])
        .select()
        .single()

      if (error) {
        console.error('Database error:', error)
        throw new Error('Failed to save extraction result')
      }

      // Return successful result
      const result: TraitExtractionResult = {
        id: extraction.id,
        bodyLength: extraction.body_length,
        withersHeight: extraction.withers_height,
        girth: extraction.girth,
        confidence: extraction.confidence,
        processingStatus: extraction.processing_status,
        timestamp: extraction.created_at,
      }

      return new Response(
        JSON.stringify({ success: true, data: result }),
        {
          headers: { ...corsHeaders, 'Content-Type': 'application/json' },
          status: 200,
        },
      )
    }

    // Method not allowed
    return new Response(
      JSON.stringify({ error: 'Method not allowed' }),
      {
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
        status: 405,
      },
    )

  } catch (error) {
    console.error('Error processing request:', error)
    
    return new Response(
      JSON.stringify({ 
        success: false, 
        error: error instanceof Error ? error.message : 'Internal server error' 
      }),
      {
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
        status: 500,
      },
    )
  }
})