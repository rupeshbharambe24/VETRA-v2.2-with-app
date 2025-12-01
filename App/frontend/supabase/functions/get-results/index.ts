import { serve } from "https://deno.land/std@0.168.0/http/server.ts"
import { createClient } from 'https://esm.sh/@supabase/supabase-js@2'

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
}

interface ResultsResponse {
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

    if (req.method === 'GET') {
      // Parse query parameters
      const url = new URL(req.url)
      const limit = parseInt(url.searchParams.get('limit') || '50')
      const offset = parseInt(url.searchParams.get('offset') || '0')
      const search = url.searchParams.get('search') || ''

      // Build query
      let query = supabaseClient
        .from('trait_extractions')
        .select('*')
        .eq('processing_status', 'completed')
        .order('created_at', { ascending: false })
        .range(offset, offset + limit - 1)

      // Add search filter if provided
      if (search) {
        query = query.or(`animal_type.ilike.%${search}%,notes.ilike.%${search}%`)
      }

      // Execute query
      const { data: extractions, error } = await query

      if (error) {
        console.error('Database error:', error)
        throw new Error('Failed to fetch extraction results')
      }

      // Transform data to match frontend interface
      const results: ResultsResponse[] = extractions.map(extraction => ({
        id: extraction.id,
        date: extraction.created_at.split('T')[0], // Format date as YYYY-MM-DD
        imageThumbnail: extraction.image_thumbnail || 'data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjAwIiBoZWlnaHQ9IjIwMCIgdmlld0JveD0iMCAwIDIwMCAyMDAiIGZpbGw9Im5vbmUiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+CjxyZWN0IHdpZHRoPSIyMDAiIGhlaWdodD0iMjAwIiBmaWxsPSIjRjNGNEY2Ii8+Cjx0ZXh0IHg9IjEwMCIgeT0iMTEwIiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBmaWxsPSIjOUI5QkEzIiBmb250LXNpemU9IjE0cHgiPk5vIEltYWdlPC90ZXh0Pgo8L3N2Zz4K',
        bodyLength: parseFloat(extraction.body_length),
        withersHeight: parseFloat(extraction.withers_height),
        girth: parseFloat(extraction.girth),
        confidence: parseFloat(extraction.confidence),
        animalType: extraction.animal_type || 'cattle',
        notes: extraction.notes || '',
      }))

      // Get total count for pagination
      const { count } = await supabaseClient
        .from('trait_extractions')
        .select('*', { count: 'exact', head: true })
        .eq('processing_status', 'completed')

      return new Response(
        JSON.stringify({ 
          success: true, 
          data: results,
          totalCount: count || 0,
          hasMore: (offset + limit) < (count || 0)
        }),
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