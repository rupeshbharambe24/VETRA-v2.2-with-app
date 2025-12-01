import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import VetraHome from "./components/VetraHome";
import TraitExtraction from "./components/TraitExtraction";
import DataRecords from "./components/DataRecords";
import Schemes from "./pages/Schemes";
import Insurance from "./pages/Insurance";
import Helpline from "./pages/Helpline";
import About from "./pages/About";
import Contact from "./pages/Contact";
import NotFound from "./pages/NotFound";

const queryClient = new QueryClient();

const App = () => (
  <QueryClientProvider client={queryClient}>
    <TooltipProvider>
      <Toaster />
      <Sonner />
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<VetraHome />} />
          <Route path="/trait-extraction" element={<TraitExtraction />} />
          <Route path="/data-records" element={<DataRecords />} />
          <Route path="/schemes" element={<Schemes />} />
          <Route path="/insurance" element={<Insurance />} />
          <Route path="/helpline" element={<Helpline />} />
          <Route path="/about" element={<About />} />
          <Route path="/contact" element={<Contact />} />
          {/* ADD ALL CUSTOM ROUTES ABOVE THE CATCH-ALL "*" ROUTE */}
          <Route path="*" element={<NotFound />} />
        </Routes>
      </BrowserRouter>
    </TooltipProvider>
  </QueryClientProvider>
);

export default App;
