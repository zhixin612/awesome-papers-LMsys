import React, { useState, useEffect, useMemo } from 'react';
import ReactDOM from 'react-dom/client';
import {
  Search,
  ExternalLink,
  Sparkles,
  Moon,
  Sun,
  Filter,
  Calendar,
  Tag,
  BookOpen,
  Github
} from 'lucide-react';


// --- Components ---

const Badge = ({ children, className = "", onClick }) => (
  <span
    onClick={onClick}
    className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium transition-colors ${onClick ? 'cursor-pointer hover:opacity-80' : ''} ${className}`}
  >
    {children}
  </span>
);

const Button = ({ children, onClick, variant = 'primary', className = "", icon: Icon }) => {
  const baseStyle = "inline-flex items-center justify-center px-4 py-2 text-sm font-medium rounded-md focus:outline-none focus:ring-2 focus:ring-offset-2 transition-all duration-200";
  const variants = {
    primary: "bg-blue-600 hover:bg-blue-700 text-white focus:ring-blue-500",
    secondary: "bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-200 border border-gray-300 dark:border-gray-600 hover:bg-gray-50 dark:hover:bg-gray-700",
    ghost: "text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-800",
    outline: "border border-gray-300 dark:border-gray-600 text-gray-700 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-800"
  };

  return (
    <button onClick={onClick} className={`${baseStyle} ${variants[variant]} ${className}`}>
      {Icon && <Icon className="w-4 h-4 mr-2" />}
      {children}
    </button>
  );
};

const PaperCard = ({ paper, language }) => {
  // Construct Gemini URL
  const prompt = `Please analyze this paper for me based on its application in Large Model System Optimization: ${paper.title}. Link: ${paper.link}`;
  const geminiUrl = `https://gemini.google.com/app?text=${encodeURIComponent(prompt)}`;

  // Handle TLDR display based on availability
  const tldrText = language === 'zh' && paper.tldr_zh ? paper.tldr_zh : paper.tldr;

  return (
    <div className="group relative flex flex-col bg-white dark:bg-gray-800 rounded-xl shadow-sm hover:shadow-md border border-gray-200 dark:border-gray-700 transition-all duration-200 overflow-hidden">
      <div className="p-5 flex-1">
        {/* Date & Categories */}
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center space-x-2 text-xs text-gray-500 dark:text-gray-400">
            <Calendar className="w-3.5 h-3.5" />
            <span>{paper.indexed_date}</span>
            <span className="text-gray-300 dark:text-gray-600">|</span>
            <span className="truncate max-w-[150px]">{paper.categories.join(', ')}</span>
          </div>
          <div className="flex space-x-1">
             {paper.tags && paper.tags.map(tag => (
               <Badge key={tag} className="bg-blue-50 text-blue-700 dark:bg-blue-900/30 dark:text-blue-300 border border-blue-100 dark:border-blue-800">
                 {tag}
               </Badge>
             ))}
          </div>
        </div>

        {/* Title */}
        <h3 className="text-lg font-bold text-gray-900 dark:text-gray-100 mb-2 leading-snug group-hover:text-blue-600 dark:group-hover:text-blue-400 transition-colors">
          <a href={paper.link} target="_blank" rel="noopener noreferrer">
            {paper.title}
          </a>
        </h3>

        {/* Authors */}
        <p className="text-sm text-gray-600 dark:text-gray-400 mb-4 line-clamp-1" title={paper.authors.join(', ')}>
          {paper.authors.join(', ')}
        </p>

        {/* TLDR Section */}
        {tldrText && (
          <div className="bg-gray-50 dark:bg-gray-700/50 rounded-lg p-3 mb-4 border border-gray-100 dark:border-gray-700">
            <div className="flex items-center mb-1 text-xs font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wider">
              <Sparkles className="w-3 h-3 mr-1 text-yellow-500" />
              TL;DR
            </div>
            <p className="text-sm text-gray-700 dark:text-gray-300 leading-relaxed">
              {tldrText}
            </p>
          </div>
        )}
      </div>

      {/* Footer Actions */}
      <div className="p-4 bg-gray-50 dark:bg-gray-800/80 border-t border-gray-100 dark:border-gray-700 flex items-center justify-between gap-3">
        <a
          href={paper.link}
          target="_blank"
          rel="noopener noreferrer"
          className="flex-1"
        >
          <Button variant="secondary" className="w-full text-xs" icon={ExternalLink}>
            ArXiv PDF
          </Button>
        </a>
        <a
          href={geminiUrl}
          target="_blank"
          rel="noopener noreferrer"
          className="flex-1"
        >
          <Button variant="outline" className="w-full text-xs group/gemini" icon={Sparkles}>
            <span className="bg-gradient-to-r from-blue-500 to-purple-600 bg-clip-text text-transparent font-bold group-hover/gemini:opacity-80">
              Ask AI
            </span>
          </Button>
        </a>
      </div>
    </div>
  );
};

// --- Main Application ---

const App = () => {
  const [papers, setPapers] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // UI State
  const [darkMode, setDarkMode] = useState(() => window.matchMedia('(prefers-color-scheme: dark)').matches);
  const [language, setLanguage] = useState('en'); // 'en' or 'zh'

  // Filter State
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedTags, setSelectedTags] = useState([]);
  const [sortOrder, setSortOrder] = useState('newest'); // 'newest' | 'oldest'

  // Load Data
  useEffect(() => {
    const loadData = async () => {
      try {
        // Try fetching from root first (production), then fallback to tools (development)
        // In GitHub Pages production, index.json will be copied to root or accessible path
        const pathsToTry = ['./index.json', '../tools/index.json', 'https://raw.githubusercontent.com/USER/REPO/main/tools/index.json'];

        let data = null;
        for (const path of pathsToTry) {
            try {
                const res = await fetch(path);
                if (res.ok) {
                    data = await res.json();
                    break;
                }
            } catch (e) {
                console.warn(`Failed to load from ${path}`);
            }
        }

        if (!data) throw new Error("Could not load paper data.");

        // Convert Object to Array and filter only relevant if needed
        // Assuming the JSON is keyed by ID
        const paperArray = Object.values(data).filter(p => p.relevant !== false); // Default show true or undefined
        setPapers(paperArray);
      } catch (err) {
        console.error(err);
        setError("Failed to load papers. Please check the network.");
      } finally {
        setLoading(false);
      }
    };
    loadData();
  }, []);

  // Toggle Dark Mode
  useEffect(() => {
    if (darkMode) {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
  }, [darkMode]);

  // Derived Data: Unique Tags
  const allTags = useMemo(() => {
    const tags = new Set();
    papers.forEach(p => p.tags && p.tags.forEach(t => tags.add(t)));
    return Array.from(tags).sort();
  }, [papers]);

  // Filter & Sort Logic
  const filteredPapers = useMemo(() => {
    return papers
      .filter(paper => {
        // Search Filter
        const query = searchQuery.toLowerCase();
        const matchesSearch =
          paper.title.toLowerCase().includes(query) ||
          paper.abstract.toLowerCase().includes(query) ||
          paper.authors.some(a => a.toLowerCase().includes(query));

        // Tag Filter (Union / OR logic)
        const matchesTags = selectedTags.length === 0 ||
          (paper.tags && paper.tags.some(t => selectedTags.includes(t)));

        return matchesSearch && matchesTags;
      })
      .sort((a, b) => {
        // Sort by Date
        const dateA = new Date(a.indexed_date);
        const dateB = new Date(b.indexed_date);
        return sortOrder === 'newest' ? dateB - dateA : dateA - dateB;
      });
  }, [papers, searchQuery, selectedTags, sortOrder]);

  const toggleTag = (tag) => {
    setSelectedTags(prev =>
      prev.includes(tag) ? prev.filter(t => t !== tag) : [...prev, tag]
    );
  };

  if (loading) return (
    <div className="min-h-screen flex items-center justify-center bg-gray-50 dark:bg-gray-900">
      <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
    </div>
  );

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900 pb-20">
      {/* Navbar */}
      <header className="sticky top-0 z-30 bg-white/80 dark:bg-gray-900/80 backdrop-blur-md border-b border-gray-200 dark:border-gray-700">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <BookOpen className="w-6 h-6 text-blue-600 dark:text-blue-400" />
            <h1 className="text-xl font-bold text-gray-900 dark:text-white hidden sm:block">
              Daily ArXiv <span className="text-gray-400 font-normal">System Opt</span>
            </h1>
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={() => setLanguage(l => l === 'en' ? 'zh' : 'en')}
              className="px-3 py-1 text-xs font-medium rounded-md bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-700 transition-colors"
            >
              {language === 'en' ? '中' : 'EN'}
            </button>
            <button
              onClick={() => setDarkMode(!darkMode)}
              className="p-2 rounded-md text-gray-500 hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
            >
              {darkMode ? <Sun className="w-5 h-5" /> : <Moon className="w-5 h-5" />}
            </button>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Controls Section */}
        <div className="mb-8 space-y-4">

          {/* Search & Sort */}
          <div className="flex flex-col md:flex-row gap-4">
            <div className="relative flex-1">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400" />
              <input
                type="text"
                placeholder={language === 'en' ? "Search title, abstract, authors..." : "搜索标题、摘要、作者..."}
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="w-full pl-10 pr-4 py-2 rounded-lg border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100 focus:ring-2 focus:ring-blue-500 focus:border-transparent outline-none transition-all"
              />
            </div>

            <div className="flex gap-2 shrink-0">
               <select
                 value={sortOrder}
                 onChange={(e) => setSortOrder(e.target.value)}
                 className="px-4 py-2 rounded-lg border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-300 text-sm focus:ring-2 focus:ring-blue-500 outline-none"
               >
                 <option value="newest">Latest First</option>
                 <option value="oldest">Oldest First</option>
               </select>
            </div>
          </div>

          {/* Tags Filter */}
          {allTags.length > 0 && (
            <div className="flex flex-wrap gap-2 items-center">
              <span className="text-sm text-gray-500 dark:text-gray-400 flex items-center">
                <Filter className="w-4 h-4 mr-1" />
                Filter:
              </span>
              {allTags.map(tag => (
                <button
                  key={tag}
                  onClick={() => toggleTag(tag)}
                  className={`px-3 py-1 rounded-full text-xs font-medium border transition-colors ${
                    selectedTags.includes(tag)
                      ? 'bg-blue-600 text-white border-blue-600'
                      : 'bg-white dark:bg-gray-800 text-gray-600 dark:text-gray-300 border-gray-200 dark:border-gray-700 hover:border-blue-300 dark:hover:border-blue-700'
                  }`}
                >
                  {tag}
                </button>
              ))}
              {selectedTags.length > 0 && (
                <button
                  onClick={() => setSelectedTags([])}
                  className="text-xs text-red-500 hover:underline ml-2"
                >
                  Clear
                </button>
              )}
            </div>
          )}
        </div>

        {/* Stats */}
        <div className="mb-6 text-sm text-gray-500 dark:text-gray-400">
          Showing {filteredPapers.length} papers
        </div>

        {/* Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-6">
          {filteredPapers.map(paper => (
            <PaperCard key={paper.id} paper={paper} language={language} />
          ))}
        </div>

        {/* Empty State */}
        {filteredPapers.length === 0 && (
          <div className="text-center py-20">
            <div className="bg-gray-100 dark:bg-gray-800 rounded-full w-16 h-16 flex items-center justify-center mx-auto mb-4">
              <Search className="w-8 h-8 text-gray-400" />
            </div>
            <h3 className="text-lg font-medium text-gray-900 dark:text-white">No papers found</h3>
            <p className="text-gray-500 dark:text-gray-400">Try adjusting your search or filters.</p>
          </div>
        )}
      </main>
    </div>
  );
};

// Create Root
const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);