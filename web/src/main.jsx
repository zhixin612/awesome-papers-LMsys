import React, { useState, useEffect, useMemo, useRef } from 'react';
import ReactDOM from 'react-dom/client';
import {
  Search,
  ExternalLink,
  Sparkles,
  Moon,
  Sun,
  Filter,
  Calendar,
  ChevronLeft,
  ChevronRight,
  FileText,
  User,
  X,
  Star,
  Quote,
  Copy,
  Github,
  Check
} from 'lucide-react';

// --- Components ---

const Badge = ({ children, className = "", onClick }) => (
  <span
    onClick={onClick}
    className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium transition-colors border ${onClick ? 'cursor-pointer hover:opacity-80' : ''} ${className}`}
  >
    {children}
  </span>
);

const Button = ({ children, onClick, variant = 'primary', className = "", icon: Icon, disabled, title }) => {
  const baseStyle = "inline-flex items-center justify-center px-3 py-1.5 text-sm font-medium rounded-md focus:outline-none focus:ring-2 focus:ring-offset-2 transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed";
  const variants = {
    primary: "bg-blue-600 hover:bg-blue-700 text-white focus:ring-blue-500 border-transparent",
    secondary: "bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-200 border border-gray-300 dark:border-gray-600 hover:bg-gray-50 dark:hover:bg-gray-700",
    ghost: "text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-800 border-transparent",
    outline: "border border-gray-300 dark:border-gray-600 text-gray-700 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-800",
    danger: "bg-red-50 text-red-600 hover:bg-red-100 border border-red-200 dark:bg-red-900/20 dark:text-red-300 dark:border-red-800"
  };

  return (
    <button onClick={onClick} disabled={disabled} className={`${baseStyle} ${variants[variant]} ${className}`} title={title}>
      {Icon && <Icon className="w-4 h-4 mr-2" />}
      {children}
    </button>
  );
};

// 分页组件
const PaginationControls = ({ currentPage, totalPages, onPageChange, itemsPerPage, onItemsPerPageChange }) => {
  const [inputPage, setInputPage] = useState(currentPage);

  useEffect(() => {
    setInputPage(currentPage);
  }, [currentPage]);

  const handleKeyDown = (e) => {
    if (e.key === 'Enter') {
      const page = parseInt(inputPage);
      if (!isNaN(page) && page >= 1 && page <= totalPages) {
        onPageChange(page);
      } else {
        setInputPage(currentPage);
      }
    }
  };

  if (totalPages <= 1 && itemsPerPage >= 100) return null;

  return (
    <div className="flex flex-wrap items-center gap-4 text-sm">
      <div className="flex items-center gap-2">
        <span className="text-gray-500 dark:text-gray-400 hidden sm:inline">Show:</span>
        <select
          value={itemsPerPage}
          onChange={(e) => onItemsPerPageChange(Number(e.target.value))}
          className="bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-700 text-gray-700 dark:text-gray-200 text-xs rounded-md py-1 px-2 focus:ring-1 focus:ring-blue-500 outline-none cursor-pointer"
        >
          <option value={10}>10</option>
          <option value={30}>30</option>
          <option value={50}>50</option>
          <option value={100}>100</option>
        </select>
      </div>

      <div className="flex items-center bg-white dark:bg-gray-800 rounded-md border border-gray-300 dark:border-gray-700 shadow-sm">
        <button
          onClick={() => onPageChange(currentPage - 1)}
          disabled={currentPage === 1}
          className="p-1.5 hover:bg-gray-50 dark:hover:bg-gray-700 rounded-l-md disabled:opacity-30 transition-colors"
        >
          <ChevronLeft className="w-4 h-4 text-gray-600 dark:text-gray-300" />
        </button>

        <div className="flex items-center border-x border-gray-300 dark:border-gray-700 px-1">
            <input
                type="number"
                min="1"
                max={totalPages}
                value={inputPage}
                onChange={(e) => setInputPage(e.target.value)}
                onKeyDown={handleKeyDown}
                className="w-10 text-center py-1 text-gray-700 dark:text-gray-200 bg-transparent outline-none appearance-none font-medium"
            />
            <span className="text-gray-400 dark:text-gray-500 px-1">/ {totalPages || 1}</span>
        </div>

        <button
          onClick={() => onPageChange(currentPage + 1)}
          disabled={currentPage === totalPages || totalPages === 0}
          className="p-1.5 hover:bg-gray-50 dark:hover:bg-gray-700 rounded-r-md disabled:opacity-30 transition-colors"
        >
          <ChevronRight className="w-4 h-4 text-gray-600 dark:text-gray-300" />
        </button>
      </div>
    </div>
  );
};

// --- Helper Functions ---

const generateBibTeX = (paper) => {
  const year = paper.indexed_date ? paper.indexed_date.split('-')[0] : new Date().getFullYear();
  const firstAuthor = paper.authors && paper.authors.length > 0 ? paper.authors[0].split(' ').pop() : 'Unknown';
  const id = paper.id ? paper.id.split('/').pop() : 'unknown';
  const citeKey = `${firstAuthor}${year}${paper.title.split(' ')[0].replace(/[^a-zA-Z]/g, '')}`;

  return `@article{${citeKey},
  title={${paper.title}},
  author={${paper.authors.join(' and ')}},
  journal={arXiv preprint arXiv:${id}},
  year={${year}},
  url={${paper.link}}
}`;
};

const extractCodeLink = (abstract) => {
    const githubRegex = /https?:\/\/(www\.)?github\.com\/[a-zA-Z0-9-]+\/[a-zA-Z0-9-._]+/gi;
    const match = abstract.match(githubRegex);
    return match ? match[0] : null;
};

// --- Paper Card Component ---

const PaperCard = ({ paper, language, isStarred, toggleStar }) => {
  const [showPdf, setShowPdf] = useState(false);
  const [copied, setCopied] = useState(null); // 'bibtex' or 'share'

  // Safe Access
  const title = paper.title || "Untitled Paper";
  const link = paper.link || "#";
  const authors = Array.isArray(paper.authors) && paper.authors.length > 0 ? paper.authors : ["Unknown Author"];
  const categories = Array.isArray(paper.categories) && paper.categories.length > 0 ? paper.categories : ["Uncategorized"];
  const tags = Array.isArray(paper.tags) ? paper.tags : [];
  const indexedDate = paper.indexed_date || "Unknown Date";

  // Safe TLDR
  const tldrEn = paper.tldr || paper.abstract || "No summary available.";
  const tldrZh = paper.tldr_zh || tldrEn;
  const tldrText = language === 'zh' ? tldrZh : tldrEn;

  // Logic
  const prompt = `Please analyze this paper for me based on its application in Large Model System Optimization: ${title}. Link: ${link}`;
  const geminiUrl = `https://gemini.google.com/app?text=${encodeURIComponent(prompt)}`;
  const pdfUrl = link.replace(/^http:/, 'https:').replace('/abs/', '/pdf/') + ".pdf";
  const formatCategory = (cat) => cat ? cat.replace(/^cs\./, '') : 'N/A';
  const codeLink = extractCodeLink(paper.abstract || "");

  // Handlers
  const handleCopyBibtex = () => {
    const bib = generateBibTeX(paper);
    navigator.clipboard.writeText(bib);
    setCopied('bibtex');
    setTimeout(() => setCopied(null), 2000);
  };

  const handleCopyShare = () => {
    const text = `${title}\n${link}`;
    navigator.clipboard.writeText(text);
    setCopied('share');
    setTimeout(() => setCopied(null), 2000);
  };

  return (
    <div className={`group relative flex flex-col bg-white dark:bg-gray-800 rounded-xl shadow-sm border transition-all duration-200 overflow-hidden ${showPdf ? 'ring-2 ring-blue-500/20 border-blue-500/30' : 'hover:shadow-md border-gray-200 dark:border-gray-700'}`}>
      <div className="p-5 flex flex-col gap-3">

        {/* Top Row */}
        <div className="flex flex-wrap items-center justify-between gap-y-2 gap-x-4">
          <div className="flex items-center gap-3 text-xs overflow-hidden">
            <div className="flex items-center text-gray-500 dark:text-gray-400 whitespace-nowrap shrink-0">
              <Calendar className="w-3.5 h-3.5 mr-1" />
              <span>{indexedDate}</span>
            </div>
            <span className="text-gray-300 dark:text-gray-600">|</span>
            <div className="flex gap-1 shrink-0">
              {categories.map(cat => (
                <span key={cat} className="font-mono text-gray-600 dark:text-gray-300 bg-gray-100 dark:bg-gray-700 px-1.5 rounded text-[10px] font-bold">
                  {formatCategory(cat)}
                </span>
              ))}
            </div>
          </div>

          <div className="flex items-center gap-2">
             <div className="flex flex-wrap gap-1">
                {tags.map(tag => (
                <Badge key={tag} className="bg-blue-50 text-blue-700 dark:bg-blue-900/30 dark:text-blue-300 border-blue-100 dark:border-blue-800">
                    {tag}
                </Badge>
                ))}
             </div>
             {/* Star Button */}
             <button
                onClick={toggleStar}
                className={`p-1 rounded-full transition-colors ${isStarred ? 'text-yellow-400 bg-yellow-400/10' : 'text-gray-300 hover:text-yellow-400 hover:bg-gray-100 dark:hover:bg-gray-700'}`}
                title={isStarred ? "Remove from favorites" : "Save for later"}
             >
                <Star className={`w-4 h-4 ${isStarred ? 'fill-yellow-400' : ''}`} />
             </button>
          </div>
        </div>

        {/* Title & Actions */}
        <div className="flex flex-col md:flex-row md:items-start md:justify-between gap-4">
            <h3 className="text-lg font-bold text-gray-900 dark:text-gray-100 leading-snug group-hover:text-blue-600 dark:group-hover:text-blue-400 transition-colors flex-1 flex items-start gap-2">
              <a href={link} target="_blank" rel="noopener noreferrer">
                {title}
              </a>
              <button onClick={handleCopyShare} className="opacity-0 group-hover:opacity-100 transition-opacity p-1 text-gray-400 hover:text-blue-500" title="Copy Title & Link">
                 {copied === 'share' ? <Check className="w-4 h-4 text-green-500" /> : <Copy className="w-4 h-4" />}
              </button>
            </h3>

            <div className="flex gap-2 shrink-0">
                {codeLink && (
                    <a href={codeLink} target="_blank" rel="noopener noreferrer">
                        <Button variant="secondary" className="h-8 text-xs border-green-200 dark:border-green-900 text-green-700 dark:text-green-400 bg-green-50 dark:bg-green-900/20" icon={Github}>
                            Code
                        </Button>
                    </a>
                )}

                <Button
                    variant={showPdf ? "danger" : "secondary"}
                    className="h-8 text-xs w-24"
                    onClick={() => setShowPdf(!showPdf)}
                    icon={showPdf ? X : FileText}
                >
                    {showPdf ? "Close" : "Read PDF"}
                </Button>

                <div className="flex items-center bg-white dark:bg-gray-800 rounded-md border border-gray-300 dark:border-gray-600 h-8">
                     <a href={link} target="_blank" rel="noopener noreferrer" className="h-full flex items-center px-2.5 hover:bg-gray-50 dark:hover:bg-gray-700 border-r border-gray-300 dark:border-gray-600 transition-colors" title="ArXiv Page">
                        <ExternalLink className="w-4 h-4 text-gray-500" />
                     </a>
                     <button onClick={handleCopyBibtex} className="h-full flex items-center px-2.5 hover:bg-gray-50 dark:hover:bg-gray-700 border-r border-gray-300 dark:border-gray-600 transition-colors" title="Copy BibTeX">
                        {copied === 'bibtex' ? <Check className="w-4 h-4 text-green-500" /> : <Quote className="w-4 h-4 text-gray-500" />}
                     </button>
                     <a href={geminiUrl} target="_blank" rel="noopener noreferrer" className="h-full flex items-center px-2.5 hover:bg-gray-50 dark:hover:bg-gray-700 text-blue-600 dark:text-blue-400 transition-colors" title="Ask Gemini">
                        <Sparkles className="w-4 h-4" />
                     </a>
                </div>
            </div>
        </div>

        {/* Authors */}
        <div className="text-sm text-gray-600 dark:text-gray-400 overflow-x-auto whitespace-nowrap pb-1 -mb-1 scrollbar-hide flex items-center">
          <User className="w-3.5 h-3.5 mr-1.5 shrink-0 opacity-50" />
          {authors.join(', ')}
        </div>

        {/* TLDR */}
        <div className="bg-gray-50 dark:bg-gray-700/30 rounded-lg px-4 py-3 border border-gray-100 dark:border-gray-700 text-sm leading-relaxed text-gray-700 dark:text-gray-300">
          <span className="inline-flex items-center font-bold text-gray-900 dark:text-gray-100 mr-2 select-none">
              <Sparkles className="w-3.5 h-3.5 text-yellow-500 mr-1" />
              TL;DR:
          </span>
          {tldrText}
        </div>

        {/* PDF Viewer */}
        {showPdf && (
            <div className="mt-2 animate-in fade-in slide-in-from-top-2 duration-300">
                <div className="relative w-full h-[600px] bg-gray-100 dark:bg-gray-900 rounded-lg border border-gray-200 dark:border-gray-700 overflow-hidden">
                    <iframe src={pdfUrl} className="w-full h-full" title="PDF Viewer" />
                    <div className="absolute top-0 right-0 p-2 opacity-0 hover:opacity-100 transition-opacity bg-black/50 backdrop-blur-sm rounded-bl-lg pointer-events-none">
                        <span className="text-white text-xs">If PDF fails to load, click external icon</span>
                    </div>
                </div>
            </div>
        )}
      </div>
    </div>
  );
};

// --- Main Application ---

const App = () => {
  const [papers, setPapers] = useState([]);
  const [loading, setLoading] = useState(true);

  // UI State
  const [darkMode, setDarkMode] = useState(() => {
    if (typeof window !== 'undefined') {
        return window.matchMedia('(prefers-color-scheme: dark)').matches;
    }
    return false;
  });
  const [language, setLanguage] = useState('en');

  // Filter State
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedTags, setSelectedTags] = useState([]);
  const [sortOrder, setSortOrder] = useState('newest');
  const [showFavoritesOnly, setShowFavoritesOnly] = useState(false); // New Filter

  // User Preferences State (LocalStorage)
  const [favorites, setFavorites] = useState(() => {
    if (typeof window !== 'undefined') {
        const saved = localStorage.getItem('daily_arxiv_favorites');
        return saved ? JSON.parse(saved) : [];
    }
    return [];
  });

  // Save favorites to localStorage
  useEffect(() => {
    localStorage.setItem('daily_arxiv_favorites', JSON.stringify(favorites));
  }, [favorites]);

  const toggleFavorite = (id) => {
    setFavorites(prev =>
        prev.includes(id) ? prev.filter(fid => fid !== id) : [...prev, id]
    );
  };

  // Pagination State
  const [currentPage, setCurrentPage] = useState(1);
  const [itemsPerPage, setItemsPerPage] = useState(30);

  // Load Data
  useEffect(() => {
    const loadData = async () => {
      try {
        const pathsToTry = ['./index.json', '../tools/index.json', 'https://raw.githubusercontent.com/zhixin612/awesome-papers-LMsys/main/tools/index.json'];

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

        const paperArray = Object.values(data).filter(p => p && typeof p === 'object' && p.relevant !== false);
        setPapers(paperArray);
      } catch (err) {
        console.error(err);
      } finally {
        setLoading(false);
      }
    };
    loadData();
  }, []);

  useEffect(() => {
    if (darkMode) {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
  }, [darkMode]);

  const allTags = useMemo(() => {
    const tags = new Set();
    papers.forEach(p => {
        if (Array.isArray(p.tags)) {
            p.tags.forEach(t => tags.add(t));
        }
    });
    return Array.from(tags).sort();
  }, [papers]);

  const filteredPapers = useMemo(() => {
    return papers
      .filter(paper => {
        // ID safe check for favorites
        const paperId = paper.id || "";
        if (showFavoritesOnly && !favorites.includes(paperId)) return false;

        const title = (paper.title || "").toLowerCase();
        const abstract = (paper.abstract || "").toLowerCase();
        const authors = Array.isArray(paper.authors) ? paper.authors : [];
        const paperTags = Array.isArray(paper.tags) ? paper.tags : [];

        const query = searchQuery.toLowerCase();
        const matchesSearch =
          title.includes(query) ||
          abstract.includes(query) ||
          authors.some(a => a.toLowerCase().includes(query));

        const matchesTags = selectedTags.length === 0 ||
          paperTags.some(t => selectedTags.includes(t));

        return matchesSearch && matchesTags;
      })
      .sort((a, b) => {
        const dateA = new Date(a.indexed_date || 0);
        const dateB = new Date(b.indexed_date || 0);
        return sortOrder === 'newest' ? dateB - dateA : dateA - dateB;
      });
  }, [papers, searchQuery, selectedTags, sortOrder, showFavoritesOnly, favorites]);

  const totalPages = Math.ceil(filteredPapers.length / itemsPerPage);

  useEffect(() => {
    setCurrentPage(1);
  }, [searchQuery, selectedTags, sortOrder, itemsPerPage, showFavoritesOnly]);

  const currentPapers = useMemo(() => {
    const indexOfLastItem = currentPage * itemsPerPage;
    const indexOfFirstItem = indexOfLastItem - itemsPerPage;
    return filteredPapers.slice(indexOfFirstItem, indexOfLastItem);
  }, [filteredPapers, currentPage, itemsPerPage]);

  const toggleTag = (tag) => {
    setSelectedTags(prev =>
      prev.includes(tag) ? prev.filter(t => t !== tag) : [...prev, tag]
    );
  };

  const mainRef = useRef(null);
  const handlePageChange = (page) => {
    setCurrentPage(page);
    if (mainRef.current) {
        mainRef.current.scrollIntoView({ behavior: 'smooth' });
    } else {
        window.scrollTo({ top: 0, behavior: 'smooth' });
    }
  };

  if (loading) return (
    <div className="min-h-screen flex items-center justify-center bg-gray-50 dark:bg-gray-900">
      <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
    </div>
  );

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900 pb-20 font-sans transition-colors duration-200">
      <header className="sticky top-0 z-30 bg-white/90 dark:bg-gray-900/90 backdrop-blur-md border-b border-gray-200 dark:border-gray-700 shadow-sm">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
          <div className="flex items-center gap-2 cursor-pointer" onClick={() => {window.scrollTo({top:0, behavior:'smooth'})}}>
            <div className="bg-blue-600 p-1.5 rounded-lg">
                <Calendar className="w-5 h-5 text-white" />
            </div>
            <h1 className="text-xl font-bold text-gray-900 dark:text-white tracking-tight hidden sm:block">
              Daily <span className="text-blue-600 dark:text-blue-400">System Opt</span>
            </h1>
          </div>

          <div className="flex items-center gap-3">
            <button
              onClick={() => setLanguage(l => l === 'en' ? 'zh' : 'en')}
              className="px-3 py-1.5 text-xs font-bold rounded-lg bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-700 transition-colors border border-gray-200 dark:border-gray-700"
            >
              {language === 'en' ? '中' : 'EN'}
            </button>
            <button
              onClick={() => setDarkMode(!darkMode)}
              className="p-2 rounded-lg text-gray-500 hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
            >
              {darkMode ? <Sun className="w-5 h-5" /> : <Moon className="w-5 h-5" />}
            </button>
          </div>
        </div>
      </header>

      <main ref={mainRef} className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-6">
        {/* Controls */}
        <div className="bg-white dark:bg-gray-800 rounded-xl p-4 shadow-sm border border-gray-200 dark:border-gray-700 space-y-4">
          <div className="flex flex-col md:flex-row gap-4">
            <div className="relative flex-1">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400" />
              <input
                type="text"
                placeholder={language === 'en' ? "Search title, abstract, authors..." : "搜索标题、摘要、作者..."}
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="w-full pl-10 pr-4 py-2.5 rounded-lg border border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-900 text-gray-900 dark:text-gray-100 focus:ring-2 focus:ring-blue-500 outline-none transition-all"
              />
            </div>
            <div className="flex gap-2 shrink-0">
               <button
                 onClick={() => setShowFavoritesOnly(!showFavoritesOnly)}
                 className={`px-4 py-2.5 rounded-lg border text-sm font-medium transition-all flex items-center gap-2 ${showFavoritesOnly ? 'bg-yellow-50 border-yellow-200 text-yellow-700 dark:bg-yellow-900/20 dark:border-yellow-800 dark:text-yellow-400' : 'bg-gray-50 border-gray-200 text-gray-700 dark:bg-gray-900 dark:border-gray-700 dark:text-gray-300'}`}
               >
                 <Star className={`w-4 h-4 ${showFavoritesOnly ? 'fill-current' : ''}`} />
                 <span className="hidden sm:inline">Favorites</span>
               </button>
               <select
                 value={sortOrder}
                 onChange={(e) => setSortOrder(e.target.value)}
                 className="px-4 py-2.5 rounded-lg border border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-900 text-gray-700 dark:text-gray-300 text-sm focus:ring-2 focus:ring-blue-500 outline-none cursor-pointer"
               >
                 <option value="newest">Latest First</option>
                 <option value="oldest">Oldest First</option>
               </select>
            </div>
          </div>

          {allTags.length > 0 && (
            <div className="flex flex-wrap gap-2 items-center pt-2 border-t border-gray-100 dark:border-gray-700">
              <span className="text-xs font-semibold text-gray-500 dark:text-gray-400 flex items-center uppercase tracking-wide">
                <Filter className="w-3 h-3 mr-1" />
                Tags:
              </span>
              {allTags.map(tag => (
                <button
                  key={tag}
                  onClick={() => toggleTag(tag)}
                  className={`px-2.5 py-1 rounded-md text-xs font-medium transition-all ${
                    selectedTags.includes(tag)
                      ? 'bg-blue-600 text-white shadow-sm'
                      : 'bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-600'
                  }`}
                >
                  {tag}
                </button>
              ))}
              {selectedTags.length > 0 && (
                <button onClick={() => setSelectedTags([])} className="text-xs text-red-500 hover:text-red-600 font-medium ml-2 underline decoration-dashed underline-offset-4">
                  Reset
                </button>
              )}
            </div>
          )}
        </div>

        {/* Top Pagination */}
        <div className="flex flex-col sm:flex-row items-center justify-between gap-4 px-1">
            <div className="text-sm text-gray-500 dark:text-gray-400 font-medium">
                Found <span className="text-gray-900 dark:text-white font-bold">{filteredPapers.length}</span> papers
            </div>
            <PaginationControls
                currentPage={currentPage}
                totalPages={totalPages}
                onPageChange={handlePageChange}
                itemsPerPage={itemsPerPage}
                onItemsPerPageChange={setItemsPerPage}
            />
        </div>

        {/* Cards */}
        <div className="flex flex-col gap-4">
          {currentPapers.map((paper, idx) => (
            <PaperCard
                key={paper.id || idx}
                paper={paper}
                language={language}
                isStarred={favorites.includes(paper.id)}
                toggleStar={() => toggleFavorite(paper.id)}
            />
          ))}
        </div>

        {/* Bottom Pagination */}
        <div className="flex justify-center pt-8 pb-4">
             <PaginationControls
                currentPage={currentPage}
                totalPages={totalPages}
                onPageChange={handlePageChange}
                itemsPerPage={itemsPerPage}
                onItemsPerPageChange={setItemsPerPage}
            />
        </div>

        {/* Empty State */}
        {filteredPapers.length === 0 && !loading && (
          <div className="text-center py-20 bg-white dark:bg-gray-800 rounded-xl border border-dashed border-gray-300 dark:border-gray-700">
            <div className="bg-gray-50 dark:bg-gray-900 rounded-full w-16 h-16 flex items-center justify-center mx-auto mb-4">
              {showFavoritesOnly ? <Star className="w-8 h-8 text-yellow-400" /> : <Search className="w-8 h-8 text-gray-400" />}
            </div>
            <h3 className="text-lg font-medium text-gray-900 dark:text-white">No papers found</h3>
            <p className="text-gray-500 dark:text-gray-400">
                {showFavoritesOnly ? "You haven't stared any papers yet." : "Try adjusting your search or filters."}
            </p>
          </div>
        )}
      </main>
    </div>
  );
};

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);