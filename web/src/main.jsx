import React, { useState, useEffect, useMemo, useRef, useCallback } from 'react';
import { createRoot } from 'react-dom/client';
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
  ChevronDown, // 新增：用于下拉菜单
  FileText,
  User,
  X,
  Star,
  Copy,
  Github,
  Check,
  GripHorizontal,
  Bot
} from 'lucide-react';

// --- Components ---

const Badge = ({ children, className = "", onClick }) => (
  <span
    onClick={onClick}
    className={`inline-flex items-center px-2 py-0.5 rounded text-sm font-medium transition-colors border ${onClick ? 'cursor-pointer hover:opacity-80' : ''} ${className}`}
  >
    {children}
  </span>
);

const Button = ({ children, onClick, variant = 'primary', className = "", icon: Icon, disabled, title }) => {
  const baseStyle = "inline-flex items-center justify-center px-2 py-1 text-xs font-medium rounded-md focus:outline-none focus:ring-2 focus:ring-offset-1 transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed";
  const variants = {
    primary: "bg-blue-600 hover:bg-blue-700 text-white focus:ring-blue-500 border-transparent",
    secondary: "bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-200 border border-gray-300 dark:border-gray-600 hover:bg-gray-50 dark:hover:bg-gray-700",
    ghost: "text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-800 border-transparent",
    outline: "border border-gray-300 dark:border-gray-600 text-gray-700 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-800",
    danger: "bg-red-50 text-red-600 hover:bg-red-100 border border-red-200 dark:bg-red-900/20 dark:text-red-300 dark:border-red-800"
  };

  return (
    <button onClick={onClick} disabled={disabled} className={`${baseStyle} ${variants[variant]} ${className}`} title={title}>
      {Icon && <Icon className="w-3.5 h-3.5 mr-1.5" />}
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

const extractCodeLink = (abstract) => {
    const githubRegex = /https?:\/\/(www\.)?github\.com\/[a-zA-Z0-9-]+\/[a-zA-Z0-9-._]+/gi;
    const match = abstract.match(githubRegex);
    return match ? match[0] : null;
};

// --- Paper Card Component ---

const PaperCard = ({ paper, language, isStarred, toggleStar, aiModel }) => {
  const [showPdf, setShowPdf] = useState(false);
  const [copied, setCopied] = useState(null);

  // Resizable PDF State
  const [pdfHeight, setPdfHeight] = useState(600);
  const isResizing = useRef(false);

  // Safe Access
  const title = paper.title || "Untitled Paper";
  const link = paper.link || "#";
  const authors = Array.isArray(paper.authors) && paper.authors.length > 0 ? paper.authors : ["Unknown Author"];
  const categories = Array.isArray(paper.categories) && paper.categories.length > 0 ? paper.categories : ["Uncategorized"];
  const tags = Array.isArray(paper.tags) ? paper.tags : [];
  const submitDate = paper.submit_date || "Unknown Date";

  // Logic
  const tldrEn = paper.tldr || null;
  const tldrZh = paper.tldr_zh || tldrEn;
  const tldrText = language === 'zh' ? tldrZh : tldrEn;

  // AI Prompt & Link Generation
  const prompt = `Please analyze this paper for me based on its application in Large Model System Optimization: ${title}. Link: ${link}`;

  const handleAskAI = (e) => {
    e.preventDefault();
    if (aiModel !== 'gemini') {
        navigator.clipboard.writeText(prompt);
    }

    let url = "";
    switch (aiModel) {
        case 'gemini':
            url = `https://gemini.google.com/app?text=${encodeURIComponent(prompt)}`;
            break;
        case 'gpt':
            url = `https://chatgpt.com/`;
            break;
        case 'kimi':
            url = `https://kimi.moonshot.cn/`;
            break;
        default:
            url = `https://gemini.google.com/app?text=${encodeURIComponent(prompt)}`;
    }
    window.open(url, '_blank');
  };

  const pdfUrl = link.replace(/^http:/, 'https:').replace('/abs/', '/pdf/') + ".pdf";
  const formatCategory = (cat) => cat ? cat.replace(/^cs\./, '') : 'N/A';
  const codeLink = extractCodeLink(paper.abstract || "");

  const handleCopyShare = () => {
    const text = `${title}\n${link}`;
    navigator.clipboard.writeText(text);
    setCopied('share');
    setTimeout(() => setCopied(null), 2000);
  };

  // --- Resizable Logic ---
  const handleMouseDown = useCallback((e) => {
    isResizing.current = true;
    e.preventDefault();
  }, []);

  useEffect(() => {
    const handleMouseMove = (e) => {
      if (!isResizing.current) return;
      setPdfHeight(prev => Math.max(200, Math.min(1200, prev + e.movementY)));
    };

    const handleMouseUp = () => {
      isResizing.current = false;
    };

    if (showPdf) {
        window.addEventListener('mousemove', handleMouseMove);
        window.addEventListener('mouseup', handleMouseUp);
    }
    return () => {
        window.removeEventListener('mousemove', handleMouseMove);
        window.removeEventListener('mouseup', handleMouseUp);
    };
  }, [showPdf]);

  return (
    <div className={`group relative flex flex-col bg-white dark:bg-gray-800 rounded-xl shadow-sm border transition-all duration-200 overflow-hidden ${showPdf ? 'ring-2 ring-blue-500/20 border-blue-500/30' : 'hover:shadow-md border-gray-200 dark:border-gray-700'}`}>
      <div className="p-5 flex flex-col gap-4">

        {/* --- Top Row: Metadata (Left) + Actions (Right) --- */}
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-y-3 gap-x-2 border-b border-gray-100 dark:border-gray-700/50 pb-3">

          {/* Left: Date | Categories | Tags (Updated Fonts) */}
          <div className="flex items-center flex-wrap gap-2 text-sm text-gray-600 dark:text-gray-400">
            {/* Date (Bold & Larger) */}
            <div className="flex items-center whitespace-nowrap font-bold text-gray-800 dark:text-gray-200">
              <Calendar className="w-4 h-4 mr-1.5 text-blue-600" />
              <span>{submitDate}</span>
            </div>

            <span className="text-gray-300 dark:text-gray-600">|</span>

            {/* Categories */}
            <div className="flex gap-1.5">
              {categories.map(cat => (
                <span key={cat} className="font-mono bg-gray-100 dark:bg-gray-700 px-2 py-0.5 rounded text-xs font-bold text-gray-700 dark:text-gray-300">
                  {formatCategory(cat)}
                </span>
              ))}
            </div>

            <span className="text-gray-300 dark:text-gray-600">|</span>

            {/* Tags */}
            <div className="flex flex-wrap gap-1.5">
                {tags.map(tag => (
                <Badge key={tag} className="bg-blue-50 text-blue-700 dark:bg-blue-900/30 dark:text-blue-300 border-blue-100 dark:border-blue-800 text-xs">
                    {tag}
                </Badge>
                ))}
            </div>
          </div>

          {/* Right: Action Buttons Group */}
          <div className="flex items-center justify-end gap-2 shrink-0">
             {/* Copy Title & Link */}
             <button
                onClick={handleCopyShare}
                className="p-1.5 text-gray-400 hover:text-blue-600 dark:hover:text-blue-400 transition-colors rounded-md hover:bg-gray-100 dark:hover:bg-gray-700"
                title="Copy Title & Link"
             >
                {copied === 'share' ? <Check className="w-4 h-4 text-green-500" /> : <Copy className="w-4 h-4" />}
             </button>

             {/* Code Link */}
             {codeLink && (
                <a href={codeLink} target="_blank" rel="noopener noreferrer" title="View Code">
                    <button className="p-1.5 text-gray-400 hover:text-green-600 dark:hover:text-green-400 transition-colors rounded-md hover:bg-green-50 dark:hover:bg-green-900/30">
                        <Github className="w-4 h-4" />
                    </button>
                </a>
             )}

             {/* Toggle PDF */}
             <Button
                variant={showPdf ? "danger" : "secondary"}
                className="h-8 text-xs px-2.5"
                onClick={() => setShowPdf(!showPdf)}
                icon={showPdf ? X : FileText}
             >
                {showPdf ? "Close" : "Read PDF"}
             </Button>

             {/* Actions Group (Bordered) */}
             <div className="flex items-center bg-gray-50 dark:bg-gray-700/50 rounded-md border border-gray-200 dark:border-gray-600 h-8 overflow-hidden">
                 {/* External Link */}
                 <a href={link} target="_blank" rel="noopener noreferrer" className="h-full flex items-center px-2.5 hover:bg-white dark:hover:bg-gray-600 border-r border-gray-200 dark:border-gray-600 transition-colors text-gray-500 dark:text-gray-300" title="ArXiv Page">
                    <ExternalLink className="w-4 h-4" />
                 </a>

                 {/* Ask AI (Dynamic) */}
                 <button
                    onClick={handleAskAI}
                    className="h-full flex items-center px-2.5 hover:bg-white dark:hover:bg-gray-600 border-r border-gray-200 dark:border-gray-600 transition-colors text-purple-600 dark:text-purple-400 group/ai"
                    title={`Ask ${aiModel.toUpperCase()}`}
                 >
                    <Sparkles className="w-4 h-4 mr-1 group-hover/ai:animate-pulse" />
                    <span className="text-xs font-bold uppercase hidden xl:inline">{aiModel}</span>
                 </button>

                 {/* Star */}
                 <button
                    onClick={toggleStar}
                    className={`h-full flex items-center px-2.5 transition-colors ${isStarred ? 'bg-yellow-50 dark:bg-yellow-900/20 text-yellow-500 hover:bg-yellow-100 dark:hover:bg-yellow-900/40' : 'hover:bg-white dark:hover:bg-gray-600 text-gray-400 hover:text-yellow-500'}`}
                    title={isStarred ? "Remove from favorites" : "Save for later"}
                 >
                    <Star className={`w-4 h-4 ${isStarred ? 'fill-current' : ''}`} />
                 </button>
             </div>
          </div>
        </div>

        {/* Title */}
        <h3 className="text-xl font-bold text-gray-900 dark:text-gray-100 leading-snug">
            <a href={link} target="_blank" rel="noopener noreferrer" className="hover:text-blue-600 dark:hover:text-blue-400 transition-colors">
            {title}
            </a>
        </h3>

        {/* Authors */}
        <div className="flex flex-wrap items-center gap-x-1.5 gap-y-1 text-sm text-gray-600 dark:text-gray-400 leading-relaxed">
          <User className="w-4 h-4 mr-1 shrink-0 opacity-50" />
          {authors.map((author, index) => (
            <React.Fragment key={index}>
                <a
                    href={`https://arxiv.org/search/?searchtype=author&query=${encodeURIComponent(author)}`}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="hover:text-blue-600 dark:hover:text-blue-400 hover:underline decoration-blue-300 underline-offset-2 transition-colors"
                >
                    {author}
                </a>
                {index < authors.length - 1 && <span className="text-gray-400">,</span>}
            </React.Fragment>
          ))}
        </div>

        {/* TLDR */}
        <div className="bg-gray-50 dark:bg-gray-700/30 rounded-lg px-4 py-3 border border-gray-100 dark:border-gray-700 text-sm leading-relaxed text-gray-700 dark:text-gray-300">
          <span className="inline-flex items-center font-bold text-gray-900 dark:text-gray-100 mr-2 select-none">
              <Sparkles className="w-4 h-4 text-yellow-500 mr-1" />
              TL;DR:
          </span>
          {tldrText ? tldrText : <span className="italic text-gray-400">No TL;DR available for this paper.</span>}
        </div>

        {/* PDF Viewer */}
        {showPdf && (
            <div className="mt-2 animate-in fade-in slide-in-from-top-2 duration-300">
                <div
                    className="relative w-full bg-gray-100 dark:bg-gray-900 rounded-lg border border-gray-200 dark:border-gray-700 overflow-hidden flex flex-col"
                    style={{ height: `${pdfHeight}px` }}
                >
                    <iframe src={pdfUrl} className="w-full flex-1" title="PDF Viewer" />

                    {/* Fallback Overlay */}
                    <div className="absolute top-0 right-0 p-2 opacity-0 hover:opacity-100 transition-opacity bg-black/50 backdrop-blur-sm rounded-bl-lg pointer-events-none z-10">
                        <span className="text-white text-xs">If PDF fails to load, click external icon</span>
                    </div>

                    {/* Resizer Handle */}
                    <div
                        onMouseDown={handleMouseDown}
                        className="h-4 bg-gray-200 dark:bg-gray-700 hover:bg-blue-100 dark:hover:bg-blue-900/50 cursor-row-resize flex items-center justify-center transition-colors shrink-0 z-20"
                        title="Drag to resize"
                    >
                        <GripHorizontal className="w-4 h-4 text-gray-400" />
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
  const [error, setError] = useState(null);
  const [debugInfo, setDebugInfo] = useState("");

  // UI State
  const [darkMode, setDarkMode] = useState(() => {
    if (typeof window !== 'undefined') {
        return window.matchMedia('(prefers-color-scheme: dark)').matches;
    }
    return false;
  });
  const [language, setLanguage] = useState('en');
  const [aiModel, setAiModel] = useState('gemini'); // Default AI Model

  // Filter State
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedTags, setSelectedTags] = useState([]);
  const [sortOrder, setSortOrder] = useState('newest');
  const [showFavoritesOnly, setShowFavoritesOnly] = useState(false);

  // Favorites
  const [favorites, setFavorites] = useState(() => {
    if (typeof window !== 'undefined') {
        const saved = localStorage.getItem('daily_arxiv_favorites');
        return saved ? JSON.parse(saved) : [];
    }
    return [];
  });

  useEffect(() => {
    localStorage.setItem('daily_arxiv_favorites', JSON.stringify(favorites));
  }, [favorites]);

  const toggleFavorite = (id) => {
    setFavorites(prev =>
        prev.includes(id) ? prev.filter(fid => fid !== id) : [...prev, id]
    );
  };

  const [currentPage, setCurrentPage] = useState(1);
  const [itemsPerPage, setItemsPerPage] = useState(30);

  // Data Loading
  useEffect(() => {
    const loadData = async () => {
      try {
        setLoading(true);
        setError(null);

        const pathsToTry = [
            'https://raw.githubusercontent.com/zhixin612/awesome-papers-LMsys/main/tools/index.json',
            './index.json',
            '/index.json',
            '../tools/index.json',
        ];

        let data = null;
        let lastError = null;

        for (const path of pathsToTry) {
            try {
                const res = await fetch(path);
                if (res.ok) {
                    const text = await res.text();
                    try {
                        data = JSON.parse(text);
                        break;
                    } catch (parseErr) {
                        console.warn(`Failed to parse JSON from ${path}:`, parseErr);
                    }
                }
            } catch (e) {
                lastError = e;
            }
        }

        if (!data) {
            throw new Error(`Failed to load data. Last check: ${lastError ? lastError.message : 'File not found'}`);
        }

        const rawArray = Array.isArray(data) ? data : Object.values(data);
        const validPapers = rawArray.filter(p => p && typeof p === 'object');

        console.log(`Loaded ${validPapers.length} papers`);
        setDebugInfo(`Source loaded: ${validPapers.length} items`);
        setPapers(validPapers);

      } catch (err) {
        console.error(err);
        setError(err.message);
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
        const dateA = new Date(a.submit_date || 0);
        const dateB = new Date(b.submit_date || 0);
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

  // --- Error View ---
  if (error) return (
    <div className="min-h-screen flex flex-col items-center justify-center bg-gray-50 dark:bg-gray-900 p-4">
      <div className="bg-red-50 border border-red-200 text-red-700 px-6 py-4 rounded-lg max-w-lg w-full">
        <h3 className="font-bold text-lg mb-2">Error Loading Data</h3>
        <p className="font-mono text-sm break-all">{error}</p>
        <button onClick={() => window.location.reload()} className="mt-4 px-4 py-2 bg-red-600 text-white rounded hover:bg-red-700 transition-colors text-sm">
          Retry
        </button>
      </div>
    </div>
  );

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
            {/* Updated Title with Link */}
            <a
              href="https://github.com/zhixin612"
              target="_blank"
              rel="noopener noreferrer"
              className="flex flex-col sm:flex-row sm:items-baseline sm:gap-2 group cursor-pointer"
            >
                <h1 className="text-xl font-bold text-gray-900 dark:text-white tracking-tight group-hover:text-blue-600 dark:group-hover:text-blue-400 transition-colors">
                Daily Arxiv: LLM Systems
                </h1>
                <span className="text-xs text-gray-500 group-hover:text-blue-600 dark:group-hover:text-blue-400 transition-colors">
                    by zhixin
                </span>
            </a>
          </div>

          <div className="flex items-center gap-2">
            {/* AI Model Selector (Beautified) */}
            <div className="relative group">
                <div className="hidden sm:flex items-center bg-gray-100 dark:bg-gray-800 rounded-lg pl-2 pr-1 border border-gray-200 dark:border-gray-700 h-8 hover:border-gray-300 dark:hover:border-gray-600 transition-colors cursor-pointer">
                    <Bot className="w-3.5 h-3.5 text-gray-500 mr-1.5" />
                    <select
                        value={aiModel}
                        onChange={(e) => setAiModel(e.target.value)}
                        className="bg-transparent text-xs font-semibold text-gray-700 dark:text-gray-300 outline-none cursor-pointer appearance-none pr-6 z-10"
                    >
                        <option value="gemini">Gemini</option>
                        <option value="gpt">GPT</option>
                        <option value="kimi">Kimi</option>
                    </select>
                    {/* Custom Arrow Icon */}
                    <div className="absolute right-2 top-1/2 -translate-y-1/2 pointer-events-none text-gray-500">
                        <ChevronDown className="w-3 h-3" />
                    </div>
                </div>
            </div>

            <button
              onClick={() => setLanguage(l => l === 'en' ? 'zh' : 'en')}
              className="px-2.5 py-1.5 text-xs font-bold rounded-lg bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-700 transition-colors border border-gray-200 dark:border-gray-700"
            >
              {language === 'en' ? '中' : 'EN'}
            </button>
            <button
              onClick={() => setDarkMode(!darkMode)}
              className="p-1.5 rounded-lg text-gray-500 hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
            >
              {darkMode ? <Sun className="w-4 h-4" /> : <Moon className="w-4 h-4" />}
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
              <span className="text-sm font-semibold text-gray-500 dark:text-gray-400 flex items-center uppercase tracking-wide">
                <Filter className="w-4 h-4 mr-1.5" />
                Tags:
              </span>
              {allTags.map(tag => (
                <button
                  key={tag}
                  onClick={() => toggleTag(tag)}
                  className={`px-3 py-1 rounded-md text-sm font-medium transition-all ${
                    selectedTags.includes(tag)
                      ? 'bg-blue-600 text-white shadow-sm'
                      : 'bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-600'
                  }`}
                >
                  {tag}
                </button>
              ))}
              {selectedTags.length > 0 && (
                <button onClick={() => setSelectedTags([])} className="text-sm text-red-500 hover:text-red-600 font-medium ml-2 underline decoration-dashed underline-offset-4">
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
                aiModel={aiModel}
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
            <p className="text-gray-500 dark:text-gray-400 mb-2">
                {showFavoritesOnly ? "You haven't stared any papers yet." : "Try adjusting your search or filters."}
            </p>
            <div className="text-xs font-mono text-gray-400 bg-gray-50 dark:bg-gray-900 inline-block px-2 py-1 rounded">
                Debug: {debugInfo}
            </div>
          </div>
        )}
      </main>
    </div>
  );
};

const root = createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);