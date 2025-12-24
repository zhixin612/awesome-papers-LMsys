import React, { useState, useEffect, useMemo, useRef, useCallback, useDeferredValue } from 'react';
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
  ChevronDown,
  ChevronUp,
  FileText,
  User,
  X,
  Star,
  Copy,
  Github,
  Check,
  GripHorizontal,
  Bot,
  Settings,
  MessageSquareText,
  Send,
  RefreshCw,
  Zap
} from 'lucide-react';

// --- Constants ---

const DEFAULT_PROMPT = "把你自己当成论文作者，用费曼学习法向我解释一下这篇论文，不要用类比";

const API_PROVIDERS = {
  siliconflow: {
    name: "SiliconFlow",
    url: "https://api.siliconflow.cn/v1/chat/completions",
    defaultModel: "moonshotai/Kimi-K2-Thinking"
  },
  openrouter: {
    name: "OpenRouter",
    url: "https://openrouter.ai/api/v1/chat/completions",
    defaultModel: "openai/gpt-5.2"
  }
};

const REDIRECTION_MODELS = {
  chatgpt: "ChatGPT",
  kimi: "Kimi"
};

// --- Components ---

const Badge = React.memo(({ children, className = "", onClick }) => (
  <span
    onClick={onClick}
    className={`inline-flex items-center px-2 py-0.5 rounded text-sm font-medium transition-colors border ${onClick ? 'cursor-pointer hover:opacity-80' : ''} ${className}`}
  >
    {children}
  </span>
));

const Button = React.memo(({ children, onClick, variant = 'primary', className = "", icon: Icon, disabled, title }) => {
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
});

// --- Simple Markdown Renderer ---
const SimpleMarkdown = React.memo(({ text }) => {
  if (!text) return null;
  const parts = text.split(/(```[\s\S]*?```)/g);
  return (
    <div className="space-y-3 text-sm leading-relaxed text-gray-800 dark:text-gray-200">
      {parts.map((part, index) => {
        if (part.startsWith('```')) {
          const content = part.replace(/^```\w*\n?|```$/g, '');
          return (
            <div key={index} className="bg-gray-100 dark:bg-gray-800 p-3 rounded-md overflow-x-auto border border-gray-200 dark:border-gray-700">
              <pre className="text-xs font-mono text-gray-800 dark:text-gray-200">{content}</pre>
            </div>
          );
        }
        return part.split(/\n\n+/).map((paragraph, pIndex) => {
          if (!paragraph.trim()) return null;
          const elements = [];
          let lastIndex = 0;
          const regex = /(\*\*.*?\*\*|`.*?`)/g;
          let match;
          while ((match = regex.exec(paragraph)) !== null) {
            if (match.index > lastIndex) elements.push(paragraph.substring(lastIndex, match.index));
            const matchedStr = match[0];
            if (matchedStr.startsWith('**')) elements.push(<strong key={match.index}>{matchedStr.slice(2, -2)}</strong>);
            else if (matchedStr.startsWith('`')) elements.push(<code key={match.index} className="bg-gray-100 dark:bg-gray-800 px-1 py-0.5 rounded font-mono text-xs text-red-500 dark:text-red-400">{matchedStr.slice(1, -1)}</code>);
            lastIndex = regex.lastIndex;
          }
          if (lastIndex < paragraph.length) elements.push(paragraph.substring(lastIndex));
          return <p key={`${index}-${pIndex}`}>{elements}</p>;
        });
      })}
    </div>
  );
});

// --- Settings Modal ---

const AISettingsModal = ({ isOpen, onClose, settings, onSave }) => {
  const [formData, setFormData] = useState(settings);

  useEffect(() => {
    if (isOpen) setFormData(settings);
  }, [isOpen, settings]);

  if (!isOpen) return null;

  const handleChange = (field, value) => {
    setFormData(prev => ({ ...prev, [field]: value }));
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm p-4 animate-in fade-in duration-200">
      <div className="bg-white dark:bg-gray-900 rounded-xl shadow-2xl w-full max-w-md overflow-hidden border border-gray-200 dark:border-gray-700 flex flex-col max-h-[90vh]">
        <div className="px-6 py-4 border-b border-gray-200 dark:border-gray-700 flex justify-between items-center bg-gray-50 dark:bg-gray-800/50">
          <h3 className="font-bold text-lg text-gray-900 dark:text-white flex items-center">
            <Settings className="w-5 h-5 mr-2 text-blue-600" />
            AI Configuration
          </h3>
          <button onClick={onClose} className="text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200">
            <X className="w-5 h-5" />
          </button>
        </div>

        <div className="p-6 overflow-y-auto space-y-8">
          {/* Explain Settings */}
          <div className="space-y-4">
            <div className="flex items-center gap-2 pb-2 border-b border-gray-100 dark:border-gray-800">
                <MessageSquareText className="w-4 h-4 text-blue-500" />
                <h4 className="text-xs font-bold uppercase tracking-wider text-gray-500 dark:text-gray-400">AI Settings for Explain</h4>
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">API Provider</label>
              <select
                value={formData.provider}
                onChange={(e) => {
                  const newProvider = e.target.value;
                  handleChange('provider', newProvider);
                  handleChange('model', API_PROVIDERS[newProvider].defaultModel);
                }}
                className="w-full rounded-lg border border-gray-300 dark:border-gray-700 bg-white dark:bg-gray-800 px-3 py-2 text-sm focus:ring-2 focus:ring-blue-500 outline-none transition-all dark:text-white"
              >
                {Object.entries(API_PROVIDERS).map(([key, val]) => (
                  <option key={key} value={key}>{val.name}</option>
                ))}
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">API Key</label>
              <input
                type="password"
                value={formData.apiKey}
                onChange={(e) => handleChange('apiKey', e.target.value)}
                placeholder="sk-..."
                className="w-full rounded-lg border border-gray-300 dark:border-gray-700 bg-white dark:bg-gray-800 px-3 py-2 text-sm focus:ring-2 focus:ring-blue-500 outline-none transition-all dark:text-white"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Model Name</label>
              <input
                type="text"
                value={formData.model}
                onChange={(e) => handleChange('model', e.target.value)}
                className="w-full rounded-lg border border-gray-300 dark:border-gray-700 bg-white dark:bg-gray-800 px-3 py-2 text-sm focus:ring-2 focus:ring-blue-500 outline-none transition-all dark:text-white"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Custom System Prompt</label>
              <textarea
                value={formData.customPrompt}
                onChange={(e) => handleChange('customPrompt', e.target.value)}
                rows={3}
                placeholder={DEFAULT_PROMPT}
                className="w-full rounded-lg border border-gray-300 dark:border-gray-700 bg-white dark:bg-gray-800 px-3 py-2 text-sm focus:ring-2 focus:ring-blue-500 outline-none transition-all dark:text-white resize-none"
              />
            </div>
          </div>

          {/* Redirection Settings */}
          <div className="space-y-4">
            <div className="flex items-center gap-2 pb-2 border-b border-gray-100 dark:border-gray-800">
                <ExternalLink className="w-4 h-4 text-purple-500" />
                <h4 className="text-xs font-bold uppercase tracking-wider text-gray-500 dark:text-gray-400">AI Settings for Redirection</h4>
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Target Model</label>
              <select
                value={formData.redirectionModel}
                onChange={(e) => handleChange('redirectionModel', e.target.value)}
                className="w-full rounded-lg border border-gray-300 dark:border-gray-700 bg-white dark:bg-gray-800 px-3 py-2 text-sm focus:ring-2 focus:ring-blue-500 outline-none transition-all dark:text-white"
              >
                {Object.entries(REDIRECTION_MODELS).map(([key, name]) => (
                  <option key={key} value={key}>{name}</option>
                ))}
              </select>
              <p className="text-xs text-gray-500 mt-1">Controls where the robot icon button redirects.</p>
            </div>
          </div>
        </div>

        <div className="px-6 py-4 border-t border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-800/50 flex justify-end gap-2">
          <Button variant="ghost" onClick={onClose}>Cancel</Button>
          <Button onClick={() => { onSave(formData); onClose(); }}>Save Settings</Button>
        </div>
      </div>
    </div>
  );
};

// --- Explain Panel Component ---

const ExplainPanel = ({ paper, settings, className }) => {
  const [response, setResponse] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const [error, setError] = useState(null);
  const abortControllerRef = useRef(null);
  const hasAutoStarted = useRef(false);

  const handleExplain = useCallback(async () => {
    if (!settings.apiKey) {
      setError("Please configure your API Key in Settings first.");
      return;
    }

    setIsStreaming(true);
    setResponse("");
    setError(null);

    if (abortControllerRef.current) {
        abortControllerRef.current.abort();
    }
    abortControllerRef.current = new AbortController();

    const providerConfig = API_PROVIDERS[settings.provider];
    const promptText = (settings.customPrompt || DEFAULT_PROMPT) + `\n\nPaper Title: ${paper.title}\nLink: ${paper.link}`;

    try {
      const res = await fetch(providerConfig.url, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Authorization": `Bearer ${settings.apiKey}`,
          ...(settings.provider === 'openrouter' && {
            "HTTP-Referer": window.location.href,
            "X-Title": "Daily MLsys"
          })
        },
        body: JSON.stringify({
          model: settings.model,
          messages: [{ role: "user", content: promptText }],
          stream: true
        }),
        signal: abortControllerRef.current.signal
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.error?.message || `API Error: ${res.status}`);
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder("utf-8");

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value, { stream: true });
        const lines = chunk.split("\n");

        for (const line of lines) {
          if (line.startsWith("data: ") && line !== "data: [DONE]") {
            try {
              const data = JSON.parse(line.slice(6));
              const content = data.choices?.[0]?.delta?.content || "";
              setResponse(prev => prev + content);
            } catch (e) {
              // Ignore partial chunks
            }
          }
        }
      }
    } catch (err) {
      if (err.name !== 'AbortError') {
        setError(err.message);
      }
    } finally {
      setIsStreaming(false);
    }
  }, [paper, settings]);

  const handleStop = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      setIsStreaming(false);
    }
  };

  useEffect(() => {
    if (!hasAutoStarted.current) {
        hasAutoStarted.current = true;
        handleExplain();
    }
    // Cleanup implies aborting, BUT we want caching.
    // Since we are using "hidden" display style in parent instead of unmounting,
    // this component will stay mounted, so we don't need to abort on "view switch".
    // We only abort on unmount (e.g. pagination change).
    return () => {
        if (abortControllerRef.current) {
            abortControllerRef.current.abort();
        }
    };
  }, [handleExplain]);

  return (
    <div className={`mt-2 animate-in fade-in slide-in-from-top-2 duration-300 ${className}`}>
      <div className="relative w-full bg-gray-50 dark:bg-gray-900 rounded-lg border border-gray-200 dark:border-gray-700 overflow-hidden flex flex-col p-4 min-h-[200px]">
        {/* Output Area (Markdown Supported) */}
        <div className="flex-1 font-sans">
          {response ? (
            <SimpleMarkdown text={response} />
          ) : (!isStreaming && !error && (
            <div className="flex flex-col items-center justify-center h-40 text-gray-400">
              <Bot className="w-8 h-8 mb-2 opacity-50" />
              <p>Waiting to start...</p>
            </div>
          ))}
          {error && (
            <div className="p-3 bg-red-50 dark:bg-red-900/20 text-red-600 dark:text-red-400 rounded-md text-sm border border-red-200 dark:border-red-800">
              Error: {error}
            </div>
          )}
        </div>

        {/* Controls */}
        <div className="mt-4 pt-3 border-t border-gray-200 dark:border-gray-700 flex justify-between items-center">
            <div className="text-xs text-gray-400 flex items-center">
                <Zap className="w-3 h-3 mr-1" />
                {settings.model}
            </div>
            <div className="flex gap-2">
                {isStreaming ? (
                    <Button onClick={handleStop} variant="danger" icon={X}>Stop</Button>
                ) : (
                    <Button onClick={handleExplain} variant="primary" icon={response ? RefreshCw : Send}>
                        {response ? "Regenerate" : "Generate"}
                    </Button>
                )}
            </div>
        </div>
      </div>
    </div>
  );
};

// --- Pagination (Unchanged) ---
const PaginationControls = React.memo(({ currentPage, totalPages, onPageChange, itemsPerPage, onItemsPerPageChange }) => {
  const [inputPage, setInputPage] = useState(currentPage);
  useEffect(() => { setInputPage(currentPage); }, [currentPage]);
  const handleKeyDown = (e) => {
    if (e.key === 'Enter') {
      const page = parseInt(inputPage);
      if (!isNaN(page) && page >= 1 && page <= totalPages) onPageChange(page);
      else setInputPage(currentPage);
    }
  };
  if (totalPages <= 1 && itemsPerPage >= 100) return null;
  return (
    <div className="flex flex-wrap items-center gap-4 text-sm">
      <div className="flex items-center gap-2">
        <span className="text-gray-500 dark:text-gray-400 hidden sm:inline">Show:</span>
        <select value={itemsPerPage} onChange={(e) => onItemsPerPageChange(Number(e.target.value))} className="bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-700 text-gray-700 dark:text-gray-200 text-xs rounded-md py-1 px-2 focus:ring-1 focus:ring-blue-500 outline-none cursor-pointer">
          <option value={10}>10</option>
          <option value={30}>30</option>
          <option value={50}>50</option>
          <option value={100}>100</option>
        </select>
      </div>
      <div className="flex items-center bg-white dark:bg-gray-800 rounded-md border border-gray-300 dark:border-gray-700 shadow-sm">
        <button onClick={() => onPageChange(currentPage - 1)} disabled={currentPage === 1} className="p-1.5 hover:bg-gray-50 dark:hover:bg-gray-700 rounded-l-md disabled:opacity-30 transition-colors">
          <ChevronLeft className="w-4 h-4 text-gray-600 dark:text-gray-300" />
        </button>
        <div className="flex items-center border-x border-gray-300 dark:border-gray-700 px-1">
            <input type="number" min="1" max={totalPages} value={inputPage} onChange={(e) => setInputPage(e.target.value)} onKeyDown={handleKeyDown} className="w-10 text-center py-1 text-gray-700 dark:text-gray-200 bg-transparent outline-none appearance-none font-medium"/>
            <span className="text-gray-400 dark:text-gray-500 px-1">/ {totalPages || 1}</span>
        </div>
        <button onClick={() => onPageChange(currentPage + 1)} disabled={currentPage === totalPages || totalPages === 0} className="p-1.5 hover:bg-gray-50 dark:hover:bg-gray-700 rounded-r-md disabled:opacity-30 transition-colors">
          <ChevronRight className="w-4 h-4 text-gray-600 dark:text-gray-300" />
        </button>
      </div>
    </div>
  );
});

// --- Helper Functions ---
const extractCodeLink = (abstract) => {
    const githubRegex = /https?:\/\/(www\.)?github\.com\/[a-zA-Z0-9-]+\/[a-zA-Z0-9-._]+/gi;
    const match = abstract.match(githubRegex);
    return match ? match[0] : null;
};

// --- Paper Card Component ---

const PaperCard = React.memo(({ paper, isStarred, toggleStar, aiSettings }) => {
  const [activeView, setActiveView] = useState('none');
  const [hasExplainStarted, setHasExplainStarted] = useState(false); // Cache: Track if explain has started
  const [copied, setCopied] = useState(null);

  // Resizable PDF
  const [pdfHeight, setPdfHeight] = useState(600);
  const isResizing = useRef(false);

  // Data
  const title = paper.title || "Untitled Paper";
  const link = paper.link || "#";
  const authors = Array.isArray(paper.authors) && paper.authors.length > 0 ? paper.authors : ["Unknown Author"];
  const categories = Array.isArray(paper.categories) && paper.categories.length > 0 ? paper.categories : ["Uncategorized"];
  const tags = Array.isArray(paper.tags) ? paper.tags : [];
  const submitDate = paper.submit_date || "Unknown Date";
  const tldrText = paper.tldr || null;

  // Prompt Construction for External Links
  const prompt = `${aiSettings.customPrompt || DEFAULT_PROMPT}\n\nPaper Title: ${title}\nLink: ${link}`;

  const handleAskAI = useCallback((e) => {
    e.preventDefault();
    const encodedPrompt = encodeURIComponent(prompt);
    let url = "";

    switch (aiSettings.redirectionModel) {
        case 'kimi':
            url = `http://kimi.com/_prefill_chat?prefill_prompt=${encodedPrompt}&send_immediately=true&force_search=false&enable_reasoning=false`;
            break;
        case 'chatgpt':
            url = `https://chatgpt.com/?q=${encodedPrompt}`;
            break;
        default:
            url = `https://chatgpt.com/?q=${encodedPrompt}`;
    }
    window.open(url, '_blank');
  }, [aiSettings.redirectionModel, prompt]);

  const pdfUrl = link.replace(/^http:/, 'https:').replace('/abs/', '/pdf/') + ".pdf";
  const formatCategory = (cat) => cat ? cat.replace(/^cs\./, '') : 'N/A';
  const codeLink = extractCodeLink(paper.abstract || "");

  const handleCopyShare = useCallback(() => {
    const text = `${title}\n${link}`;
    navigator.clipboard.writeText(text);
    setCopied('share');
    setTimeout(() => setCopied(null), 2000);
  }, [title, link]);

  // Resizing logic
  const handleMouseDown = useCallback((e) => {
    isResizing.current = true;
    e.preventDefault();
  }, []);

  useEffect(() => {
    const handleMouseMove = (e) => {
      if (!isResizing.current) return;
      setPdfHeight(prev => Math.max(200, Math.min(1200, prev + e.movementY)));
    };
    const handleMouseUp = () => { isResizing.current = false; };
    if (activeView === 'pdf') {
        window.addEventListener('mousemove', handleMouseMove);
        window.addEventListener('mouseup', handleMouseUp);
    }
    return () => {
        window.removeEventListener('mousemove', handleMouseMove);
        window.removeEventListener('mouseup', handleMouseUp);
    };
  }, [activeView]);

  const toggleView = (view) => {
    // If switching to explain, mark as started to keep it mounted
    if (view === 'explain') {
        setHasExplainStarted(true);
    }
    setActiveView(prev => prev === view ? 'none' : view);
  };

  return (
    <div className={`group relative flex flex-col bg-white dark:bg-gray-800 rounded-xl shadow-sm border transition-all duration-200 overflow-hidden ${activeView !== 'none' ? 'ring-2 ring-blue-500/20 border-blue-500/30' : 'hover:shadow-md border-gray-200 dark:border-gray-700'}`}>
      <div className="p-5 flex flex-col gap-4">

        {/* Top Row */}
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-y-3 gap-x-2 border-b border-gray-100 dark:border-gray-700/50 pb-3">
          <div className="flex items-center flex-wrap gap-2 text-sm text-gray-600 dark:text-gray-400">
            <div className="flex items-center whitespace-nowrap font-bold text-gray-800 dark:text-gray-200">
              <Calendar className="w-4 h-4 mr-1.5 text-blue-600" />
              <span>{submitDate}</span>
            </div>
            <span className="text-gray-300 dark:text-gray-600">|</span>
            <div className="flex gap-1.5">
              {categories.map(cat => (
                <span key={cat} className="font-mono bg-gray-100 dark:bg-gray-700 px-2 py-0.5 rounded text-xs font-bold text-gray-700 dark:text-gray-300">{formatCategory(cat)}</span>
              ))}
            </div>
            <span className="text-gray-300 dark:text-gray-600">|</span>
            <div className="flex flex-wrap gap-1.5">
                {tags.map(tag => (
                <Badge key={tag} className="bg-blue-50 text-blue-700 dark:bg-blue-900/30 dark:text-blue-300 border-blue-100 dark:border-blue-800 text-xs">{tag}</Badge>
                ))}
            </div>
          </div>

          <div className="flex items-center justify-end gap-2 shrink-0">
             <button onClick={handleCopyShare} className="p-1.5 text-gray-400 hover:text-blue-600 dark:hover:text-blue-400 transition-colors rounded-md hover:bg-gray-100 dark:hover:bg-gray-700" title="Copy Title & Link">
                {copied === 'share' ? <Check className="w-4 h-4 text-green-500" /> : <Copy className="w-4 h-4" />}
             </button>

             {codeLink && (
                <a href={codeLink} target="_blank" rel="noopener noreferrer" title="View Code">
                    <button className="p-1.5 text-gray-400 hover:text-green-600 dark:hover:text-green-400 transition-colors rounded-md hover:bg-green-50 dark:hover:bg-green-900/30">
                        <Github className="w-4 h-4" />
                    </button>
                </a>
             )}

             {/* Unified PDF/Explain Group */}
             <div className="flex items-center bg-white dark:bg-gray-700/50 rounded-md border border-gray-200 dark:border-gray-600 h-8 overflow-hidden">
                <Button
                    variant={activeView === 'pdf' ? "danger" : "ghost"}
                    className={`h-full text-xs px-2.5 rounded-none border-r border-gray-200 dark:border-gray-600 ${activeView === 'pdf' ? '' : 'hover:bg-gray-50 dark:hover:bg-gray-600'}`}
                    onClick={() => toggleView('pdf')}
                    icon={activeView === 'pdf' ? X : FileText}
                >
                    {activeView === 'pdf' ? "Close" : "Read PDF"}
                </Button>
                <Button
                    variant={activeView === 'explain' ? "danger" : "ghost"}
                    className={`h-full text-xs px-2.5 rounded-none ${activeView === 'explain' ? '' : 'hover:bg-gray-50 dark:hover:bg-gray-600'}`}
                    onClick={() => toggleView('explain')}
                    icon={activeView === 'explain' ? X : MessageSquareText}
                >
                    {activeView === 'explain' ? "Close" : "Explain"}
                </Button>
             </div>

             <div className="flex items-center bg-gray-50 dark:bg-gray-700/50 rounded-md border border-gray-200 dark:border-gray-600 h-8 overflow-hidden">
                 <a href={link} target="_blank" rel="noopener noreferrer" className="h-full flex items-center px-2.5 hover:bg-white dark:hover:bg-gray-600 border-r border-gray-200 dark:border-gray-600 transition-colors text-gray-500 dark:text-gray-300" title="ArXiv Page">
                    <ExternalLink className="w-4 h-4" />
                 </a>
                 <button onClick={handleAskAI} className="h-full flex items-center px-2.5 hover:bg-white dark:hover:bg-gray-600 border-r border-gray-200 dark:border-gray-600 transition-colors text-purple-600 dark:text-purple-400 group/ai" title={`Ask ${aiSettings.redirectionModel === 'kimi' ? 'Kimi' : 'ChatGPT'}`}>
                    <Bot className="w-4 h-4 group-hover/ai:animate-bounce" />
                 </button>
                 <button onClick={toggleStar} className={`h-full flex items-center px-2.5 transition-colors ${isStarred ? 'bg-yellow-50 dark:bg-yellow-900/20 text-yellow-500 hover:bg-yellow-100 dark:hover:bg-yellow-900/40' : 'hover:bg-white dark:hover:bg-gray-600 text-gray-400 hover:text-yellow-500'}`} title={isStarred ? "Remove from favorites" : "Save for later"}>
                    <Star className={`w-4 h-4 ${isStarred ? 'fill-current' : ''}`} />
                 </button>
             </div>
          </div>
        </div>

        <h3 className="text-xl font-bold text-gray-900 dark:text-gray-100 leading-snug">
            <a href={link} target="_blank" rel="noopener noreferrer" className="hover:text-blue-600 dark:hover:text-blue-400 transition-colors">{title}</a>
        </h3>

        <div className="flex flex-wrap items-center gap-x-1.5 gap-y-1 text-sm text-gray-600 dark:text-gray-400 leading-relaxed">
          <User className="w-4 h-4 mr-1 shrink-0 opacity-50" />
          {authors.map((author, index) => (
            <React.Fragment key={index}>
                <a href={`https://arxiv.org/search/?searchtype=author&query=${encodeURIComponent(author)}`} target="_blank" rel="noopener noreferrer" className="hover:text-blue-600 dark:hover:text-blue-400 hover:underline decoration-blue-300 underline-offset-2 transition-colors">{author}</a>
                {index < authors.length - 1 && <span className="text-gray-400">,</span>}
            </React.Fragment>
          ))}
        </div>

        <div className="bg-gray-50 dark:bg-gray-700/30 rounded-lg px-4 py-3 border border-gray-100 dark:border-gray-700 text-sm leading-relaxed text-gray-700 dark:text-gray-300">
          <span className="inline-flex items-center font-bold text-gray-900 dark:text-gray-100 mr-2 select-none">
              <Sparkles className="w-4 h-4 text-yellow-500 mr-1" />
              TL;DR:
          </span>
          {tldrText ? tldrText : <span className="italic text-gray-400">No TL;DR available for this paper.</span>}
        </div>

        {/* Dynamic Content Area */}
        {/* PDF View (Conditional Render to save memory on close) */}
        {activeView === 'pdf' && (
            <div className="mt-2 animate-in fade-in slide-in-from-top-2 duration-300">
                <div className="relative w-full bg-gray-100 dark:bg-gray-900 rounded-lg border border-gray-200 dark:border-gray-700 overflow-hidden flex flex-col" style={{ height: `${pdfHeight}px` }}>
                    <iframe src={pdfUrl} className="w-full flex-1" title="PDF Viewer" />
                    <div className="absolute top-0 right-0 p-2 opacity-0 hover:opacity-100 transition-opacity bg-black/50 backdrop-blur-sm rounded-bl-lg pointer-events-none z-10">
                        <span className="text-white text-xs">If PDF fails to load, click external icon</span>
                    </div>
                    <div onMouseDown={handleMouseDown} className="h-4 bg-gray-200 dark:bg-gray-700 hover:bg-blue-100 dark:hover:bg-blue-900/50 cursor-row-resize flex items-center justify-center transition-colors shrink-0 z-20" title="Drag to resize">
                        <GripHorizontal className="w-4 h-4 text-gray-400" />
                    </div>
                </div>
            </div>
        )}

        {/* Explain View (Hidden Render for Caching) */}
        {hasExplainStarted && (
            <div className={activeView === 'explain' ? 'block' : 'hidden'}>
                <ExplainPanel paper={paper} settings={aiSettings} />
            </div>
        )}
      </div>
    </div>
  );
});

// --- Main Application ---

const App = () => {
  const [papers, setPapers] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [debugInfo, setDebugInfo] = useState("");

  const [darkMode, setDarkMode] = useState(() => window.matchMedia('(prefers-color-scheme: dark)').matches);

  // AI API Settings
  const [aiSettings, setAiSettings] = useState(() => {
    const saved = localStorage.getItem('daily_arxiv_ai_settings');
    const defaultSettings = {
        provider: 'siliconflow',
        apiKey: '',
        model: 'moonshotai/Kimi-K2-Thinking',
        customPrompt: '',
        redirectionModel: 'chatgpt'
    };
    return saved ? { ...defaultSettings, ...JSON.parse(saved) } : defaultSettings;
  });
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);

  useEffect(() => {
    localStorage.setItem('daily_arxiv_ai_settings', JSON.stringify(aiSettings));
  }, [aiSettings]);

  useEffect(() => {
    if (darkMode) document.documentElement.classList.add('dark');
    else document.documentElement.classList.remove('dark');
  }, [darkMode]);

  // Filter State
  const [searchQuery, setSearchQuery] = useState('');
  const deferredSearchQuery = useDeferredValue(searchQuery);
  const [selectedTags, setSelectedTags] = useState([]);
  const [sortOrder, setSortOrder] = useState('newest');
  const [showFavoritesOnly, setShowFavoritesOnly] = useState(false);
  const [tagsExpanded, setTagsExpanded] = useState(false);

  // Favorites
  const [favorites, setFavorites] = useState(() => {
    const saved = localStorage.getItem('daily_arxiv_favorites');
    return saved ? JSON.parse(saved) : [];
  });

  useEffect(() => {
    localStorage.setItem('daily_arxiv_favorites', JSON.stringify(favorites));
  }, [favorites]);

  const toggleFavorite = useCallback((id) => {
    setFavorites(prev => prev.includes(id) ? prev.filter(fid => fid !== id) : [...prev, id]);
  }, []);

  // Pagination
  const [currentPage, setCurrentPage] = useState(1);
  const [itemsPerPage, setItemsPerPage] = useState(30);

  // Load Data
  useEffect(() => {
    const loadData = async () => {
      try {
        setLoading(true);
        setError(null);
        const pathsToTry = [
            '[https://raw.githubusercontent.com/zhixin612/awesome-papers-LMsys/main/tools/index.json](https://raw.githubusercontent.com/zhixin612/awesome-papers-LMsys/main/tools/index.json)',
            './index.json', '/index.json', '../tools/index.json',
        ];
        let data = null;
        let lastError = null;
        for (const path of pathsToTry) {
            try {
                const res = await fetch(path);
                if (res.ok) {
                    const text = await res.text();
                    data = JSON.parse(text);
                    break;
                }
            } catch (e) { lastError = e; }
        }
        if (!data) throw new Error(`Failed to load data. ${lastError ? lastError.message : ''}`);
        const rawArray = Array.isArray(data) ? data : Object.values(data);
        const validPapers = rawArray.filter(p => p && typeof p === 'object' && p.relevant !== false);
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

  // Sorted Tags
  const allTags = useMemo(() => {
    const counts = {};
    papers.forEach(p => {
        if (Array.isArray(p.tags)) {
            p.tags.forEach(t => {
                counts[t] = (counts[t] || 0) + 1;
            });
        }
    });
    // Sort by count desc, then alpha
    return Object.keys(counts).sort((a, b) => {
        const diff = counts[b] - counts[a];
        return diff !== 0 ? diff : a.localeCompare(b);
    });
  }, [papers]);

  const filteredPapers = useMemo(() => {
    return papers.filter(paper => {
        const paperId = paper.id || "";
        if (showFavoritesOnly && !favorites.includes(paperId)) return false;
        const title = (paper.title || "").toLowerCase();
        const abstract = (paper.abstract || "").toLowerCase();
        const authors = Array.isArray(paper.authors) ? paper.authors : [];
        const paperTags = Array.isArray(paper.tags) ? paper.tags : [];
        const query = deferredSearchQuery.toLowerCase();
        const matchesSearch = title.includes(query) || abstract.includes(query) || authors.some(a => a.toLowerCase().includes(query));
        const matchesTags = selectedTags.length === 0 || paperTags.some(t => selectedTags.includes(t));
        return matchesSearch && matchesTags;
      }).sort((a, b) => {
        const dateA = new Date(a.submit_date || 0);
        const dateB = new Date(b.submit_date || 0);
        return sortOrder === 'newest' ? dateB - dateA : dateA - dateB;
      });
  }, [papers, deferredSearchQuery, selectedTags, sortOrder, showFavoritesOnly, favorites]);

  const totalPages = Math.ceil(filteredPapers.length / itemsPerPage);
  const currentPapers = useMemo(() => {
    const indexOfLastItem = currentPage * itemsPerPage;
    const indexOfFirstItem = indexOfLastItem - itemsPerPage;
    return filteredPapers.slice(indexOfFirstItem, indexOfLastItem);
  }, [filteredPapers, currentPage, itemsPerPage]);

  useEffect(() => { setCurrentPage(1); }, [deferredSearchQuery, selectedTags, sortOrder, itemsPerPage, showFavoritesOnly]);

  const toggleTag = useCallback((tag) => {
    setSelectedTags(prev => prev.includes(tag) ? prev.filter(t => t !== tag) : [...prev, tag]);
  }, []);

  const mainRef = useRef(null);
  const handlePageChange = useCallback((page) => {
    setCurrentPage(page);
    if (mainRef.current) mainRef.current.scrollIntoView({ behavior: 'smooth' });
  }, []);

  if (error) return (
    <div className="min-h-screen flex flex-col items-center justify-center bg-gray-50 dark:bg-gray-900 p-4">
      <div className="bg-red-50 border border-red-200 text-red-700 px-6 py-4 rounded-lg max-w-lg w-full">
        <h3 className="font-bold text-lg mb-2">Error Loading Data</h3>
        <p className="font-mono text-sm break-all">{error}</p>
        <button onClick={() => window.location.reload()} className="mt-4 px-4 py-2 bg-red-600 text-white rounded hover:bg-red-700 transition-colors text-sm">Retry</button>
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
            <a href="[https://github.com/zhixin612](https://github.com/zhixin612)" target="_blank" rel="noopener noreferrer" className="flex flex-col sm:flex-row sm:items-baseline sm:gap-2 group cursor-pointer">
                <h1 className="text-xl font-bold text-gray-900 dark:text-white tracking-tight group-hover:text-blue-600 dark:group-hover:text-blue-400 transition-colors">Daily Arxiv: LLM Systems</h1>
                <span className="text-xs text-gray-500 group-hover:text-blue-600 dark:group-hover:text-blue-400 transition-colors">by zhixin</span>
            </a>
          </div>

          <div className="flex items-center gap-2">
            <button onClick={() => setIsSettingsOpen(true)} className="p-2 rounded-lg text-gray-500 hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors" title="Configure AI Settings">
                <Settings className="w-5 h-5" />
            </button>

            <button onClick={() => setDarkMode(!darkMode)} className="p-2 rounded-lg text-gray-500 hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors">
              {darkMode ? <Sun className="w-5 h-5" /> : <Moon className="w-5 h-5" />}
            </button>
          </div>
        </div>
      </header>

      <main ref={mainRef} className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-6">
        <div className="bg-white dark:bg-gray-800 rounded-xl p-4 shadow-sm border border-gray-200 dark:border-gray-700 space-y-4">
          <div className="flex flex-col md:flex-row gap-4">
            <div className="relative flex-1">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400" />
              <input type="text" placeholder="Search title, abstract, authors..." value={searchQuery} onChange={(e) => setSearchQuery(e.target.value)} className="w-full pl-10 pr-4 py-2.5 rounded-lg border border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-900 text-gray-900 dark:text-gray-100 focus:ring-2 focus:ring-blue-500 outline-none transition-all" />
            </div>
            <div className="flex gap-2 shrink-0">
               <button onClick={() => setShowFavoritesOnly(!showFavoritesOnly)} className={`px-4 py-2.5 rounded-lg border text-sm font-medium transition-all flex items-center gap-2 ${showFavoritesOnly ? 'bg-yellow-50 border-yellow-200 text-yellow-700 dark:bg-yellow-900/20 dark:border-yellow-800 dark:text-yellow-400' : 'bg-gray-50 border-gray-200 text-gray-700 dark:bg-gray-900 dark:border-gray-700 dark:text-gray-300'}`}>
                 <Star className={`w-4 h-4 ${showFavoritesOnly ? 'fill-current' : ''}`} />
                 <span className="hidden sm:inline">Favorites</span>
               </button>
               <select value={sortOrder} onChange={(e) => setSortOrder(e.target.value)} className="px-4 py-2.5 rounded-lg border border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-900 text-gray-700 dark:text-gray-300 text-sm focus:ring-2 focus:ring-blue-500 outline-none cursor-pointer">
                 <option value="newest">Latest First</option>
                 <option value="oldest">Oldest First</option>
               </select>
            </div>
          </div>
          {allTags.length > 0 && (
            <div className="pt-2 border-t border-gray-100 dark:border-gray-700 relative">
              <div className={`flex flex-wrap gap-2 items-center text-sm transition-all duration-300 ease-in-out overflow-hidden ${tagsExpanded ? 'max-h-[1000px]' : 'max-h-[32px]'}`}>
                  <span className="font-semibold text-gray-500 dark:text-gray-400 flex items-center uppercase tracking-wide shrink-0 mr-1">
                    <Filter className="w-4 h-4 mr-1.5" />Tags:
                  </span>
                  {allTags.map(tag => (
                    <button key={tag} onClick={() => toggleTag(tag)} className={`px-2 py-0.5 rounded-md text-xs font-medium transition-all border ${selectedTags.includes(tag) ? 'bg-blue-600 text-white border-blue-600' : 'bg-gray-50 dark:bg-gray-800 text-gray-600 dark:text-gray-300 border-gray-200 dark:border-gray-700 hover:border-gray-300 dark:hover:border-gray-600'}`}>{tag}</button>
                  ))}
                  <button onClick={() => setSelectedTags([])} className="text-xs text-red-500 hover:text-red-600 font-medium underline decoration-dashed underline-offset-4 ml-2">Reset</button>
              </div>

              {/* Expand/Collapse Button - Centered and Larger */}
              <div className="flex justify-center mt-1 border-t border-dashed border-gray-100 dark:border-gray-800 pt-1">
                  <button
                    onClick={() => setTagsExpanded(!tagsExpanded)}
                    className="w-full flex justify-center py-1 hover:bg-gray-50 dark:hover:bg-gray-800/50 text-gray-400 hover:text-gray-600 dark:hover:text-gray-300 transition-colors rounded-b-lg group"
                    title={tagsExpanded ? "Show Less" : "Show More"}
                  >
                      {tagsExpanded ? <ChevronUp className="w-5 h-5 group-hover:-translate-y-0.5 transition-transform" /> : <ChevronDown className="w-5 h-5 group-hover:translate-y-0.5 transition-transform" />}
                  </button>
              </div>
            </div>
          )}
        </div>

        <div className="flex flex-col sm:flex-row items-center justify-between gap-4 px-1">
            <div className="text-sm text-gray-500 dark:text-gray-400 font-medium">Found <span className="text-gray-900 dark:text-white font-bold">{filteredPapers.length}</span> papers</div>
            <PaginationControls currentPage={currentPage} totalPages={totalPages} onPageChange={handlePageChange} itemsPerPage={itemsPerPage} onItemsPerPageChange={setItemsPerPage} />
        </div>

        <div className="flex flex-col gap-4">
          {currentPapers.map((paper, idx) => (
            <PaperCard key={paper.id || idx} paper={paper} isStarred={favorites.includes(paper.id)} toggleStar={() => toggleFavorite(paper.id)} aiSettings={aiSettings} />
          ))}
        </div>

        <div className="flex justify-center pt-8 pb-4">
             <PaginationControls currentPage={currentPage} totalPages={totalPages} onPageChange={handlePageChange} itemsPerPage={itemsPerPage} onItemsPerPageChange={setItemsPerPage} />
        </div>

        {filteredPapers.length === 0 && !loading && (
          <div className="text-center py-20 bg-white dark:bg-gray-800 rounded-xl border border-dashed border-gray-300 dark:border-gray-700">
            <div className="bg-gray-50 dark:bg-gray-900 rounded-full w-16 h-16 flex items-center justify-center mx-auto mb-4">
              {showFavoritesOnly ? <Star className="w-8 h-8 text-yellow-400" /> : <Search className="w-8 h-8 text-gray-400" />}
            </div>
            <h3 className="text-lg font-medium text-gray-900 dark:text-white">No papers found</h3>
            <p className="text-gray-500 dark:text-gray-400 mb-2">{showFavoritesOnly ? "You haven't stared any papers yet." : "Try adjusting your search or filters."}</p>
            <div className="text-xs font-mono text-gray-400 bg-gray-50 dark:bg-gray-900 inline-block px-2 py-1 rounded">Debug: {debugInfo}</div>
          </div>
        )}
      </main>

      <AISettingsModal isOpen={isSettingsOpen} onClose={() => setIsSettingsOpen(false)} settings={aiSettings} onSave={setAiSettings} />
    </div>
  );
};

const root = createRoot(document.getElementById('root'));
root.render(<React.StrictMode><App /></React.StrictMode>);