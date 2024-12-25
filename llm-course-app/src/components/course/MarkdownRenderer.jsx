import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';
import { Card } from '@/components/ui/card';
import Citation from './Citation';

const MarkdownRenderer = ({ content }) => {
  const components = {
    code: ({ node, inline, className, children, ...props }) => {
      const match = /language-(\w+)/.exec(className || '');
      const language = match ? match[1] : '';
      return !inline ? (
        <div className="relative group">
          <SyntaxHighlighter
            style={vscDarkPlus}
            language={language}
            PreTag="div"
            className="mb-4 mt-6 overflow-x-auto rounded-lg border bg-muted p-4 whitespace-pre"
            {...props}
          >
            {String(children)
              .replace(/\n$/, '')
              .replace(/&gt;/g, '>')
              .replace(/&lt;/g, '<')
              .replace(/&amp;/g, '&')
            }
          </SyntaxHighlighter>
          <button
            className="absolute right-4 top-4 opacity-0 group-hover:opacity-100 transition-opacity"
            onClick={() => {
              navigator.clipboard.writeText(String(children));
            }}
          >
            <svg
              xmlns="http://www.w3.org/2000/svg"
              width="24"
              height="24"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
              className="h-4 w-4"
            >
              <rect width="8" height="4" x="8" y="2" rx="1" ry="1" />
              <path d="M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2" />
            </svg>
          </button>
        </div>
      ) : (
        <code className="relative rounded bg-muted px-[0.3rem] py-[0.2rem] font-mono text-sm">
          {children}
        </code>
      );
    },
    blockquote: Citation,
    h1: ({ children }) => (
      <h1 className="scroll-m-20 text-4xl font-extrabold tracking-tight lg:text-5xl mb-8">
        {children}
      </h1>
    ),
    h2: ({ children }) => (
      <h2 className="scroll-m-20 border-b pb-2 text-3xl font-semibold tracking-tight first:mt-0 mt-10 mb-4">
        {children}
      </h2>
    ),
    h3: ({ children }) => (
      <h3 className="scroll-m-20 text-2xl font-semibold tracking-tight mt-8 mb-4">
        {children}
      </h3>
    ),
    h4: ({ children }) => (
      <h4 className="scroll-m-20 text-xl font-semibold tracking-tight mt-6 mb-4">
        {children}
      </h4>
    ),
    p: ({ children }) => (
      <p className="leading-7 [&:not(:first-child)]:mt-6">
        {children}
      </p>
    ),
    ul: ({ children }) => (
      <ul className="my-6 ml-6 list-disc [&>li]:mt-2">
        {children}
      </ul>
    ),
    ol: ({ children }) => (
      <ol className="my-6 ml-6 list-decimal [&>li]:mt-2">
        {children}
      </ol>
    ),
    img: ({ src, alt }) => (
      <img
        src={src}
        alt={alt}
        className="rounded-lg border my-6 max-w-full h-auto"
      />
    ),
    a: ({ href, children }) => (
      <a
        href={href}
        target="_blank"
        rel="noopener noreferrer"
        className="font-medium text-primary underline underline-offset-4 hover:text-primary/80"
      >
        {children}
      </a>
    ),
  };

  return (
    <Card className="p-6">
      <div className="markdown-content prose prose-slate max-w-none dark:prose-invert">
        <ReactMarkdown 
          remarkPlugins={[remarkGfm]} 
          components={{
            ...components,
            img: ({ src, alt }) => (
              <div className="flex flex-col items-center">
                <img
                  src={src}
                  alt={alt}
                  className="rounded-lg border my-6 max-w-full h-auto shadow-lg"
                  loading="lazy"
                />
                {alt && <em className="text-sm text-gray-500 mt-2">{alt}</em>}
              </div>
            ),
          }}
        >
          {content}
        </ReactMarkdown>
      </div>
    </Card>
  );
};

export default MarkdownRenderer;
