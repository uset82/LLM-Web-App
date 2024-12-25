import { useState } from 'react';
import { Check, Copy } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { cn } from '@/lib/utils';

const CodeBlock = ({ children, className }) => {
  const [copied, setCopied] = useState(false);
  const code = children?.props?.children || '';

  const copyToClipboard = async () => {
    await navigator.clipboard.writeText(code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="relative group">
      <pre className={cn(
        "mb-4 mt-6 overflow-x-auto rounded-lg border bg-muted p-4 font-mono text-sm",
        className
      )}>
        <Button
          variant="ghost"
          size="icon"
          className="absolute right-4 top-4 opacity-0 group-hover:opacity-100 transition-opacity"
          onClick={copyToClipboard}
        >
          {copied ? (
            <Check className="h-4 w-4 text-green-500" />
          ) : (
            <Copy className="h-4 w-4" />
          )}
        </Button>
        <code className="relative rounded px-[0.3rem] py-[0.2rem] font-mono text-sm">
          {code}
        </code>
      </pre>
    </div>
  );
};

export default CodeBlock;
