import { useEffect, useState } from 'react';

export const useMarkdownContent = (contentPath: string) => {
  const [content, setContent] = useState<string>('');
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(true);

  useEffect(() => {
    const loadContent = async () => {
      try {
        setLoading(true);
        // Handle both content/ and labs/ paths
        const normalizedPath = contentPath.startsWith('/') 
          ? contentPath.substring(1) 
          : contentPath;
        
        const response = await fetch(`/${normalizedPath}`, {
          credentials: 'same-origin',
          headers: {
            'Accept': 'text/markdown'
          }
        });
        
        if (!response.ok) {
          throw new Error(`Failed to load content: ${response.statusText}`);
        }
        
        let text = await response.text();
        
        // Update image paths to ensure they're relative to /content/images/
        text = text.replace(
          /!\[(.*?)\]\((\/content\/images\/.*?)\)/g,
          '![$1]($2)'
        ).replace(
          /!\[(.*?)\]\(((?!\/content\/images).*?\.(?:svg|png|jpg|jpeg|gif))\)/g,
          '![$1](/content/images/$2)'
        );
        
        setContent(text);
        setError(null);
      } catch (err) {
        console.error('Content loading error:', err);
        setError(err instanceof Error ? err.message : 'Failed to load content');
        setContent('');
      } finally {
        setLoading(false);
      }
    };

    loadContent();
  }, [contentPath]);

  return { content, error, loading };
};
