import { promises as fs } from 'fs';
import path from 'path';

export default async function handler(req, res) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  try {
    const { query } = req.body;
    const contentDir = path.join(process.cwd(), 'public/content/daily');
    
    // Read all markdown files
    const files = await fs.readdir(contentDir);
    const markdownFiles = files.filter(file => file.endsWith('.md'));
    
    let relevantContent = [];
    
    // Search through each file for relevant content
    for (const file of markdownFiles) {
      const content = await fs.readFile(path.join(contentDir, file), 'utf-8');
      
      // Simple keyword matching - in production, use proper vector similarity
      if (content.toLowerCase().includes(query.toLowerCase())) {
        // Extract the section containing the query
        const lines = content.split('\n');
        const queryIndex = lines.findIndex(line => 
          line.toLowerCase().includes(query.toLowerCase())
        );
        
        if (queryIndex !== -1) {
          // Get context around the matching line (5 lines before and after)
          const contextStart = Math.max(0, queryIndex - 5);
          const contextEnd = Math.min(lines.length, queryIndex + 5);
          const context = lines.slice(contextStart, contextEnd).join('\n');
          
          relevantContent.push({
            file,
            context
          });
        }
      }
    }
    
    // Combine and truncate relevant content to avoid token limits
    const combinedContent = relevantContent
      .slice(0, 3) // Take top 3 most relevant sections
      .map(({ file, context }) => `From ${file}:\n${context}`)
      .join('\n\n')
      .slice(0, 4000); // Approximate 1000 token limit
    
    res.status(200).json({ content: combinedContent });
  } catch (error) {
    console.error('Error searching content:', error);
    res.status(500).json({ error: 'Failed to search content' });
  }
}
