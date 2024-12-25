import { Card } from "@/components/ui/card";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { AlertCircle } from "lucide-react";
import MarkdownRenderer from '../components/course/MarkdownRenderer';
import { useMarkdownContent } from '../utils/contentLoader';

const Home = () => {
  const { content, error, loading } = useMarkdownContent('content/index.md');

  if (error) {
    return (
      <div className="container max-w-4xl mx-auto space-y-6">
        <h1 className="scroll-m-20 text-4xl font-bold tracking-tight">
          Building LLM Applications
        </h1>
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      </div>
    );
  }

  return (
    <div className="container max-w-4xl mx-auto space-y-6">
      <h1 className="scroll-m-20 text-4xl font-bold tracking-tight">
        Building LLM Applications
      </h1>
      <Card className="p-6">
        <div className="prose prose-lg max-w-none dark:prose-invert">
          {loading ? (
            <div className="h-[200px] animate-pulse bg-muted rounded-lg" />
          ) : (
            <MarkdownRenderer content={content} />
          )}
        </div>
      </Card>
    </div>
  );
};

export default Home;
