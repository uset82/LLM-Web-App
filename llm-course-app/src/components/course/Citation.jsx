import { Card } from '@/components/ui/card';
import { cn } from '@/lib/utils';

const Citation = ({ children, className }) => {
  return (
    <Card className={cn(
      "my-6 border-l-4 border-l-primary bg-muted/50 p-4",
      className
    )}>
      <blockquote className="text-muted-foreground">
        {children}
      </blockquote>
    </Card>
  );
};

export default Citation;
