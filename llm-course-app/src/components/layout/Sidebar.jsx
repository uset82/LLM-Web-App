import { Link, useLocation } from 'react-router-dom';
import { Book, Beaker, FileText } from 'lucide-react';
import { cn } from '@/lib/utils';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Button } from '@/components/ui/button';

const Sidebar = () => {
  const weeks = [1, 2, 3, 4];
  const location = useLocation();

  const isActive = (path) => location.pathname === path;

  return (
    <div className="border-r bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
      <ScrollArea className="h-full w-64">
        <div className="space-y-4 py-4">
          <div className="px-3 py-2">
            <Button
              variant={isActive('/') ? 'secondary' : 'ghost'}
              className="w-full justify-start"
              asChild
            >
              <Link to="/">
                <Book className="mr-2 h-4 w-4" />
                Course Overview
              </Link>
            </Button>
          </div>
          {weeks.map((week) => (
            <div key={week} className="px-3 py-2">
              <h3 className="mb-2 px-4 text-sm font-semibold tracking-tight">
                Week {week}
              </h3>
              <div className="space-y-1">
                <Button
                  variant={isActive(`/week/${week}/content`) ? 'secondary' : 'ghost'}
                  className="w-full justify-start"
                  asChild
                >
                  <Link to={`/week/${week}/content`}>
                    <FileText className="mr-2 h-4 w-4" />
                    Content
                  </Link>
                </Button>
                <Button
                  variant={isActive(`/week/${week}/labs`) ? 'secondary' : 'ghost'}
                  className="w-full justify-start"
                  asChild
                >
                  <Link to={`/week/${week}/labs`}>
                    <Beaker className="mr-2 h-4 w-4" />
                    Labs
                  </Link>
                </Button>
              </div>
            </div>
          ))}
          <div className="px-3 py-2">
            <Button
              variant={isActive('/references') ? 'secondary' : 'ghost'}
              className="w-full justify-start"
              asChild
            >
              <Link to="/references">
                <FileText className="mr-2 h-4 w-4" />
                References
              </Link>
            </Button>
          </div>
        </div>
      </ScrollArea>
    </div>
  );
};

export default Sidebar;
