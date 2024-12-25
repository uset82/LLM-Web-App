import { Menu } from 'lucide-react';
import { Button } from '@/components/ui/button';

const Navbar = () => {
  return (
    <nav className="sticky top-0 z-50 border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
      <div className="container flex h-14 items-center">
        <div className="flex">
          <div className="flex items-center">
            <h1 className="scroll-m-20 text-2xl font-semibold tracking-tight">
              Building LLM Applications
            </h1>
          </div>
        </div>
        <div className="flex flex-1 items-center justify-end">
          <Button variant="ghost" size="icon">
            <Menu className="h-5 w-5" />
          </Button>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;
