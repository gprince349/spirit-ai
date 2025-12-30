# Spirit AI - UI Client

A simple, minimalistic Next.js UI project with shadcn/ui components.

## Features

- ⚡ Next.js 14 with App Router
- 🎨 Tailwind CSS for styling
- 🧩 shadcn/ui component library
- 📱 Responsive design
- 🌙 Dark mode support
- 🚀 TypeScript support

## Getting Started

### Prerequisites

- Node.js 18+ 
- npm or yarn

### Installation

1. Install dependencies:
```bash
npm install
```

2. Run the development server:
```bash
npm run dev
```

3. Open [http://localhost:3000](http://localhost:3000) in your browser.

## Available Scripts

- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm run start` - Start production server
- `npm run lint` - Run ESLint

## Project Structure

```
src/
├── app/                 # Next.js App Router
│   ├── globals.css     # Global styles
│   ├── layout.tsx      # Root layout
│   └── page.tsx        # Home page
├── components/          # React components
│   └── ui/             # shadcn/ui components
└── lib/                 # Utility functions
```

## Adding More Components

To add more shadcn/ui components:

```bash
npx shadcn@latest add [component-name]
```

## Customization

The project uses a clean, minimalistic design with:
- Subtle gradients and shadows
- Glassmorphism effects
- Responsive card layouts
- Accessible color schemes
