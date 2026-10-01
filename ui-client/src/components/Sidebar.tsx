
"use client";

import Link from "next/link";
import Image from "next/image";
import { usePathname } from "next/navigation";
import {
    Home,
    MessageCircle,
    BookOpen,
    Bookmark,
    Users,
    CircleUserRound,
    Settings,
    LogOut,
} from "lucide-react";

const navItems = [
    { label: "Home", href: "/", icon: Home },
    { label: "Conversations", href: "/Conversation", icon: MessageCircle },
    { label: "Journal", href: "/Journal", icon: BookOpen },
    { label: "Saved Wisdom", href: "/Dashboard", icon: Bookmark },
    { label: "Mentors", href: "/Chat", icon: Users },
    { label: "Profile", href: "/Profile", icon: CircleUserRound },
];

const Sidebar = () => {
    const pathname = usePathname();

    return (
        <aside className="flex h-screen w-64 shrink-0 flex-col justify-between border-r border-[var(--sidebar-border)] bg-[var(--sidebar)] px-4 py-6">
            <div>
                <div className="mb-8 flex items-center gap-2 px-2">
                    <Image
                        src="/spiritual_lotus.png"
                        alt="Spiritual AI"
                        width={28}
                        height={28}
                    />
                    <span className="font-serif text-lg font-semibold text-[var(--primary)]">
                        Spiritual AI
                    </span>
                </div>

                <nav className="flex flex-col gap-1">
                    {navItems.map(({ label, href, icon: Icon }) => {
                        const isActive =
                            href === "/" ? pathname === href : pathname?.startsWith(href);

                        return (
                            <Link
                                key={label}
                                href={href}
                                className={`flex items-center gap-3 rounded-xl px-3 py-2.5 text-sm transition-colors ${
                                    isActive
                                        ? "bg-[var(--sidebar-accent)] font-medium text-[var(--primary)]"
                                        : "text-[var(--sidebar-foreground)] hover:bg-[var(--sidebar-accent)]/60"
                                }`}
                            >
                                <Icon className="size-4.5" />
                                {label}
                            </Link>
                        );
                    })}
                </nav>
            </div>

            <div className="flex flex-col gap-1 border-t border-[var(--sidebar-border)] pt-4">
                <Link
                    href="/Settings"
                    className="flex items-center gap-3 rounded-xl px-3 py-2.5 text-sm text-[var(--sidebar-foreground)] transition-colors hover:bg-[var(--sidebar-accent)]/60"
                >
                    <Settings className="size-4.5" />
                    Settings
                </Link>
                <button
                    type="button"
                    className="flex items-center gap-3 rounded-xl px-3 py-2.5 text-left text-sm text-[var(--sidebar-foreground)] transition-colors hover:bg-[var(--sidebar-accent)]/60"
                >
                    <LogOut className="size-4.5" />
                    Log out
                </button>
            </div>
        </aside>
    );
};

export default Sidebar;