import Image from "next/image";
import lotusImage from "../../assests/lotus.png";

const STAR_SHAPE =
    "[clip-path:polygon(50%_0%,61%_39%,100%_50%,61%_61%,50%_100%,39%_61%,0%_50%,39%_39%)]";

// drifting particles move outward from the medallion center via these per-particle CSS vars
type ParticleStyle = React.CSSProperties & { "--tx"?: string; "--ty"?: string };

const driftParticles: { size: string; tx: string; ty: string; duration: string; delay: string }[] = [
    { size: "h-1.5 w-1.5", tx: "42px", ty: "-34px", duration: "3.4s", delay: "0s" },
    { size: "h-1 w-1", tx: "-38px", ty: "-22px", duration: "3.8s", delay: "0.4s" },
    { size: "h-1.5 w-1.5", tx: "30px", ty: "40px", duration: "3.1s", delay: "0.8s" },
    { size: "h-1 w-1", tx: "-30px", ty: "38px", duration: "4.1s", delay: "1.2s" },
    { size: "h-1 w-1", tx: "48px", ty: "6px", duration: "3.6s", delay: "1.6s" },
    { size: "h-1.5 w-1.5", tx: "-46px", ty: "4px", duration: "3.3s", delay: "2s" },
    { size: "h-1 w-1", tx: "8px", ty: "-46px", duration: "3.9s", delay: "0.2s" },
    { size: "h-1 w-1", tx: "-6px", ty: "44px", duration: "3.5s", delay: "1.4s" },
    { size: "h-1 w-1", tx: "20px", ty: "-46px", duration: "3.7s", delay: "0.6s" },
    { size: "h-1.5 w-1.5", tx: "-18px", ty: "-44px", duration: "4s", delay: "1.8s" },
    { size: "h-1 w-1", tx: "44px", ty: "22px", duration: "3.2s", delay: "1s" },
    { size: "h-1 w-1", tx: "-44px", ty: "24px", duration: "3.9s", delay: "0.3s" },
    { size: "h-1.5 w-1.5", tx: "14px", ty: "48px", duration: "3.6s", delay: "2.2s" },
    { size: "h-1 w-1", tx: "-14px", ty: "-50px", duration: "3.4s", delay: "0.9s" },
    { size: "h-1 w-1", tx: "50px", ty: "-10px", duration: "4.2s", delay: "1.5s" },
    { size: "h-1 w-1", tx: "-50px", ty: "-6px", duration: "3.1s", delay: "0.1s" },
];

// deterministic PRNG so the dust field is identical between server render and client hydration
function seededRandom(seed: number) {
    let value = seed;
    return () => {
        value = (value * 9301 + 49297) % 233280;
        return value / 233280;
    };
}

// a dense field of tiny shimmering dots filling the whole medallion, like a sound-reactive particle sphere
const rand = seededRandom(13);
const dustParticles = Array.from({ length: 70 }, () => {
    const angle = rand() * Math.PI * 2;
    const radius = Math.sqrt(rand()) * 44; // sqrt keeps the spread even across the full disc, not bunched at center
    return {
        left: 50 + Math.cos(angle) * radius,
        top: 50 + Math.sin(angle) * radius,
        size: 1 + rand() * 1.4,
        duration: 1.8 + rand() * 2.2,
        delay: rand() * 3,
    };
});

export default function VoiceIndicator(){
    return (
        <div className="flex items-center justify-center" role="img" aria-label="Lotus voice indicator">
            <div className="relative flex aspect-square w-[min(72vw,22rem)] items-center justify-center">
                {/* ambient ripple rings, filled with the medallion's own gold so they read as a fading glow */}
                <div aria-hidden="true" className="absolute inset-[2%] rounded-full border border-[#d7c097]/60" />
                <div
                    aria-hidden="true"
                    className="absolute inset-[10%] rounded-full bg-[#cf9646]/10 [animation:ripple-pulse_3.2s_ease-out_infinite]"
                />
                <div
                    aria-hidden="true"
                    className="absolute inset-[18%] rounded-full bg-[#cf9646]/10 [animation:ripple-pulse_3.2s_ease-out_infinite_1.1s]"
                />

                <div className="relative aspect-square w-[64%] overflow-hidden rounded-full bg-[radial-gradient(circle_at_50%_38%,#f4d795_0%,#cf9646_54%,#94602f_80%,#70472c_100%)] shadow-[0_18px_48px_rgba(112,71,44,0.24),inset_0_0_22px_rgba(255,245,218,0.56)]">
                    <div
                        aria-hidden="true"
                        className="absolute -inset-[12%] -z-10 rounded-full bg-[#d9ad72]/35 blur-2xl"
                    />
                    {/* image bg is opaque white, so multiply-blend it away and let the gold circle show through */}
                    <div className="absolute inset-[10%]">
                        <Image
                            src={lotusImage}
                            alt=""
                            fill
                            sizes="(max-width: 640px) 40vw, 160px"
                            className="object-contain mix-blend-multiply"
                            priority
                        />
                    </div>

                    {/* twinkling sparkles sit over the gold face, where they contrast and read clearly */}
                    <span aria-hidden="true" className={`absolute left-[30%] top-[12%] z-10 h-2.5 w-2.5 bg-[#fff8e6] shadow-[0_0_8px_2px_rgba(255,248,230,0.9)] ${STAR_SHAPE} [animation:sparkle-twinkle_2.6s_ease-in-out_infinite]`} />
                    <span aria-hidden="true" className={`absolute right-[16%] top-[28%] z-10 h-2 w-2 bg-[#fff8e6] shadow-[0_0_8px_2px_rgba(255,248,230,0.9)] ${STAR_SHAPE} [animation:sparkle-twinkle_2.6s_ease-in-out_infinite_0.6s]`} />
                    <span aria-hidden="true" className={`absolute left-[18%] bottom-[24%] z-10 h-1.5 w-1.5 bg-[#fff8e6] shadow-[0_0_8px_2px_rgba(255,248,230,0.9)] ${STAR_SHAPE} [animation:sparkle-twinkle_2.6s_ease-in-out_infinite_1.3s]`} />
                    <span aria-hidden="true" className={`absolute right-[22%] bottom-[16%] z-10 h-2 w-2 bg-[#fff8e6] shadow-[0_0_8px_2px_rgba(255,248,230,0.9)] ${STAR_SHAPE} [animation:sparkle-twinkle_2.6s_ease-in-out_infinite_1.9s]`} />

                    {/* particles drifting outward from the center in every direction, Perplexity-orb style */}
                    {driftParticles.map((particle, index) => (
                        <span
                            key={index}
                            aria-hidden="true"
                            className={`absolute left-1/2 top-1/2 z-10 ${particle.size} bg-[#fff8e6] shadow-[0_0_6px_1px_rgba(255,248,230,0.85)] ${STAR_SHAPE} [animation-name:particle-drift] [animation-timing-function:ease-out] [animation-iteration-count:infinite]`}
                            style={{
                                "--tx": particle.tx,
                                "--ty": particle.ty,
                                animationDuration: particle.duration,
                                animationDelay: particle.delay,
                            } as ParticleStyle}
                        />
                    ))}

                    {/* dense shimmering dust field filling the sphere, for the "voice is coming from it" feel */}
                    {dustParticles.map((dot, index) => (
                        <span
                            key={index}
                            aria-hidden="true"
                            className="absolute rounded-full bg-[#fff8e6] [animation-name:sparkle-twinkle] [animation-timing-function:ease-in-out] [animation-iteration-count:infinite]"
                            style={{
                                left: `${dot.left}%`,
                                top: `${dot.top}%`,
                                width: `${dot.size}px`,
                                height: `${dot.size}px`,
                                animationDuration: `${dot.duration}s`,
                                animationDelay: `${dot.delay}s`,
                            }}
                        />
                    ))}

                    <div aria-hidden="true" className="pointer-events-none absolute inset-[4%] rounded-full border border-white/40" />
                </div>
            </div>
        </div>
    );
}