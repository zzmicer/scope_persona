import { useCallback, useEffect, useState } from "react";
import { Button } from "./ui/button";

const EVENTS = [
  {
    key: "1",
    label: "Run Through Hair",
    prompt:
      "gently runs one hand through her hair and looks naturally toward the camera",
  },
  {
    key: "2",
    label: "Rest Chin In Hands",
    prompt:
      "leans closer and rests her chin softly in both hands while looking at the camera",
  },
  {
    key: "3",
    label: "Hold Candle",
    prompt:
      "holds a small lit candle carefully in both hands; its warm flame illuminates her face",
  },
  {
    key: "F",
    label: "A Butterfly Alights",
    prompt: "watches as a delicate butterfly alights gently on her hand",
  },
  {
    key: "G",
    label: "Snow Blankets Room",
    prompt: "watches soft snow begin to fall and blanket the same room",
  },
] as const;

interface LingbotEventPanelProps {
  disabled: boolean;
  onEvent: (prompt: string) => void;
}

export function LingbotEventPanel({
  disabled,
  onEvent,
}: LingbotEventPanelProps) {
  const [activeKey, setActiveKey] = useState<string | null>(null);

  const trigger = useCallback(
    (event: (typeof EVENTS)[number]) => {
      if (disabled) return;
      setActiveKey(event.key);
      onEvent(event.prompt);
    },
    [disabled, onEvent]
  );

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.repeat || event.target instanceof HTMLInputElement) return;
      const proposal = EVENTS.find(
        item => item.key.toLowerCase() === event.key.toLowerCase()
      );
      if (proposal) trigger(proposal);
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [trigger]);

  return (
    <section className="rounded-lg border bg-card p-3">
      <div className="mb-1 text-sm font-semibold">Event Proposals</div>
      <p className="mb-3 text-xs text-muted-foreground">
        Beauty demo actions · camera: WASD + mouse
      </p>
      <div className="space-y-2">
        {EVENTS.map(event => (
          <Button
            key={event.key}
            variant={activeKey === event.key ? "default" : "outline"}
            className="h-10 w-full justify-start"
            disabled={disabled}
            onClick={() => trigger(event)}
          >
            <span className="flex size-6 items-center justify-center rounded bg-primary/15 text-xs">
              {event.key}
            </span>
            {event.label}
          </Button>
        ))}
      </div>
      {!disabled && activeKey && (
        <Button
          variant="ghost"
          size="sm"
          className="mt-2 w-full"
          onClick={() => {
            setActiveKey(null);
            onEvent("");
          }}
        >
          Clear event
        </Button>
      )}
    </section>
  );
}
