interface TabOption {
  id: string;
  label: string;
}

interface SegmentedTabsProps {
  options: TabOption[];
  activeId: string;
  onChange: (id: string) => void;
}

export function SegmentedTabs({ options, activeId, onChange }: SegmentedTabsProps) {
  return (
    <div className="segmented-tabs" role="tablist">
      {options.map((option) => (
        <button
          key={option.id}
          type="button"
          role="tab"
          aria-selected={activeId === option.id}
          className={activeId === option.id ? "segmented-tab segmented-tab-active" : "segmented-tab"}
          onClick={() => onChange(option.id)}
        >
          {option.label}
        </button>
      ))}
    </div>
  );
}
