export interface AutoFormSection {
  type: "panel" | "dock_area" | "value" | "choice" | "toggle" | "toggle_row" | "button_row" | "table" | "custom" | "info" | string;
  title?: string;
  label?: string;
  description?: string;
  attr?: string;
  target?: string;
  kind?: "int" | "float" | "str" | "text" | "file" | "directory" | string;
  minimum?: number;
  maximum?: number;
  step?: number;
  options?: any;

  sections?: AutoFormSection[];
  collapsed?: boolean;
  n_col?: number;
  key?: string;
}

export interface AutoFormSpec {
  locale?: string;
  title?: string;
  sections: AutoFormSection[];
  _comment?: string;
}

export interface AutoFormProps {
  spec: AutoFormSpec;
  data: Record<string, any>;
  onChange: (path: string, value: any) => void;
  onSave?: (path: string) => void;
  onAction?: (actionName: string, payload?: any) => void;
  catalog?: Record<string, any>;
}
