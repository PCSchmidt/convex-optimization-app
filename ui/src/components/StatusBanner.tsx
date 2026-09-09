export type Status = "idle" | "loading" | "success" | "error";

const STATUS_TEXT: Record<Status, string> = {
  idle: "Enter a configuration and solve.",
  loading: "Solving on the backend...",
  success: "Solve complete.",
  error: "The solve failed - see the message below.",
};

/** Fixed-height status region: no layout shift when the state changes. */
export default function StatusBanner({ status, message }: { status: Status; message: string | null }) {
  return (
    <p
      data-testid="status-banner"
      role={status === "error" ? "alert" : "status"}
      className={`status-banner status-${status}`}
    >
      {message ?? STATUS_TEXT[status]}
    </p>
  );
}
