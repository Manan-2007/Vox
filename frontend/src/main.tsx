import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { createBrowserRouter, RouterProvider } from "react-router-dom";
import { SessionPage } from "./pages/SessionPage";
import { NotFoundPage } from "./pages/NotFoundPage";
import "./index.css";

// One real route today. The router is here so the speech and settings views
// have somewhere to land without reshaping the app.
const router = createBrowserRouter([
  { path: "/", element: <SessionPage /> },
  { path: "*", element: <NotFoundPage /> },
]);

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <RouterProvider router={router} />
  </StrictMode>,
);
