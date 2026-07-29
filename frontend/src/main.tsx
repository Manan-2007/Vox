import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { createBrowserRouter, RouterProvider } from "react-router-dom";
import { LandingPage } from "./pages/LandingPage";
import { SessionPage } from "./pages/SessionPage";
import { NotFoundPage } from "./pages/NotFoundPage";
import "./index.css";

const router = createBrowserRouter([
  { path: "/", element: <LandingPage /> },
  { path: "/session", element: <SessionPage /> },
  { path: "*", element: <NotFoundPage /> },
]);

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <RouterProvider router={router} />
  </StrictMode>,
);
