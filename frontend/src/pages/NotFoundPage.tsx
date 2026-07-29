import { Link } from "react-router-dom";

export function NotFoundPage() {
  return (
    <div className="notfound">
      <h1 className="notfound__title">Not found</h1>
      <p className="notfound__body">There is nothing at this address.</p>
      <Link className="button button--primary" to="/">
        Back to session
      </Link>
    </div>
  );
}
