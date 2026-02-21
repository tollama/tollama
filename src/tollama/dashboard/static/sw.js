const CACHE_NAME = "tollama-dashboard-v1";
const SHELL_ASSETS = [
  "/dashboard",
  "/dashboard/static/index.html",
  "/dashboard/static/css/dashboard.css",
  "/dashboard/static/js/app.js",
  "/dashboard/static/js/forecast.js",
  "/dashboard/static/js/comparison.js",
  "/dashboard/static/js/htmx.min.js",
  "/dashboard/static/js/alpine.min.js",
  "/dashboard/static/js/chart.min.js",
  "/dashboard/static/manifest.json",
  "/dashboard/static/img/favicon.svg"
];

self.addEventListener("install", (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => cache.addAll(SHELL_ASSETS)),
  );
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(
        keys
          .filter((key) => key !== CACHE_NAME)
          .map((key) => caches.delete(key)),
      ),
    ),
  );
});

self.addEventListener("fetch", (event) => {
  const requestUrl = new URL(event.request.url);
  if (requestUrl.pathname.startsWith("/api/")) {
    event.respondWith(
      fetch(event.request).catch(() => caches.match(event.request)),
    );
    return;
  }

  event.respondWith(
    caches.match(event.request).then((cached) => cached || fetch(event.request)),
  );
});
