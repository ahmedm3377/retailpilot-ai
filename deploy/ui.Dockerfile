# The web UI: the production build of frontend/, served by nginx, which also proxies /api and /health
# to the API on 127.0.0.1:8080 (they share one network namespace; see deploy/compose.yaml).
FROM node:22-bookworm-slim AS build
WORKDIR /ui
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

FROM nginx:1.29-alpine
COPY deploy/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /ui/dist /usr/share/nginx/html
