# -- Base --
FROM node:22-alpine AS base
WORKDIR /app

# -- Dependencies --
FROM base AS deps
COPY package.json package-lock.json ./
COPY apps/server/package.json apps/server/
COPY packages/types/package.json packages/types/
COPY packages/engine/package.json packages/engine/
COPY packages/config/package.json packages/config/
COPY packages/shared/package.json packages/shared/
RUN npm ci --workspace=apps/server

# -- Build --
FROM base AS builder
COPY --from=deps /app/node_modules ./node_modules
COPY . .
RUN npm run build --workspace=apps/server

# -- Production --
FROM base AS runner
ENV NODE_ENV=production
RUN addgroup --system --gid 1001 appgroup
RUN adduser --system --uid 1001 appuser
COPY --from=builder /app/apps/server/dist ./dist
COPY --from=builder /app/node_modules ./node_modules
USER appuser
EXPOSE 4000
ENV PORT=4000
CMD ["node", "dist/index.js"]
