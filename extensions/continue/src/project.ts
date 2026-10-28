import { existsSync, readFileSync } from "node:fs";
import { mkdir, writeFile } from "node:fs/promises";
import { dirname, isAbsolute, join, normalize, relative, resolve, sep } from "node:path";
import type { HistoryOutputBudget, PromptPassUsageTelemetry, ResolvedProjectContext } from "./types.ts";

interface ExecApi {
	exec(command: string, args: string[], options?: { cwd?: string; timeout?: number }): Promise<{
		stdout: string;
		code: number;
	}>;
}

const mutationQueues = new Map<string, Promise<void>>();

async function withMutationQueue(path: string, work: () => Promise<void>): Promise<void> {
	const previous = mutationQueues.get(path) ?? Promise.resolve();
	const next = previous.then(work, work);
	mutationQueues.set(path, next);
	try {
		await next;
	} finally {
		if (mutationQueues.get(path) === next) mutationQueues.delete(path);
	}
}

function trimTrailingWhitespace(value: string): string {
	return value
		.replace(/\r\n/g, "\n")
		.split("\n")
		.map((line) => line.replace(/[ \t]+$/g, ""))
		.join("\n")
		.trim();
}

/** Normalize markdown content before diffing or writing. */
export function normalizeMarkdownContent(value: string): string {
	return `${trimTrailingWhitespace(value)}\n`;
}

async function getGitRoot(pi: ExecApi, cwd: string): Promise<string | undefined> {
	const result = await pi.exec("git", ["rev-parse", "--show-toplevel"], { cwd, timeout: 4000 });
	if (result.code !== 0) return undefined;
	const root = result.stdout.trim();
	return root.length > 0 ? root : undefined;
}

export async function resolveProjectRoot(pi: ExecApi, cwd: string): Promise<string> {
	return (await getGitRoot(pi, cwd)) ?? cwd;
}

function sanitizeRepoRelativePath(projectRoot: string, configuredPath: string, fallback: string): string {
	const trimmed = normalize(configuredPath.trim()).replace(/^\.\//, "");
	if (trimmed.length === 0) return fallback;
	if (isAbsolute(trimmed)) return fallback;
	const resolved = resolve(projectRoot, trimmed);
	const rel = relative(projectRoot, resolved);
	if (rel === "" || rel === ".." || rel.startsWith(`..${sep}`)) return fallback;
	return trimmed;
}

function readOptionalFile(path: string): string | undefined {
	return existsSync(path) ? readFileSync(path, "utf8") : undefined;
}

export function encodeSessionIdForArtifactPath(sessionId: string): string {
	const encoded = Buffer.from(sessionId, "utf8").toString("base64url");
	return encoded.length > 0 ? encoded : "empty-session-id";
}

export function buildContinuationArtifactPath(projectRoot: string, sessionId: string): string {
	return join(projectRoot, ".pi", "continue", `${encodeSessionIdForArtifactPath(sessionId)}.md`);
}

export interface SynthesisFailureArtifactDetails {
	pass: "history";
	code: string;
	requestedModel?: string;
	responseModel?: string;
	httpStatus?: number;
	usage?: PromptPassUsageTelemetry;
	outputBudget?: HistoryOutputBudget;
	rawText: string;
}

/** Filesystem-safe timestamp for a diagnostic filename: 2026-09-01T08-53-58-123Z. */
function sanitizeTimestampForFilename(timestamp: Date): string {
	return timestamp.toISOString().replace(/[:.]/g, "-");
}

export function buildSynthesisFailureArtifactPath(projectRoot: string, timestamp: Date, code: string): string {
	return join(projectRoot, ".pi", "continue", "failures", `${sanitizeTimestampForFilename(timestamp)}-${code}.md`);
}

function renderSynthesisFailureArtifactHeader(timestamp: Date, details: SynthesisFailureArtifactDetails): string {
	const lines = [
		"# pi-continue synthesis failure",
		"",
		"The history pass produced a response that failed to parse or validate.",
		"Raw model output follows the `---` marker below, unmodified.",
		"",
		`- timestamp: ${timestamp.toISOString()}`,
		`- pass: ${details.pass}`,
		`- code: ${details.code}`,
	];
	if (details.requestedModel !== undefined) lines.push(`- requested model: ${details.requestedModel}`);
	if (details.responseModel !== undefined) lines.push(`- response model: ${details.responseModel}`);
	if (details.httpStatus !== undefined) lines.push(`- HTTP status: ${details.httpStatus}`);
	if (details.usage) {
		lines.push(
			`- usage: input=${details.usage.input}, output=${details.usage.output}, total=${details.usage.totalTokens}`,
		);
	}
	if (details.outputBudget) {
		const budget = details.outputBudget;
		lines.push(
			`- output budget: requested=${budget.requestedTokens}, effective=${budget.effectiveTokens}, modelMax=${budget.modelMaxTokens ?? "unknown"}, clampedByModel=${budget.clampedByModel}`,
		);
	}
	lines.push("", "---", "");
	return lines.join("\n");
}

/**
 * Best-effort diagnostic capture for a synthesis pass that failed parsing/validation.
 * Writes the raw model output alongside request telemetry so a failure can be inspected
 * after the fact instead of vanishing — parseHistoryArtifacts' recovery heuristics only
 * cover known-good shapes, so an unrecovered failure is otherwise unobservable state.
 * Never throws: a write failure here must not mask or replace the original synthesis error.
 */
export async function writeSynthesisFailureArtifact(
	projectRoot: string,
	details: SynthesisFailureArtifactDetails,
): Promise<string | undefined> {
	const timestamp = new Date();
	const path = buildSynthesisFailureArtifactPath(projectRoot, timestamp, details.code);
	try {
		await mkdir(dirname(path), { recursive: true });
		await writeFile(path, `${renderSynthesisFailureArtifactHeader(timestamp, details)}${details.rawText}\n`, "utf8");
		return path;
	} catch {
		return undefined;
	}
}

/** Resolve the project root, package-owned continuation artifact path, and configured agent guide. */
export async function resolveProjectContext(
	pi: ExecApi,
	cwd: string,
	sessionId: string,
	configuredAgentGuidePath = "AGENTS.md",
): Promise<ResolvedProjectContext> {
	const projectRoot = await resolveProjectRoot(pi, cwd);
	const repoRelativeAgentGuidePath = sanitizeRepoRelativePath(projectRoot, configuredAgentGuidePath, "AGENTS.md");
	const agentGuidePath = join(projectRoot, repoRelativeAgentGuidePath);
	return {
		projectRoot,
		continuationArtifactPath: buildContinuationArtifactPath(projectRoot, sessionId),
		agentGuidePath,
		existingAgentGuide: readOptionalFile(agentGuidePath),
	};
}

/** Write a normalized Markdown file only when normalized content changes. */
export async function writeNormalizedMarkdownFile(path: string, content: string): Promise<"updated" | "unchanged"> {
	const normalized = normalizeMarkdownContent(content);
	const existing = existsSync(path) ? readFileSync(path, "utf8") : undefined;
	if (existing !== undefined && normalizeMarkdownContent(existing) === normalized) {
		return "unchanged";
	}
	await withMutationQueue(path, async () => {
		await mkdir(dirname(path), { recursive: true });
		await writeFile(path, normalized, "utf8");
	});
	return "updated";
}
