import { existsSync, realpathSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

interface PiInternals {
	prepareCompaction: (entries: unknown[], settings: { enabled: boolean; reserveTokens: number; keepRecentTokens: number }) => unknown;
	estimateTokens: (message: unknown) => number;
	estimateContextTokens: (messages: unknown[]) => {
		tokens: number;
		usageTokens: number;
		trailingTokens: number;
		lastUsageIndex: number | null;
	};
	convertToLlm: (messages: unknown[]) => unknown[];
	serializeConversation: (messages: unknown[]) => string;
}

const COMPACTION_ENTRY = join("core", "compaction", "compaction.js");

/**
 * Locate the installed pi-coding-agent dist directory. import.meta.resolve works when this
 * file runs as plain ESM, but Pi 1.x loads extensions through jiti, whose resolver does not
 * serve the host package to import.meta.resolve ("Cannot find module"). Fall back to walking
 * up from the running Pi CLI entry point (dist/bundle/cli.js), which sits inside that dist.
 */
function resolvePiDistRoot(): string {
	try {
		return dirname(fileURLToPath(import.meta.resolve("@earendil-works/pi-coding-agent")));
	} catch {
		// fall through to the CLI-entry lookup
	}
	const entry = process.argv[1];
	if (entry) {
		let dir = dirname(realpathSync(entry));
		for (let depth = 0; depth < 6; depth += 1) {
			if (existsSync(join(dir, COMPACTION_ENTRY))) return dir;
			const parent = dirname(dir);
			if (parent === dir) break;
			dir = parent;
		}
	}
	throw new Error("pi-continue could not locate the installed @earendil-works/pi-coding-agent dist");
}

let cachedInternals: Promise<PiInternals> | undefined;

/** Resolve Pi internal compaction helpers from the installed package at runtime. */
export async function loadPiInternals(): Promise<PiInternals> {
	if (!cachedInternals) {
		cachedInternals = (async () => {
			const distRoot = resolvePiDistRoot();
			const [compactionModule, messagesModule, utilsModule] = await Promise.all([
				import(pathToFileURL(join(distRoot, "core", "compaction", "compaction.js")).href),
				import(pathToFileURL(join(distRoot, "core", "messages.js")).href),
				import(pathToFileURL(join(distRoot, "core", "compaction", "utils.js")).href),
			]);
			return {
				prepareCompaction: compactionModule.prepareCompaction,
				estimateTokens: compactionModule.estimateTokens,
				estimateContextTokens: compactionModule.estimateContextTokens,
				convertToLlm: messagesModule.convertToLlm,
				serializeConversation: utilsModule.serializeConversation,
			};
		})();
	}
	return cachedInternals;
}
