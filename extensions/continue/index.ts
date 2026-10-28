import { randomUUID } from "node:crypto";
import type { AssistantMessage } from "@earendil-works/pi-ai";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { loadHistoryPromptAssets } from "./src/assets.ts";
import { parseHistoryArtifacts } from "./src/blocks.ts";
import { splitContinueSubcommand, shouldOpenContinuePalette } from "./src/command-shape.ts";
import { runContinuePaletteResult, runEnabledContinuationCommand } from "./src/command-runner.ts";
import { runLedgerCommand, runPreviewCommand, runResetCommand, runSettingsDialog, runStatusCommand } from "./src/commands.ts";
import { getContinueArgumentCompletions } from "./src/completions.ts";
import { normalizeCompactionPreparation, snapshotFileOperations, stripCompactionPreparationMessages } from "./src/compaction-preparation.ts";
import { composeCompactionSummary } from "./src/compose.ts";
import { DEFAULT_CONTINUE_CONFIG, loadContinuationConfig } from "./src/config.ts";
import {
	abandonActiveContinuationEvent,
	failPendingOutputWritesForEvent,
	getActiveContinuationEventId,
	isActiveRunningContinuationEvent,
	markContinuationArtifact,
	planContinuationOutputWrites,
	recordContinuationSynthesisFailure,
	recordContinuationSynthesisTelemetry,
	recordOutputWriteResult,
} from "./src/continuation-event.ts";
import { buildContinuationDetails, buildContinuationSynthesisTelemetry, parseContinuationDetails } from "./src/details.ts";
import { buildLedgerSnapshot, createContinuationLedgerOverlayController } from "./src/ledger-viewer.ts";
import { runMidRunGuard } from "./src/mid-run-guard.ts";
import { PromptPassError, runPromptPass } from "./src/model.ts";
import { loadPiInternals } from "./src/pi-internals.ts";
import { compileHistoryPrompt } from "./src/prompt.ts";
import { resolveProjectContext, writeNormalizedMarkdownFile, writeSynthesisFailureArtifact } from "./src/project.ts";
import { isContinuationPromptUserMessage } from "./src/prompt-dispatch.ts";
import { showContinuePalette } from "./src/palette.ts";
import {
	CONTINUATION_PROMPT,
	armDeferredResumeStartTimeout,
	clearResumeStartTimeout,
	createContinuationRuntimeState,
	dispatchVerifiedContinuationResume,
	failContinuationCompactionProof,
	failRunningAwaitingContinuationResume,
	markAwaitingContinuationResumeStarted,
	settleAwaitingContinuationResumeFromAssistant,
	type ContinuationRuntimeState,
	verifyContinuationCompactionProof,
} from "./src/runtime.ts";
import {
	INVALID_COMPACTION_PROOF_FAILURE,
	NATIVE_COMPACTION_FALLBACK_FAILURE,
	clearPendingResumeDispatch,
} from "./src/resume-proof.ts";
import type { AgentGuideWriteStatus, ContinuationSynthesisFailure, ContinuationSynthesisTelemetry, ParsedHistoryArtifacts, PendingOutputWrite, WriteMode } from "./src/types.ts";
import {
	clearWorkingVisuals,
	settleWorkingVisuals,
	updateWorkingVisuals,
} from "./src/working-ui.ts";

function decideAgentGuideWriteStatus(writeMode: WriteMode, agentGuideMd: string | undefined): AgentGuideWriteStatus {
	if (writeMode === "off") return "write-off";
	return agentGuideMd ? "replacement-pending" : "no-replacement";
}

function isAssistantMessage(message: unknown): message is AssistantMessage {
	return typeof message === "object" && message !== null && "role" in message && message.role === "assistant";
}

const OUTPUT_WRITE_FAILURE = "Output write failed; check the configured path and permissions.";
// Extra headroom beyond the configured synthesis timeout for the rest of the
// compaction pipeline around the LLM call itself: project context resolution,
// artifact parsing, appending the compaction entry, and (when continuing)
// dispatching the resume prompt.
const SESSION_SHUTDOWN_COMPACTION_WAIT_BUFFER_MS = 30_000;
const PENDING_OUTPUT_WRITE_FAILURE = "Output write did not complete before continuation failed.";
// Used when the history-pass synthesis fails (bad JSON, model ignored the
// instruction, request-level error, etc). Resuming without a structured brief
// beats stopping cold: the whole point of automatic continuation is that the
// session keeps going unattended, and a human isn't there to notice a stall.
const DEGRADED_CONTINUATION_MESSAGE =
	"Automatic continuation could not produce a structured handoff this cycle. Continue the task using the current file state and recent tool/session output; do not assume specific facts from before this compaction without re-verifying them.";

class ArtifactParseError extends Error {
	readonly failure: ContinuationSynthesisFailure;

	constructor(failure: ContinuationSynthesisFailure) {
		super(failure.code);
		this.failure = failure;
	}
}

function normalizeSynthesisFailure(error: unknown): ContinuationSynthesisFailure {
	if (error instanceof PromptPassError) {
		return {
			kind: "model-provider-call",
			code: error.code,
			pass: "history",
			requestedModel: error.requestedModel,
			httpStatus: error.httpStatus,
		};
	}
	if (error instanceof ArtifactParseError) return error.failure;
	return { kind: "internal", code: "internal-error", pass: "history" };
}

export default function (pi: ExtensionAPI) {
	const pendingOutputWrites = new Map<string, PendingOutputWrite>();
	const runtime = createContinuationRuntimeState();
	const ledgerOverlay = createContinuationLedgerOverlayController();

	function cleanupPendingOutputWrites(eventId: string): void {
		let removed = false;
		for (const [writeId, pending] of pendingOutputWrites) {
			if (pending.eventId !== eventId) continue;
			pendingOutputWrites.delete(writeId);
			removed = true;
		}
		if (removed) {
			failPendingOutputWritesForEvent(runtime, eventId, PENDING_OUTPUT_WRITE_FAILURE);
		}
	}

	pi.registerCommand("continue", {
		description: "Save a same-session handoff, resume this run, or inspect continuation settings.",
		getArgumentCompletions: getContinueArgumentCompletions,
		handler: async (args, ctx) => {
			if (shouldOpenContinuePalette(args, ctx.hasUI)) {
				const palette = await showContinuePalette(pi, ctx, runtime);
				if (palette.supported) {
					if (palette.result) {
						await runContinuePaletteResult(pi, ctx, runtime, ledgerOverlay, palette.result, (eventId) => cleanupPendingOutputWrites(eventId));
					}
					return;
				}
			}
			const subcommand = splitContinueSubcommand(args);
			if (subcommand?.name === "status") {
				await runStatusCommand(pi, ctx, runtime);
				return;
			}
			if (subcommand?.name === "ledger") {
				await runLedgerCommand(ctx, runtime, ledgerOverlay);
				return;
			}
			if (subcommand?.name === "settings") {
				await runSettingsDialog(pi, ctx, subcommand.rest);
				return;
			}
			if (subcommand?.name === "reset") {
				await runResetCommand(pi, ctx, subcommand.rest);
				return;
			}
			if (subcommand?.name === "preview") {
				await runPreviewCommand(pi, ctx, subcommand.rest);
				return;
			}
			await runEnabledContinuationCommand(
				pi,
				ctx,
				runtime,
				args,
				(eventId) => cleanupPendingOutputWrites(eventId),
			);
		},
	});

	pi.on("before_agent_start", async (event, ctx) => {
		if (event.prompt !== CONTINUATION_PROMPT) return;
		const eventId = markAwaitingContinuationResumeStarted(runtime);
		if (!eventId) return;
		updateWorkingVisuals(ctx, runtime, eventId, "pi-continue resume running");
	});

	pi.on("message_start", async (event, ctx) => {
		if (isContinuationPromptUserMessage(event.message, CONTINUATION_PROMPT)) {
			const eventId = markAwaitingContinuationResumeStarted(runtime);
			if (eventId) {
				updateWorkingVisuals(ctx, runtime, eventId, "pi-continue resume running");
			}
			return;
		}
		if (!isAssistantMessage(event.message)) return;
		const eventId = runtime.awaitingResumeEventId;
		if (!eventId || runtime.latestEvent?.id !== eventId || runtime.latestEvent.resume.status !== "running") return;
		updateWorkingVisuals(ctx, runtime, eventId, "pi-continue resume running");
	});

	pi.on("agent_end", async (_event, ctx) => {
		const settlement = failRunningAwaitingContinuationResume(runtime, "Continuation resume did not produce an assistant response.");
		if (settlement) {
			settleWorkingVisuals(ctx, runtime, settlement.eventId);
			return;
		}
		armDeferredResumeStartTimeout(ctx, runtime);
	});

	pi.on("message_end", async (event, ctx) => {
		if (!isAssistantMessage(event.message)) return;
		const settlement = settleAwaitingContinuationResumeFromAssistant(runtime, event.message);
		if (!settlement) return;
		settleWorkingVisuals(ctx, runtime, settlement.eventId);
	});

	pi.on("context", async (event, ctx) => {
		await runMidRunGuard(pi, ctx, runtime, event.messages, (eventId) => cleanupPendingOutputWrites(eventId));
	});

	pi.on("session_before_compact", async (event, ctx) => {
		const sessionId = ctx.sessionManager.getSessionId();
		const ownerEventId = getActiveContinuationEventId(runtime);
		// pi-continue always drives compaction through ctx.compact(), which core
		// always tags reason:"manual". Any other reason (e.g. "threshold" /
		// "overflow") means core's own native auto-compaction fired on its own —
		// it mirrors pi-continue's threshold by design, so the two can trigger
		// within moments of each other. Only claim the event pi-continue itself
		// started; otherwise let core run its own default compaction untouched
		// instead of racing two compaction attempts against the same event.
		// `reason` is a required field on SessionBeforeCompactEvent as of the
		// pi-coding-agent version we actually run against (>=0.84.0), but our pinned
		// devDependency floor (^0.74.0, used only for local type-checking) predates
		// it -- hence the cast rather than a plain property access.
		if ((event as { reason?: string }).reason !== "manual") return undefined;
		if (ownerEventId === undefined) return undefined;
		const ownerStillActive = () => isActiveRunningContinuationEvent(runtime, ownerEventId);
		const ownerLostResult = () => ownerStillActive() ? undefined : { cancel: true as const };
		const projectContext = await resolveProjectContext(pi, ctx.cwd, sessionId);
		const ownerLostAfterProjectContext = ownerLostResult();
		if (ownerLostAfterProjectContext) return ownerLostAfterProjectContext;
		const config = loadContinuationConfig(projectContext.projectRoot);
		if (!config.enabled) return { cancel: true };
		const resolvedProjectContext = await resolveProjectContext(pi, ctx.cwd, sessionId, config.agentGuidePath);
		const ownerLostAfterResolvedContext = ownerLostResult();
		if (ownerLostAfterResolvedContext) return ownerLostAfterResolvedContext;
		const normalizedPreparation = normalizeCompactionPreparation(event.preparation, event.branchEntries);
		if (normalizedPreparation.repairedProviderUnsafeSuffix && ctx.hasUI) {
			ctx.ui.notify("pi-continue summarized a provider-unsafe kept suffix before resuming.", "warning");
		} else if (normalizedPreparation.repairedNoOpCut && ctx.hasUI) {
			ctx.ui.notify("pi-continue moved the handoff to a safer checkpoint before resuming.", "warning");
		}
		// Strip pi-continue's own receiver prompt so the synthesizer never mistakes
		// our resume wrapper ("Continue from the same-session pi-continue/v4 handoff…")
		// for user content. Otherwise the synthesizer can promote our wrapper sentences
		// as `forbid` or `task` entries.
		const preparation = stripCompactionPreparationMessages(normalizedPreparation, (message) =>
			isContinuationPromptUserMessage(message, CONTINUATION_PROMPT)
		);
		const fileOpsSnapshot = snapshotFileOperations(preparation.fileOps);
		const internals = await loadPiInternals();
		const ownerLostAfterInternals = ownerLostResult();
		if (ownerLostAfterInternals) return ownerLostAfterInternals;
		const messagesToSummarize = preparation.messagesToSummarize;
		const turnPrefixMessages = preparation.turnPrefixMessages;
		const turnPrefixTranscript = preparation.isSplitTurn && turnPrefixMessages.length > 0
			? internals.serializeConversation(internals.convertToLlm(turnPrefixMessages))
			: undefined;
		let historyPrompt = compileHistoryPrompt(
			loadHistoryPromptAssets(
				resolvedProjectContext.projectRoot,
				config.promptOverridePolicy,
				preparation.previousSummary ? "update" : "initial",
			),
			{
				scenario: preparation.previousSummary ? "update" : "initial",
				projectRoot: resolvedProjectContext.projectRoot,
				agentGuidePath: resolvedProjectContext.agentGuidePath,
				existingAgentGuide: resolvedProjectContext.existingAgentGuide,
				previousSummary: preparation.previousSummary,
				historyTranscript: internals.serializeConversation(internals.convertToLlm(messagesToSummarize)),
				turnPrefixTranscript,
				customInstructions: event.customInstructions,
				fileOps: fileOpsSnapshot,
			},
		);
		let historyArtifacts: ParsedHistoryArtifacts | undefined;
		let synthesis: ContinuationSynthesisTelemetry | undefined;
		let synthesisSucceeded = false;
		const maxRetries = config.synthesisMaxRetries ?? 0;
		for (let attempt = 0; attempt <= maxRetries && !synthesisSucceeded; attempt++) {
			try {
				if (attempt > 0) {
					// Recompile with stricter repair instructions
					const repairInstructions = [
						event.customInstructions,
						"CRITICAL RETRY: Your previous response was not valid JSON matching pi-continue-artifacts/v4.",
						"Output ONLY raw JSON. No markdown fences, no explanation, no preamble, no apology.",
						'version MUST be exactly "pi-continue-artifacts/v4".',
						"Include ALL required keys: version, brief(task,done_when,forbid,established,learned,open,next), agentGuideUpdate(content,reason).",
						"Do NOT invent other versions like pi-continue-memento/v1.",
					].filter(Boolean).join(" ");
					historyPrompt = compileHistoryPrompt(
						loadHistoryPromptAssets(
							resolvedProjectContext.projectRoot,
							config.promptOverridePolicy,
							preparation.previousSummary ? "update" : "initial",
						),
						{
							scenario: preparation.previousSummary ? "update" : "initial",
							projectRoot: resolvedProjectContext.projectRoot,
							agentGuidePath: resolvedProjectContext.agentGuidePath,
							existingAgentGuide: resolvedProjectContext.existingAgentGuide,
							previousSummary: preparation.previousSummary,
							historyTranscript: internals.serializeConversation(internals.convertToLlm(messagesToSummarize)),
							turnPrefixTranscript,
							customInstructions: repairInstructions,
							fileOps: fileOpsSnapshot,
						},
					);
					if (ctx.hasUI) {
						ctx.ui.notify(`pi-continue: retrying synthesis (${attempt}/${maxRetries}) with stricter JSON instructions`, "warning");
					}
				}
				const historyOutput = await runPromptPass(pi, ctx, config, historyPrompt, preparation.settings.reserveTokens, event.signal);
				const ownerLostAfterSynthesis = ownerLostResult();
				if (ownerLostAfterSynthesis) return ownerLostAfterSynthesis;
				synthesis = buildContinuationSynthesisTelemetry(historyOutput);
				recordContinuationSynthesisTelemetry(runtime, ownerEventId, synthesis);
				const parsedHistoryArtifacts = parseHistoryArtifacts(historyOutput.text);
				if (!parsedHistoryArtifacts.ok) {
					if (attempt < maxRetries) {
						continue;
					}
					await writeSynthesisFailureArtifact(resolvedProjectContext.projectRoot, {
						pass: "history",
						code: parsedHistoryArtifacts.code,
						requestedModel: historyOutput.requestedModel,
						responseModel: historyOutput.responseModel,
						httpStatus: historyOutput.httpStatus,
						usage: historyOutput.usage,
						outputBudget: historyOutput.outputBudget,
						rawText: historyOutput.text,
					});
					throw new ArtifactParseError({
						kind: "artifact-parse-validation",
						code: parsedHistoryArtifacts.code,
						pass: "history",
						requestedModel: historyOutput.requestedModel,
						httpStatus: historyOutput.httpStatus,
					});
				}
				historyArtifacts = parsedHistoryArtifacts.artifacts;
				markContinuationArtifact(runtime, ownerEventId, "modeled", undefined);
				synthesisSucceeded = true;
			} catch (error) {
				if ((error instanceof ArtifactParseError) && attempt < maxRetries) {
					continue;
				}
				// A newer continuation event already superseded this one (e.g. shutdown
				// abandoned it, or a chained mid-run-guard trigger took over) -- injecting a
				// stale fallback here would fight the event that's actually in charge, so
				// this is the one case that still cancels outright rather than degrading.
				if (!ownerStillActive()) return { cancel: true };
				recordContinuationSynthesisFailure(runtime, ownerEventId, normalizeSynthesisFailure(error));
				markContinuationArtifact(runtime, ownerEventId, "modeled", DEGRADED_CONTINUATION_MESSAGE);
				const fallbackDetails = buildContinuationDetails(
					preparation.fileOps,
					undefined,
					undefined,
					undefined,
					undefined,
					synthesis,
					ownerEventId,
				);
				return {
					compaction: {
						summary: composeCompactionSummary(DEGRADED_CONTINUATION_MESSAGE, fallbackDetails, {
							appendCompactionMetadata: config.appendCompactionMetadata,
							appendReadFileTags: config.appendReadFileTags,
							appendModifiedFileTags: config.appendModifiedFileTags,
						}),
						firstKeptEntryId: preparation.firstKeptEntryId,
						tokensBefore: preparation.tokensBefore,
						details: fallbackDetails,
					},
				};
			}
		}
		if (!historyArtifacts) {
			if (!ownerStillActive()) return { cancel: true };
			recordContinuationSynthesisFailure(runtime, ownerEventId, "artifact-invalid-shape");
			markContinuationArtifact(runtime, ownerEventId, "modeled", DEGRADED_CONTINUATION_MESSAGE);
			const fallbackDetails = buildContinuationDetails(preparation.fileOps, undefined, undefined, undefined, undefined, synthesis, ownerEventId);
			return {
				compaction: {
					summary: composeCompactionSummary(DEGRADED_CONTINUATION_MESSAGE, fallbackDetails, {
						appendCompactionMetadata: config.appendCompactionMetadata,
						appendReadFileTags: config.appendReadFileTags,
						appendModifiedFileTags: config.appendModifiedFileTags,
					}),
					firstKeptEntryId: preparation.firstKeptEntryId,
					tokensBefore: preparation.tokensBefore,
					details: fallbackDetails,
				},
			};
		}
		const continuationArtifactWriteId = config.continuationArtifactMode === "always" ? randomUUID() : undefined;
		if (continuationArtifactWriteId) {
			pendingOutputWrites.set(continuationArtifactWriteId, {
				path: resolvedProjectContext.continuationArtifactPath,
				content: historyArtifacts.briefMarkdown,
				label: "continuation artifact",
				target: "continuation-artifact",
				eventId: ownerEventId,
			});
		}
		const agentGuideWriteStatus = decideAgentGuideWriteStatus(config.agentGuideSyncMode, historyArtifacts.agentGuideMd);
		const agentGuideWriteId = agentGuideWriteStatus === "replacement-pending" ? randomUUID() : undefined;
		if (agentGuideWriteId && historyArtifacts.agentGuideMd) {
			pendingOutputWrites.set(agentGuideWriteId, {
				path: resolvedProjectContext.agentGuidePath,
				content: historyArtifacts.agentGuideMd,
				label: "agent guide",
				target: "agent-guide",
				eventId: ownerEventId,
			});
		}
		planContinuationOutputWrites(
			runtime,
			ownerEventId,
			{
				continuationArtifact: continuationArtifactWriteId ? "pending" : "off",
				agentGuide: agentGuideWriteStatus === "replacement-pending"
					? "pending"
					: agentGuideWriteStatus === "no-replacement"
						? "no-replacement"
						: "off",
			},
		);
		const details = buildContinuationDetails(
			preparation.fileOps,
			continuationArtifactWriteId,
			agentGuideWriteId,
			agentGuideWriteStatus,
			historyArtifacts.agentGuideChangeReason,
			synthesis,
			ownerEventId,
		);
		return {
			compaction: {
				summary: composeCompactionSummary(historyArtifacts.briefMarkdown, details, {
					appendCompactionMetadata: config.appendCompactionMetadata,
					appendReadFileTags: config.appendReadFileTags,
					appendModifiedFileTags: config.appendModifiedFileTags,
				}),
				firstKeptEntryId: preparation.firstKeptEntryId,
				tokensBefore: preparation.tokensBefore,
				details,
			},
		};
	});

	pi.on("session_compact", async (event, ctx) => {
		const activeEventId = getActiveContinuationEventId(runtime);
		const activeCompactionProofVerified = activeEventId !== undefined
			&& runtime.latestEvent?.id === activeEventId
			&& runtime.latestEvent.compactionProof.status === "verified";
		if (activeEventId && !event.fromExtension) {
			if (!activeCompactionProofVerified) {
				failContinuationCompactionProof(ctx, runtime, activeEventId, NATIVE_COMPACTION_FALLBACK_FAILURE);
			}
			return;
		}
		if (!event.fromExtension) return;
		const details = parseContinuationDetails(event.compactionEntry.details);
		if (activeEventId && !details) {
			if (!activeCompactionProofVerified) {
				failContinuationCompactionProof(ctx, runtime, activeEventId, INVALID_COMPACTION_PROOF_FAILURE);
			}
			return;
		}
		if (!details) return;
		if (!details.continuationEventId) {
			if (activeEventId && !activeCompactionProofVerified) {
				failContinuationCompactionProof(ctx, runtime, activeEventId, INVALID_COMPACTION_PROOF_FAILURE);
			}
			return;
		}
		if (activeEventId && details.continuationEventId !== activeEventId) return;
		const acceptedActiveProof = activeEventId !== undefined && details.continuationEventId === activeEventId
			? verifyContinuationCompactionProof(ctx, runtime, activeEventId, event.compactionEntry.id)
			: false;
		const ledgerOwnerId = details.continuationEventId;
		const canUpdateLedger = isActiveRunningContinuationEvent(runtime, ledgerOwnerId);
		const ledger = canUpdateLedger
			? buildLedgerSnapshot(event.compactionEntry.summary, ledgerOwnerId, event.compactionEntry.id)
			: undefined;
		if (ledger) {
			runtime.latestLedger = ledger;
			const projectContext = await resolveProjectContext(pi, ctx.cwd, ctx.sessionManager.getSessionId());
			const config = loadContinuationConfig(projectContext.projectRoot);
			if (config.enabled && config.showAfterCompact) {
				ledgerOverlay.showSoon(ctx, ledger, (reason) => {
					if (ctx.hasUI) ctx.ui.notify(`Could not open Continuation Ledger: ${reason}`, "error");
				});
			}
		}
		for (const writeId of [details.continuationArtifactWriteId, details.agentGuideWriteId]) {
			if (!writeId) continue;
			const pending = pendingOutputWrites.get(writeId);
			pendingOutputWrites.delete(writeId);
			if (!pending) continue;
			if (!isActiveRunningContinuationEvent(runtime, pending.eventId)) continue;
			try {
				const result = await writeNormalizedMarkdownFile(pending.path, pending.content);
				recordOutputWriteResult(runtime, pending.eventId, pending.target, result, undefined);
				if (ctx.hasUI) {
					ctx.ui.notify(
						result === "updated"
							? `Updated ${pending.label}.`
							: `${pending.label} was already up to date.`,
						"info",
					);
				}
			} catch {
				recordOutputWriteResult(runtime, pending.eventId, pending.target, "failed", OUTPUT_WRITE_FAILURE);
				if (ctx.hasUI) ctx.ui.notify(`Could not update ${pending.label}: ${OUTPUT_WRITE_FAILURE}`, "error");
			}
		}
		if (ctx.hasUI && details.agentGuideWriteStatus === "no-replacement" && details.agentGuideChangeReason) {
			ctx.ui.notify("Agent guide unchanged; no full replacement was produced.", "info");
		}
		if (acceptedActiveProof && activeEventId) {
			dispatchVerifiedContinuationResume(ctx, runtime, activeEventId);
		}
	});

	pi.on("session_shutdown", async (_event, ctx) => {
		// ctx.compact() runs detached from session.prompt() (see runtime.ts), so a
		// host like print-mode can call session.dispose() -> abortCompaction()
		// while pi-continue's synthesis request is still in flight, killing it
		// with "This operation was aborted" even though a valid handoff was
		// moments from completing. session_shutdown handlers are awaited by core
		// before it disposes the session, so wait here (bounded, best-effort) for
		// any in-flight compaction to settle on its own before tearing down. The
		// bound must cover the full synthesis call (which can legitimately take
		// close to config.synthesisTimeoutMs on a slow/local model) plus the rest
		// of the compaction pipeline around it, not an arbitrary short guess --
		// otherwise this abandons a continuation that would have completed fine
		// moments later, defeating the same-session auto-resume this waits for.
		if (runtime.activeCompactionSettle) {
			const projectContext = await resolveProjectContext(pi, ctx.cwd, ctx.sessionManager.getSessionId());
			const config = loadContinuationConfig(projectContext.projectRoot);
			const waitMs = config.synthesisTimeoutMs + SESSION_SHUTDOWN_COMPACTION_WAIT_BUFFER_MS;
			// Promise.race doesn't cancel the losing side: if activeCompactionSettle wins,
			// this setTimeout would otherwise keep running (and keep the event loop alive)
			// for the rest of its multi-minute duration regardless.
			let fallbackTimer: ReturnType<typeof setTimeout> | undefined;
			await Promise.race([
				runtime.activeCompactionSettle,
				new Promise<void>((resolve) => {
					fallbackTimer = setTimeout(resolve, waitMs);
				}),
			]);
			clearTimeout(fallbackTimer);
		}
		abandonActiveContinuationEvent(runtime, "Pi session shut down before continuation finished settling.");
		pendingOutputWrites.clear();
		clearWorkingVisuals(ctx, runtime);
		ledgerOverlay.clear();
		runtime.compactionRunning = false;
		runtime.guardFailureKey = undefined;
		runtime.lastNoCompactableGuardKey = undefined;
		clearResumeStartTimeout(runtime);
		clearPendingResumeDispatch(runtime);
		runtime.awaitingResumeEventId = undefined;
	});
}
