import { existsSync, mkdirSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { Agent } from "@earendil-works/pi-agent-core";
import {
	type AssistantMessage,
	createAssistantMessageEventStream,
	getModel,
	type Model,
} from "@earendil-works/pi-ai/compat";
import { afterEach, describe, expect, it, vi } from "vitest";
import { AgentSession } from "../src/core/agent-session.ts";
import type { AgentSessionRuntime } from "../src/core/agent-session-runtime.ts";
import { AuthStorage } from "../src/core/auth-storage.ts";
import type { LoadExtensionsResult } from "../src/core/extensions/index.ts";
import { SessionManager } from "../src/core/session-manager.ts";
import { SettingsManager } from "../src/core/settings-manager.ts";
import { runRpcMode } from "../src/modes/rpc/rpc-mode.ts";
import { createInMemoryModelRegistry, getModelRuntime } from "./model-runtime-test-utils.ts";
import { createTestExtensionsResult, createTestResourceLoader } from "./utilities.ts";

const rpcIo = vi.hoisted(() => ({
	outputLines: [] as string[],
	lineHandler: undefined as ((line: string) => void) | undefined,
}));

vi.mock("../src/core/output-guard.js", () => ({
	flushRawStdout: vi.fn(async () => {}),
	takeOverStdout: vi.fn(),
	waitForRawStdoutBackpressure: vi.fn(async () => {}),
	writeRawStdout: (line: string) => {
		rpcIo.outputLines.push(line);
	},
}));

vi.mock("../src/modes/interactive/theme/theme.js", () => ({ theme: {} }));

vi.mock("../src/modes/rpc/jsonl.js", () => ({
	attachJsonlLineReader: vi.fn((_stream: NodeJS.ReadableStream, onLine: (line: string) => void) => {
		rpcIo.lineHandler = onLine;
		return () => {};
	}),
	serializeJsonLine: (value: unknown) => `${JSON.stringify(value)}\n`,
}));

function createAssistantMessage(text: string): AssistantMessage {
	return {
		role: "assistant",
		content: [{ type: "text", text }],
		api: "anthropic-messages",
		provider: "anthropic",
		model: "claude-sonnet-4-5",
		usage: {
			input: 0,
			output: 0,
			cacheRead: 0,
			cacheWrite: 0,
			totalTokens: 0,
			cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
		},
		stopReason: "stop",
		timestamp: Date.now(),
	};
}

type ParsedOutputLine = Record<string, unknown>;

function parseOutputLines(outputLines: string[]): ParsedOutputLine[] {
	return outputLines
		.flatMap((line) => line.split("\n"))
		.filter((line) => line.trim().length > 0)
		.map((line) => JSON.parse(line) as ParsedOutputLine);
}

function getPromptResponses(outputLines: string[], id: string, command = "prompt"): ParsedOutputLine[] {
	return parseOutputLines(outputLines).filter(
		(record) => record.id === id && record.type === "response" && record.command === command,
	);
}

function sleep(ms: number): Promise<void> {
	return new Promise((resolve) => setTimeout(resolve, ms));
}

async function createRuntimeHost(options: {
	withAuth: boolean;
	responseDelayMs: number;
	model?: Model<any>;
	extensionsResult?: LoadExtensionsResult;
	sessionManager?: SessionManager;
	responseGate?: Promise<void>;
}): Promise<{
	runtimeHost: AgentSessionRuntime;
	cleanup: () => Promise<void>;
}> {
	const tempDir = join(tmpdir(), `pi-rpc-prompt-${Date.now()}-${Math.random().toString(36).slice(2)}`);
	mkdirSync(tempDir, { recursive: true });

	const model = options.model ?? getModel("anthropic", "claude-sonnet-4-5");
	if (!model) {
		throw new Error("Test model not found");
	}

	const agent = new Agent({
		getApiKey: () => "test-key",
		initialState: {
			model,
			systemPrompt: "Test",
			tools: [],
		},
		streamFn: (_model, _context, _options) => {
			const stream = createAssistantMessageEventStream();
			queueMicrotask(() => {
				stream.push({ type: "start", partial: createAssistantMessage("") });
				setTimeout(() => {
					void (options.responseGate ?? Promise.resolve()).then(() =>
						stream.push({ type: "done", reason: "stop", message: createAssistantMessage("done") }),
					);
				}, options.responseDelayMs);
			});
			return stream;
		},
	});

	const sessionManager = options.sessionManager ?? SessionManager.inMemory();
	const settingsManager = SettingsManager.create(tempDir, tempDir);
	const authStorage = AuthStorage.create(join(tempDir, "auth.json"));
	const modelRegistry = await createInMemoryModelRegistry(authStorage);
	if (options.withAuth) {
		await authStorage.modify("anthropic", async () => ({ type: "api_key", key: "test-key" }));
	}

	const session = new AgentSession({
		agent,
		sessionManager,
		settingsManager,
		cwd: tempDir,
		modelRuntime: getModelRuntime(modelRegistry),
		resourceLoader: createTestResourceLoader({ extensionsResult: options.extensionsResult }),
	});

	const runtimeHost = {
		session,
		newSession: vi.fn(async () => ({ cancelled: true })),
		switchSession: vi.fn(async () => ({ cancelled: true })),
		fork: vi.fn(async () => ({ cancelled: true, selectedText: "" })),
		dispose: vi.fn(async () => {}),
		setRebindSession: vi.fn(),
	} as unknown as AgentSessionRuntime;

	return {
		runtimeHost,
		cleanup: async () => {
			try {
				await session.abort();
			} catch {
				// ignore test cleanup failures
			}
			session.dispose();
			if (existsSync(tempDir)) {
				rmSync(tempDir, { recursive: true });
			}
		},
	};
}

async function startRpcMode(options: Parameters<typeof createRuntimeHost>[0]): Promise<{
	lineHandler: (line: string) => void;
	runtimeHost: AgentSessionRuntime;
	cleanup: () => Promise<void>;
}> {
	rpcIo.outputLines = [];
	rpcIo.lineHandler = undefined;

	const { runtimeHost, cleanup } = await createRuntimeHost(options);
	const signals = new Map(
		(["SIGTERM", "SIGHUP"] as const).map((signal) => [signal, new Set(process.listeners(signal))] as const),
	);
	const stdinEndListeners = new Set(process.stdin.listeners("end"));
	void runRpcMode(runtimeHost);
	await vi.waitFor(() => expect(rpcIo.lineHandler).toBeDefined());

	return {
		lineHandler: rpcIo.lineHandler!,
		runtimeHost,
		cleanup: async () => {
			for (const [signal, existing] of signals) {
				for (const listener of process.listeners(signal)) {
					if (!existing.has(listener)) process.removeListener(signal, listener);
				}
			}
			for (const listener of process.stdin.listeners("end")) {
				if (!stdinEndListeners.has(listener)) process.stdin.removeListener("end", listener as () => void);
			}
			await cleanup();
		},
	};
}

describe("RPC prompt response semantics", () => {
	afterEach(() => {
		rpcIo.outputLines = [];
		rpcIo.lineHandler = undefined;
	});

	it("records a keyed user entry once before replying and reuses it after reopening", async () => {
		const sessionDir = join(tmpdir(), `pi-rpc-keyed-${crypto.randomUUID()}`);
		mkdirSync(sessionDir);
		const sessionManager = SessionManager.create(sessionDir, sessionDir);
		const file = sessionManager.getSessionFile()!;
		let release!: () => void;
		const gate = new Promise<void>((resolve) => {
			release = resolve;
		});
		const clientContent = JSON.stringify([{ type: "resource_link", name: "fixture", uri: "fixture://original" }]);
		const request = (id: string, message = "Hello", content = clientContent) =>
			JSON.stringify({ id, type: "prompt_with_id", clientMessageId: "one", message, clientContent: content });
		const receipts = (id: string) => getPromptResponses(rpcIo.outputLines, id, "prompt_with_id");
		let entryId!: string;
		try {
			const first = await startRpcMode({ withAuth: true, responseDelayMs: 0, responseGate: gate, sessionManager });
			try {
				first.lineHandler(request("original"));
				first.lineHandler(request("duplicate", "Do not insert this", "changed client content"));
				await vi.waitFor(() => {
					expect(receipts("original")).toHaveLength(1);
					expect(receipts("duplicate")).toHaveLength(1);
				});
				const entries = SessionManager.open(file).getEntries();
				const users = entries.filter((entry) => entry.type === "message" && entry.message.role === "user");
				expect(users).toHaveLength(1);
				expect(users[0]).toMatchObject({
					message: { content: [{ type: "text", text: "Hello" }] },
					clientSubmission: { sessionId: sessionManager.getSessionId(), messageId: "one", clientContent },
				});
				entryId = users[0]!.id;
				expect(receipts("original")[0]).toMatchObject({
					success: true,
					data: { clientMessageId: "one", entryId, disposition: "inserted", clientContent },
				});
				expect(receipts("duplicate")[0]).toMatchObject({
					success: true,
					data: { clientMessageId: "one", entryId, disposition: "reused", clientContent },
				});
				expect(parseOutputLines(rpcIo.outputLines).some((record) => record.type === "agent_settled")).toBe(false);
				release();
				await vi.waitFor(() =>
					expect(parseOutputLines(rpcIo.outputLines).some((record) => record.type === "agent_settled")).toBe(true),
				);
			} finally {
				release();
				await first.cleanup();
			}
			const reopened = await startRpcMode({
				withAuth: false,
				responseDelayMs: 0,
				sessionManager: SessionManager.open(file),
			});
			try {
				reopened.lineHandler(request("retry", "Do not insert this either", "changed again"));
				await vi.waitFor(() =>
					expect(receipts("retry")).toEqual([
						expect.objectContaining({
							success: true,
							data: { clientMessageId: "one", entryId, disposition: "reused", clientContent },
						}),
					]),
				);
				expect(parseOutputLines(rpcIo.outputLines).some((record) => record.type === "agent_start")).toBe(false);
			} finally {
				await reopened.cleanup();
			}
		} finally {
			rmSync(sessionDir, { recursive: true, force: true });
		}
	});

	it("limits keyed steering to the native run and retains its committed receipt", async () => {
		const directory = join(tmpdir(), `pi-steering-keyed-${crypto.randomUUID()}`);
		mkdirSync(directory);
		const sessionManager = SessionManager.create(directory, directory);
		const file = sessionManager.getSessionFile()!;
		const gate = Promise.withResolvers<void>();
		const request = (id: string, clientMessageId = "steer-one") =>
			JSON.stringify({ id, type: "steer_with_id", clientMessageId, message: "Change direction" });
		const replies = (id: string) => getPromptResponses(rpcIo.outputLines, id, "steer_with_id");
		const first = await startRpcMode({
			withAuth: true,
			responseDelayMs: 0,
			responseGate: gate.promise,
			sessionManager,
		});
		try {
			first.lineHandler(request("idle"));
			await vi.waitFor(() => expect(replies("idle")[0]).toMatchObject({ success: true, data: { receipt: null } }));
			expect(parseOutputLines(rpcIo.outputLines).filter((record) => record.type === "agent_start")).toHaveLength(0);

			first.lineHandler(JSON.stringify({ id: "prompt", type: "prompt", message: "Start" }));
			await vi.waitFor(() =>
				expect(parseOutputLines(rpcIo.outputLines).filter((record) => record.type === "agent_start")).toHaveLength(
					1,
				),
			);
			first.lineHandler(request("active"));
			first.lineHandler(request("duplicate"));
			await vi.waitFor(() => expect(first.runtimeHost.session.agent.hasQueuedMessages()).toBe(true));
			expect(replies("active")).toHaveLength(0);
			gate.resolve();
			await vi.waitFor(() => {
				expect(replies("active")[0]).toMatchObject({
					success: true,
					data: { receipt: { disposition: "inserted" } },
				});
				expect(replies("duplicate")[0]).toMatchObject({
					success: true,
					data: { receipt: { disposition: "reused" } },
				});
			});
			await vi.waitFor(() =>
				expect(
					parseOutputLines(rpcIo.outputLines).filter((record) => record.type === "agent_settled"),
				).toHaveLength(1),
			);
			const rows = SessionManager.open(file)
				.getEntries()
				.filter(
					(entry) =>
						entry.type === "message" &&
						entry.message.role === "user" &&
						entry.clientSubmission?.messageId === "steer-one",
				);
			expect(rows).toHaveLength(1);
			expect(rows[0]).toMatchObject({ message: { content: [{ type: "text", text: "Change direction" }] } });
			await first.cleanup();

			const reopened = await startRpcMode({
				withAuth: false,
				responseDelayMs: 0,
				sessionManager: SessionManager.open(file),
			});
			try {
				reopened.lineHandler(request("cold"));
				await vi.waitFor(() =>
					expect(replies("cold")[0]).toMatchObject({
						success: true,
						data: { receipt: { entryId: rows[0]!.id, disposition: "reused" } },
					}),
				);
				expect(parseOutputLines(rpcIo.outputLines).filter((record) => record.type === "agent_start")).toHaveLength(
					0,
				);
			} finally {
				await reopened.cleanup();
			}
		} finally {
			gate.resolve();
			await first.cleanup();
			rmSync(directory, { recursive: true, force: true });
		}
	});

	it("emits one failure response when prompt preflight rejects", async () => {
		const { lineHandler, cleanup } = await startRpcMode({
			withAuth: false,
			responseDelayMs: 0,
			model: {
				id: "fake-model",
				name: "Fake Model",
				api: "openai-completions",
				provider: "fake-provider",
				baseUrl: "https://example.invalid",
				reasoning: false,
				input: [],
				cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
				contextWindow: 0,
				maxTokens: 0,
			},
		});

		try {
			lineHandler(JSON.stringify({ id: "b1", type: "prompt", message: "Hello" }));

			await vi.waitFor(() => {
				const responses = getPromptResponses(rpcIo.outputLines, "b1");
				expect(responses).toHaveLength(1);
				expect(responses[0]).toMatchObject({
					id: "b1",
					type: "response",
					command: "prompt",
					success: false,
					error: expect.stringContaining(
						"No API key found for fake-provider.\n\nUse /login to log into a provider via OAuth or API key. See:",
					),
				});
			});
		} finally {
			await cleanup();
		}
	});

	// #9098: a successful prompt may start an agent run or be consumed by an extension.
	it("emits one started response when prompt preflight succeeds", async () => {
		const { lineHandler, cleanup } = await startRpcMode({ withAuth: true, responseDelayMs: 0 });

		try {
			lineHandler(JSON.stringify({ id: "b2", type: "prompt", message: "Hello" }));

			await vi.waitFor(() => {
				const responses = getPromptResponses(rpcIo.outputLines, "b2");
				expect(responses).toHaveLength(1);
				expect(responses[0]).toMatchObject({
					id: "b2",
					type: "response",
					command: "prompt",
					success: true,
					data: { disposition: "started" },
				});
			});
		} finally {
			await cleanup();
		}
	});

	it("reports extension commands and intercepted input as handled without starting a run", async () => {
		const { lineHandler, cleanup } = await startRpcMode({
			withAuth: false,
			responseDelayMs: 0,
			extensionsResult: await createTestExtensionsResult([
				(pi) => {
					pi.registerCommand("handled", { handler: async () => {} });
					pi.on("input", (event) => {
						if (event.text === "handled input") return { action: "handled" };
					});
				},
			]),
		});

		try {
			for (const [id, message] of [
				["command", "/handled"],
				["input", "handled input"],
			]) {
				lineHandler(JSON.stringify({ id, type: "prompt", message }));
				await vi.waitFor(() => {
					expect(getPromptResponses(rpcIo.outputLines, id)).toEqual([
						{ id, type: "response", command: "prompt", success: true, data: { disposition: "handled" } },
					]);
				});
			}
			expect(parseOutputLines(rpcIo.outputLines).filter((line) => line.type === "agent_start")).toHaveLength(0);
		} finally {
			await cleanup();
		}
	});

	it("emits one success response when prompt is queued during streaming", async () => {
		const { lineHandler, cleanup } = await startRpcMode({ withAuth: true, responseDelayMs: 100 });

		try {
			lineHandler(JSON.stringify({ id: "b3-start", type: "prompt", message: "Start" }));
			await vi.waitFor(() => {
				expect(getPromptResponses(rpcIo.outputLines, "b3-start")).toHaveLength(1);
			});

			rpcIo.outputLines = [];
			lineHandler(
				JSON.stringify({
					id: "b3",
					type: "prompt",
					message: "Queue this",
					streamingBehavior: "followUp",
				}),
			);

			await vi.waitFor(() => {
				const responses = getPromptResponses(rpcIo.outputLines, "b3");
				expect(responses).toHaveLength(1);
				expect(responses[0]).toMatchObject({
					id: "b3",
					type: "response",
					command: "prompt",
					success: true,
					data: { disposition: "queued" },
				});
			});

			await sleep(150);
		} finally {
			await cleanup();
		}
	});

	// #9803: a handler can consume A while independently queueing B; report A's outcome.
	it.each(["steer", "follow_up"] as const)(
		"reports %s as handled even when an extension queues another message",
		async (type) => {
			const { lineHandler, cleanup } = await startRpcMode({
				withAuth: true,
				responseDelayMs: 500,
				extensionsResult: await createTestExtensionsResult([
					(pi) => {
						pi.on("input", (event) => {
							if (event.text === "A" && event.source === "rpc") {
								pi.sendUserMessage("B", { deliverAs: type === "steer" ? "steer" : "followUp" });
								return { action: "handled" };
							}
						});
					},
				]),
			});

			try {
				lineHandler(JSON.stringify({ id: "start", type: "prompt", message: "Start" }));
				await vi.waitFor(() => expect(getPromptResponses(rpcIo.outputLines, "start")).toHaveLength(1));

				lineHandler(JSON.stringify({ id: "A", type, message: "A" }));
				await vi.waitFor(() => {
					expect(parseOutputLines(rpcIo.outputLines)).toContainEqual({
						id: "A",
						type: "response",
						command: type,
						success: true,
						data: { disposition: "handled" },
					});
					expect(parseOutputLines(rpcIo.outputLines)).toContainEqual({
						type: "queue_update",
						steering: type === "steer" ? ["B"] : [],
						followUp: type === "follow_up" ? ["B"] : [],
					});
				});
				await vi.waitFor(() => {
					expect(
						parseOutputLines(rpcIo.outputLines).filter((line) => line.type === "response" && line.id === "A"),
					).toHaveLength(1);
				});
			} finally {
				await cleanup();
			}
		},
	);

	it.each(["steer", "follow_up"] as const)("reports %s as queued after input transformation", async (type) => {
		const { lineHandler, cleanup } = await startRpcMode({
			withAuth: false,
			responseDelayMs: 0,
			extensionsResult: await createTestExtensionsResult([
				(pi) => {
					pi.on("input", (event) => {
						if (event.text === "A") return { action: "transform", text: "B" };
					});
				},
			]),
		});

		try {
			lineHandler(JSON.stringify({ id: "A", type, message: "A" }));
			await vi.waitFor(() => {
				expect(parseOutputLines(rpcIo.outputLines)).toContainEqual({
					id: "A",
					type: "response",
					command: type,
					success: true,
					data: { disposition: "queued" },
				});
				expect(parseOutputLines(rpcIo.outputLines)).toContainEqual({
					type: "queue_update",
					steering: type === "steer" ? ["B"] : [],
					followUp: type === "follow_up" ? ["B"] : [],
				});
			});
		} finally {
			await cleanup();
		}
	});

	it("reports an opted-in extension confirmation withdrawal with the original dialog ID", async () => {
		const controller = new AbortController();
		const result = vi.fn();
		const { lineHandler, cleanup } = await startRpcMode({
			withAuth: false,
			responseDelayMs: 0,
			extensionsResult: await createTestExtensionsResult([
				(pi) => {
					pi.on("input", async (_event, ctx) => {
						result(await ctx.ui.confirm("Confirm", "Test", { signal: controller.signal }));
						return { action: "handled" };
					});
				},
			]),
		});

		vi.stubEnv("PI_RPC_EXTENSION_UI_WITHDRAWAL", "1");
		try {
			lineHandler(JSON.stringify({ id: "prompt-id", type: "prompt", message: "Test" }));
			const dialog = await vi.waitFor(() => {
				const request = parseOutputLines(rpcIo.outputLines).find((line) => line.type === "extension_ui_request");
				expect(request).toMatchObject({ id: expect.any(String), method: "confirm" });
				return request!;
			});
			controller.abort();
			lineHandler(JSON.stringify({ type: "extension_ui_response", id: dialog.id, confirmed: true }));
			await vi.waitFor(() => expect(result.mock.calls).toEqual([[false]]));
			expect(parseOutputLines(rpcIo.outputLines).filter((line) => line.type === "extension_ui_cancelled")).toEqual([
				{ type: "extension_ui_cancelled", id: dialog.id },
			]);
		} finally {
			vi.unstubAllEnvs();
			controller.abort();
			await cleanup();
		}
	});

	it("returns and clears queued steering and follow-up messages", async () => {
		const { lineHandler, cleanup } = await startRpcMode({ withAuth: true, responseDelayMs: 500 });

		try {
			lineHandler(JSON.stringify({ id: "clear-start", type: "prompt", message: "Start" }));
			await vi.waitFor(() => {
				expect(getPromptResponses(rpcIo.outputLines, "clear-start")).toHaveLength(1);
			});

			lineHandler(
				JSON.stringify({
					id: "clear-steering",
					type: "prompt",
					message: "Change direction",
					streamingBehavior: "steer",
				}),
			);
			await vi.waitFor(() => {
				expect(getPromptResponses(rpcIo.outputLines, "clear-steering")).toMatchObject([
					{ data: { disposition: "queued" } },
				]);
			});

			lineHandler(
				JSON.stringify({
					id: "clear-follow-up",
					type: "prompt",
					message: "Summarize when finished",
					streamingBehavior: "followUp",
				}),
			);
			await vi.waitFor(() => {
				expect(getPromptResponses(rpcIo.outputLines, "clear-follow-up")).toHaveLength(1);
			});

			lineHandler(JSON.stringify({ id: "clear", type: "clear_queue" }));
			await vi.waitFor(() => {
				expect(parseOutputLines(rpcIo.outputLines)).toContainEqual({
					id: "clear",
					type: "response",
					command: "clear_queue",
					success: true,
					data: {
						steering: ["Change direction"],
						followUp: ["Summarize when finished"],
					},
				});
			});

			await sleep(600);
			expect(parseOutputLines(rpcIo.outputLines).filter((record) => record.type === "agent_start")).toHaveLength(1);
		} finally {
			await cleanup();
		}
	});
});
