/**
 * Document attachments (PDFs) on user messages.
 *
 * Flue's message content has one binary carrier, `ImageContent`
 * (`{ type: 'image', data, mimeType }`). Flue carries a document through its
 * messages as an `ImageContent` whose `mimeType` is a supported document
 * type, and `toModelRequest` turns it into a TanStack `document` part for
 * each model request.
 *
 * Everything durable is already media-type agnostic (`AttachmentRef` carries
 * `mimeType`; the attachment store, the `file` UI part, and the hosted
 * attachment route never look at the media type), so documents need no
 * canonical-record change: a PDF is an attachment whose `mimeType` is
 * `application/pdf`.
 *
 * Native document input, by model API (the TanStack adapters send the part):
 * - `anthropic-messages`: a `document` block.
 * - `openai-responses` / `azure-openai-responses`: an `input_file` part.
 * - `google-generative-ai` / `google-vertex`: `inlineData`.
 *
 * Every other model API gets a text placeholder instead, as for images sent
 * to a model that cannot read them.
 */
import type { ImageContent } from './llm-types.ts';
import type { PromptDocument, PromptImage } from './types.ts';

/** Document MIME types accepted on attachments and operation `documents`. */
export const DOCUMENT_MIME_TYPES = ['application/pdf'] as const;

const documentMimeTypes: ReadonlySet<string> = new Set(DOCUMENT_MIME_TYPES);

/** Model APIs that receive documents as native document content. */
export const NATIVE_DOCUMENT_APIS: ReadonlySet<string> = new Set([
	'anthropic-messages',
	'openai-responses',
	'azure-openai-responses',
	'google-generative-ai',
	'google-vertex',
]);

/**
 * A document as it rides Flue's model context: the `ImageContent` carrier
 * with a document `mimeType`, plus the uploader's filename when known
 * (`toModelRequest` reads it for the document part).
 */
export type DocumentContextBlock = ImageContent & { filename?: string };

export function isDocumentMimeType(mimeType: string): boolean {
	return documentMimeTypes.has(mimeType);
}

/** True for a context block that carries a document rather than an image. */
export function isDocumentContextBlock(block: unknown): block is DocumentContextBlock {
	if (!block || typeof block !== 'object') return false;
	const candidate = block as {
		type?: unknown;
		mimeType?: unknown;
		data?: unknown;
	};
	return (
		candidate.type === 'image' &&
		typeof candidate.data === 'string' &&
		typeof candidate.mimeType === 'string' &&
		isDocumentMimeType(candidate.mimeType)
	);
}

/** Reject unsupported document MIME types on the in-code operation surface. */
export function assertSupportedDocuments(documents: readonly PromptDocument[] | undefined): void {
	for (const document of documents ?? []) {
		if (!isDocumentMimeType(document.mimeType)) {
			throw new Error(
				`[flue] Unsupported document mimeType "${document.mimeType}". ` +
					`Supported: ${DOCUMENT_MIME_TYPES.join(', ')}.`,
			);
		}
	}
}

/**
 * Fold an operation's `images` and `documents` into the single binary list
 * the session plumbing carries (documents ride the `ImageContent` carrier —
 * see the module doc). Returns `images` unchanged when there are no documents.
 */
export function mergeOperationAttachments(
	images: readonly PromptImage[] | undefined,
	documents: readonly PromptDocument[] | undefined,
): PromptImage[] | undefined {
	assertSupportedDocuments(documents);
	if (!documents?.length) return images as PromptImage[] | undefined;
	return [
		...(images ?? []),
		...documents.map((document) => ({
			type: 'image' as const,
			data: document.data,
			mimeType: document.mimeType,
		})),
	];
}

/**
 * Convert a carrier block back to its public shape — a document block when
 * its MIME type is a document type, else the image unchanged. Used where the
 * runtime reconstructs a `DeliveredMessage` from the model context or the
 * attachment store (task delegation, delivery-cursor restore).
 */
export function toPublicAttachment<T extends { mimeType: string; data: string }>(
	block: T,
): (T & { type: 'image' }) | (T & { type: 'document' }) {
	return isDocumentMimeType(block.mimeType)
		? { ...block, type: 'document' as const }
		: { ...block, type: 'image' as const };
}

/** Text that replaces a document for a model API without native document input. */
export function documentOmittedPlaceholder(api: string, filename?: string): string {
	const name = filename ? ` "${filename}"` : '';
	return `(document${name} omitted: model API "${api}" does not support document input)`;
}

const warnedApis = new Set<string>();

/** Warn once per model API that its documents become placeholders. */
export function warnDocumentsOmitted(api: string): void {
	if (warnedApis.has(api)) return;
	warnedApis.add(api);
	console.warn(
		`[flue] Model API "${api}" does not support native document input; ` +
			'document attachments are replaced with a text placeholder in the model context. ' +
			`Native document input is supported on: ${[...NATIVE_DOCUMENT_APIS].join(', ')}.`,
	);
}
