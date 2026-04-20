// SPDX-License-Identifier: GPL-2.0
/*
 * @7fce48e6-7c3f-4a41-8f2e-7734c3154e24/samples/fanotify/fs-monitor.c
 * @brief Simple filesystem monitor utilizing the fanotify API for error reporting.
 *
 * Functional Intent: Demonstrates the use of the FAN_FS_ERROR mask to capture 
 * and decode filesystem-level error notifications. It showcases how to retrieve 
 * file handles (FID) and error metadata (error code, error count) from fanotify events.
 *
 * Copyright 2021, Collabora Ltd.
 */

#define _GNU_SOURCE
#include <errno.h>
#include <err.h>
#include <stdlib.h>
#include <stdio.h>
#include <fcntl.h>
#include <sys/fanotify.h>
#include <sys/types.h>
#include <unistd.h>
#ifndef __GLIBC__
#include <asm-generic/int-ll64.h>
#endif

#ifndef FAN_FS_ERROR
#define FAN_FS_ERROR		0x00008000
#define FAN_EVENT_INFO_TYPE_ERROR	5

/**
 * struct fanotify_event_info_error - Metadata for filesystem error events.
 * @hdr: Standard fanotify event information header.
 * @error: The specific errno reported by the filesystem.
 * @error_count: Number of errors that occurred since the last notification.
 */
struct fanotify_event_info_error {
	struct fanotify_event_info_header hdr;
	__s32 error;
	__u32 error_count;
};
#endif

#ifndef FILEID_INO32_GEN
#define FILEID_INO32_GEN	1
#endif

#ifndef FILEID_INVALID
#define	FILEID_INVALID		0xff
#endif

/**
 * print_fh - Decodes and prints a file handle retrieved from a fanotify event.
 * @fh: Pointer to the file_handle structure.
 * 
 * Functional Utility: Extracts inode and generation numbers if the handle type 
 * is standard (FILEID_INO32_GEN), or identifies superblock/unknown errors.
 */
static void print_fh(struct file_handle *fh)
{
	int i;
	uint32_t *h = (uint32_t *) fh->f_handle;

	printf("\tfh: ");
	for (i = 0; i < fh->handle_bytes; i++)
		printf("%hhx", fh->f_handle[i]);
	printf("\n");

	printf("\tdecoded fh: ");
	if (fh->handle_type == FILEID_INO32_GEN)
		printf("inode=%u gen=%u\n", h[0], h[1]);
	else if (fh->handle_type == FILEID_INVALID && !fh->handle_bytes)
		printf("Type %d (Superblock error)\n", fh->handle_type);
	else
		printf("Type %d (Unknown)\n", fh->handle_type);

}

/**
 * handle_notifications - Processes a batch of fanotify events from the read buffer.
 * @buffer: The raw data buffer containing event metadata and info records.
 * @len: The length of the data in the buffer.
 * 
 * Block Logic: Iterates through the event stream, validating event masks and 
 * parsing nested information records (Error info and FID/File Handle info).
 */
static void handle_notifications(char *buffer, int len)
{
	struct fanotify_event_metadata *event =
		(struct fanotify_event_metadata *) buffer;
	struct fanotify_event_info_header *info;
	struct fanotify_event_info_error *err;
	struct fanotify_event_info_fid *fid;
	int off;

	// Invariant: FAN_EVENT_OK ensures we don't read beyond the available buffer.
	for (; FAN_EVENT_OK(event, len); event = FAN_EVENT_NEXT(event, len)) {

		if (event->mask != FAN_FS_ERROR) {
			printf("unexpected FAN MARK: %llx\n",
							(unsigned long long)event->mask);
			goto next_event;
		}

		if (event->fd != FAN_NOFD) {
			printf("Unexpected fd (!= FAN_NOFD)\n");
			goto next_event;
		}

		printf("FAN_FS_ERROR (len=%d)\n", event->event_len);

		// Block Logic: Parsing variable-length information records following the metadata.
		for (off = sizeof(*event) ; off < event->event_len;
		     off += info->len) {
			info = (struct fanotify_event_info_header *)
				((char *) event + off);

			switch (info->info_type) {
			case FAN_EVENT_INFO_TYPE_ERROR:
				err = (struct fanotify_event_info_error *) info;

				printf("\tGeneric Error Record: len=%d\n",
				       err->hdr.len);
				printf("\terror: %d\n", err->error);
				printf("\terror_count: %d\n", err->error_count);
				break;

			case FAN_EVENT_INFO_TYPE_FID:
				fid = (struct fanotify_event_info_fid *) info;

				printf("\tfsid: %x%x\n",
#if defined(__GLIBC__)
				       fid->fsid.val[0], fid->fsid.val[1]);
#else
				       fid->fsid.__val[0], fid->fsid.__val[1]);
#endif
				print_fh((struct file_handle *) &fid->handle);
				break;

			default:
				printf("\tUnknown info type=%d len=%d:\n",
				       info->info_type, info->len);
			}
		}
next_event:
		printf("---\n\n");
	}
}

/**
 * main - Entry point for the filesystem monitor.
 * @argc: Argument count.
 * @argv: Argument vector, expecting a mount point or directory path.
 * 
 * Functional Intent: Initializes fanotify, sets a filesystem-level mark 
 * for error monitoring, and enters an infinite event processing loop.
 */
int main(int argc, char **argv)
{
	int fd;

	char buffer[BUFSIZ];

	if (argc < 2) {
		printf("Missing path argument\n");
		return 1;
	}

	// Logic: Initialize fanotify with notification class and FID reporting enabled.
	fd = fanotify_init(FAN_CLASS_NOTIF|FAN_REPORT_FID, O_RDONLY);
	if (fd < 0)
		errx(1, "fanotify_init");

	// Logic: Attach a filesystem mark to the specified path to listen for FAN_FS_ERROR.
	if (fanotify_mark(fd, FAN_MARK_ADD|FAN_MARK_FILESYSTEM,
			  FAN_FS_ERROR, AT_FDCWD, argv[1])) {
		errx(1, "fanotify_mark");
	}

	// Block Logic: Event retrieval loop.
	// Invariant: Blocks until at least one event is available in the fanotify descriptor.
	while (1) {
		int n = read(fd, buffer, BUFSIZ);

		if (n < 0)
			errx(1, "read");

		handle_notifications(buffer, n);
	}

	return 0;
}
