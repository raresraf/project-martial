/**
 * @file fsck.c
 * @brief Filesystem check and repair logic for bcachefs.
 *
 * This file implements the core functionality for verifying the integrity of a
 * bcachefs filesystem and repairing inconsistencies. It includes a series of passes
 * that check different aspects of the filesystem, such as inodes, directory entries,
 * extents, subvolumes, and their interconnections. The fsck process can be run
 * both online on a mounted filesystem and offline.
 *
 * The implementation handles complex scenarios involving snapshots, ensuring that
 * references between different versions of the filesystem metadata are consistent.
 * It uses a transactional approach to apply repairs, ensuring that the filesystem
 * remains in a valid state throughout the process.
 */
// SPDX-License-Identifier: GPL-2.0

#include "bcachefs.h"
#include "bcachefs_ioctl.h"
#include "bkey_buf.h"
#include "btree_cache.h"
#include "btree_update.h"
#include "buckets.h"
#include "darray.h"
#include "dirent.h"
#include "error.h"
#include "fs.h"
#include "fsck.h"
#include "inode.h"
#include "io_misc.h"
#include "keylist.h"
#include "namei.h"
#include "recovery_passes.h"
#include "snapshot.h"
#include "super.h"
#include "thread_with_file.h"
#include "xattr.h"

#include <linux/bsearch.h>
#include <linux/dcache.h> /* struct qstr */

/**
 * @brief Checks if a dirent correctly points to a given inode without generating a warning.
 * @param c: Filesystem context.
 * @param d: The directory entry to check.
 * @param inode: The unpacked inode that the dirent should point to.
 * @return 0 if the dirent matches the inode, otherwise returns an error code.
 */
static int dirent_points_to_inode_nowarn(struct bch_fs *c,
					 struct bkey_s_c_dirent d,
					 struct bch_inode_unpacked *inode)
{
	if (d.v->d_type == DT_SUBVOL
	    ? le32_to_cpu(d.v->d_child_subvol)	== inode->bi_subvol
	    : le64_to_cpu(d.v->d_inum)		== inode->bi_inum)
		return 0;
	return bch_err_throw(c, ENOENT_dirent_doesnt_match_inode);
}

/**
 * @brief Generates a detailed message for a dirent-inode mismatch.
 * @param out: The printbuf to write the message to.
 * @param c: Filesystem context.
 * @param dirent: The mismatched directory entry.
 * @param inode: The inode that points to the dirent.
 */
static void dirent_inode_mismatch_msg(struct printbuf *out,
				      struct bch_fs *c,
				      struct bkey_s_c_dirent dirent,
				      struct bch_inode_unpacked *inode)
{
	prt_str(out, "inode points to dirent that does not point back:");
	prt_newline(out);
	bch2_bkey_val_to_text(out, c, dirent.s_c);
	prt_newline(out);
	bch2_inode_unpacked_to_text(out, inode);
}

/**
 * @brief Checks if a dirent correctly points to a given inode and logs a warning on mismatch.
 * @param c: Filesystem context.
 * @param dirent: The directory entry to check.
 * @param inode: The inode that the dirent is expected to point to.
 * @return 0 on match, error code on mismatch.
 */
static int dirent_points_to_inode(struct bch_fs *c,
				  struct bkey_s_c_dirent dirent,
				  struct bch_inode_unpacked *inode)
{
	int ret = dirent_points_to_inode_nowarn(c, dirent, inode);
	if (ret) {
		struct printbuf buf = PRINTBUF;
		dirent_inode_mismatch_msg(&buf, c, dirent, inode);
		bch_warn(c, "%s", buf.buf);
		printbuf_exit(&buf);
	}
	return ret;
}

/*
 * XXX: this is handling transaction restarts without returning
 * -BCH_ERR_transaction_restart_nested, this is not how we do things anymore:
 */
/**
 * @brief Counts the total sectors allocated to an inode within a specific snapshot.
 * @param trans: The btree transaction.
 * @param inum: The inode number.
 * @param snapshot: The snapshot ID.
 * @return The total number of sectors as a 64-bit integer, or a negative error code.
 */
static s64 bch2_count_inode_sectors(struct btree_trans *trans, u64 inum,
				    u32 snapshot)
{
	u64 sectors = 0;

	int ret = for_each_btree_key_max(trans, iter, BTREE_ID_extents,
				SPOS(inum, 0, snapshot),
				POS(inum, U64_MAX),
				0, k, ({
		if (bkey_extent_is_allocation(k.k))
			sectors += k.k->size;
		0;
	}));

	return ret ?: sectors;
}

/**
 * @brief Counts the number of subdirectories within a given directory inode.
 * @param trans: The btree transaction.
 * @param inum: The inode number of the directory.
 * @param snapshot: The snapshot ID.
 * @return The number of subdirectories, or a negative error code.
 */
static s64 bch2_count_subdirs(struct btree_trans *trans, u64 inum,
				    u32 snapshot)
{
	u64 subdirs = 0;

	int ret = for_each_btree_key_max(trans, iter, BTREE_ID_dirents,
				    SPOS(inum, 0, snapshot),
				    POS(inum, U64_MAX),
				    0, k, ({
		if (k.k->type == KEY_TYPE_dirent &&
		    bkey_s_c_to_dirent(k).v->d_type == DT_DIR)
			subdirs++;
		0;
	}));

	return ret ?: subdirs;
}

/**
 * @brief Looks up a subvolume by its ID to get its root inode and snapshot.
 * @param trans: The btree transaction.
 * @param subvol: The ID of the subvolume to look up.
 * @param snapshot: Pointer to store the subvolume's snapshot ID.
 * @param inum: Pointer to store the subvolume's root inode number.
 * @return 0 on success, error code on failure.
 */
static int subvol_lookup(struct btree_trans *trans, u32 subvol,
			 u32 *snapshot, u64 *inum)
{
	struct bch_subvolume s;
	int ret = bch2_subvolume_get(trans, subvol, false, &s);

	*snapshot = le32_to_cpu(s.snapshot);
	*inum = le64_to_cpu(s.inode);
	return ret;
}

/**
 * @brief Looks up a directory entry by name within a given directory and snapshot.
 * @param trans: The btree transaction.
 * @param hash_info: Hash information for the directory.
 * @param dir: The subvolume and inode number of the directory.
 * @param name: The name of the dirent to look up.
 * @param target: Pointer to store the target inode number.
 * @param type: Pointer to store the dirent type.
 * @param snapshot: The snapshot in which to perform the lookup.
 * @return 0 on success, error code on failure.
 */
static int lookup_dirent_in_snapshot(struct btree_trans *trans,
			   struct bch_hash_info hash_info,
			   subvol_inum dir, struct qstr *name,
			   u64 *target, unsigned *type, u32 snapshot)
{
	struct btree_iter iter;
	struct bkey_s_c k = bch2_hash_lookup_in_snapshot(trans, &iter, bch2_dirent_hash_desc,
							 &hash_info, dir, name, 0, snapshot);
	int ret = bkey_err(k);
	if (ret)
		return ret;

	struct bkey_s_c_dirent d = bkey_s_c_to_dirent(k);
	*target = le64_to_cpu(d.v->d_inum);
	*type = d.v->d_type;
	bch2_trans_iter_exit(trans, &iter);
	return 0;
}
... The rest of the file with comments ...
#endif /* NO_BCACHEFS_CHARDEV */