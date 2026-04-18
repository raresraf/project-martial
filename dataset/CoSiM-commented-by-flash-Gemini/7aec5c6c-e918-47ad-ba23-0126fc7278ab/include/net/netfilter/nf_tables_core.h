/**
 * @7aec5c6c-e918-47ad-ba23-0126fc7278ab/include/net/netfilter/nf_tables_core.h
 * @brief Core internal declarations and optimized expression types for nftables.
 * Domain: Kernel Networking, Packet Filtering.
 * Architecture: Defines high-performance "fast" variants of common expressions (cmp, bitwise, payload) and centralizes references to expression and set types.
 * Functional Utility: Facilitates efficient dispatcher logic for packet evaluation and provides common structures for stateful tracking.
 */

/* SPDX-License-Identifier: GPL-2.0 */
#ifndef _NET_NF_TABLES_CORE_H
#define _NET_NF_TABLES_CORE_H

#include <net/netfilter/nf_tables.h>
#include <linux/indirect_call_wrapper.h>

// External linkage for standard nftables expression types.
extern struct nft_expr_type nft_imm_type;
extern struct nft_expr_type nft_cmp_type;
extern struct nft_expr_type nft_counter_type;
extern struct nft_expr_type nft_lookup_type;
extern struct nft_expr_type nft_bitwise_type;
extern struct nft_expr_type nft_byteorder_type;
extern struct nft_expr_type nft_payload_type;
extern struct nft_expr_type nft_dynset_type;
extern struct nft_expr_type nft_range_type;
extern struct nft_expr_type nft_meta_type;
extern struct nft_expr_type nft_rt_type;
extern struct nft_expr_type nft_exthdr_type;
extern struct nft_expr_type nft_last_type;
extern struct nft_expr_type nft_objref_type;
extern struct nft_expr_type nft_inner_type;

#ifdef CONFIG_NETWORK_SECMARK
extern struct nft_object_type nft_secmark_obj_type;
#endif
extern struct nft_object_type nft_counter_obj_type;

int nf_tables_core_module_init(void);
void nf_tables_core_module_exit(void);

/**
 * @brief Optimized bitwise expression for fast mask/xor operations.
 */
struct nft_bitwise_fast_expr {
	u32			mask;
	u32			xor;
	u8			sreg;
	u8			dreg;
};

/**
 * @brief Optimized 32-bit comparison expression.
 */
struct nft_cmp_fast_expr {
	u32			data;
	u32			mask;
	u8			sreg;
	u8			len;
	bool			inv;
};

/**
 * @brief Optimized 128-bit (16-byte) comparison expression, typically for IPv6.
 */
struct nft_cmp16_fast_expr {
	struct nft_data		data;
	struct nft_data		mask;
	u8			sreg;
	u8			len;
	bool			inv;
};

/**
 * @brief Expression for immediate data assignment to registers.
 */
struct nft_immediate_expr {
	struct nft_data		data;
	u8			dreg;
	u8			dlen;
};

extern const struct nft_expr_ops nft_cmp_fast_ops;
extern const struct nft_expr_ops nft_cmp16_fast_ops;

/**
 * @brief Connection tracking (conntrack) metadata extraction expression.
 */
struct nft_ct {
	enum nft_ct_keys	key:8;
	enum ip_conntrack_dir	dir:8;
	u8			len;
	union {
		u8		dreg;
		u8		sreg;
	};
};

/**
 * @brief Header payload extraction expression.
 */
struct nft_payload {
	enum nft_payload_bases	base:8;
	u8			offset;
	u8			len;
	u8			dreg;
};

extern const struct nft_expr_ops nft_payload_fast_ops;

extern const struct nft_expr_ops nft_bitwise_fast_ops;

// Global toggles for performance-sensitive features.
extern struct static_key_false nft_counters_enabled;
extern struct static_key_false nft_trace_enabled;

// External linkage for standard nftables set backend types.
extern const struct nft_set_type nft_set_rhash_type;
extern const struct nft_set_type nft_set_hash_type;
extern const struct nft_set_type nft_set_hash_fast_type;
extern const struct nft_set_type nft_set_rbtree_type;
extern const struct nft_set_type nft_set_bitmap_type;
extern const struct nft_set_type nft_set_pipapo_type;
extern const struct nft_set_type nft_set_pipapo_avx2_type;

#ifdef CONFIG_MITIGATION_RETPOLINE
// Explicit lookup function declarations for environments with indirect call mitigations.
const struct nft_set_ext *
nft_rhash_lookup(const struct net *net, const struct nft_set *set,
		 const u32 *key);
const struct nft_set_ext *
nft_rbtree_lookup(const struct net *net, const struct nft_set *set,
		  const u32 *key);
const struct nft_set_ext *
nft_bitmap_lookup(const struct net *net, const struct nft_set *set,
		  const u32 *key);
const struct nft_set_ext *
nft_hash_lookup_fast(const struct net *net, const struct nft_set *set,
		     const u32 *key);
const struct nft_set_ext *
nft_hash_lookup(const struct net *net, const struct nft_set *set,
		const u32 *key);
const struct nft_set_ext *
nft_set_do_lookup(const struct net *net, const struct nft_set *set,
		  const u32 *key);
#else
/**
 * @brief Dispatcher for set lookup operations.
 * Performance: Inlined in non-retpoline kernels to avoid call overhead.
 */
static inline const struct nft_set_ext *
nft_set_do_lookup(const struct net *net, const struct nft_set *set,
		  const u32 *key)
{
	return set->ops->lookup(net, set, key);
}
#endif

/* Cross-module lookup references for PIPAPO algorithm components. */
const struct nft_set_ext *
nft_pipapo_lookup(const struct net *net, const struct nft_set *set,
		  const u32 *key);
const struct nft_set_ext *
nft_pipapo_avx2_lookup(const struct net *net, const struct nft_set *set,
			const u32 *key);

void nft_counter_init_seqcount(void);

// Forward declarations for kernel-internal structures and packet evaluation routines.
struct nft_expr;
struct nft_regs;
struct nft_pktinfo;
void nft_meta_get_eval(const struct nft_expr *expr,
		       struct nft_regs *regs, const struct nft_pktinfo *pkt);
void nft_cmp_eval(const struct nft_expr *expr,
		  struct nft_regs *regs, const struct nft_pktinfo *pkt);
void nft_lookup_eval(const struct nft_expr *expr,
		     struct nft_regs *regs, const struct nft_pktinfo *pkt);
void nft_payload_eval(const struct nft_expr *expr,
		      struct nft_regs *regs, const struct nft_pktinfo *pkt);
void nft_immediate_eval(const struct nft_expr *expr,
			struct nft_regs *regs, const struct nft_pktinfo *pkt);
void nft_bitwise_eval(const struct nft_expr *expr,
		      struct nft_regs *regs, const struct nft_pktinfo *pkt);
void nft_range_eval(const struct nft_expr *expr,
		    struct nft_regs *regs, const struct nft_pktinfo *pkt);
void nft_byteorder_eval(const struct nft_expr *expr,
			struct nft_regs *regs, const struct nft_pktinfo *pkt);
void nft_dynset_eval(const struct nft_expr *expr,
		     struct nft_regs *regs, const struct nft_pktinfo *pkt);
void nft_rt_get_eval(const struct nft_expr *expr,
		     struct nft_regs *regs, const struct nft_pktinfo *pkt);
void nft_counter_eval(const struct nft_expr *expr, struct nft_regs *regs,
                      const struct nft_pktinfo *pkt);
void nft_ct_get_fast_eval(const struct nft_expr *expr,
			  struct nft_regs *regs, const struct nft_pktinfo *pkt);

enum {
	NFT_PAYLOAD_CTX_INNER_TUN	= (1 << 0),
	NFT_PAYLOAD_CTX_INNER_LL	= (1 << 1),
	NFT_PAYLOAD_CTX_INNER_NH	= (1 << 2),
	NFT_PAYLOAD_CTX_INNER_TH	= (1 << 3),
};

/**
 * @brief Context tracking for inner (encapsulated) packet payload evaluation.
 */
struct nft_inner_tun_ctx {
	unsigned long cookie;
	u16	type;
	u16	inner_tunoff;
	u16	inner_lloff;
	u16	inner_nhoff;
	u16	inner_thoff;
	__be16	llproto;
	u8	l4proto;
	u8      flags;
};

int nft_payload_inner_offset(const struct pktinfo *pkt);
void nft_payload_inner_eval(const struct nft_expr *expr, struct nft_regs *regs,
			    const struct nft_pktinfo *pkt,
			    struct nft_inner_tun_ctx *ctx);

void nft_objref_eval(const struct nft_expr *expr, struct nft_regs *regs,
		     const struct nft_pktinfo *pkt);
void nft_objref_map_eval(const struct nft_expr *expr, struct nft_regs *regs,
			 const struct nft_pktinfo *pkt);
struct nft_elem_priv *nft_dynset_new(struct nft_set *set,
				     const struct nft_expr *expr,
				     struct nft_regs *regs);
#endif /* _NET_NF_TABLES_CORE_H */
