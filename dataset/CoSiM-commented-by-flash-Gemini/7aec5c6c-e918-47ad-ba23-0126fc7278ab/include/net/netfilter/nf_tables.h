/**
 * @7aec5c6c-e918-47ad-ba23-0126fc7278ab/include/net/netfilter/nf_tables.h
 * @brief Primary internal API and data structure definitions for the nftables framework.
 * Domain: Kernel Networking, Packet Classification Engine.
 * Architecture: Implements a hierarchical object model (Tables -> Chains -> Rules -> Expressions) for extensible packet processing.
 * Functional Utility: Provides the foundational types for the nftables virtual machine, including registers, verdicts, and transactional state.
 * Synchronization: Heavily relies on RCU (Read-Copy-Update) for lock-less packet evaluation and generation-based atomic ruleset updates.
 */

/* SPDX-License-Identifier: GPL-2.0 */
#ifndef _NET_NF_TABLES_H
#define _NET_NF_TABLES_H

#include <linux/unaligned.h>
#include <linux/list.h>
#include <linux/netfilter.h>
#include <linux/netfilter/nfnetlink.h>
#include <linux/netfilter/x_tables.h>
#include <linux/netfilter/nf_tables.h>
#include <linux/u64_stats_sync.h>
#include <linux/rhashtable.h>
#include <net/netfilter/nf_flow_table.h>
#include <net/netlink.h>
#include <net/flow_offload.h>
#include <net/netns/generic.h>

#define NFT_MAX_HOOKS	(NF_INET_INGRESS + 1)

struct module;

#define NFT_JUMP_STACK_SIZE	16

enum {
	NFT_PKTINFO_L4PROTO	= (1 << 0),
	NFT_PKTINFO_INNER	= (1 << 1),
	NFT_PKTINFO_INNER_FULL	= (1 << 2),
};

/**
 * @brief Metadata container for the currently processed packet.
 * State: Caches header offsets and netfilter hook state to minimize redundant parsing.
 */
struct nft_pktinfo {
	struct sk_buff			*skb;
	const struct nf_hook_state	*state;
	u8				flags;
	u8				tprot;
	u16				fragoff;
	u16				thoff;
	u16				inneroff;
};

static inline struct sock *nft_sk(const struct nft_pktinfo *pkt)
{
	return pkt->state->sk;
}

static inline unsigned int nft_thoff(const struct nft_pktinfo *pkt)
{
	return pkt->thoff;
}

static inline struct net *nft_net(const struct nft_pktinfo *pkt)
{
	return pkt->state->net;
}

static inline unsigned int nft_hook(const struct nft_pktinfo *pkt)
{
	return pkt->state->hook;
}

static inline u8 nft_pf(const struct nft_pktinfo *pkt)
{
	return pkt->state->pf;
}

static inline const struct net_device *nft_in(const struct nft_pktinfo *pkt)
{
	return pkt->state->in;
}

static inline const struct net_device *nft_out(const struct nft_pktinfo *pkt)
{
	return pkt->state->out;
}

static inline void nft_set_pktinfo(struct nft_pktinfo *pkt,
				   struct sk_buff *skb,
				   const struct nf_hook_state *state)
{
	pkt->skb = skb;
	pkt->state = state;
}

static inline void nft_set_pktinfo_unspec(struct nft_pktinfo *pkt)
{
	pkt->flags = 0;
	pkt->tprot = 0;
	pkt->thoff = 0;
	pkt->fragoff = 0;
}

/**
 * @brief Encapsulates the result of a packet evaluation step.
 * Mapping: maps to immediate actions (ACCEPT/DROP) or control flow transitions (JUMP/GOTO).
 */
struct nft_verdict {
	u32				code;
	struct nft_chain		*chain;
};

/**
 * @brief Generic data container for nftables registers.
 * Memory Layout: Aligned to 64-bits for efficient atomic operations and aligned memory access.
 */
struct nft_data {
	union {
		u32			data[4];
		struct nft_verdict	verdict;
	};
} __attribute__((aligned(__alignof__(u64))));

#define NFT_REG32_NUM		20

/**
 * @brief Virtual register set for the nftables execution environment.
 * Architecture: The first 4 registers (16 bytes) are aliased to the verdict register.
 */
struct nft_regs {
	union {
		u32			data[NFT_REG32_NUM];
		struct nft_verdict	verdict;
	};
};

/**
 * @brief Tracking structure for register-level constant/expression propagation.
 * Optimization: Used to reduce redundant expressions and enable hardware offload hints.
 */
struct nft_regs_track {
	struct {
		const struct nft_expr		*selector;
		const struct nft_expr		*bitwise;
		u8				num_reg;
	} regs[NFT_REG32_NUM];

	const struct nft_expr			*cur;
	const struct nft_expr			*last;
};

/* Register manipulation primitives for sub-word and multi-word data. */

static inline void nft_reg_store8(u32 *dreg, u8 val)
{
	*dreg = 0;
	*(u8 *)dreg = val;
}

static inline u8 nft_reg_load8(const u32 *sreg)
{
	return *(u8 *)sreg;
}

static inline void nft_reg_store16(u32 *dreg, u16 val)
{
	*dreg = 0;
	*(u16 *)dreg = val;
}

static inline void nft_reg_store_be16(u32 *dreg, __be16 val)
{
	nft_reg_store16(dreg, (__force __u16)val);
}

static inline u16 nft_reg_load16(const u32 *sreg)
{
	return *(u16 *)sreg;
}

static inline __be16 nft_reg_load_be16(const u32 *sreg)
{
	return (__force __be16)nft_reg_load16(sreg);
}

static inline __be32 nft_reg_load_be32(const u32 *sreg)
{
	return *(__force __be32 *)sreg;
}

static inline void nft_reg_store64(u64 *dreg, u64 val)
{
	put_unaligned(val, dreg);
}

static inline u64 nft_reg_load64(const u32 *sreg)
{
	return get_unaligned((u64 *)sreg);
}

static inline void nft_data_copy(u32 *dst, const struct nft_data *src,
				 unsigned int len)
{
	if (len % NFT_REG32_SIZE)
		dst[len / NFT_REG32_SIZE] = 0;
	memcpy(dst, src, len);
}

/**
 * @brief Contextual state for rule and set manipulation operations.
 * State: Bundles net namespace, table/chain hierarchy, and Netlink sequence info.
 */
struct nft_ctx {
	struct net			*net;
	struct nft_table		*table;
	struct nft_chain		*chain;
	const struct nlattr * const 	*nla;
	u32				portid;
	u32				seq;
	u16				flags;
	u8				family;
	u8				level;
	bool				report;
	DECLARE_BITMAP(reg_inited, NFT_REG32_NUM);
};

enum nft_data_desc_flags {
	NFT_DATA_DESC_SETELEM	= (1 << 0),
};

/**
 * @brief Metadata describing the type and constraints of a data object.
 */
struct nft_data_desc {
	enum nft_data_types		type;
	unsigned int			size;
	unsigned int			len;
	unsigned int			flags;
};

int nft_data_init(const struct nft_ctx *ctx, struct nft_data *data,
		  struct nft_data_desc *desc, const struct nlattr *nla);
void nft_data_hold(const struct nft_data *data, enum nft_data_types type);
void nft_data_release(const struct nft_data *data, enum nft_data_types type);
int nft_data_dump(struct sk_buff *skb, int attr, const struct nft_data *data,
		  enum nft_data_types type, unsigned int len);

static inline enum nft_data_types nft_dreg_to_type(enum nft_registers reg)
{
	return reg == NFT_REG_VERDICT ? NFT_DATA_VERDICT : NFT_DATA_VALUE;
}

static inline enum nft_registers nft_type_to_reg(enum nft_data_types type)
{
	return type == NFT_DATA_VERDICT ? NFT_REG_VERDICT : NFT_REG_1 * NFT_REG_SIZE / NFT_REG32_SIZE;
}

int nft_parse_u32_check(const struct nlattr *attr, int max, u32 *dest);
int nft_dump_register(struct sk_buff *skb, unsigned int attr, unsigned int reg);

int nft_parse_register_load(const struct nft_ctx *ctx,
			    const struct nlattr *attr, u8 *sreg, u32 len);
int nft_parse_register_store(const struct nft_ctx *ctx,
			     const struct nlattr *attr, u8 *dreg,
			     const struct nft_data *data,
			     enum nft_data_types type, unsigned int len);

/**
 * @brief Container for arbitrary user-defined metadata attached to objects.
 */
struct nft_userdata {
	u8			len;
	unsigned char		data[];
};

/* placeholder structure for opaque set element backend representation. */
struct nft_elem_priv { };

/**
 * @brief Generic representation of an entry within an nftables set.
 * Domain: Set Management.
 * Logic: Supports exact keys, range ends (for intervals), and associated mapping data.
 */
struct nft_set_elem {
	union {
		u32		buf[NFT_DATA_VALUE_MAXLEN / sizeof(u32)];
		struct nft_data	val;
	} key;
	union {
		u32		buf[NFT_DATA_VALUE_MAXLEN / sizeof(u32)];
		struct nft_data	val;
	} key_end;
	union {
		u32		buf[NFT_DATA_VALUE_MAXLEN / sizeof(u32)];
		struct nft_data val;
	} data;
	struct nft_elem_priv	*priv;
};

static inline void *nft_elem_priv_cast(const struct nft_elem_priv *priv)
{
	return (void *)priv;
}


/**
 * @brief Enumeration of supported set iteration modes.
 */
enum nft_iter_type {
	NFT_ITER_UNSPEC,
	NFT_ITER_READ,
	NFT_ITER_UPDATE,
};

struct nft_set;
/**
 * @brief Descriptor for traversing set elements.
 * Synchronization: genmask ensures iteration only sees elements valid in the target generation.
 */
struct nft_set_iter {
	u8		genmask;
	enum nft_iter_type type:8;
	unsigned int	count;
	unsigned int	skip;
	int		err;
	int		(*fn)(const struct nft_ctx *ctx,
			      struct nft_set *set,
			      const struct nft_set_iter *iter,
			      struct nft_elem_priv *elem_priv);
};

/**
 * @brief Schema definition for elements within a specific set.
 */
struct nft_set_desc {
	u32			ktype;
	unsigned int		klen;
	u32			dtype;
	unsigned int		dlen;
	u32			objtype;
	unsigned int		size;
	u32			policy;
	u32			gc_int;
	u64			timeout;
	u8			field_len[NFT_REG32_COUNT];
	u8			field_count;
	bool			expr;
};

/**
 * @brief Classification of set backend performance profiles.
 */
enum nft_set_class {
	NFT_SET_CLASS_O_1,
	NFT_SET_CLASS_O_LOG_N,
	NFT_SET_CLASS_O_N,
};

/**
 * @brief Estimated operational overhead for a chosen set backend.
 */
struct nft_set_estimate {
	u64			size;
	enum nft_set_class	lookup;
	enum nft_set_class	space;
};

#define NFT_EXPR_MAXATTR		16
#define NFT_EXPR_SIZE(size)		(sizeof(struct nft_expr) + \
					 ALIGN(size, __alignof__(struct nft_expr)))

/**
 * @brief Base polymorphic structure for packet processing logic.
 * State: Dynamically-sized private data following the ops pointer.
 */
struct nft_expr {
	const struct nft_expr_ops	*ops;
	unsigned char			data[]
		__attribute__((aligned(__alignof__(u64))));
};

static inline void *nft_expr_priv(const struct nft_expr *expr)
{
	return (void *)expr->data;
}

struct nft_expr_info;

int nft_expr_inner_parse(const struct nft_ctx *ctx, const struct nlattr *nla,
			 struct nft_expr_info *info);
int nft_expr_clone(struct nft_expr *dst, struct nft_expr *src, gfp_t gfp);
void nft_expr_destroy(const struct nft_ctx *ctx, struct nft_expr *expr);
int nft_expr_dump(struct sk_buff *skb, unsigned int attr,
		  const struct nft_expr *expr, bool reset);
bool nft_expr_reduce_bitwise(struct nft_regs_track *track,
			     const struct nft_expr *expr);

struct nft_set_ext;

/**
 * @brief Function table for set backend implementations (hash, rbtree, bitmap, etc.).
 * Functional Utility: Abstracts the storage and retrieval logic from the core netfilter pipeline.
 */
struct nft_set_ops {
	const struct nft_set_ext *	(*lookup)(const struct net *net,
						  const struct nft_set *set,
						  const u32 *key);
	const struct nft_set_ext *	(*update)(struct nft_set *set,
						  const u32 *key,
						  const struct nft_expr *expr,
						  struct nft_regs *regs);
	bool				(*delete)(const struct nft_set *set,
						  const u32 *key);

	int				(*insert)(const struct net *net,
						  const struct nft_set *set,
						  const struct nft_set_elem *elem,
						  struct nft_elem_priv **priv);
	void				(*activate)(const struct net *net,
						    const struct nft_set *set,
						    struct nft_elem_priv *elem_priv);
	struct nft_elem_priv *		(*deactivate)(const struct net *net,
						      const struct nft_set *set,
						      const struct nft_set_elem *elem);
	void				(*flush)(const struct net *net,
						 const struct nft_set *set,
						 struct nft_elem_priv *priv);
	void				(*remove)(const struct net *net,
						  const struct nft_set *set,
						  struct nft_elem_priv *elem_priv);
	void				(*walk)(const struct nft_ctx *ctx,
						struct nft_set *set,
						struct nft_set_iter *iter);
	struct nft_elem_priv *		(*get)(const struct net *net,
					       const struct nft_set *set,
					       const struct nft_set_elem *elem,
					       unsigned int flags);
	u32				(*ksize)(u32 size);
	u32				(*usize)(u32 size);
	u32				(*adjust_maxsize)(const struct nft_set *set);
	void				(*commit)(struct nft_set *set);
	void				(*abort)(const struct nft_set *set);
	u64				(*privsize)(const struct nlattr * const nla[],
						    const struct nft_set_desc *desc);
	bool				(*estimate)(const struct nft_set_desc *desc,
						    u32 features,
						    struct nft_set_estimate *est);
	int				(*init)(const struct nft_set *set,
						const struct nft_set_desc *desc,
						const struct nlattr * const nla[]);
	void				(*destroy)(const struct nft_ctx *ctx,
						   const struct nft_set *set);
	void				(*gc_init)(const struct nft_set *set);

	unsigned int			elemsize;
};

/**
 * @brief Metadata defining a supported set backend type.
 */
struct nft_set_type {
	const struct nft_set_ops	ops;
	u32				features;
};
#define to_set_type(o) container_of(o, struct nft_set_type, ops)

/**
 * @brief Container for stateful expressions attached to a set element.
 */
struct nft_set_elem_expr {
	u8				size;
	unsigned char			data[]
		__attribute__((aligned(__alignof__(struct nft_expr))));
};

#define nft_setelem_expr_at(__elem_expr, __offset)			\
	((struct nft_expr *)&__elem_expr->data[__offset])

#define nft_setelem_expr_foreach(__expr, __elem_expr, __size)		\
	for (__expr = nft_setelem_expr_at(__elem_expr, 0), __size = 0;	\
	     __size < (__elem_expr)->size;				\
	     __size += (__expr)->ops->size, __expr = ((void *)(__expr)) + (__expr)->ops->size)

#define NFT_SET_EXPR_MAX	2

/**
 * @brief Runtime instance of an nftables set.
 * Domain: Dynamic State Management.
 * State: Tracks membership, handles, timeouts, and stateful expressions (counters, etc.).
 * Synchronization: uses ____cacheline_aligned on performance-critical 'ops' pointer.
 */
struct nft_set {
	struct list_head		list;
	struct list_head		bindings;
	refcount_t			refs;
	struct nft_table		*table;
	possible_net_t			net;
	char				*name;
	u64				handle;
	u32				ktype;
	u32				dtype;
	u32				objtype;
	u32				size;
	u8				field_len[NFT_REG32_COUNT];
	u8				field_count;
	u32				use;
	atomic_t			nelems;
	u32				ndeact;
	u64				timeout;
	u32				gc_int;
	u16				policy;
	u16				udlen;
	unsigned char			*udata;
	struct list_head		pending_update;
	/* runtime data below here */
	const struct nft_set_ops	*ops ____cacheline_aligned;
	u16				flags:13,
					dead:1,
					genmask:2;
	u8				klen;
	u8				dlen;
	u8				num_exprs;
	struct nft_expr			*exprs[NFT_SET_EXPR_MAX];
	struct list_head		catchall_list;
	unsigned char			data[]
		__attribute__((aligned(__alignof__(u64))));
};

static inline bool nft_set_is_anonymous(const struct nft_set *set)
{
	return set->flags & NFT_SET_ANONYMOUS;
}

static inline void *nft_set_priv(const struct nft_set *set)
{
	return (void *)set->data;
}

static inline enum nft_data_types nft_set_datatype(const struct nft_set *set)
{
	return set->dtype == NFT_DATA_VERDICT ? NFT_DATA_VERDICT : NFT_DATA_VALUE;
}

static inline bool nft_set_gc_is_pending(const struct nft_set *s)
{
	return refcount_read(&s->refs) != 1;
}

static inline struct nft_set *nft_set_container_of(const void *priv)
{
	return (void *)priv - offsetof(struct nft_set, data);
}

struct nft_set *nft_set_lookup_global(const struct net *net,
				      const struct nft_table *table,
				      const struct nlattr *nla_set_name,
				      const struct nlattr *nla_set_id,
				      u8 genmask);

struct nft_set_ext *nft_set_catchall_lookup(const struct net *net,
					    const struct nft_set *set);

static inline unsigned long nft_set_gc_interval(const struct nft_set *set)
{
	u32 gc_int = READ_ONCE(set->gc_int);

	return gc_int ? msecs_to_jiffies(gc_int) : HZ;
}

/**
 * @brief Representation of a reference to a set from a chain/rule.
 */
struct nft_set_binding {
	struct list_head		list;
	const struct nft_chain		*chain;
	u32				flags;
};

enum nft_trans_phase;
void nf_tables_activate_set(const struct nft_ctx *ctx, struct nft_set *set);
void nf_tables_deactivate_set(const struct nft_ctx *ctx, struct nft_set *set,
			      struct nft_set_binding *binding,
			      enum nft_trans_phase phase);
int nf_tables_bind_set(const struct nft_ctx *ctx, struct nft_set *set,
		       struct nft_set_binding *binding);
void nf_tables_destroy_set(const struct nft_ctx *ctx, struct nft_set *set);

/**
 * @brief Identifier enumeration for set element extensions.
 */
enum nft_set_extensions {
	NFT_SET_EXT_KEY,
	NFT_SET_EXT_KEY_END,
	NFT_SET_EXT_DATA,
	NFT_SET_EXT_FLAGS,
	NFT_SET_EXT_TIMEOUT,
	NFT_SET_EXT_USERDATA,
	NFT_SET_EXT_EXPRESSIONS,
	NFT_SET_EXT_OBJREF,
	NFT_SET_EXT_NUM
};

/**
 * @brief Typing metadata for set extensions.
 */
struct nft_set_ext_type {
	u8	len;
	u8	align;
};

extern const struct nft_set_ext_type nft_set_ext_types[];

/**
 * @brief Blueprint for allocating memory and tracking offsets for set element extensions.
 */
struct nft_set_ext_tmpl {
	u16	len;
	u8	offset[NFT_SET_EXT_NUM];
	u8	ext_len[NFT_SET_EXT_NUM];
};

/**
 * @brief Compact storage for optional set element features.
 * Memory Layout: Uses a header with bit-offsets to locate extensions within an opaque data block.
 * Synchronization: Alignment constrained to word-size for atomic bit-ops on 'genmask'.
 */
struct nft_set_ext {
	u8	genmask;
	u8	offset[NFT_SET_EXT_NUM];
	char	data[];
} __aligned(BITS_PER_LONG / 8);

static inline void nft_set_ext_prepare(struct nft_set_ext_tmpl *tmpl)
{
	memset(tmpl, 0, sizeof(*tmpl));
	tmpl->len = sizeof(struct nft_set_ext);
}

static inline int nft_set_ext_add_length(struct nft_set_ext_tmpl *tmpl, u8 id,
					 unsigned int len)
{
	tmpl->len	 = ALIGN(tmpl->len, nft_set_ext_types[id].align);
	if (tmpl->len > U8_MAX)
		return -EINVAL;

	tmpl->offset[id] = tmpl->len;
	tmpl->ext_len[id] = nft_set_ext_types[id].len + len;
	tmpl->len	+= tmpl->ext_len[id];

	return 0;
}

static inline int nft_set_ext_add(struct nft_set_ext_tmpl *tmpl, u8 id)
{
	return nft_set_ext_add_length(tmpl, id, 0);
}

static inline void nft_set_ext_init(struct nft_set_ext *ext,
				    const struct nft_set_ext_tmpl *tmpl)
{
	memcpy(ext->offset, tmpl->offset, sizeof(ext->offset));
}

static inline bool __nft_set_ext_exists(const struct nft_set_ext *ext, u8 id)
{
	return !!ext->offset[id];
}

static inline bool nft_set_ext_exists(const struct nft_set_ext *ext, u8 id)
{
	return ext && __nft_set_ext_exists(ext, id);
}

static inline void *nft_set_ext(const struct nft_set_ext *ext, u8 id)
{
	return (void *)ext + ext->offset[id];
}

static inline struct nft_data *nft_set_ext_key(const struct nft_set_ext *ext)
{
	return nft_set_ext(ext, NFT_SET_EXT_KEY);
}

static inline struct nft_data *nft_set_ext_key_end(const struct nft_set_ext *ext)
{
	return nft_set_ext(ext, NFT_SET_EXT_KEY_END);
}

static inline struct nft_data *nft_set_ext_data(const struct nft_set_ext *ext)
{
	return nft_set_ext(ext, NFT_SET_EXT_DATA);
}

static inline u8 *nft_set_ext_flags(const struct nft_set_ext *ext)
{
	return nft_set_ext(ext, NFT_SET_EXT_FLAGS);
}

/**
 * @brief Representation of element-level expiration metadata.
 */
struct nft_timeout {
	u64	timeout;
	u64	expiration;
};

static inline struct nft_timeout *nft_set_ext_timeout(const struct nft_set_ext *ext)
{
	return nft_set_ext(ext, NFT_SET_EXT_TIMEOUT);
}

static inline struct nft_userdata *nft_set_ext_userdata(const struct nft_set_ext *ext)
{
	return nft_set_ext(ext, NFT_SET_EXT_USERDATA);
}

static inline struct nft_set_elem_expr *nft_set_ext_expr(const struct nft_set_ext *ext)
{
	return nft_set_ext(ext, NFT_SET_EXT_EXPRESSIONS);
}

static inline bool __nft_set_elem_expired(const struct nft_set_ext *ext,
					  u64 tstamp)
{
	if (!nft_set_ext_exists(ext, NFT_SET_EXT_TIMEOUT) ||
	    READ_ONCE(nft_set_ext_timeout(ext)->timeout) == 0)
		return false;

	return time_after_eq64(tstamp, READ_ONCE(nft_set_ext_timeout(ext)->expiration));
}

static inline bool nft_set_elem_expired(const struct nft_set_ext *ext)
{
	return __nft_set_elem_expired(ext, get_jiffies_64());
}

static inline struct nft_set_ext *nft_set_elem_ext(const struct nft_set *set,
						   const struct nft_elem_priv *elem_priv)
{
	return (void *)elem_priv + set->ops->elemsize;
}

static inline struct nft_object **nft_set_ext_obj(const struct nft_set_ext *ext)
{
	return nft_set_ext(ext, NFT_SET_EXT_OBJREF);
}

struct nft_expr *nft_set_elem_expr_alloc(const struct nft_ctx *ctx,
					 const struct nft_set *set,
					 const struct nlattr *attr);

struct nft_elem_priv *nft_set_elem_init(const struct nft_set *set,
					const struct nft_set_ext_tmpl *tmpl,
					const u32 *key, const u32 *key_end,
					const u32 *data,
					u64 timeout, u64 expiration, gfp_t gfp);
int nft_set_elem_expr_clone(const struct nft_ctx *ctx, struct nft_set *set,
			    struct nft_expr *expr_array[]);
void nft_set_elem_destroy(const struct nft_set *set,
			  const struct nft_elem_priv *elem_priv,
			  bool destroy_expr);
void nf_tables_set_elem_destroy(const struct nft_ctx *ctx,
				const struct nft_set *set,
				const struct nft_elem_priv *elem_priv);

struct nft_expr_ops;
/**
 * @brief Metadata for expression types (e.g. counter, cmp, meta).
 */
struct nft_expr_type {
	const struct nft_expr_ops	*(*select_ops)(const struct nft_ctx *,
						       const struct nlattr * const tb[]);
	void				(*release_ops)(const struct nft_expr_ops *ops);
	const struct nft_expr_ops	*ops;
	const struct nft_expr_ops	*inner_ops;
	struct list_head		list;
	const char			*name;
	struct module			*owner;
	const struct nla_policy		*policy;
	unsigned int			maxattr;
	u8				family;
	u8				flags;
};

#define NFT_EXPR_STATEFUL		0x1
#define NFT_EXPR_GC			0x2

enum nft_trans_phase {
	NFT_TRANS_PREPARE,
	NFT_TRANS_PREPARE_ERROR,
	NFT_TRANS_ABORT,
	NFT_TRANS_COMMIT,
	NFT_TRANS_RELEASE
};

struct nft_flow_rule;
struct nft_offload_ctx;

/**
 * @brief Logic table for expression-specific behavior.
 * Mapping: defines how each expression evaluates, clones, and interacts with hardware offload.
 */
struct nft_expr_ops {
	void				(*eval)(const struct nft_expr *expr,
						struct nft_regs *regs,
						const struct nft_pktinfo *pkt);
	int				(*clone)(struct nft_expr *dst,
						 const struct nft_expr *src, gfp_t gfp);
	unsigned int			size;

	int				(*init)(const struct nft_ctx *ctx,
						const struct nft_expr *expr,
						const struct nlattr * const tb[]);
	void				(*activate)(const struct nft_ctx *ctx,
						    const struct nft_expr *expr);
	void				(*deactivate)(const struct nft_ctx *ctx,
						      const struct nft_expr *expr,
						      enum nft_trans_phase phase);
	void				(*destroy)(const struct nft_ctx *ctx,
						   const struct nft_expr *expr);
	void				(*destroy_clone)(const struct nft_ctx *ctx,
							 const struct nft_expr *expr);
	int				(*dump)(struct sk_buff *skb,
						const struct nft_expr *expr,
						bool reset);
	int				(*validate)(const struct nft_ctx *ctx,
						    const struct nft_expr *expr);
	bool				(*reduce)(struct nft_regs_track *track,
						  const struct nft_expr *expr);
	bool				(*gc)(struct net *net,
					      const struct nft_expr *expr);
	int				(*offload)(struct nft_offload_ctx *ctx,
						   struct nft_flow_rule *flow,
						   const struct nft_expr *expr);
	bool				(*offload_action)(const struct nft_expr *expr);
	void				(*offload_stats)(struct nft_expr *expr,
							 const struct flow_stats *stats);
	const struct nft_expr_type	*type;
	void				*data;
};

/**
 * @brief Representation of a packet classification rule.
 * Architecture: Packed binary blob containing a sequence of expressions terminated by dlen.
 */
struct nft_rule {
	struct list_head		list;
	u64				handle:42,
					genmask:2,
					dlen:12,
					udata:1;
	unsigned char			data[]
		__attribute__((aligned(__alignof__(struct nft_expr))));
};

static inline struct nft_expr *nft_expr_first(const struct nft_rule *rule)
{
	return (struct nft_expr *)&rule->data[0];
}

static inline struct nft_expr *nft_expr_next(const struct nft_expr *expr)
{
	return ((void *)expr) + expr->ops->size;
}

static inline struct nft_expr *nft_expr_last(const struct nft_rule *rule)
{
	return (struct nft_expr *)&rule->data[rule->dlen];
}

static inline bool nft_expr_more(const struct nft_rule *rule,
				 const struct nft_expr *expr)
{
	return expr != nft_expr_last(rule) && expr->ops;
}

static inline struct nft_userdata *nft_userdata(const struct nft_rule *rule)
{
	return (void *)&rule->data[rule->dlen];
}

void nft_rule_expr_activate(const struct nft_ctx *ctx, struct nft_rule *rule);
void nft_rule_expr_deactivate(const struct nft_ctx *ctx, struct nft_rule *rule,
			      enum nft_trans_phase phase);
void nf_tables_rule_destroy(const struct nft_ctx *ctx, struct nft_rule *rule);

/**
 * @brief Iteratively executes stateful expressions attached to a set element.
 */
static inline void nft_set_elem_update_expr(const struct nft_set_ext *ext,
					    struct nft_regs *regs,
					    const struct nft_pktinfo *pkt)
{
	struct nft_set_elem_expr *elem_expr;
	struct nft_expr *expr;
	u32 size;

	if (__nft_set_ext_exists(ext, NFT_SET_EXT_EXPRESSIONS)) {
		elem_expr = nft_set_ext_expr(ext);
		nft_setelem_expr_foreach(expr, elem_expr, size) {
			expr->ops->eval(expr, regs, pkt);
			if (regs->verdict.code == NFT_BREAK)
				return;
		}
	}
}

/* Macro for traversing the expression list within a rule. */
#define nft_rule_for_each_expr(expr, last, rule) \
	for ((expr) = nft_expr_first(rule), (last) = nft_expr_last(rule); \
	     (expr) != (last); \
	     (expr) = nft_expr_next(expr))

#define NFT_CHAIN_POLICY_UNSET		U8_MAX

/**
 * @brief Runtime-optimized representation of rules within the packet path.
 */
struct nft_rule_dp {
	u64				is_last:1,
					dlen:12,
					handle:42;	/* for tracing */
	unsigned char			data[]
		__attribute__((aligned(__alignof__(struct nft_expr))));
};

struct nft_rule_dp_last {
	struct nft_rule_dp end;		/* end of nft_rule_blob marker */
	struct rcu_head h;		/* call_rcu head */
	struct nft_rule_blob *blob;	/* ptr to free via call_rcu */
	const struct nft_chain *chain;	/* for nftables tracing */
};

static inline const struct nft_rule_dp *nft_rule_next(const struct nft_rule_dp *rule)
{
	return (void *)rule + sizeof(*rule) + rule->dlen;
}

/**
 * @brief Binary blob containing a linear sequence of data-plane rules.
 */
struct nft_rule_blob {
	unsigned long			size;
	unsigned char			data[]
		__attribute__((aligned(__alignof__(struct nft_rule_dp))));
};

/**
 * @brief A container (list) of rules evaluated in sequence.
 * Architecture: Supports generation-based double-buffering (blob_gen_0/1) for zero-latency ruleset swaps.
 * State: Includes handle for management and tracking of jump references (use counter).
 */
struct nft_chain {
	struct nft_rule_blob		__rcu *blob_gen_0;
	struct nft_rule_blob		__rcu *blob_gen_1;
	struct list_head		rules;
	struct list_head		list;
	struct rhlist_head		rhlhead;
	struct nft_table		*table;
	u64				handle;
	u32				use;
	u8				flags:5,
					bound:1,
					genmask:2;
	char				*name;
	u16				udlen;
	u8				*udata;

	/* Only used during control plane commit phase: */
	struct nft_rule_blob		*blob_next;
};

int nft_chain_validate(const struct nft_ctx *ctx, const struct nft_chain *chain);
int nft_setelem_validate(const struct nft_ctx *ctx, struct nft_set *set,
			 const struct nft_set_iter *iter,
			 struct nft_elem_priv *elem_priv);
int nft_set_catchall_validate(const struct nft_ctx *ctx, struct nft_set *set);
int nf_tables_bind_chain(const struct nft_ctx *ctx, struct nft_chain *chain);
void nf_tables_unbind_chain(const struct nft_ctx *ctx, struct nft_chain *chain);

enum nft_chain_types {
	NFT_CHAIN_T_DEFAULT = 0,
	NFT_CHAIN_T_ROUTE,
	NFT_CHAIN_T_NAT,
	NFT_CHAIN_T_MAX
};

/**
 * @brief Metadata for base chains linked to netfilter hooks.
 */
struct nft_chain_type {
	const char			*name;
	enum nft_chain_types		type;
	int				family;
	struct module			*owner;
	unsigned int			hook_mask;
	nf_hookfn			*hooks[NFT_MAX_HOOKS];
	int				(*ops_register)(struct net *net, const struct nf_hook_ops *ops);
	void				(*ops_unregister)(struct net *net, const struct nf_hook_ops *ops);
};

int nft_chain_validate_dependency(const struct nft_chain *chain,
				  enum nft_chain_types type);
int nft_chain_validate_hooks(const struct nft_chain *chain,
                             unsigned int hook_flags);

static inline bool nft_chain_binding(const struct nft_chain *chain)
{
	return chain->flags & NFT_CHAIN_BINDING;
}

static inline bool nft_chain_is_bound(struct nft_chain *chain)
{
	return (chain->flags & NFT_CHAIN_BINDING) && chain->bound;
}

int nft_chain_add(struct nft_table *table, struct nft_chain *chain);
void nft_chain_del(struct nft_chain *chain);
void nf_tables_chain_destroy(struct nft_chain *chain);

/**
 * @brief Performance statistics tracking for chains.
 */
struct nft_stats {
	u64			bytes;
	u64			pkts;
	struct u64_stats_sync	syncp;
};

/**
 * @brief Instance of a hook linkage for netdev protocol families.
 */
struct nft_hook {
	struct list_head	list;
	struct list_head	ops_list;
	struct rcu_head		rcu;
	char			ifname[IFNAMSIZ];
	u8			ifnamelen;
};

struct nf_hook_ops *nft_hook_find_ops(const struct nft_hook *hook,
				      const struct net_device *dev);
struct nf_hook_ops *nft_hook_find_ops_rcu(const struct nft_hook *hook,
					  const struct net_device *dev);

/**
 * @brief Specialized chain type that directly interfaces with netfilter engine hooks.
 */
struct nft_base_chain {
	struct nf_hook_ops		ops;
	struct list_head		hook_list;
	const struct nft_chain_type	*type;
	u8				policy;
	u8				flags;
	struct nft_stats __percpu	*stats;
	struct nft_chain		chain;
	struct flow_block		flow_block;
};

static inline struct nft_base_chain *nft_base_chain(const struct nft_chain *chain)
{
	return container_of(chain, struct nft_base_chain, chain);
}

static inline bool nft_is_base_chain(const struct nft_chain *chain)
{
	return chain->flags & NFT_CHAIN_BASE;
}

unsigned int nft_do_chain(struct nft_pktinfo *pkt, void *priv);

static inline bool nft_use_inc(u32 *use)
{
	if (*use == UINT_MAX)
		return false;

	(*use)++;

	return true;
}

static inline void nft_use_dec(u32 *use)
{
	WARN_ON_ONCE((*use)-- == 0);
}

/* For error and abort path: restore use counter to previous state. */
static inline void nft_use_inc_restore(u32 *use)
{
	WARN_ON_ONCE(!nft_use_inc(use));
}

#define nft_use_dec_restore	nft_use_dec

/**
 * @brief Top-level namespace for nftables objects within a protocol family.
 */
struct nft_table {
	struct list_head		list;
	struct rhltable			chains_ht;
	struct list_head		chains;
	struct list_head		sets;
	struct list_head		objects;
	struct list_head		flowtables;
	u64				hgenerator;
	u64				handle;
	u32				use;
	u16				family:6,
					flags:8,
					genmask:2;
	u32				nlpid;
	char				*name;
	u16				udlen;
	u8				*udata;
	u8				validate_state;
};

static inline bool nft_table_has_owner(const struct nft_table *table)
{
	return table->flags & NFT_TABLE_F_OWNER;
}

static inline bool nft_table_is_orphan(const struct nft_table *table)
{
	return (table->flags & (NFT_TABLE_F_OWNER | NFT_TABLE_F_PERSIST)) ==
			NFT_TABLE_F_PERSIST;
}

static inline bool nft_base_chain_netdev(int family, u32 hooknum)
{
	return family == NFPROTO_NETDEV ||
	       (family == NFPROTO_INET && hooknum == NF_INET_INGRESS);
}

void nft_register_chain_type(const struct nft_chain_type *);
void nft_unregister_chain_type(const struct nft_chain_type *);

int nft_register_expr(struct nft_expr_type *);
void nft_unregister_expr(struct nft_expr_type *);

int nft_verdict_dump(struct sk_buff *skb, int type,
		     const struct nft_verdict *v);

/**
 * @brief Internal key structure for hashing stateful objects.
 */
struct nft_object_hash_key {
	const char                      *name;
	const struct nft_table          *table;
};

/**
 * @brief Polymorphic container for persistent rule state (e.g., quotas, limiters).
 * Strategy: Decouples rule logic from stateful metrics to allow shared state across rules.
 */
struct nft_object {
	struct list_head		list;
	struct rhlist_head		rhlhead;
	struct nft_object_hash_key	key;
	u32				genmask:2;
	u32				use;
	u64				handle;
	u16				udlen;
	u8				*udata;
	/* runtime data below here */
	const struct nft_object_ops	*ops ____cacheline_aligned;
	unsigned char			data[]
		__attribute__((aligned(__alignof__(u64))));
};

static inline void *nft_obj_data(const struct nft_object *obj)
{
	return (void *)obj->data;
}

#define nft_expr_obj(expr)	*((struct nft_object **)nft_expr_priv(expr))

struct nft_object *nft_obj_lookup(const struct net *net,
				  const struct nft_table *table,
				  const struct nlattr *nla, u32 objtype,
				  u8 genmask);

void nft_obj_notify(struct net *net, const struct nft_table *table,
		    struct nft_object *obj, u32 portid, u32 seq,
		    int event, u16 flags, int family, int report, gfp_t gfp);

/**
 * @brief Registry info for stateful object types.
 */
struct nft_object_type {
	const struct nft_object_ops	*(*select_ops)(const struct nft_ctx *,
						       const struct nlattr * const tb[]);
	const struct nft_object_ops	*ops;
	struct list_head		list;
	u32				type;
	unsigned int                    maxattr;
	u8				family;
	struct module			*owner;
	const struct nla_policy		*policy;
};

/**
 * @brief Operations table for stateful objects.
 */
struct nft_object_ops {
	void				(*eval)(struct nft_object *obj,
						struct nft_regs *regs,
						const struct nft_pktinfo *pkt);
	unsigned int			size;
	int				(*init)(const struct nft_ctx *ctx,
						const struct nlattr *const tb[],
						struct nft_object *obj);
	void				(*destroy)(const struct nft_ctx *ctx,
						   struct nft_object *obj);
	int				(*dump)(struct sk_buff *skb,
						struct nft_object *obj,
						bool reset);
	void				(*update)(struct nft_object *obj,
						  struct nft_object *newobj);
	const struct nft_object_type	*type;
};

int nft_register_obj(struct nft_object_type *obj_type);
void nft_unregister_obj(struct nft_object_type *obj_type);

#define NFT_NETDEVICE_MAX	256

/**
 * @brief Representation of an accelerated flow-offload table.
 */
struct nft_flowtable {
	struct list_head		list;
	struct nft_table		*table;
	char				*name;
	int				hooknum;
	int				ops_len;
	u32				genmask:2;
	u32				use;
	u64				handle;
	/* runtime data below here */
	struct list_head		hook_list ____cacheline_aligned;
	struct nf_flowtable		data;
};

struct nft_flowtable *nft_flowtable_lookup(const struct net *net,
					   const struct nft_table *table,
					   const struct nlattr *nla,
					   u8 genmask);

void nf_tables_deactivate_flowtable(const struct nft_ctx *ctx,
				    struct nft_flowtable *flowtable,
				    enum nft_trans_phase phase);

void nft_register_flowtable_type(struct nf_flowtable_type *type);
void nft_unregister_flowtable_type(struct nf_flowtable_type *type);

/**
 * @brief Snapshot of packet path tracing information.
 */
struct nft_traceinfo {
	bool				trace;
	bool				nf_trace;
	bool				packet_dumped;
	enum nft_trace_types		type:8;
	u32				skbid;
	const struct nft_base_chain	*basechain;
};

void nft_trace_init(struct nft_traceinfo *info, const struct nft_pktinfo *pkt,
		    const struct nft_chain *basechain);

void nft_trace_notify(const struct nft_pktinfo *pkt,
		      const struct nft_verdict *verdict,
		      const struct nft_rule_dp *rule,
		      struct nft_traceinfo *info);

#define MODULE_ALIAS_NFT_CHAIN(family, name) \
	MODULE_ALIAS("nft-chain-" __stringify(family) "-" name)

#define MODULE_ALIAS_NFT_AF_EXPR(family, name) \
	MODULE_ALIAS("nft-expr-" __stringify(family) "-" name)

#define MODULE_ALIAS_NFT_EXPR(name) \
	MODULE_ALIAS("nft-expr-" name)

#define MODULE_ALIAS_NFT_OBJ(type) \
	MODULE_ALIAS("nft-obj-" __stringify(type))

#if IS_ENABLED(CONFIG_NF_TABLES)

/*
 * Generation cursors and masks for transactional atomicity.
 * Logic: Implements a 2-bit generation scheme. 00 indicates active in all generations.
 * 01 or 10 indicate inactivity in specific generations (current vs next).
 */
static inline unsigned int nft_gencursor_next(const struct net *net)
{
	return net->nft.gencursor + 1 == 1 ? 1 : 0;
}

static inline u8 nft_genmask_next(const struct net *net)
{
	return 1 << nft_gencursor_next(net);
}

static inline u8 nft_genmask_cur(const struct net *net)
{
	/* Use READ_ONCE() to prevent refetching the value for atomicity */
	return 1 << READ_ONCE(net->nft.gencursor);
}

#define NFT_GENMASK_ANY		((1 << 0) | (1 << 1))

/*
 * Transaction state helpers for determining object visibility.
 */

#define nft_is_active(__net, __obj)				\
	(((__obj)->genmask & nft_genmask_cur(__net)) == 0)

#define nft_is_active_next(__net, __obj)			\
	(((__obj)->genmask & nft_genmask_next(__net)) == 0)

#define nft_activate_next(__net, __obj)				\
	(__obj)->genmask = nft_genmask_cur(__net)

#define nft_deactivate_next(__net, __obj)			\
        (__obj)->genmask = nft_genmask_next(__net)

#define nft_clear(__net, __obj)					\
	(__obj)->genmask &= ~nft_genmask_next(__net)
#define nft_active_genmask(__obj, __genmask)			\
	!((__obj)->genmask & __genmask)

static inline bool nft_set_elem_active(const struct nft_set_ext *ext,
				       u8 genmask)
{
	return !(ext->genmask & genmask);
}

static inline void nft_set_elem_change_active(const struct net *net,
					      const struct nft_set *set,
					      struct nft_set_ext *ext)
{
	ext->genmask ^= nft_genmask_next(net);
}

#endif /* IS_ENABLED(CONFIG_NF_TABLES) */

#define NFT_SET_ELEM_DEAD_MASK	(1 << 2)

#if defined(__LITTLE_ENDIAN_BITFIELD)
#define NFT_SET_ELEM_DEAD_BIT	2
#elif defined(__BIG_ENDIAN_BITFIELD)
#define NFT_SET_ELEM_DEAD_BIT	(BITS_PER_LONG - BITS_PER_BYTE + 2)
#else
#error
#endif

/**
 * @brief Logical deletion primitive for set elements.
 * Optimization: Uses atomic bitops on aligned extension data.
 */
static inline void nft_set_elem_dead(struct nft_set_ext *ext)
{
	unsigned long *word = (unsigned long *)ext;

	BUILD_BUG_ON(offsetof(struct nft_set_ext, genmask) != 0);
	set_bit(NFT_SET_ELEM_DEAD_BIT, word);
}

static inline int nft_set_elem_is_dead(const struct nft_set_ext *ext)
{
	unsigned long *word = (unsigned long *)ext;

	BUILD_BUG_ON(offsetof(struct nft_set_ext, genmask) != 0);
	return test_bit(NFT_SET_ELEM_DEAD_BIT, word);
}

/**
 * @brief Base structure for objects participating in an nftables Netlink transaction.
 */
struct nft_trans {
	struct list_head		list;
	struct net			*net;
	struct nft_table		*table;
	int				msg_type;
	u32				seq;
	u16				flags;
	u8				report:1;
	u8				put_net:1;
};

/**
 * @brief Transaction base type for objects with parent-chain bindings.
 */
struct nft_trans_binding {
	struct nft_trans nft_trans;
	struct list_head binding_list;
};

struct nft_trans_rule {
	struct nft_trans		nft_trans;
	struct nft_rule			*rule;
	struct nft_chain		*chain;
	struct nft_flow_rule		*flow;
	u32				rule_id;
	bool				bound;
};

#define nft_trans_container_rule(trans)			\
	container_of(trans, struct nft_trans_rule, nft_trans)
#define nft_trans_rule(trans)				\
	nft_trans_container_rule(trans)->rule
#define nft_trans_flow_rule(trans)			\
	nft_trans_container_rule(trans)->flow
#define nft_trans_rule_id(trans)			\
	nft_trans_container_rule(trans)->rule_id
#define nft_trans_rule_bound(trans)			\
	nft_trans_container_rule(trans)->bound
#define nft_trans_rule_chain(trans)	\
	nft_trans_container_rule(trans)->chain

struct nft_trans_set {
	struct nft_trans_binding	nft_trans_binding;
	struct list_head		list_trans_newset;
	struct nft_set			*set;
	u32				set_id;
	u32				gc_int;
	u64				timeout;
	bool				update;
	bool				bound;
	u32				size;
};

#define nft_trans_container_set(t)	\
	container_of(t, struct nft_trans_set, nft_trans_binding.nft_trans)
#define nft_trans_set(trans)				\
	nft_trans_container_set(trans)->set
#define nft_trans_set_id(trans)				\
	nft_trans_container_set(trans)->set_id
#define nft_trans_set_bound(trans)			\
	nft_trans_container_set(trans)->bound
#define nft_trans_set_update(trans)			\
	nft_trans_container_set(trans)->update
#define nft_trans_set_timeout(trans)			\
	nft_trans_container_set(trans)->timeout
#define nft_trans_set_gc_int(trans)			\
	nft_trans_container_set(trans)->gc_int
#define nft_trans_set_size(trans)			\
	nft_trans_container_set(trans)->size

struct nft_trans_chain {
	struct nft_trans_binding	nft_trans_binding;
	struct nft_chain		*chain;
	char				*name;
	struct nft_stats __percpu	*stats;
	u8				policy;
	bool				update;
	bool				bound;
	u32				chain_id;
	struct nft_base_chain		*basechain;
	struct list_head		hook_list;
};

#define nft_trans_container_chain(t)	\
	container_of(t, struct nft_trans_chain, nft_trans_binding.nft_trans)
#define nft_trans_chain(trans)				\
	nft_trans_container_chain(trans)->chain
#define nft_trans_chain_update(trans)			\
	nft_trans_container_chain(trans)->update
#define nft_trans_chain_name(trans)			\
	nft_trans_container_chain(trans)->name
#define nft_trans_chain_stats(trans)			\
	nft_trans_container_chain(trans)->stats
#define nft_trans_chain_policy(trans)			\
	nft_trans_container_chain(trans)->policy
#define nft_trans_chain_bound(trans)			\
	nft_trans_container_chain(trans)->bound
#define nft_trans_chain_id(trans)			\
	nft_trans_container_chain(trans)->chain_id
#define nft_trans_basechain(trans)			\
	nft_trans_container_chain(trans)->basechain
#define nft_trans_chain_hooks(trans)			\
	nft_trans_container_chain(trans)->hook_list

struct nft_trans_table {
	struct nft_trans		nft_trans;
	bool				update;
};

#define nft_trans_container_table(trans)		\
	container_of(trans, struct nft_trans_table, nft_trans)
#define nft_trans_table_update(trans)			\
	nft_trans_container_table(trans)->update

enum nft_trans_elem_flags {
	NFT_TRANS_UPD_TIMEOUT		= (1 << 0),
	NFT_TRANS_UPD_EXPIRATION	= (1 << 1),
};

struct nft_elem_update {
	u64				timeout;
	u64				expiration;
	u8				flags;
};

struct nft_trans_one_elem {
	struct nft_elem_priv		*priv;
	struct nft_elem_update		*update;
};

struct nft_trans_elem {
	struct nft_trans		nft_trans;
	struct nft_set			*set;
	bool				bound;
	unsigned int			nelems;
	struct nft_trans_one_elem	elems[] __counted_by(nelems);
};

#define nft_trans_container_elem(t)			\
	container_of(t, struct nft_trans_elem, nft_trans)
#define nft_trans_elem_set(trans)			\
	nft_trans_container_elem(trans)->set
#define nft_trans_elem_set_bound(trans)			\
	nft_trans_container_elem(trans)->bound

struct nft_trans_obj {
	struct nft_trans		nft_trans;
	struct nft_object		*obj;
	struct nft_object		*newobj;
	bool				update;
};

#define nft_trans_container_obj(t)			\
	container_of(t, struct nft_trans_obj, nft_trans)
#define nft_trans_obj(trans)				\
	nft_trans_container_obj(trans)->obj
#define nft_trans_obj_newobj(trans)			\
	nft_trans_container_obj(trans)->newobj
#define nft_trans_obj_update(trans)			\
	nft_trans_container_obj(trans)->update

struct nft_trans_flowtable {
	struct nft_trans		nft_trans;
	struct nft_flowtable		*flowtable;
	struct list_head		hook_list;
	u32				flags;
	bool				update;
};

#define nft_trans_container_flowtable(t)		\
	container_of(t, struct nft_trans_flowtable, nft_trans)
#define nft_trans_flowtable(trans)			\
	nft_trans_container_flowtable(trans)->flowtable
#define nft_trans_flowtable_update(trans)		\
	nft_trans_container_flowtable(trans)->update
#define nft_trans_flowtable_hooks(trans)		\
	nft_trans_container_flowtable(trans)->hook_list
#define nft_trans_flowtable_flags(trans)		\
	nft_trans_container_flowtable(trans)->flags

#define NFT_TRANS_GC_BATCHCOUNT	256

/**
 * @brief Batch container for asynchronous set element garbage collection.
 */
struct nft_trans_gc {
	struct list_head	list;
	struct net		*net;
	struct nft_set		*set;
	u32			seq;
	u16			count;
	struct nft_elem_priv	*priv[NFT_TRANS_GC_BATCHCOUNT];
	struct rcu_head		rcu;
};

static inline void nft_ctx_update(struct nft_ctx *ctx,
				  const struct nft_trans *trans)
{
	switch (trans->msg_type) {
	case NFT_MSG_NEWRULE:
	case NFT_MSG_DELRULE:
	case NFT_MSG_DESTROYRULE:
		ctx->chain = nft_trans_rule_chain(trans);
		break;
	case NFT_MSG_NEWCHAIN:
	case NFT_MSG_DELCHAIN:
	case NFT_MSG_DESTROYCHAIN:
		ctx->chain = nft_trans_chain(trans);
		break;
	default:
		ctx->chain = NULL;
		break;
	}

	ctx->net = trans->net;
	ctx->table = trans->table;
	ctx->family = trans->table->family;
	ctx->report = trans->report;
	ctx->flags = trans->flags;
	ctx->seq = trans->seq;
}

struct nft_trans_gc *nft_trans_gc_alloc(struct nft_set *set,
					unsigned int gc_seq, gfp_t gfp);
void nft_trans_gc_destroy(struct nft_trans_gc *trans);

struct nft_trans_gc *nft_trans_gc_queue_async(struct nft_trans_gc *gc,
					      unsigned int gc_seq, gfp_t gfp);
void nft_trans_gc_queue_async_done(struct nft_trans_gc *gc);

struct nft_trans_gc *nft_trans_gc_queue_sync(struct nft_trans_gc *gc, gfp_t gfp);
void nft_trans_gc_queue_sync_done(struct nft_trans_gc *trans);

void nft_trans_gc_elem_add(struct nft_trans_gc *gc, void *priv);

struct nft_trans_gc *nft_trans_gc_catchall_async(struct nft_trans_gc *gc,
						 unsigned int gc_seq);
struct nft_trans_gc *nft_trans_gc_catchall_sync(struct nft_trans_gc *gc);

void nft_setelem_data_deactivate(const struct net *net,
				 const struct nft_set *set,
				 struct nft_elem_priv *elem_priv);

int __init nft_chain_filter_init(void);
void nft_chain_filter_fini(void);

void __init nft_chain_route_init(void);
void nft_chain_route_fini(void);

void nf_tables_trans_destroy_flush_work(struct net *net);

int nf_msecs_to_jiffies64(const struct nlattr *nla, u64 *result);
__be64 nf_jiffies64_to_msecs(u64 input);

#ifdef CONFIG_MODULES
__printf(2, 3) int nft_request_module(struct net *net, const char *fmt, ...);
#else
static inline int nft_request_module(struct net *net, const char *fmt, ...) { return -ENOENT; }
#endif

/**
 * @brief Per-network-namespace metadata for the nftables subsystem.
 * State: Maintains global lists of tables, transaction status, and GC sequence numbers.
 * Synchronization: Uses 'commit_mutex' to serialize control-plane operations across Netlink tasks.
 */
struct nftables_pernet {
	struct list_head	tables;
	struct list_head	commit_list;
	struct list_head	destroy_list;
	struct list_head	commit_set_list;
	struct list_head	binding_list;
	struct list_head	module_list;
	struct list_head	notify_list;
	struct mutex		commit_mutex;
	u64			table_handle;
	u64			tstamp;
	unsigned int		base_seq;
	unsigned int		gc_seq;
	u8			validate_state;
	struct work_struct	destroy_work;
};

extern unsigned int nf_tables_net_id;

static inline struct nftables_pernet *nft_pernet(const struct net *net)
{
	return net_generic(net, nf_tables_net_id);
}

static inline u64 nft_net_tstamp(const struct net *net)
{
	return nft_pernet(net)->tstamp;
}

#define __NFT_REDUCE_READONLY	1UL
#define NFT_REDUCE_READONLY	(void *)__NFT_REDUCE_READONLY

void nft_reg_track_update(struct nft_regs_track *track,
			  const struct nft_expr *expr, u8 dreg, u8 len);
void nft_reg_track_cancel(struct nft_regs_track *track, u8 dreg, u8 len);
void __nft_reg_track_cancel(struct nft_regs_track *track, u8 dreg);

static inline bool nft_reg_track_cmp(struct nft_regs_track *track,
				     const struct nft_expr *expr, u8 dreg)
{
	return track->regs[dreg].selector &&
	       track->regs[dreg].selector->ops == expr->ops &&
	       track->regs[dreg].num_reg == 0;
}

#endif /* _NET_NF_TABLES_H */
