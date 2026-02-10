/* SPDX-License-Identifier: GPL-2.0 */
/**
 * @file asm-generic/hugetlb.h
 * @brief Generic architecture-independent HugeTLB helper functions.
 *
 * @details This file provides a generic, portable implementation for a set of
 * functions and macros that manipulate HugeTLB page table entries (PTEs). It
 * serves as a fallback for architectures that do not need or do not provide
 * their own custom, optimized versions of these helpers.
 *
 * The functions defined here typically wrap standard page table operations
 * (e.g., pte_mkwrite, pte_dirty) to provide a consistent API for the core
 * HugeTLB subsystem. Architectures can override these weak definitions by
 * defining their own versions, often prefixed with `__HAVE_ARCH_*`. This
 * allows for architecture-specific optimizations while maintaining common
 * kernel code.
 */
#ifndef _ASM_GENERIC_HUGETLB_H
#define _ASM_GENERIC_HUGETLB_H

#include <linux/swap.h>
#include <linux/swapops.h>

/**
 * @brief Checks if a huge PTE has the write permission bit set.
 * @param pte The huge page table entry.
 * @return True if the entry is writable, false otherwise.
 */
static inline unsigned long huge_pte_write(pte_t pte)
{
	return pte_write(pte);
}

/**
 * @brief Checks if a huge PTE is marked as dirty.
 * @param pte The huge page table entry.
 * @return True if the entry is dirty, false otherwise.
 */
static inline unsigned long huge_pte_dirty(pte_t pte)
{
	return pte_dirty(pte);
}

/**
 * @brief Makes a huge PTE writable.
 * @param pte The huge page table entry.
 * @return A new PTE with the write permission bit set.
 */
static inline pte_t huge_pte_mkwrite(pte_t pte)
{
	return pte_mkwrite_novma(pte);
}

#ifndef __HAVE_ARCH_HUGE_PTE_WRPROTECT
/**
 * @brief Removes write permissions from a huge PTE.
 * @param pte The huge page table entry.
 * @return A new PTE with the write permission bit cleared.
 */
static inline pte_t huge_pte_wrprotect(pte_t pte)
{
	return pte_wrprotect(pte);
}
#endif

/**
 * @brief Marks a huge PTE as dirty.
 * @param pte The huge page table entry.
 * @return A new PTE with the dirty bit set.
 */
static inline pte_t huge_pte_mkdirty(pte_t pte)
{
	return pte_mkdirty(pte);
}

/**
 * @brief Modifies a huge PTE with new protection attributes.
 * @param pte The original huge page table entry.
 * @param newprot The new page protection attributes.
 * @return A new PTE with the updated protection.
 */
static inline pte_t huge_pte_modify(pte_t pte, pgprot_t newprot)
{
	return pte_modify(pte, newprot);
}

#ifndef __HAVE_ARCH_HUGE_PTE_MKUFFD_WP
/**
 * @brief Marks a huge PTE for userfaultfd write protection.
 * @param pte The huge page table entry.
 * @return A new PTE with userfaultfd write-protection bits set.
 */
static inline pte_t huge_pte_mkuffd_wp(pte_t pte)
{
	return huge_pte_wrprotect(pte_mkuffd_wp(pte));
}
#endif

#ifndef __HAVE_ARCH_HUGE_PTE_CLEAR_UFFD_WP
/**
 * @brief Clears userfaultfd write protection from a huge PTE.
 * @param pte The huge page table entry.
 * @return A new PTE with userfaultfd write-protection bits cleared.
 */
static inline pte_t huge_pte_clear_uffd_wp(pte_t pte)
{
	return pte_clear_uffd_wp(pte);
}
#endif

#ifndef __HAVE_ARCH_HUGE_PTE_UFFD_WP
/**
 * @brief Checks if a huge PTE is marked for userfaultfd write protection.
 * @param pte The huge page table entry.
 * @return True if the userfaultfd write-protection bit is set, false otherwise.
 */
static inline int huge_pte_uffd_wp(pte_t pte)
{
	return pte_uffd_wp(pte);
}
#endif

#ifndef __HAVE_ARCH_HUGE_PTE_CLEAR
/**
 * @brief Clears a huge PTE from the page table.
 * @param mm The memory management structure.
 * @param addr The virtual address of the mapping.
 * @param ptep Pointer to the huge page table entry.
 * @param sz The size of the huge page.
 */
static inline void huge_pte_clear(struct mm_struct *mm, unsigned long addr,
		    pte_t *ptep, unsigned long sz)
{
	pte_clear(mm, addr, ptep);
}
#endif

#ifndef __HAVE_ARCH_HUGE_SET_HUGE_PTE_AT
/**
 * @brief Sets a huge PTE at a specific location in the page table.
 * @param mm The memory management structure.
 * @param addr The virtual address of the mapping.
 * @param ptep Pointer to the huge page table entry.
 * @param pte The new PTE value to set.
 * @param sz The size of the huge page.
 */
static inline void set_huge_pte_at(struct mm_struct *mm, unsigned long addr,
		pte_t *ptep, pte_t pte, unsigned long sz)
{
	set_pte_at(mm, addr, ptep, pte);
}
#endif

#ifndef __HAVE_ARCH_HUGE_PTEP_GET_AND_CLEAR
/**
 * @brief Atomically gets and clears a huge PTE.
 * @param mm The memory management structure.
 * @param addr The virtual address of the mapping.
 * @param ptep Pointer to the huge page table entry.
 * @param sz The size of the huge page.
 * @return The old PTE value.
 */
static inline pte_t huge_ptep_get_and_clear(struct mm_struct *mm,
		unsigned long addr, pte_t *ptep, unsigned long sz)
{
	return ptep_get_and_clear(mm, addr, ptep);
}
#endif

#ifndef __HAVE_ARCH_HUGE_PTEP_CLEAR_FLUSH
/**
 * @brief Clears a huge PTE and flushes the corresponding TLB entry.
 * @param vma The virtual memory area.
 * @param addr The virtual address.
 * @param ptep Pointer to the huge page table entry.
 * @return The old PTE value.
 */
static inline pte_t huge_ptep_clear_flush(struct vm_area_struct *vma,
		unsigned long addr, pte_t *ptep)
{
	return ptep_clear_flush(vma, addr, ptep);
}
#endif

#ifndef __HAVE_ARCH_HUGE_PTE_NONE
/**
 * @brief Checks if a huge PTE is not present.
 * @param pte The huge page table entry.
 * @return True if the PTE is zero (not present).
 */
static inline int huge_pte_none(pte_t pte)
{
	return pte_none(pte);
}
#endif

/* Please refer to comments above pte_none_mostly() for the usage */
#ifndef __HAVE_ARCH_HUGE_PTE_NONE_MOSTLY
/**
 * @brief Checks if a huge PTE is mostly not present (i.e., zero or a migration/swap marker).
 * @param pte The huge page table entry.
 * @return True if the PTE is empty or contains a non-present marker.
 */
static inline int huge_pte_none_mostly(pte_t pte)
{
	return huge_pte_none(pte) || is_pte_marker(pte);
}
#endif

#ifndef __HAVE_ARCH_HUGE_PTEP_SET_WRPROTECT
/**
 * @brief Sets write protection on a huge PTE.
 * @param mm The memory management structure.
 * @param addr The virtual address.
 * @param ptep Pointer to the huge page table entry.
 */
static inline void huge_ptep_set_wrprotect(struct mm_struct *mm,
		unsigned long addr, pte_t *ptep)
{
	ptep_set_wrprotect(mm, addr, ptep);
}
#endif

#ifndef __HAVE_ARCH_HUGE_PTEP_SET_ACCESS_FLAGS
/**
 * @brief Sets access and dirty flags on a huge PTE.
 * @param vma The virtual memory area.
 * @param addr The virtual address.
 * @param ptep Pointer to the huge page table entry.
 * @param pte The new PTE value.
 * @param dirty Whether to set the dirty flag.
 * @return 1 if the PTE was changed, 0 otherwise.
 */
static inline int huge_ptep_set_access_flags(struct vm_area_struct *vma,
		unsigned long addr, pte_t *ptep,
		pte_t pte, int dirty)
{
	return ptep_set_access_flags(vma, addr, ptep, pte, dirty);
}
#endif

#ifndef __HAVE_ARCH_HUGE_PTEP_GET
/**
 * @brief Atomically gets the value of a huge PTE.
 * @param mm The memory management structure.
 * @param addr The virtual address.
 * @param ptep Pointer to the huge page table entry.
 * @return The PTE value.
 */
static inline pte_t huge_ptep_get(struct mm_struct *mm, unsigned long addr, pte_t *ptep)
{
	return ptep_get(ptep);
}
#endif

#ifndef __HAVE_ARCH_GIGANTIC_PAGE_RUNTIME_SUPPORTED
/**
 * @brief Checks if gigantic pages are supported at runtime.
 * @return True if the architecture has build-time support for gigantic pages.
 * This can be overridden by architectures that determine support at boot time.
 */
static inline bool gigantic_page_runtime_supported(void)
{
	return IS_ENABLED(CONFIG_ARCH_HAS_GIGANTIC_PAGE);
}
#endif /* __HAVE_ARCH_GIGANTIC_PAGE_RUNTIME_SUPPORTED */

#endif /* _ASM_GENERIC_HUGETLB_H */
