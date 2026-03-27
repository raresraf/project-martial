/**
 * @file hugetlb.h
 * @brief Generic HugeTLB (Huge Translation Lookaside Buffer) definitions and inline functions.
 *
 * This header provides a common interface for managing huge pages across different
 * architectures within the Linux kernel. It defines inline functions that wrap
 * architecture-specific Page Table Entry (PTE) operations or provide default
 * implementations for huge page attributes like write protection, dirty status,
 * and access flags.
 *
 * Domain: Linux Kernel, Memory Management, HugeTLB, Page Tables, Architecture-Generic.
 */
/* SPDX-License-Identifier: GPL-2.0 */
#ifndef _ASM_GENERIC_HUGETLB_H
#define _ASM_GENERIC_HUGETLB_H

#include <linux/swap.h>
#include <linux/swapops.h>

/**
 * @brief Checks if a HugeTLB PTE is writable.
 * @param pte The HugeTLB page table entry.
 * @return True if the PTE is writable, false otherwise.
 * Functional Utility: Abstracts architecture-specific check for PTE writability.
 */
static inline unsigned long huge_pte_write(pte_t pte)
{
	return pte_write(pte);
}

/**
 * @brief Checks if a HugeTLB PTE is marked dirty.
 * @param pte The HugeTLB page table entry.
 * @return True if the PTE is dirty, false otherwise.
 * Functional Utility: Abstracts architecture-specific check for PTE dirty status.
 */
static inline unsigned long huge_pte_dirty(pte_t pte)
{
	return pte_dirty(pte);
}

/**
 * @brief Marks a HugeTLB PTE as writable, without VMA considerations.
 * @param pte The HugeTLB page table entry.
 * @return The modified HugeTLB PTE.
 * Functional Utility: Abstracts architecture-specific PTE modification to grant write permissions.
 */
static inline pte_t huge_pte_mkwrite(pte_t pte)
{
	return pte_mkwrite_novma(pte);
}

#ifndef __HAVE_ARCH_HUGE_PTE_WRPROTECT
/**
 * @brief Write-protects a HugeTLB PTE.
 * @param pte The HugeTLB page table entry.
 * @return The modified HugeTLB PTE, with write access removed.
 * Functional Utility: Abstracts architecture-specific PTE modification to remove write permissions.
 */
static inline pte_t huge_pte_wrprotect(pte_t pte)
{
	return pte_wrprotect(pte);
}
#endif

/**
 * @brief Marks a HugeTLB PTE as dirty.
 * @param pte The HugeTLB page table entry.
 * @return The modified HugeTLB PTE.
 * Functional Utility: Abstracts architecture-specific PTE modification to set the dirty bit.
 */
static inline pte_t huge_pte_mkdirty(pte_t pte)
{
	return pte_mkdirty(pte);
}

/**
 * @brief Modifies the protection flags of a HugeTLB PTE.
 * @param pte The HugeTLB page table entry.
 * @param newprot The new page protection flags.
 * @return The modified HugeTLB PTE with the new protection.
 * Functional Utility: Abstracts architecture-specific PTE modification to change protection attributes.
 */
static inline pte_t huge_pte_modify(pte_t pte, pgprot_t newprot)
{
	return pte_modify(pte, newprot);
}

#ifndef __HAVE_ARCH_HUGE_PTE_MKUFFD_WP
/**
 * @brief Marks a HugeTLB PTE for userfaultfd write-protection.
 * @param pte The HugeTLB page table entry.
 * @return The modified HugeTLB PTE, marked for userfaultfd write-protection.
 * Functional Utility: Combines write-protection with userfaultfd tracking.
 */
static inline pte_t huge_pte_mkuffd_wp(pte_t pte)
{
	return huge_pte_wrprotect(pte_mkuffd_wp(pte));
}
#endif

#ifndef __HAVE_ARCH_HUGE_PTE_CLEAR_UFFD_WP
/**
 * @brief Clears the userfaultfd write-protection flag from a HugeTLB PTE.
 * @param pte The HugeTLB page table entry.
 * @return The modified HugeTLB PTE, with userfaultfd write-protection cleared.
 * Functional Utility: Removes userfaultfd write-protection from a PTE.
 */
static inline pte_t huge_pte_clear_uffd_wp(pte_t pte)
{
	return pte_clear_uffd_wp(pte);
}
#endif

#ifndef __HAVE_ARCH_HUGE_PTE_UFFD_WP
/**
 * @brief Checks if a HugeTLB PTE is marked for userfaultfd write-protection.
 * @param pte The HugeTLB page table entry.
 * @return True if the PTE is userfaultfd write-protected, false otherwise.
 * Functional Utility: Abstracts architecture-specific check for userfaultfd write-protection.
 */
static inline int huge_pte_uffd_wp(pte_t pte)
{
	return pte_uffd_wp(pte);
}
#endif

#ifndef __HAVE_ARCH_HUGE_PTE_CLEAR
/**
 * @brief Clears a HugeTLB PTE.
 * @param mm The memory descriptor.
 * @param addr The virtual address for the PTE.
 * @param ptep Pointer to the HugeTLB page table entry.
 * @param sz Size of the huge page.
 * Functional Utility: Abstracts architecture-specific clearing of a PTE.
 */
static inline void huge_pte_clear(struct mm_struct *mm, unsigned long addr,
		    pte_t *ptep, unsigned long sz)
{
	pte_clear(mm, addr, ptep);
}
#endif

#ifndef __HAVE_ARCH_HUGE_SET_HUGE_PTE_AT
/**
 * @brief Sets a HugeTLB PTE at a specific address.
 * @param mm The memory descriptor.
 * @param addr The virtual address for the PTE.
 * @param ptep Pointer to the HugeTLB page table entry.
 * @param pte The new HugeTLB PTE to set.
 * @param sz Size of the huge page.
 * Functional Utility: Abstracts architecture-specific setting of a PTE.
 */
static inline void set_huge_pte_at(struct mm_struct *mm, unsigned long addr,
		pte_t *ptep, pte_t pte, unsigned long sz)
{
	set_pte_at(mm, addr, ptep, pte);
}
#endif

#ifndef __HAVE_ARCH_HUGE_PTEP_GET_AND_CLEAR
/**
 * @brief Atomically gets and clears a HugeTLB PTE.
 * @param mm The memory descriptor.
 * @param addr The virtual address for the PTE.
 * @param ptep Pointer to the HugeTLB page table entry.
 * @param sz Size of the huge page.
 * @return The original HugeTLB PTE before clearing.
 * Functional Utility: Atomically retrieves the PTE and then clears it.
 */
static inline pte_t huge_ptep_get_and_clear(struct mm_struct *mm,
		unsigned long addr, pte_t *ptep, unsigned long sz)
{
	return ptep_get_and_clear(mm, addr, ptep);
}
#endif

#ifndef __HAVE_ARCH_HUGE_PTEP_CLEAR_FLUSH
/**
 * @brief Atomically clears and flushes a HugeTLB PTE from the TLB.
 * @param vma The virtual memory area structure.
 * @param addr The virtual address for the PTE.
 * @param ptep Pointer to the HugeTLB page table entry.
 * @return The original HugeTLB PTE before clearing.
 * Functional Utility: Atomically clears the PTE and ensures TLB consistency.
 */
static inline pte_t huge_ptep_clear_flush(struct vm_area_struct *vma,
		unsigned long addr, pte_t *ptep)
{
	return ptep_clear_flush(vma, addr, ptep);
}
#endif

#ifndef __HAVE_ARCH_HUGE_PTE_NONE
/**
 * @brief Checks if a HugeTLB PTE is effectively null (no page mapped).
 * @param pte The HugeTLB page table entry.
 * @return True if the PTE is null, false otherwise.
 * Functional Utility: Abstracts architecture-specific check for a null PTE.
 */
static inline int huge_pte_none(pte_t pte)
{
	return pte_none(pte);
}
#endif

/* Please refer to comments above pte_none_mostly() for the usage */
#ifndef __HAVE_ARCH_HUGE_PTE_NONE_MOSTLY
/**
 * @brief Checks if a HugeTLB PTE is mostly null (no page mapped or a marker).
 * @param pte The HugeTLB page table entry.
 * @return True if the PTE is null or a marker, false otherwise.
 * Functional Utility: Expands on `huge_pte_none` to also identify marker PTEs.
 */
static inline int huge_pte_none_mostly(pte_t pte)
{
	return huge_pte_none(pte) || is_pte_marker(pte);
}
#endif

#ifndef __HAVE_ARCH_HUGE_PTEP_SET_WRPROTECT
/**
 * @brief Atomically sets write-protection on a HugeTLB PTE.
 * @param mm The memory descriptor.
 * @param addr The virtual address for the PTE.
 * @param ptep Pointer to the HugeTLB page table entry.
 * Functional Utility: Atomically removes write permissions from a PTE.
 */
static inline void huge_ptep_set_wrprotect(struct mm_struct *mm,
		unsigned long addr, pte_t *ptep)
{
	ptep_set_wrprotect(mm, addr, ptep);
}
#endif

#ifndef __HAVE_ARCH_HUGE_PTEP_SET_ACCESS_FLAGS
/**
 * @brief Atomically sets access flags for a HugeTLB PTE.
 * @param vma The virtual memory area structure.
 * @param addr The virtual address for the PTE.
 * @param ptep Pointer to the HugeTLB page table entry.
 * @param pte The new HugeTLB PTE with desired access flags.
 * @param dirty Flag indicating if the page should be marked dirty.
 * @return True if the PTE's dirty or accessed state was changed, false otherwise.
 * Functional Utility: Atomically updates PTE access flags (e.g., accessed, dirty bits).
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
 * @brief Gets a HugeTLB PTE.
 * @param mm The memory descriptor.
 * @param addr The virtual address for the PTE.
 * @param ptep Pointer to the HugeTLB page table entry.
 * @return The HugeTLB PTE.
 * Functional Utility: Abstracts architecture-specific retrieval of a PTE.
 */
static inline pte_t huge_ptep_get(struct mm_struct *mm, unsigned long addr, pte_t *ptep)
{
	return ptep_get(ptep);
}
#endif

#ifndef __HAVE_ARCH_GIGANTIC_PAGE_RUNTIME_SUPPORTED
/**
 * @brief Checks if gigantic pages are supported at runtime.
 * @return True if gigantic pages are supported, false otherwise.
 * Functional Utility: Provides a runtime check for a specific huge page size.
 */
static inline bool gigantic_page_runtime_supported(void)
{
	return IS_ENABLED(CONFIG_ARCH_HAS_GIGANTIC_PAGE);
}
#endif /* __HAVE_ARCH_GIGANTIC_PAGE_RUNTIME_SUPPORTED */

#endif /* _ASM_GENERIC_HUGETLB_H */
