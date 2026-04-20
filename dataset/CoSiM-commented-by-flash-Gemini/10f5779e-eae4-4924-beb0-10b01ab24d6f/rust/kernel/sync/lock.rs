// SPDX-License-Identifier: GPL-2.0

/**
 * @10f5779e-eae4-4924-beb0-10b01ab24d6f/rust/kernel/sync/lock.rs
 * @brief Generic abstraction for kernel-level mutual exclusion primitives in Rust.
 * 
 * Functional Intent: Provides a type-safe, backend-agnostic framework for kernel 
 * synchronization (Mutexes, Spinlocks, etc.). It leverages Rust's ownership and 
 * RAII models to ensure that locks are correctly initialized, held during 
 * data access, and automatically released. It interfaces directly with C-side 
 * kernel infrastructure like lockdep for deadlock detection and validation.
 * 
 * Domain: Kernel Synchronization, Memory Safety, FFI Interop.
 */

use super::LockClassKey;
use crate::{
    str::CStr,
    types::{NotThreadSafe, Opaque, ScopeGuard},
};
use core::{cell::UnsafeCell, marker::PhantomPinned, pin::Pin};
use pin_init::{pin_data, pin_init, PinInit};

pub mod mutex;
pub mod spinlock;

pub(super) mod global;
pub use global::{GlobalGuard, GlobalLock, GlobalLockBackend, GlobalLockedBy};

/**
 * @trait Backend
 * @brief Strategy interface for physical lock implementations.
 * 
 * Functional Utility: Decouples the Rust-facing Lock/Guard API from the 
 * underlying kernel synchronization mechanism (e.g., raw spinlocks vs. 
 * sleepable mutexes). 
 * 
 * Safety Invariant: Implementers must guarantee strict mutual exclusion between 
 * 'lock' and 'unlock' calls across all execution contexts (threads/CPUs).
 */
pub unsafe trait Backend {
    /// Internal kernel state (e.g., struct mutex, spinlock_t).
    type State;

    /// Transient state managed by the guard (e.g., interrupt flags).
    type GuardState;

    /// @brief Orchestrates C-level initialization of the lock structure.
    unsafe fn init(
        ptr: *mut Self::State,
        name: *const crate::ffi::c_char,
        key: *mut bindings::lock_class_key,
    );

    /// @brief Primary acquisition primitive.
    #[must_use]
    unsafe fn lock(ptr: *mut Self::State) -> Self::GuardState;

    /// @brief Non-blocking acquisition attempt.
    unsafe fn try_lock(ptr: *mut Self::State) -> Option<Self::GuardState>;

    /// @brief Primary release primitive.
    unsafe fn unlock(ptr: *mut Self::State, guard_state: &Self::GuardState);

    /// @brief Re-establishes a previously released lock state.
    unsafe fn relock(ptr: *mut Self::State, guard_state: &mut Self::GuardState) {
        *guard_state = unsafe { Self::lock(ptr) };
    }

    /// @brief Validation hook for lockdep integration.
    unsafe fn assert_is_held(ptr: *mut Self::State);
}

/**
 * @struct Lock
 * @brief RAII container for data protected by a kernel synchronization primitive.
 * 
 * Logic: Wraps the protected data in an UnsafeCell and the backend state in 
 * an Opaque wrapper. It enforces pinning to accommodate self-referential C 
 * structures and architecture-specific alignment requirements.
 */
#[repr(C)]
#[pin_data]
pub struct Lock<T: ?Sized, B: Backend> {
    #[pin]
    state: Opaque<B::State>,

    #[pin]
    _pin: PhantomPinned,

    pub(crate) data: UnsafeCell<T>,
}

unsafe impl<T: ?Sized + Send, B: Backend> Send for Lock<T, B> {}
unsafe impl<T: ?Sized + Send, B: Backend> Sync for Lock<T, B> {}

impl<T, B: Backend> Lock<T, B> {
    /**
     * new - Multi-stage initialization of the lock and its contents.
     * Logic: Returns a PinInit implementation that ensures the lock state 
     * is correctly registered with the kernel's lockdep framework during 
     * the construction phase.
     */
    pub fn new(t: T, name: &'static CStr, key: Pin<&'static LockClassKey>) -> impl PinInit<Self> {
        pin_init!(Self {
            data: UnsafeCell::new(t),
            _pin: PhantomPinned,
            state <- Opaque::ffi_init(|slot| unsafe {
                B::init(slot, name.as_char_ptr(), key.as_ptr())
            }),
        })
    }
}

impl<B: Backend> Lock<(), B> {
    /**
     * from_raw - Zero-copy wrapper for pre-existing C-level locks.
     * Logic: Casts a raw pointer to a Rust Lock reference. This is only sound 
     * for zero-sized data protected by the lock, as it assumes the layout 
     * matches the backend state exactly.
     */
    pub unsafe fn from_raw<'a>(ptr: *mut B::State) -> &'a Self {
        unsafe { &*ptr.cast() }
    }
}

impl<T: ?Sized, B: Backend> Lock<T, B> {
    /**
     * lock - Transitions the caller to owner state.
     * Logic: Invokes the backend acquisition and wraps the resulting 
     * lifetime-bound state in a RAII Guard.
     */
    pub fn lock(&self) -> Guard<'_, T, B> {
        let state = unsafe { B::lock(self.state.get()) };
        unsafe { Guard::new(self, state) }
    }

    /**
     * try_lock - Opportunistic acquisition attempt.
     */
    #[must_use = "if unused, the lock will be immediately unlocked"]
    pub fn try_lock(&self) -> Option<Guard<'_, T, B>> {
        unsafe { B::try_lock(self.state.get()).map(|state| Guard::new(self, state)) }
    }
}

/**
 * @struct Guard
 * @brief Scoped accessor providing mutually exclusive access to data.
 * 
 * Functional Utility: Implements Deref to allow 'safe' access to the 
 * underlying data while the lock is held. On Drop, it automatically 
 * invokes the backend release logic to prevent deadlock.
 */
#[must_use = "the lock unlocks immediately when the guard is unused"]
pub struct Guard<'a, T: ?Sized, B: Backend> {
    pub(crate) lock: &'a Lock<T, B>,
    pub(crate) state: B::GuardState,
    _not_send: NotThreadSafe,
}

unsafe impl<T: Sync + ?Sized, B: Backend> Sync for Guard<'_, T, B> {}

impl<'a, T: ?Sized, B: Backend> Guard<'a, T, B> {
    pub fn lock_ref(&self) -> &'a Lock<T, B> {
        self.lock
    }

    /**
     * do_unlocked - Temporary suspension of the lock for non-reentrant operations.
     * Logic: Releases the lock, executes the provided closure, and uses 
     * a ScopeGuard to ensure the lock is reacquired even if the closure panics.
     */
    pub(crate) fn do_unlocked<U>(&mut self, cb: impl FnOnce() -> U) -> U {
        unsafe { B::unlock(self.lock.state.get(), &self.state) };

        let _relock = ScopeGuard::new(||
                unsafe { B::relock(self.lock.state.get(), &mut self.state) });

        cb()
    }
}

impl<T: ?Sized, B: Backend> core::ops::Deref for Guard<'_, T, B> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        // Block Logic: Safe data projection.
        // Invariant: Memory access is only possible while the guard exists.
        unsafe { &*self.lock.data.get() }
    }
}

impl<T: ?Sized, B: Backend> core::ops::DerefMut for Guard<'_, T, B> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { &mut *self.lock.data.get() }
    }
}

impl<T: ?Sized, B: Backend> Drop for Guard<'_, T, B> {
    /**
     * drop - Automatic unlock on scope exit.
     */
    fn drop(&mut self) {
        unsafe { B::unlock(self.lock.state.get(), &self.state) };
    }
}

impl<'a, T: ?Sized, B: Backend> Guard<'a, T, B> {
    /**
     * @brief Factory for internal guard creation.
     */
    pub unsafe fn new(lock: &'a Lock<T, B>, state: B::GuardState) -> Self {
        unsafe { B::assert_is_held(lock.state.get()) };

        Self {
            lock,
            state,
            _not_send: NotThreadSafe,
        }
    }
}
