
/**
 * @file consumer.py
 * @brief Concurrent marketplace simulation with per-producer synchronization and transaction logging.
 * 
 * Functional Intent: Provides a multi-threaded framework for a marketplace where 
 * Producers provide inventory and Consumers perform transactional shopping. 
 * The system employs fine-grained mutexes (one per producer) to minimize 
 * contention during parallel stock updates. It ensures atomic state transitions 
 * between 'available', 'reserved', and 'finalized' items while maintaining 
 * a production-grade audit log.
 * 
 * Domain: Production Systems, Concurrency, Resource Orchestration.
 */

from threading import Thread, Lock
from typing import List, Dict
from time import sleep


class Consumer(Thread):
    /**
     * @class Consumer
     * @brief Customer thread that executes a scripted sequence of shopping cart operations.
     * 
     * Logic: For each cart, the consumer opens a session in the marketplace. 
     * It implements a blocking retry loop for 'add' operations when stock is 
     * temporarily exhausted. Finalizes the entire cart in one atomic block (place_order).
     */
    
    # Synchronization: Class-level mutex for serialized console output.
    print_lock = Lock()

    def __init__(self, carts: List[List[Dict]], marketplace: 'Marketplace',
                 retry_wait_time: float, **kwargs):
        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time

    def run(self):
        /**
         * Block Logic: Shopping execution lifecycle.
         * Invariant: Every product added to a cart is either purchased 
         * or explicitly released back to the supplier list.
         */
        for op_list in self.carts:
            cart_id = self.marketplace.new_cart()
            for operation in op_list:
                if operation['type'] == 'add':
                    # Block Logic: Transactional reservation with polling retry.
                    for _ in range(operation['quantity']):
                        retval = self.marketplace.add_to_cart(cart_id, operation['product'])
                        while not retval:
                            # Logic: Backoff delay before re-attempting reservation.
                            sleep(self.retry_wait_time)
                            retval = self.marketplace.add_to_cart(cart_id, operation['product'])
                elif operation['type'] == 'remove':
                    # Logic: Reverses a product reservation within the current cart context.
                    for _ in range(operation['quantity']):
                        self.marketplace.remove_from_cart(cart_id, operation['product'])

            # Serialization: Commit items and synchronize output to avoid interleaved prints.
            msg = "\n".join([f'{self.name} bought {str(prod)}'
                             for prod in self.marketplace.place_order(cart_id)])
            with Consumer.print_lock:
                print(msg)


from logging import Logger, Formatter
from logging.handlers import RotatingFileHandler
import time
from threading import Lock
from typing import Dict, List, Tuple
import unittest


class Marketplace:
    /**
     * @class Marketplace
     * @brief Centralized thread-safe registry for producers, consumers, and inventory.
     * 
     * Logic: Distributes state into multiple mutex-protected domains. 
     * Uses independent locks for ID generation and a per-producer lock for 
     * inventory operations to maximize parallel throughput.
     */

    def __init__(self, queue_size_per_producer: int):
        self.queue_size_per_producer = queue_size_per_producer
        
        # Synchronization: Atomic ID generators.
        self.producer_id_generator_lock = Lock()
        self.producer_count = 0
        self.cart_id_generator_lock = Lock()
        self.cart_count = 0
        
        # Invariant: Maps cart identifiers to their list of (Product, SourceProducer) tuples.
        self.carts: Dict[int, List[Tuple['Product', str]]] = {}
        
        # Invariant: Maps producer IDs to a tuple of (Lock, List of [is_reserved, Product]).
        self.products: Dict[str, Tuple[Lock, List[Tuple[bool, 'Product']]]] = {}
        
        # Logging: Initializing production audit trail via rotating file handler.
        self.logger = Logger("marketplace logger", level="INFO")
        fmt = Formatter(fmt='%(asctime)s %(message)s', datefmt="%Y-%m-%d %H:%M:%S")
        fmt.converter = time.gmtime
        rfh = RotatingFileHandler("marketplace.log", delay=True)
        rfh.formatter = fmt
        self.logger.addHandler(rfh)

    def register_producer(self):
        /**
         * register_producer - Assigns a unique ID and initializes producer-specific storage.
         */
        self.logger.info('entry: register_producer')
        with self.producer_id_generator_lock:
            producer_id = str(self.producer_count)
            self.producer_count += 1
        
        # Invariant: Each producer has a dedicated Lock for fine-grained synchronization.
        self.products[producer_id] = (Lock(), [])
        self.logger.info('exit: register_producer')
        return producer_id

    def publish(self, producer_id: str, product: 'Product'):
        /**
         * publish - Injects a new product unit into the supplier's pool.
         * 
         * Logic: Rejects publication if the supplier's queue is at capacity. 
         * Initial state is always 'unreserved' (False).
         */
        self.logger.info('enter: publish %s %s', producer_id, product)
        lock, plist = self.products[producer_id]
        with lock:
            if len(plist) == self.queue_size_per_producer:
                self.logger.info('exit_fail: publish %s %s - queue full', producer_id, product)
                return False
            plist.append([False, product])
        self.logger.info('exit_success: publish %s %s', producer_id, product)
        return True

    def new_cart(self):
        self.logger.info('enter: new_cart')
        with self.cart_id_generator_lock:
            cart_id = self.cart_count
            self.cart_count += 1
        
        self.carts[cart_id] = []
        self.logger.info('exit: new_cart')
        return cart_id

    def add_to_cart(self, cart_id: int, product: 'Product'):
        /**
         * add_to_cart - Atomically reserves a product for a specific cart.
         * 
         * Algorithm: Segmented linear search.
         * 1. Iterates through all registered producers.
         * 2. Acquires the producer-specific lock to ensure exclusive access.
         * 3. Finds the first matching 'unreserved' unit.
         * 4. Flags the unit as 'reserved' and appends to the consumer's cart.
         */
        self.logger.info('enter: add_to_cart %d %s', cart_id, product)
        if cart_id not in self.carts:
            return False

        for producer_id, (lock, plist) in self.products.items():
            with lock:
                for entry in plist:
                    if not entry[0] and entry[1] == product:
                        # Synchronization: Transition unit to 'reserved' state.
                        entry[0] = True
                        self.carts[cart_id].append((product, producer_id))
                        self.logger.info('exit_success: add_to_cart %d %s', cart_id, product)
                        return True
        return False

    def remove_from_cart(self, cart_id: int, product: 'Product'):
        /**
         * remove_from_cart - Reverses a reservation, restoring unit availability to the producer.
         */
        self.logger.info('enter: remove_from_cart %d %s', cart_id, product)
        if cart_id not in self.carts:
            return
        
        for entry_in_cart in self.carts[cart_id]:
            if entry_in_cart[0] == product:
                prod_id = entry_in_cart[1]
                lock, plist = self.products[prod_id]
                
                with lock:
                    for entry in plist:
                        if entry[0] and entry[1] == product:
                            # Logic: Reverses reservation in the source producer list.
                            entry[0] = False
                            self.carts[cart_id].remove(entry_in_cart)
                            self.logger.info('exit_success: remove_from_cart %d %s', cart_id, product)
                            return
                return

    def place_order(self, cart_id: int):
        /**
         * place_order - Finalizes the sale by removing items from producers permanently.
         * 
         * Logic: For every item in the cart, it identifies the source producer 
         * and performs a hard removal of the reserved unit, freeing up space 
         * in the producer's publication quota.
         */
        self.logger.info('enter: place_order %d', cart_id)
        if cart_id not in self.carts:
            return None
        
        final_products = []
        for product, producer in self.carts[cart_id]:
            lock, plist = self.products[producer]
            with lock:
                # Synchronization: Permanent extraction from global inventory.
                plist.remove([True, product])
            final_products.append(product)
        
        # Invariant: Clears the cart reference from the system memory.
        self.carts.pop(cart_id)
        self.logger.info('exit: place_order %d', cart_id)
        return final_products


from threading import Thread
from typing import List
from time import sleep

class Producer(Thread):
    /**
     * @class Producer
     * @brief Background worker thread responsible for populating the marketplace.
     * 
     * Logic: Continuously publishes its assigned products. It implements 
     * a blocking wait loop when the marketplace is full for its specific ID, 
     * ensuring it respects system-wide capacity constraints.
     */

    def __init__(self, products: List[Tuple['Product', int, float]], marketplace: 'Marketplace',
                 republish_wait_time: float, **kwargs):
        Thread.__init__(self, **kwargs)
        self.marketplace = marketplace
        self.products = products
        self.republish_wait_time = republish_wait_time
        # Invariant: Obtains a supplier identity for tracking its active buffer.
        self.producer_id = marketplace.register_producer()

    def run(self):
        while True:
            for prod, quant, production_time in self.products:
                for _ in range(quant):
                    # Block Logic: Quota enforcement retry.
                    ret_val = self.marketplace.publish(self.producer_id, prod)
                    while not ret_val:
                        sleep(self.republish_wait_time)
                        ret_val = self.marketplace.publish(self.producer_id, prod)
                    
                    # Logic: Simulated production delay.
                    sleep(production_time)


from dataclasses import dataclass

@dataclass(init=True, repr=True, order=False, frozen=True)
class Product:
    name: str
    price: int


@dataclass(init=True, repr=True, order=False, frozen=True)
class Tea(Product):
    type: str

    def __eq__(self, other):
        return isinstance(other, Tea) \
            and self.name == other.name \
            and self.price == other.price \
            and self.type == other.type


@dataclass(init=True, repr=True, order=False, frozen=True)
class Coffee(Product):
    acidity: str
    roast_level: str

    def __eq__(self, other):
        return isinstance(other, Coffee) \
            and self.name == other.name \
            and self.price == other.price \
            and self.acidity == other.acidity \
            and self.roast_level == other.roast_level
