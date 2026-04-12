"""
@file consumer.py
@brief Concurrent marketplace simulation utilizing the Producer-Consumer architectural pattern.
@details Implements a shared-state intermediary (Marketplace) where multiple producer 
and consumer threads interact via synchronized buffers and shopping carts.
"""

import time
from threading import Thread

class Consumer(Thread):
    """
    @brief Asynchronous client agent that performs batch purchasing operations.
    Functional Utility: Manages multiple shopping sessions (carts) and handles 
    inventory-related retries with backoff.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        @param carts Sequence of shopping lists to be processed.
        @param marketplace Shared state controller.
        @param retry_wait_time Latency interval when attempting to add unavailable items.
        """
        Thread.__init__(self)
        self.carts = carts
        self.marketplace = marketplace
        self.wait_time = retry_wait_time
        self.name = kwargs["name"]

    def run(self):
        """
        @brief Main execution loop for the consumer session.
        Invariant: For every cart, all operations are executed until completion or successful retry.
        """
        for cart in self.carts:
            # Session Initiation: Allocates a new transaction context in the marketplace.
            cart_id = self.marketplace.new_cart()
            for operation in cart:
                if operation["type"] == "add":
                    i = 0
                    while i < operation["quantity"]:
                        # Logic: Attempts to reserve an item from the global pool.
                        verify = self.marketplace.add_to_cart(cart_id, operation["product"])
                        /**
                         * Block Logic: Polling-based retry mechanism for out-of-stock items.
                         * Protocol: Waits for 'wait_time' before re-querying the marketplace.
                         */
                        while not verify:
                            time.sleep(self.wait_time)
                            verify = self.marketplace.add_to_cart(cart_id, operation["product"])
                        i += 1
                
                elif operation["type"] == "remove":
                    i = 0
                    while i < operation["quantity"]:
                        # Reversion: Returns an item from the cart to the marketplace.
                        self.marketplace.remove_from_cart(cart_id, operation["product"])
                        i += 1

            # Finalization: Commits the cart and retrieves the list of purchased goods.
            orders = self.marketplace.place_order(cart_id)
            for order in orders:
                print("%s bought %s" % (self.name, order[0]))

from threading import Lock

class Marketplace:
    """
    @brief Centralized synchronization point and inventory manager.
    Architecture: Uses isolated locks for inventory (buffers) and transactions (carts) 
    to maximize concurrency.
    """
    
    def __init__(self, queue_size_per_producer):
        self.queue_size = queue_size_per_producer
        self.producer_id = 0
        self.cart_id = 0
        
        # State: Maps producer IDs to their respective inventory lists.
        self.producers_buffers = {}
        
        # State: Maps cart IDs to lists of reserved (item, producer_id) tuples.
        self.carts_list = {}
        
        # Synchronization: Protects shared access to the producers' inventory.
        self.lock_buffers = Lock()
        
        # Synchronization: Protects shared access to consumer session metadata.
        self.lock_carts = Lock()

    def register_producer(self):
        """
        @brief Onboards a new supply entity with a unique monotonic identifier.
        """
        with self.lock_buffers:
            self.producer_id += 1
            self.producers_buffers[self.producer_id] = []
            new_id = self.producer_id
        return new_id

    def publish(self, producer_id, product):
        """
        @brief Injects a product into the producer's specific inventory buffer.
        @return True if successful, False if the buffer is at capacity.
        """
        self.lock_buffers.acquire()
        if len(self.producers_buffers[producer_id]) < self.queue_size:
            self.producers_buffers[producer_id].append(product)
            self.lock_buffers.release()
            return True

        self.lock_buffers.release()
        return False

    def new_cart(self):
        """
        @brief Generates a unique transaction identifier for a new consumer session.
        """
        with self.lock_carts:
            self.cart_id += 1
            self.carts_list[self.cart_id] = []
            new_cart = self.cart_id
        return new_cart

    def add_to_cart(self, cart_id, product):
        """
        @brief Reserves an item by transferring it from a producer's buffer to a consumer's cart.
        Logic: Performs a global search across all producers for the target product.
        """
        self.lock_buffers.acquire()
        for producer in self.producers_buffers:
            for prod in self.producers_buffers[producer]:
                if prod == product:
                    # Atomic Transfer Phase.
                    self.producers_buffers[producer].remove(prod)
                    self.carts_list[cart_id].append((prod, producer))
                    self.lock_buffers.release()
                    return True

        self.lock_buffers.release()
        return False

    def remove_from_cart(self, cart_id, product):
        """
        @brief Rollback operation: returns a reserved item to its original source.
        Functional Utility: Ensures inventory integrity during transaction cancellations.
        """
        for (prod, producer) in self.carts_list[cart_id]:
            if prod == product:
                with self.lock_carts:
                    self.carts_list[cart_id].remove((prod, producer))
                # Logic: Returns item to the specific producer it originated from.
                self.producers_buffers[producer].append(prod)
                break

    def place_order(self, cart_id):
        """
        @brief Finalizes the transaction and returns the cart contents.
        """
        return self.carts_list[cart_id]


from threading import Thread
import time

class Producer(Thread):
    """
    @brief Supply-side agent that generates and publishes products to the marketplace.
    Functional Utility: Manages the manufacturing lifecycle including production 
    latency and buffer-overflow retries.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        Thread.__init__(self, group=None, target=None, name=None, args=(), kwargs={},
                        daemon=kwargs.get("daemon"))
        self.products = products
        self.name = kwargs["name"]
        self.marketplace = marketplace      
        self.wait_time = republish_wait_time
        self.id_producer = 0

    def run(self):
        """
        @brief Main execution loop for the producer.
        Invariant: Continuously attempts to fulfill production quotas for each item.
        """
        self.id_producer = self.marketplace.register_producer()
        while True:
            for prod in self.products:
                i = 0
                while i < prod[1]:
                    # Logic: Tries to publish to the marketplace.
                    verify = self.marketplace.publish(self.id_producer, prod[0])
                    /**
                     * Block Logic: Wait-and-retry strategy for saturated marketplace buffers.
                     */
                    while not verify:
                        time.sleep(self.wait_time)
                        verify = self.marketplace.publish(self.id_producer, prod[0])
                    
                    # Production Latency: Simulates time taken to create the item.
                    time.sleep(prod[2])
                    i += 1
