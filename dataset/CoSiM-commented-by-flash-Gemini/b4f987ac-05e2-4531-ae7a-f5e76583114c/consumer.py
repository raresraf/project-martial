"""
@file consumer.py
@brief multi-threaded marketplace simulation utilizing the Producer-Consumer architectural pattern.
@details Implements a synchronized environment where asynchronous producer threads 
generate goods and consumer threads perform batch transactions through a shared 
intermediary (Marketplace). Includes logic for inventory reservation and rollback.
"""

from threading import Thread
import time

class Consumer(Thread):
    """
    @brief Asynchronous client agent that manages a sequence of shopping transactions.
    Functional Utility: Executes a set of shopping lists (carts), handling inventory 
    contention through backoff-based retries for unavailable items.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        @param carts Sequence of transaction lists to be processed.
        @param marketplace Shared state intermediary.
        @param retry_wait_time Latency interval when an operation cannot be fulfilled.
        """
        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time

    def run(self):
        """
        @brief Consumer lifecycle execution loop.
        Invariant: For every cart, all 'add' and 'remove' operations are executed 
        until the requested quantity is satisfied.
        """
        for cart in self.carts:
            # Session Initiation: Allocates a new transaction identifier.
            cart_id = self.marketplace.new_cart()

            for item in cart:
                quantity_fulfilled = 0
                while quantity_fulfilled < item["quantity"]:
                    # Logic: Dispatch to marketplace based on operation type.
                    if item["type"] == "add":
                        success = self.marketplace.add_to_cart(cart_id, item["product"])
                    else:
                        success = self.marketplace.remove_from_cart(cart_id, item["product"])

                    if success:
                        quantity_fulfilled += 1
                    else:
                        # Protocol: Polling-based retry with wait interval.
                        time.sleep(self.retry_wait_time)

            # Finalization: Commits the transaction.
            self.marketplace.place_order(cart_id)


from threading import Lock, currentThread

class Marketplace:
    """
    @brief Centralized transaction coordinator and inventory manager.
    Architecture: Uses isolated locks for session IDs and output streams to 
    ensure thread-safety while minimizing contention on data structures.
    """
    
    def __init__(self, queue_size_per_producer):
        # Bound: Maximum items allowed per individual producer buffer.
        self.max_products_per_size = queue_size_per_producer
        
        # State: Registry of active shopping sessions.
        self.carts = {}
        # State: Registry of available inventory partitioned by producer.
        self.producers = {}
        # State: Registry of reserved items (currently in carts) to allow for rollbacks.
        self.reserved = {}

        self.id_cart = 0
        self.id_producer = 0

        # Synchronization: Protects atomic ID generation and stdout.
        self.lock_id_cart = Lock()
        self.lock_id_producer = Lock()
        self.lock_print = Lock()

    def register_producer(self):
        """
        @brief Onboards a new supply entity with a unique monotonic identifier.
        """
        with self.lock_id_producer:
            self.id_producer += 1
            prod_id = self.id_producer

        self.producers[prod_id] = []
        return prod_id

    def publish(self, producer_id, product):
        """
        @brief Injects a product into the producer's specific inventory buffer.
        @return True if successful, False if the buffer is saturated.
        """
        prod_id = int(producer_id)

        # Optimization: Resource limit enforcement.
        if len(self.producers[prod_id]) >= self.max_products_per_size:
            return False

        self.producers[prod_id].append(product)
        return True

    def new_cart(self):
        """
        @brief Allocates a new transaction session for a consumer.
        """
        with self.lock_id_cart:
            self.id_cart += 1
            cart_id = self.id_cart

        self.carts[cart_id] = []
        return cart_id

    def add_to_cart(self, cart_id, product):
        """
        @brief Reserves an item by transferring it from a producer's pool to a cart.
        Logic: Linear search across all producer buffers for the requested product.
        """
        found = False
        target_producer = None

        for prod_key in self.producers:
            if product in self.producers[prod_key]:
                found = True
                target_producer = prod_key
                break

        if not found:
            return False

        # Atomic Transfer: Removes from source and tracks in 'reserved' for rollback safety.
        self.producers[target_producer].remove(product)
        
        if target_producer not in self.reserved:
            self.reserved[target_producer] = []
        self.reserved[target_producer].append(product)

        self.carts[cart_id].append(product)
        return True

    def remove_from_cart(self, cart_id, product):
        """
        @brief Rollback operation: returns a reserved item back to its original producer.
        """
        found_in_reserved = False
        rem_producer = None
        
        # Search Strategy: Identifies the originating producer from the reserved list.
        for prod_key in self.reserved:
            if product in self.reserved[prod_key]:
                found_in_reserved = True
                rem_producer = prod_key
                break

        if not found_in_reserved:
            return False

        # Atomic Reversion: Reverses the reservation.
        self.carts[cart_id].remove(product)
        self.producers[rem_producer].append(product)
        self.reserved[rem_producer].remove(product)
        
        return True

    def place_order(self, cart_id):
        """
        @brief Finalizes the consumer session and outputs the transaction log.
        """
        final_items = self.carts.pop(cart_id, [])

        for item in final_items:
            with self.lock_print:
                # Synchronization: ensures non-interleaved output in concurrent environments.
                print("{} bought {}".format(currentThread().getName(), item))

        return final_items


from threading import Thread
import time

class Producer(Thread):
    """
    @brief Autonomous manufacturing agent that generates and publishes products.
    Functional Utility: Manages the supply lifecycle, including production latency 
    and backpressure handling.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        self.prod_id = self.marketplace.register_producer()

    def run(self):
        """
        @brief Main production loop.
        Invariant: Continuously attempts to fulfill manufacturing quotas for each item.
        """
        while True:
            for (product, target_quantity, production_latency) in self.products:
                for _ in range(target_quantity):
                    # Logic: Publication attempt with backoff for full buffers.
                    if self.marketplace.publish(str(self.prod_id), product):
                        # Latency: simulates time taken to manufacture.
                        time.sleep(production_latency)
                    else:
                        # Protocol: wait for consumer demand to clear buffer space.
                        time.sleep(self.republish_wait_time)


from dataclasses import dataclass

@dataclass(init=True, repr=True, order=False, frozen=True)
class Product:
    """
    @brief immutable base entity for tradable goods.
    """
    name: str
    price: int

@dataclass(init=True, repr=True, order=False, frozen=True)
class Tea(Product):
    """
    @brief Domain-specific specialization of Product.
    """
    type: str

@dataclass(init=True, repr=True, order=False, frozen=True)
class Coffee(Product):
    """
    @brief Domain-specific specialization of Product.
    """
    acidity: str
    roast_level: str
