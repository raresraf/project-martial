"""
@file consumer.py
@brief multi-threaded marketplace simulation utilizing the Producer-Consumer design pattern.
@details Orchestrates an environment where asynchronous producer threads generate goods 
and consumer threads perform batch transactions through a shared intermediary (Marketplace).
"""

from threading import Thread
import time

# Constants: Domain-specific keys for transaction dictionaries.
QUANTITY = "quantity"
PRODUCT = "product"
TYPE = "type"
ADD = "add"
REMOVE = "remove"

class Consumer(Thread):
    """
    @brief Asynchronous agent that manages a sequence of shopping transactions.
    Functional Utility: Executes a set of shopping lists (carts), handling inventory 
    contention through backoff-based retries.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        @param carts Sequence of lists containing intended transactions.
        @param marketplace Central state intermediary.
        @param retry_wait_time Duration to wait when an item is unavailable.
        """
        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time

    def run(self):
        """
        @brief Execution loop for the consumer thread.
        Invariant: All items in each cart are successfully processed or retried indefinitely.
        """
        for cart in self.carts:
            # Session Initiation: Allocates a new transaction identifier.
            id_cart = self.marketplace.new_cart()

            for element in cart:
                counter = 0
                while counter < element[QUANTITY]:
                    success = False
                    
                    # Logic: Dispatch to marketplace based on transaction type.
                    if element[TYPE] == ADD:
                        success = self.marketplace.add_to_cart(id_cart, element[PRODUCT])
                    elif element[TYPE] == REMOVE:
                        success = self.marketplace.remove_from_cart(id_cart, element[PRODUCT])
                    
                    if success:
                        counter += 1
                    else:
                        # Protocol: Wait-and-retry strategy for missing inventory.
                        time.sleep(self.retry_wait_time)
            
            # Finalization: Commits the transaction.
            self.marketplace.place_order(id_cart)


from threading import Lock, currentThread

class Marketplace:
    """
    @brief Centralized inventory manager and transaction coordinator.
    Architecture: Uses a multi-lock strategy to ensure thread-safe access to 
    global inventory and individual consumer sessions.
    """
    
    def __init__(self, queue_size_per_producer):
        self.queue_size_per_producer = queue_size_per_producer
        
        # State: Tracks inventory levels and item registries.
        self.producer_size = []
        self.producer_products = []
        self.producers_items = {} # Maps product -> originating producer ID.
        self.cart_items = {}      # Maps cart_id -> list of reserved products.
        self.number_of_carts = 0

        # Synchronization: Specialized locks for different operational domains.
        self.lock_number_carts = Lock()
        self.lock_add = Lock()
        self.lock_remove = Lock()
        self.lock_register = Lock()
        self.lock_print = Lock()

    def register_producer(self):
        """
        @brief Onboards a new supply entity with a monotonic identifier.
        """
        with self.lock_register:
            prod_id = len(self.producer_size)
            self.producer_size.append(0)
        return prod_id

    def publish(self, producer_id, product):
        """
        @brief Injects a product into the global pool.
        @return True if successful, False if the producer's buffer is saturated.
        """
        if self.producer_size[producer_id] >= self.queue_size_per_producer:
            return False
        
        # Logic: Increment inventory level and register the item.
        self.producer_size[producer_id] += 1
        self.producer_products.append(product)
        self.producers_items[product] = producer_id

        return True

    def new_cart(self):
        """
        @brief Allocates a new transaction session for a consumer.
        """
        with self.lock_number_carts:
            self.number_of_carts += 1
            new_cart_id = self.number_of_carts 

        self.cart_items[new_cart_id] = []
        return new_cart_id

    def add_to_cart(self, cart_id, product):
        """
        @brief Atomically transfers an item from global inventory to a specific cart.
        Search Strategy: Verifies existence in the global pool.
        """
        with self.lock_add:
            if product not in self.producer_products:
                return False
            
            # Atomic Transfer: Update producer quota and remove from shared pool.
            self.producer_size[self.producers_items[product]] -= 1
            self.producer_products.remove(product)

        self.cart_items[cart_id].append(product)
        return True

    def remove_from_cart(self, cart_id, product):
        """
        @brief Rollback: returns a reserved item back to the shared inventory.
        """
        self.cart_items[cart_id].remove(product)
        self.producer_products.append(product)

        with self.lock_remove:
            self.producer_size[self.producers_items[product]] += 1
        return True
        
    def place_order(self, cart_id):
        """
        @brief Commits the transaction and outputs the result.
        """
        my_prods = self.cart_items.pop(cart_id, None)

        if my_prods:
            for elem in my_prods:
                with self.lock_print:
                    print(currentThread().getName() + " bought " + str(elem))

        return my_prods


from threading import Thread
import time

class Producer(Thread):
    """
    @brief Autonomous manufacturing agent that generates products for the marketplace.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        # Initialization: Registers self with the marketplace intermediary.
        self.id = self.marketplace.register_producer()

    def run(self):
        """
        @brief Main production loop.
        Invariant: Continuously attempts to fulfill quotas for assigned goods.
        """
        while True:
            for (product, number_product, time_to_wait) in self.products:
                counter = 0
                while counter < number_product:
                    published = self.marketplace.publish(self.id, product)

                    if published:
                        # Production Latency: simulates manufacturing time.
                        time.sleep(time_to_wait)
                        counter += 1
                    else:
                        # Saturation Handling: waits for inventory to be consumed.
                        time.sleep(self.republish_wait_time)


from dataclasses import dataclass

@dataclass(init=True, repr=True, order=False, frozen=True)
class Product:
    """
    @brief immutable base entity for tradeable items.
    """
    name: str
    price: int

@dataclass(init=True, repr=True, order=False, frozen=True)
class Tea(Product):
    """
    @brief Specialized product category.
    """
    type: str

@dataclass(init=True, repr=True, order=False, frozen=True)
class Coffee(Product):
    """
    @brief Specialized product category.
    """
    acidity: str
    roast_level: str
