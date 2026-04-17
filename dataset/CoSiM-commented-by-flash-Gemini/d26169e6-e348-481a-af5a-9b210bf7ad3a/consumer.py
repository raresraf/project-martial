"""
@d26169e6-e348-481a-af5a-9b210bf7ad3a/consumer.py
@brief Threaded marketplace simulation with partitioned inventory and dual-lock synchronization.
* Algorithm: Concurrent producer-consumer model with shared state managed via producer-specific locks and a global marketplace lock.
* Functional Utility: Facilitates a virtual market where producers generate goods and consumers manage items via a cart-based acquisition system.
"""

from threading import Thread
from time import sleep


class Consumer(Thread):
    """
    @brief Consumer entity that performs synchronized shopping operations across multiple carts.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        @brief Initializes the consumer with its assigned shopping lists.
        """
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        Thread.__init__(self, **kwargs)

    def run(self):
        """
        @brief Main execution loop for shopping activities.
        Algorithm: Iterative operation processing with busy-wait retry and backoff for failed acquisitions.
        """
        for cart in self.carts:
            # Logic: Initializes a new transaction context.
            id_cart = self.marketplace.new_cart()
            for purchase in cart:
                if purchase["type"] == 'add':
                    for _ in range(purchase["quantity"]):
                        # Logic: Continuous attempt to secure product from inventory.
                        cart_new_product = self.marketplace.add_to_cart(id_cart,
                                                                        purchase["product"])
                        while not cart_new_product:
                            # Functional Utility: Throttles retry attempts during stock-outs.
                            sleep(self.retry_wait_time)
                            cart_new_product = self.marketplace.add_to_cart(id_cart,
                                                                            purchase["product"])
                else:
                    # Logic: Returns items from cart to inventory.
                    for _ in range(purchase["quantity"]):
                        self.marketplace.remove_from_cart(id_cart, purchase["product"])
            
            # Post-condition: Completes the transaction and displays purchased items.
            order = self.marketplace.place_order(id_cart)
            for buy in order:
                print(self.name + ' bought ' + str(buy))


import threading


class Marketplace:
    """
    @brief Centralized inventory controller with multi-level locking to support concurrent producers and consumers.
    """
    
    def __init__(self, queue_size_per_producer):
        """
        @brief Initializes the marketplace and its synchronization primitives.
        """
        self.queue_size_per_producer = queue_size_per_producer
        self.contor_producer = -1
        self.contor_consumer = -1
        self.product_queue = [[]] # Intent: Buffer list for products, partitioned by producer ID.
        self.cart_queue = [[]]    # Intent: Buffer list for active consumer carts.
        self.producer_cart = [[]] # Intent: Mapping of producer ID to its items currently in consumer carts.
        self.lock = threading.Lock() # Intent: Global lock for metadata and general structural updates.
        self.producer_locks = []     # Intent: List of locks for per-producer inventory synchronization.

    def register_producer(self):
        """
        @brief Onboards a new producer and initializes its dedicated storage and synchronization objects.
        """
        with self.lock:
            self.contor_producer += 1
            tmp = self.contor_producer
            self.product_queue.append([])
            self.producer_cart.append([])
            self.producer_locks.append(threading.Lock())
        return tmp

    def publish(self, producer_id, product):
        """
        @brief Adds a product to a producer's buffer if capacity permits.
        Invariant: Uses producer-specific lock to ensure thread-safe append.
        """
        self.producer_locks[producer_id].acquire()
        if self.queue_size_per_producer > len(self.product_queue[producer_id]):
            self.product_queue[producer_id].append(product)
            self.producer_locks[producer_id].release()
            return True
        self.producer_locks[producer_id].release()
        return False

    def new_cart(self):
        """
        @brief Allocates a new transaction identifier for a consumer.
        """
        self.lock.acquire()
        self.contor_consumer += 1
        self.cart_queue.append([])
        tmp = self.contor_consumer
        self.lock.release()
        return tmp

    def add_to_cart(self, cart_id, product):
        """
        @brief Transfers a product unit from any producer to a specific consumer cart.
        Algorithm: Search-and-consume across all producer buffers.
        Invariant: Uses global lock to ensure consistent ownership transfer.
        """
        if any(product in list_products for list_products in self.product_queue):
            for products in self.product_queue:
                for prod in products:
                    if prod == product:
                        self.lock.acquire()
                        tmp = self.product_queue.index(products)
                        # Logic: Preserves producer-affinity to support consistent returns.
                        self.producer_cart[tmp].append((product, cart_id))
                        self.cart_queue[cart_id].append(product)
                        self.product_queue[tmp].remove(product)
                        self.lock.release()
                        return True
        return False

    def remove_from_cart(self, cart_id, product):
        """
        @brief Returns an item from a cart back to its original producer's buffer.
        """
        self.cart_queue[cart_id].remove(product)
        for producer in self.producer_cart:
            if (cart_id, product) in producer:
                # Logic: Identifies the original producer of the item.
                tmp = self.producer_cart.index(producer)
                self.producer_cart.remove((cart_id, product))
                # Post-condition: Restores item to its source buffer under the producer lock.
                self.producer_locks[tmp].acquire()
                self.product_queue[tmp].append(product)
                self.producer_locks[tmp].release()

    def place_order(self, cart_id):
        """
        @brief Returns the list of items finalized in the given cart.
        """
        return self.cart_queue[cart_id]


from threading import Thread
from time import sleep


class Producer(Thread):
    """
    @brief Producer agent that generates goods for the marketplace.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        @brief Initializes the producer with its production schedule.
        """
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        Thread.__init__(self, **kwargs)

    def run(self):
        """
        @brief Main production lifecycle.
        Algorithm: Iterative batch generation with fixed production latency and quota-constrained publication.
        """
        id_producer = self.marketplace.register_producer()
        while True:
            for product in self.products:
                for _ in range(product[1]):
                    # Domain: Simulation of manufacturing time.
                    sleep(product[2])
                    # Logic: Attempts to publish item; retries on buffer saturation.
                    market_confirm = self.marketplace.publish(id_producer, product[0])
                    while not market_confirm:
                        # Functional Utility: Throttles re-publication attempts.
                        sleep(self.republish_wait_time)
                        market_confirm = self.marketplace.publish(id_producer, product[0])


from dataclasses import dataclass

@dataclass(init=True, repr=True, order=False, frozen=True)
class Product:
    """
    @brief Immutable base schema for marketplace goods.
    """
    name: str
    price: int


@dataclass(init=True, repr=True, order=False, frozen=True)
class Tea(Product):
    """
    @brief Specialized product type for tea items.
    """
    type: str


@dataclass(init=True, repr=True, order=False, frozen=True)
class Coffee(Product):
    """
    @brief Specialized product type for coffee items.
    """
    acidity: str
    roast_level: str
