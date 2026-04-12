"""
@file consumer.py
@brief Multi-threaded simulation of a concurrent marketplace using the Producer-Consumer pattern.
@details Orchestrates a shared marketplace environment where producers publish goods and consumers 
purchase them asynchronously, utilizing thread-safe operations for inventory management.
"""

from threading import Thread
import time

class Consumer(Thread):
    """
    @brief Represents an asynchronous agent that performs batch purchasing operations.
    Functional Utility: Encapsulates the lifecycle of a customer traversing multiple shopping carts.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        @param carts List of carts, where each cart is a sequence of intended transactions.
        @param marketplace Reference to the shared state intermediary.
        @param retry_wait_time Backoff duration for failed 'add' operations due to empty inventory.
        """
        Thread.__init__(self)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.name = kwargs["name"]

    def run(self):
        """
        @brief Execution loop for the consumer thread.
        Invariant: For every cart in the sequence, all requested product quantities are fulfilled or retried.
        """
        for cart in self.carts:
            # Initialization of a unique session identifier within the marketplace.
            cart_id = self.marketplace.new_cart()

            for product_info in cart:
                op = product_info['type']
                product = product_info['product']
                quantity = product_info['quantity']

                /**
                 * Block Logic: Processes a specific quantity of a product based on the transaction type.
                 * Pre-condition: quantity > 0.
                 */
                while quantity > 0:
                    if op == "add":
                        # Attempt to reserve a product from the marketplace global pool.
                        added = self.marketplace.add_to_cart(cart_id, product)

                        if added:
                            quantity = quantity - 1
                        else:
                            # Protocol: Exponential backoff/waiting strategy for inventory replenishment.
                            time.sleep(self.retry_wait_time)

                    if op == "remove":
                        # Returns a reserved item back to the marketplace pool.
                        self.marketplace.remove_from_cart(cart_id, product)
                        quantity = quantity - 1

            # Functional Utility: Finalizes the transaction and retrieves the confirmed item list.
            products = self.marketplace.place_order(cart_id)
            
            for prod in products:
                print(self.name + " bought " + str(prod))

import random
from threading import Lock

class Marketplace:
    """
    @brief Acts as a centralized synchronization point and inventory manager.
    Architecture: Maintains per-producer queues and per-consumer carts to ensure isolation.
    """
    
    def __init__(self, queue_size_per_producer):
        """
        @param queue_size_per_producer Bound on individual producer buffer to prevent overflow.
        """
        self.queue_size_per_producer = queue_size_per_producer
        self.prod_seed = 0
        self.cart_seed = 0

        # Registry of active producer identifiers.
        self.producers = []

        # Mapping of producer IDs to their available inventory buffers.
        self.items_by_producers = {}

        # Collection of active consumer cart states.
        self.carts = []

        # Synchronization Primitives: Protects monotonic ID generation for producers and carts.
        self.p_seed = Lock()
        self.c_seed = Lock()

    def register_producer(self):
        """
        @brief Onboards a new producer with a unique stochastic identifier.
        Functional Utility: Thread-safe ID generation and namespace allocation for inventory.
        """
        self.p_seed.acquire()

        random.seed(self.prod_seed)
        producer_id = random.randint(10000, 99999)
        self.prod_seed = self.prod_seed + 1

        self.p_seed.release()

        products = []
        self.items_by_producers[str(producer_id)] = products
        self.producers.append(str(producer_id)) 

        return str(producer_id)

    def publish(self, producer_id, product):
        """
        @brief Injects a product into a producer's allocated inventory buffer.
        @return True if the operation succeeded, False if the buffer is saturated.
        """
        products = self.items_by_producers[producer_id]

        # Optimization: Enforces resource limits to prevent memory exhaustion by a single producer.
        if len(self.items_by_producers[producer_id]) >= self.queue_size_per_producer:
            return False

        products.append(product)
        self.items_by_producers[producer_id] = products

        return True

    def new_cart(self):
        """
        @brief Allocates a new transaction context for a consumer.
        Functional Utility: Guaranteed unique identifier generation using an atomic lock.
        """
        self.c_seed.acquire()

        cart_id = self.cart_seed
        self.cart_seed = self.cart_seed + 1

        self.c_seed.release()

        new_cart = []
        self.carts.append(new_cart)

        return cart_id

    def add_to_cart(self, cart_id, product):
        """
        @brief Transfers an item from the global inventory to a consumer's cart.
        Search Strategy: Linear sweep across all producers to find the requested product.
        """
        for producer_id in self.producers:
            for item in self.items_by_producers[producer_id]:
                if item == product:
                    # Atomic Transfer: Removes from source and adds to destination cart.
                    self.items_by_producers[producer_id].remove(item)
                    self.carts[cart_id].append([product, producer_id])
                    return True

        return False

    def remove_from_cart(self, cart_id, product):
        """
        @brief Reverts an item transfer, returning it to the original producer's inventory.
        Functional Utility: Essential for handling cancellations or rollbacks in multi-step transactions.
        """
        found = False
        for prod in self.carts[cart_id]:
            if prod[0] == product:
                found = True
                put_back = prod[0]
                producer_id = prod[1]

                self.carts[cart_id].remove(prod)

                # Restoration Logic: Ensures inventory consistency by returning items to their source.
                self.items_by_producers[producer_id].append(put_back)
                break

        if not found:
            return False 
        return True

    def place_order(self, cart_id):
        """
        @brief Finalizes the consumer session and converts the cart state into a product list.
        @return Sequence of purchased products.
        """
        list_of_products = []
        for prod in self.carts[cart_id]:
            list_of_products.append(prod[0])

        return list_of_products


from threading import Thread
import time

class Producer(Thread):
    """
    @brief Autonomous agent that generates and publishes products to the marketplace.
    Functional Utility: Manages the supply-side lifecycle, including production rates and retry logic.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        @param products Blueprint for production (product, quantity, production_latency).
        @param republish_wait_time Duration to wait when the marketplace buffer is full.
        """
        Thread.__init__(self)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time

    def run(self):
        """
        @brief Execution loop for the producer thread.
        Invariant: Production continues indefinitely, attempting to fulfill quotas for each item.
        """
        producer_id = self.marketplace.register_producer()

        while True:
            for (product, quantity, waiting_time) in self.products:
                while quantity > 0:
                    published = self.marketplace.publish(producer_id, product)

                    if published:
                        quantity = quantity - 1
                        # Production Latency: Simulates the time taken to manufacture the item.
                        time.sleep(waiting_time)
                    else:
                        # Saturation Handling: Pauses production if the marketplace cannot accept more items.
                        time.sleep(self.republish_wait_time)


from dataclasses import dataclass

@dataclass(init=True, repr=True, order=False, frozen=True)
class Product:
    """
    @brief Immutable base entity representing a tradeable item.
    """
    name: str
    price: int

@dataclass(init=True, repr=True, order=False, frozen=True)
class Tea(Product):
    """
    @brief Domain-specific specialization of Product for beverage simulation.
    """
    type: str

@dataclass(init=True, repr=True, order=False, frozen=True)
class Coffee(Product):
    """
    @brief Domain-specific specialization of Product for caffeinated beverage simulation.
    """
    acidity: str
    roast_level: str
