"""
@d3c7ed64-f6a0-4ea6-bc62-6eaa5b8067dc/consumer.py
@brief E-commerce marketplace simulation using status-based inventory tracking and multi-threaded coordination.
* Algorithm: State-based resource management (Available/Unavailable) with explicit mutual exclusion and retry logic.
* Functional Utility: Orchestrates the lifecycle of products from creation by producers to acquisition by consumers through a central market.
"""

from threading import Thread, Lock
import time

class Consumer(Thread):
    """
    @brief Consumer entity that performs synchronized shopping operations.
    """
    cart_id = -1
    name = ''
    my_lock = Lock() # Intent: Global synchronization for printing results.

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        @brief Initializes the consumer with its assigned shopping list and market connection.
        """
        Thread.__init__(self)
        self.carts = carts
        self.retry_wait_time = retry_wait_time
        self.marketplace = marketplace
        self.name = kwargs['name']

    def run(self):
        """
        @brief Main transactional loop for processing all assigned carts.
        Algorithm: Iterative operation execution with status verification and backoff.
        """
        # Block Logic: Ensures serialized processing of consumer outputs.
        self.my_lock.acquire()
        for i in range(len(self.carts)):
            self.cart_id = self.marketplace.new_cart()
            for j in range(len(self.carts[i])):
                # Logic: Dispatches add/remove operations based on cart specification.
                if self.carts[i][j]['type'] == 'add':
                    for k in range(self.carts[i][j]['quantity']):
                        verify = False
                        while not verify:
                            verify = self.marketplace.add_to_cart(self.cart_id,
                                                                  self.carts[i][j]['product']
                                                                  )
                            if not verify:
                                # Functional Utility: Prevents busy-waiting during inventory stock-out.
                                time.sleep(self.retry_wait_time)

                elif self.carts[i][j]['type'] == 'remove':
                    for k in range(self.carts[i][j]['quantity']):
                        self.marketplace.remove_from_cart(self.cart_id, self.carts[i][j]['product'])
            
            # Post-condition: Completes order and releases acquired items.
            list_1 = self.marketplace.place_order(self.cart_id)
            for k in range(len(list_1) - 1, -1, -1):
                print(self.name + ' bought ' + str(list_1[k][0]))
                self.marketplace.remove_from_cart(self.cart_id, list_1[k][0])
        self.my_lock.release()

from threading import Lock

class Marketplace:
    """
    @brief Inventory hub that manages shared product queues and consumer carts.
    * Domain: Central state manager using class-level static lists and counters.
    """
    id_producer = 0
    id_cart = 0
    queues = [] # Intent: Stores products per producer with availability status.
    carts = []  # Intent: Stores products associated with each consumer cart.
    my_Lock1 = Lock() # Intent: Synchronizes product acquisition (add_to_cart).
    my_Lock2 = Lock() # Intent: Synchronizes product return (remove_from_cart).
    done = 0

    def __init__(self, queue_size_per_producer):
        """
        @brief Initializes market instance with capacity limits.
        """
        self.queue_size_per_producer = queue_size_per_producer

    def register_producer(self):
        """
        @brief Registers a new producer and allocates a dedicated queue.
        """
        self.queues.append([])
        self.id_producer = self.id_producer + 1
        return self.id_producer - 1

    def publish(self, producer_id, product):
        """
        @brief Adds a product to a producer's queue if space is available.
        Logic: Tags products as 'Disponibil' (Available) for market consumption.
        """
        if len(self.queues[producer_id]) >= self.queue_size_per_producer:
            return False
        self.queues[producer_id].append([product, "Disponibil"])
        return True

    def new_cart(self):
        """
        @brief Assigns a new cart ID to a consumer.
        """
        self.carts.append([])
        self.id_cart = self.id_cart + 1
        return self.id_cart - 1

    def add_to_cart(self, cart_id, product):
        """
        @brief Transfers a product from any producer queue to a specific consumer cart.
        Algorithm: Exhaustive search with state verification.
        Invariant: Uses my_Lock1 to ensure atomic status transition from 'Available' to 'Unavailable'.
        """
        verify = 0
        for i in range(len(self.queues)):
            for j in range(len(self.queues[i])):
                self.my_Lock1.acquire()
                if product == self.queues[i][j][0] \
                        and self.queues[i][j][1] == 'Disponibil' \
                        and verify == 0:
                    # Logic: Record product and source producer ID for potential future returns.
                    self.carts[cart_id].append([product, i])
                    self.queues[i][j][1] = 'Indisponibil'
                    verify = 1
                    self.my_Lock1.release()
                    break
                self.my_Lock1.release()
                if verify == 1:
                    break
        if verify == 1:
            return True
        return False

    def remove_from_cart(self, cart_id, product):
        """
        @brief Returns a product from a cart back to its original producer's queue.
        Invariant: Uses my_Lock2 to ensure atomic status transition from 'Unavailable' to 'Available'.
        """
        for i in range(len(self.carts[cart_id])):
            if product == self.carts[cart_id][i][0]:
                for j in range(len(self.queues[self.carts[cart_id][i][1]])):
                    self.my_Lock2.acquire()
                    if self.queues[self.carts[cart_id][i][1]][j][0] == product \
                            and self.queues[self.carts[cart_id][i][1]][j][1] == 'Indisponibil':
                        self.queues[self.carts[cart_id][i][1]][j][1] = 'Disponibil'
                        self.carts[cart_id].remove(self.carts[cart_id][i])
                        self.my_Lock2.release()
                        return True
                    self.my_Lock2.release()
        return False

    def place_order(self, cart_id):
        """
        @brief Finalizes the shopping phase for a cart.
        """
        return self.carts[cart_id]

from threading import Thread
import time

class Producer(Thread):
    """
    @brief Background thread generating products for the marketplace.
    """
    producer_id = -1

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        @brief Initializes the producer with production parameters.
        """
        Thread.__init__(self, daemon=kwargs['daemon'])
        self.republish_wait_time = republish_wait_time
        self.marketplace = marketplace
        self.products = products

    def run(self):
        """
        @brief Main production loop.
        Algorithm: Iterative generation and publication with dynamic backoff.
        """
        self.producer_id = self.marketplace.register_producer()
        while True:
            for i in range(len(self.products)):
                for j in range(self.products[i][1]):
                    # Logic: Attempts to publish; if queue is full, waits before retrying.
                    verify = self.marketplace.publish(self.producer_id, self.products[i][0])
                    # Domain: Simulation of production time.
                    time.sleep(self.products[i][2])
                    if not verify:
                        # Functional Utility: Throttles producer attempts when market is saturated.
                        time.sleep(self.republish_wait_time)
                        break

from dataclasses import dataclass

@dataclass(init=True, repr=True, order=False, frozen=True)
class Product:
    """
    @brief Base schema for marketplace goods.
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
