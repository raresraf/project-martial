"""
@cea5fa2c-54d0-456c-aa5e-328e498c2da2/consumer.py
@brief Distributed marketplace simulation using state-based product reservation and atomic inventory transitions.
* Algorithm: Two-stage commit style transaction (Reserve -> Commit/Rollback) using a custom `PublishedProduct` wrapper with a `reserved` flag.
* Functional Utility: Facilitates a virtual market where producers generate goods and consumers manage items via a reservation system that ensures exclusive access during cart operations.
"""

from threading import Thread
from time import sleep


class Consumer(Thread):
    """
    @brief Consumer entity that performs multi-step shopping transactions across multiple lists.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        @brief Initializes the consumer with its assigned shopping lists.
        """
        Thread.__init__(self, name=kwargs["name"], kwargs=kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time

    def run(self):
        """
        @brief Main execution loop for shopping activities.
        Algorithm: Iterative operation processing with a greedy retry strategy for failed reservations.
        """
        for lista_cumparaturi in self.carts:
            # Logic: Initializes a new transaction context.
            cart_id = self.marketplace.new_cart()
            
            failed = True
            while failed:
                failed = False
                for operatie in lista_cumparaturi:
                    # Block Logic: Sequential processing of add/remove requests.
                    cantitate = operatie["quantity"]
                    for _ in range(cantitate):
                        if operatie["type"] == "add":
                            if self.marketplace.add_to_cart(cart_id, operatie["product"]):
                                # Logic: Mutates the input list to track remaining items (unsafe if shared).
                                operatie["quantity"] = operatie["quantity"] - 1
                            else:
                                # Functional Utility: Triggers a global transaction retry on failure.
                                failed = True
                                break
                        elif operatie["type"] == "remove":
                            if self.marketplace.remove_from_cart(cart_id, operatie["product"]):
                                operatie["quantity"] = operatie["quantity"] - 1
                            else:
                                failed = True
                                break
                
                if failed:
                    # Logic: Throttles retry attempts during resource contention.
                    sleep(self.retry_wait_time)
            
            # Post-condition: Permanently commits the reserved items in the cart.
            self.marketplace.place_order(cart_id)


from threading import Lock
from threading import current_thread


class PublishedProduct:
    """
    @brief Wrapper for marketplace goods that adds a 'reserved' state attribute.
    """
    
    def __init__(self, product):
        self.product = product
        self.reserved = False # Intent: Mutual exclusion flag for consumer acquisition.

    def __eq__(self, obj):
        """
        @brief Equality override to support lookup in the ProductsList.
        """
        ret = isinstance(obj, PublishedProduct) and self.reserved == obj.reserved
        return ret and obj.product == self.product

class ProductsList:
    """
    @brief Thread-safe collection of products for a specific producer.
    """
    
    def __init__(self, maxsize):
        """
        @brief Initializes the buffer with a capacity constraint.
        """
        self.lock = Lock()
        self.list = []
        self.maxsize = maxsize

    def put(self, item):
        """
        @brief Appends an item to the list if space is available.
        """
        with self.lock:
            if self.maxsize == len(self.list):
                return False
            self.list.append(item)
        return True

    def rezerva(self, item):
        """
        @brief Atomically transitions a product to the reserved state.
        Algorithm: In-place state mutation within the list using a search index.
        """
        item = PublishedProduct(item)
        with self.lock:
            if item in self.list:
                # Logic: Finds the unreserved match and locks it.
                self.list[self.list.index(item)].reserved = True
                return True
        return False

    def anuleaza_rezervarea(self, item):
        """
        @brief Rolls back a previous reservation, making the item available again.
        """
        item = PublishedProduct(item)
        item.reserved = True # Intent: Targets the previously reserved entry.
        with self.lock:
            self.list[self.list.index(item)].reserved = False

    def remove(self, item):
        """
        @brief Permanently removes a reserved item from the collection.
        Pre-condition: Product must be in the reserved state.
        """
        product = PublishedProduct(item)
        product.reserved = True
        with self.lock:
            self.list.remove(product)
            return item

class Cart:
    """
    @brief In-memory record of items currently reserved by a consumer.
    """

    def __init__(self):
        self.products = []

    def add_product(self, product, producer_id):
        """
        @brief Maps a product to its original source producer for later commitment.
        """
        self.products.append((product, producer_id))

    def remove_product(self, product):
        """
        @brief Identification and removal of a product from the local cart.
        """
        for item in self.products:
            if item[0] == product:
                self.products.remove(item)
                return item[1]
        return None

    def get_products(self):
        return self.products

class Marketplace:
    """
    @brief Centralized marketplace controller managing producer inventories and consumer sessions.
    """
    
    def __init__(self, queue_size_per_producer):
        """
        @brief Initializes the marketplace and its atomic generators.
        """
        self.print_lock = Lock()
        self.queue_size_per_producer = queue_size_per_producer
        self.producer_queues = {}
        self.generator_id_producator = 0
        self.generator_id_producator_lock = Lock()

        self.carts = {}
        self.cart_id_generator = 0
        self.cart_id_generator_lock = Lock()

    def register_producer(self):
        """
        @brief Onboards a new producer and creates its dedicated ProductsList.
        """
        id_producator = None
        with self.generator_id_producator_lock:
            id_producator = self.generator_id_producator
            self.generator_id_producator += 1
            self.producer_queues[id_producator] = ProductsList(self.queue_size_per_producer)
        return id_producator

    def publish(self, producer_id, product):
        """
        @brief Routes a publication request to the producer-specific queue.
        """
        return self.producer_queues[producer_id].put(PublishedProduct(product))

    def new_cart(self):
        """
        @brief Allocates a new transaction identifier.
        """
        with self.cart_id_generator_lock:
            current_cart_id = self.cart_id_generator
            self.cart_id_generator += 1
            self.carts[current_cart_id] = Cart()
            return current_cart_id

    def add_to_cart(self, cart_id, product):
        """
        @brief Attempts to reserve a product unit from any available producer.
        Algorithm: Linear scan across all producer catalogs with early-exit on first successful reservation.
        """
        producers_num = 0
        with self.generator_id_producator_lock:
            producers_num = self.generator_id_producator

        for producer_id in range(producers_num):
            if self.producer_queues[producer_id].rezerva(product):
                self.carts[cart_id].add_product(product, producer_id)
                return True
        return False

    def remove_from_cart(self, cart_id, product):
        """
        @brief Returns a reserved item from a cart back to the producer's available pool.
        """
        producer_id = self.carts[cart_id].remove_product(product)
        if producer_id is None:
            return False
        self.producer_queues[producer_id].anuleaza_rezervarea(product)
        return True

    def place_order(self, cart_id):
        """
        @brief Commits all reservations in a cart and finalizes the transaction.
        Invariant: Uses self.print_lock to synchronize the console output of multiple consumers.
        """
        lista = list()
        for (produs, producer_id) in self.carts[cart_id].get_products():
            # Logic: Permanent removal from inventory.
            lista.append(self.producer_queues[producer_id].remove(produs))
            with self.print_lock:
                print(f"{current_thread().getName()} bought {produs}")
        return lista


from threading import Thread
from time import sleep


class Producer(Thread):
    """
    @brief Producer agent that generates goods for the marketplace.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        @brief Initializes the producer with its production plan and market connection.
        """
        Thread.__init__(self, name=kwargs["name"], daemon=kwargs["daemon"], kwargs=kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time

    def run(self):
        """
        @brief Main production loop.
        Algorithm: Iterative generation with a fixed production delay followed by quota-constrained publication.
        """
        producer_id = self.marketplace.register_producer()
        
        while True:
            for (product, cantitate, production_time) in self.products:
                # Domain: Physical production latency.
                sleep(production_time)
                
                for _ in range(cantitate):
                    # Logic: Blocks until the marketplace accepts the item (handles full buffers).
                    while not self.marketplace.publish(producer_id, product):
                        # Functional Utility: Throttles re-publication attempts.
                        sleep(self.republish_wait_time)
