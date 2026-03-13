"""
This module simulates a multi-threaded producer-consumer marketplace.

It defines `Producer` and `Consumer` threads that interact with a central
`Marketplace` class. This implementation uses multiple locks in an attempt at
fine-grained concurrency, but it contains several significant race conditions,
design flaws, and inefficient operations.
"""
from threading import Thread, Lock, currentThread
import time
import uuid
from dataclasses import dataclass

# --- Constants for dictionary keys, improving readability ---
QUANTITY = "quantity"
PRODUCT = "product"
TYPE = "type"
ADD = "add"
REMOVE = "remove"

@dataclass(init=True, repr=True, order=False, frozen=True)
class Product:
    """A base data class for a product."""
    name: str
    price: int

@dataclass(init=True, repr=True, order=False, frozen=True)
class Tea(Product):
    """A data class for a Tea product."""
    type: str

@dataclass(init=True, repr=True, order=False, frozen=True)
class Coffee(Product):
    """A data class for a Coffee product."""
    acidity: str
    roast_level: str

class Marketplace:
    """
    The central marketplace that manages inventory and carts.

    Warning: This class contains multiple race conditions due to improper locking.
    Operations that should be atomic are split across lock boundaries. The logic
    for finding products is also highly inefficient.
    """
    def __init__(self, queue_size_per_producer):
        self.queue_size_per_producer = queue_size_per_producer
        self.number_of_carts = 0
        # --- State Management ---
        # Maps cart_id -> [producer_id, [list_of_products]]
        # Flaw: A cart can only hold items from a single producer.
        self.items_in_cart = {}
        # Maps producer_id -> [[list_of_products], count]
        # Flaw: The count is redundant with len(list_of_products).
        self.producer = {}
        # --- Locking ---
        self.lock_carts = Lock()
        self.lock_remove = Lock()
        self.lock_print = Lock()
        self.lock_add = Lock()

    def register_producer(self):
        """Registers a new producer and returns a unique ID."""
        id_producer = uuid.uuid4()
        # The state is a list containing the product list and a count.
        element = [[], 0]
        self.producer[id_producer] = element
        return id_producer

    def publish(self, producer_id, product):
        """
        Publishes a product from a producer.

        Warning: Race condition. The size check is performed before any lock is acquired.
        """
        if self.producer[producer_id][1] >= self.queue_size_per_producer:
            return False
        # State is modified without a lock, another race condition.
        self.producer[producer_id][1] += 1
        self.producer[producer_id][0].append(product)
        return True

    def new_cart(self):
        """Creates a new, empty cart and returns its ID."""
        with self.lock_carts:
            self.number_of_carts += 1
            cart_id = self.number_of_carts
        # A cart stores the producer_id and a list of products.
        element = ["", []]
        self.items_in_cart[cart_id] = element
        return cart_id

    def add_to_cart(self, cart_id, product):
        """
        Adds a product to a cart.

        Warning: Major race condition. The lock is released before the state
        (the producer's product list) is fully modified. The product lookup
        is also extremely inefficient, O(Num_Producers * Num_Items).
        """
        with self.lock_add:
            id_prod = ""
            # Inefficiently find which producer has the product.
            id_p = [x for x in self.producer.keys() if product in self.producer[x][0]]
            if len(id_p) == 0:
                return False
            id_prod = id_p[0]
            # Decrement producer's item count inside the lock.
            self.producer[id_prod][1] -= 1
        # !!! RACE CONDITION: The lock is released here.
        # Another thread could now read the state before the product is removed.
        self.producer[id_prod][0].remove(product)

        # This will overwrite the producer ID for the cart with every item added.
        self.items_in_cart[cart_id][0] = id_prod
        self.items_in_cart[cart_id][1].append(product)
        return True

    def remove_from_cart(self, cart_id, product):
        """
        Removes a product from a cart.

        Warning: Major race condition. State is modified before and after the lock.
        """
        # RACE CONDITION: State is modified without a lock.
        self.producer[self.items_in_cart[cart_id][0]][0].append(product)
        with self.lock_remove:
            self.producer[self.items_in_cart[cart_id][0]][1] += 1
        # RACE CONDITION: State is modified without a lock.
        self.items_in_cart[cart_id][1].remove(product)
        return True

    def place_order(self, cart_id):
        """Finalizes an order by printing items and clearing the cart."""
        my_prods = self.items_in_cart.pop(cart_id, None)
        if my_prods:
            for elem in my_prods[1]:
                with self.lock_print:
                    print(f"{currentThread().getName()} bought {elem}")
            return my_prods[1]
        return []

class Producer(Thread):
    """A thread that simulates a producer publishing products."""
    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        # Registers itself once upon creation, which is the correct approach.
        self.id_ = self.marketplace.register_producer()

    def run(self):
        """Continuously attempts to publish its catalog of products."""
        while True:
            for product_info in self.products:
                product, nr_products, time_wait = product_info
                for _ in range(nr_products):
                    # Busy-wait until the product can be published.
                    val = self.marketplace.publish(self.id_, product)
                    if val:
                        time.sleep(time_wait)
                    else:
                        while True:
                            time.sleep(self.republish_wait_time)
                            val = self.marketplace.publish(self.id_, product)
                            if val:
                                break

class Consumer(Thread):
    """A thread that simulates a consumer filling a cart and placing an order."""
    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time

    def do_action(self, id_element, item):
        """Helper to perform an add or remove action."""
        if item[TYPE] == ADD:
            return self.marketplace.add_to_cart(id_element, item[PRODUCT])
        else:
            return self.marketplace.remove_from_cart(id_element, item[PRODUCT])

    def run(self):
        """Processes each shopping cart configuration."""
        for cart_config in self.carts:
            cart_id = self.marketplace.new_cart()
            for item in cart_config:
                for _ in range(item[QUANTITY]):
                    # Perform the action once.
                    success = self.do_action(cart_id, item)
                    # If it fails (e.g., item not available), enter a busy-wait loop.
                    if not success:
                        while True:
                            time.sleep(self.retry_wait_time)
                            success = self.do_action(cart_id, item)
                            if success:
                                break
            self.marketplace.place_order(cart_id)
