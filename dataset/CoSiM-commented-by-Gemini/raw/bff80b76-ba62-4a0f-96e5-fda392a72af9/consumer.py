"""
This module simulates a producer-consumer marketplace using multi-threading.

It contains `Producer` and `Consumer` threads that interact with a central
`Marketplace` class. The simulation also defines data classes for the products
being traded.

NOTE: This implementation has several significant design flaws:
- A single global `Condition` lock is used for all marketplace operations,
  which serializes all access and eliminates any potential for concurrency,
  acting as a major performance bottleneck.
- The `add_to_cart` method does not remove the item from the producer's
  inventory, allowing the same item to be sold multiple times (item duplication).
- The `remove_from_cart` method does not return the item to the producer's
  inventory, causing the item to be permanently lost from the system.
- ID generation for producers and carts is inefficient and logically questionable.
"""

from threading import Thread
import time

class Consumer(Thread):
    """
    A thread that simulates a consumer purchasing items from the marketplace.

    It processes a list of "carts", where each cart is a sequence of add/remove
    operations, and then places an order.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        Initializes the Consumer thread.

        Args:
            carts (list): A list of shopping lists (each a list of operations).
            marketplace (Marketplace): The shared marketplace instance.
            retry_wait_time (float): The time to wait before retrying an operation.
            **kwargs: Keyword arguments for the `Thread` parent class.
        """
        Thread.__init__(self, **kwargs)
        self.retry_wait_time = retry_wait_time
        self.marketplace = marketplace
        self.carts = carts

    def run(self):
        """The main execution logic for the consumer thread."""
        for cos in self.carts:
            cos_id = self.marketplace.new_cart()
            for produs in cos:
                # Block Logic: Performs 'add' or 'remove' operations based on the
                # instruction in the shopping list.
                if produs['type'] == 'add':
                    contor = 0
                    # Uses a polling/busy-wait loop to retry adding to the cart.
                    while contor < produs['quantity']:
                        adaugat = self.marketplace.add_to_cart(cos_id, produs['product'])
                        while adaugat == False:
                            adaugat = self.marketplace.add_to_cart(cos_id, produs['product'])
                            time.sleep(self.retry_wait_time)
                        contor += 1
                else:
                    contor = 0
                    # This loop simply calls remove N times without checking for success.
                    while contor < produs['quantity']:
                        self.marketplace.remove_from_cart(cos_id, produs['product'])
                        contor += 1
            produse_cumparate = self.marketplace.place_order(cos_id)
            for produs_cumparat in produse_cumparate:
                print(f"{self.name} bought {produs_cumparat}")

from threading import Condition

class Marketplace:
    """
    The central marketplace, managing all shared state between producers and consumers.

    All methods are synchronized under a single `Condition` object, effectively
    making the entire marketplace a single-threaded bottleneck.
    """
    def __init__(self, queue_size_per_producer):
        """
        Initializes the Marketplace.

        Args:
            queue_size_per_producer (int): This parameter is declared but not
                                           actually used to limit production.
        """
        self.queue_size_per_producer = queue_size_per_producer
        self.producatori = dict()
        self.cosuri = dict()
        self.cond = Condition()
        self.producatori_id = []
        self.cosuri_id = []
        self.contor_producator = 1
        self.contor_cos = 1

    def register_producer(self):
        """
        Registers a new producer and assigns them a unique ID.

        Note: The ID generation logic `sum(self.producatori_id)` is highly
        inefficient and logically flawed for creating unique sequential IDs.
        """
        with self.cond:
            self.contor_producator = sum(self.producatori_id)
            self.contor_producator += 1
            self.producatori_id.append(self.contor_producator)
            producator = dict()


            producator['produse'] = []
            self.producatori[self.contor_producator] = producator
            return self.contor_producator


    def publish(self, producer_id, product):
        """
        Adds a product to a specific producer's inventory.

        Note: This is inefficiently implemented with a loop instead of a direct
        dictionary lookup.
        """
        with self.cond:
            for producator_id, lista_produse_publicate in self.producatori.items():
                if producator_id == producer_id:
                    lista_produse_publicate['produse'].append(product)
                    return True
            return False

    def new_cart(self):
        """
        Creates a new, empty cart for a consumer.

        Note: The ID generation logic is inefficient and logically flawed.
        """
        with self.cond:
            self.contor_cos = sum(self.cosuri_id)
            self.contor_cos += 2
            self.cosuri_id.append(self.contor_cos)
            cos = dict()
            cos['produse_rezervate'] = []
            self.cosuri[self.contor_cos] = cos
            return self.contor_cos


    def add_to_cart(self, cart_id, product):
        """
        Adds a product to a cart.

        MAJOR FLAW: This method finds a product in a producer's inventory but
        does NOT remove it. This allows the same product instance to be added
        to multiple carts and sold infinitely.
        """
        with self.cond:
            for cos, continut in self.cosuri.items():
                if cos == cart_id:
                    for producator, produse_publicate in self.producatori.items():
                        if product in produse_publicate['produse']:
                            continut['produse_rezervate'].append(product)
                            return True
            return False

    def remove_from_cart(self, cart_id, product):
        """
        Removes a product from a cart.

        FLAW: This method removes the product from the consumer's cart but does
        not return it to any producer's inventory. The product is permanently
        lost from the simulation.
        """
        with self.cond:
            for cos, continut in self.cosuri.items():
                if cos == cart_id:
                    continut['produse_rezervate'].remove(product)

    def place_order(self, cart_id):
        """Finalizes an order, returning the products from the cart."""
        with self.cond:
            for cos, continut in self.cosuri.items():
                if cos == cart_id:


                    return continut['produse_rezervate']
            return None


from threading import Thread
import time

class Producer(Thread):
    """
    A thread that simulates a producer creating and publishing products.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        Initializes the Producer thread.

        Args:
            products (list): A list of products this producer can create.
            marketplace (Marketplace): The shared marketplace instance.
            republish_wait_time (float): Time to wait before retrying to publish.
            **kwargs: Keyword arguments for the `Thread` parent class.
        """
        Thread.__init__(self, **kwargs)
        self.kwargs = kwargs
        self.republish_wait_time = republish_wait_time
        self.marketplace = marketplace
        self.products = products
        self.daemon = True

    def run(self):
        """
        Main execution loop for the producer. Registers itself and then enters an
        infinite loop to produce and publish products.
        """
        producator_id = self.marketplace.register_producer()
        while True:
            for produs in self.products:
                contor = 0
                while contor < produs[1]:
                    in_market = self.marketplace.publish(producator_id, produs[0])
                    time.sleep(produs[2]) # Simulates production time.
                    # This polling loop is unlikely to be needed given the flaws
                    # in the marketplace, which never stops a publish.
                    while in_market == False:
                        in_market = self.marketplace.publish(producator_id, produs[0])
                        time.sleep(self.republish_wait_time)
                    contor += 1


from dataclasses import dataclass


@dataclass(init=True, repr=True, order=False, frozen=True)
class Product:
    """A simple, immutable data class representing a generic product."""
    name: str
    price: int


@dataclass(init=True, repr=True, order=False, frozen=True)
class Tea(Product):
    """An immutable data class for a Tea product, inheriting from Product."""
    type: str


@dataclass(init=True, repr=True, order=False, frozen=True)
class Coffee(Product):
    """An immutable data class for a Coffee product, inheriting from Product."""
    acidity: str
    roast_level: str
