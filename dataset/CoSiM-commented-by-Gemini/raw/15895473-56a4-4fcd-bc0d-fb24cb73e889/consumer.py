"""
This module simulates a marketplace with producers and consumers running in
concurrent threads. It defines the central marketplace logic, as well as the
behavior for producer and consumer threads.

Note: The implementation, particularly in the Marketplace class, has significant
thread-safety issues and is prone to race conditions. The file structure is
also unconventional, with classes and imports defined in a non-standard order.
"""
from threading import Thread
from time import sleep


class Consumer(Thread):
    """
    Represents a consumer thread that interacts with the marketplace.

    A consumer processes a predefined list of shopping carts, where each cart
    contains instructions to add or remove products.
    """
    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """Initializes the Consumer thread."""
        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.name = kwargs['name']

    def run(self):
        """The main execution loop for the consumer."""
        for cart in self.carts:
            new_cart = self.marketplace.new_cart()
            for instruction in cart:
                i = 0
                product = instruction['product']
                action = instruction['type']
                while i < instruction['quantity']:
                    if action == "add":
                        if self.marketplace.add_to_cart(new_cart, product):
                            i += 1
                        else:
                            sleep(self.retry_wait_time)
                    elif action == "remove":
                        self.marketplace.remove_from_cart(new_cart, product)
                        i += 1


            new_list = self.marketplace.place_order(new_cart)
            for _, instruction in new_list:
                print(self.name, "bought", instruction)

from threading import Lock


class Marketplace:
    """
    The central marketplace shared between producers and consumers.

    This class manages product inventory and shopping carts.

    Warning: This class is not thread-safe. Methods like `add_to_cart` and
    `remove_from_cart` can cause race conditions as they modify shared
    data structures without proper locking.
    """
    def __init__(self, queue_size_per_producer):
        """Initializes the marketplace."""
        self.producer_queue_size = queue_size_per_producer
        self.producers = {}
        self.carts = {}
        self.current_producer = 1
        self.current_cart = 1
        self.register_lock = Lock()
        self.cart_lock = Lock()

    def register_producer(self):
        """Registers a new producer and returns a unique producer ID."""
        self.register_lock.acquire()
        products = []
        producer_id = self.current_producer
        self.producers[producer_id] = products
        self.current_producer += 1
        self.register_lock.release()
        return producer_id

    def publish(self, producer_id, product):
        """
        Allows a producer to publish a product to the marketplace.

        Returns:
            bool: True if the product was added, False if the producer's
                  inventory is full.
        """
        if len(self.producers[producer_id]) >= self.producer_queue_size:
            return False

        self.producers[producer_id].append(product)
        return True

    def new_cart(self):
        """Creates a new, empty shopping cart and returns its ID."""
        self.cart_lock.acquire()
        cart = []
        new_cart = self.current_cart
        self.carts[new_cart] = cart
        self.current_cart += 1
        self.cart_lock.release()
        return new_cart

    def add_to_cart(self, cart_id, product):
        """
        Adds a product to a shopping cart by taking it from a producer.

        Warning: This method is not thread-safe. It iterates over producers
        and modifies their inventory without a lock.
        """
        if cart_id not in self.carts.keys():
            return False

        for producer in self.producers:
            if product in self.producers[producer]:


                self.carts[cart_id].append([producer, product])
                self.producers[producer].remove(product)
                return True
        return False

    def remove_from_cart(self, cart_id, product):
        """
        Removes a product from a cart and returns it to the producer's inventory.

        Warning: This method is not thread-safe.
        """
        if cart_id not in self.carts.keys():
            return False



        for producer_id, prod in self.carts[cart_id]:
            if prod == product:
                self.carts[cart_id].remove([producer_id, product])
                self.producers[producer_id].append(product)
                break
        return None

    def place_order(self, cart_id):
        """Finalizes an order and returns the contents of the cart."""
        if cart_id not in self.carts.keys():
            return None
        return self.carts[cart_id]


from threading import Thread
from time import sleep


class Producer(Thread):
    """
    Represents a producer thread that publishes products to the marketplace.
    """
    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """Initializes the Producer thread."""
        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        self.daemon = kwargs['daemon']
        self.name = kwargs['name']
        self.producer_id = self.marketplace.register_producer()

    def run(self):
        """The main execution loop for the producer."""
        while True:
            for product in self.products:
                i = 0
                while i < product[1]:
                    if self.marketplace.publish(self.producer_id, product[0]):
                        sleep(product[2])
                        i += 1
                    else:
                        sleep(self.republish_wait_time)


from dataclasses import dataclass


@dataclass(init=True, repr=True, order=False, frozen=True)
class Product:
    """A base dataclass for products in the marketplace."""
    name: str
    price: int


@dataclass(init=True, repr=True, order=False, frozen=True)
class Tea(Product):
    """A dataclass for Tea products, inheriting from Product."""
    type: str


@dataclass(init=True, repr=True, order=False, frozen=True)
class Coffee(Product):
    """A dataclass for Coffee products, inheriting from Product."""
    acidity: str
    roast_level: str
