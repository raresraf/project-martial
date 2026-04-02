"""
This module implements a producer-consumer simulation for a marketplace.

It defines the core components of a concurrent e-commerce system:
- Producer threads that publish products.
- A central Marketplace that manages inventory and shopping carts.
- Consumer threads that purchase products.
- Product data structures.

The simulation uses threading and locks to manage concurrent access to shared
resources, modeling a real-world scenario of multiple producers and consumers
interacting in a marketplace.
"""

from threading import Thread
from time import sleep


class Consumer(Thread):
    """
    Represents a consumer that purchases products from the marketplace.

    Each consumer runs in its own thread, simulating a user adding and removing
    products from a shopping cart before placing an order.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        Initializes a Consumer thread.

        Args:
            carts (list): A list of shopping carts, where each cart is a list of
                          product operations (add/remove).
            marketplace (Marketplace): The marketplace instance from which the
                                       consumer will buy products.
            retry_wait_time (float): Time in seconds to wait before retrying a
                                     failed operation (e.g., product unavailable).
            **kwargs: Additional keyword arguments, including 'name' for the thread.
        """
        Thread.__init__(self)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.kwargs = kwargs

    def run(self):
        """
        The main execution logic for the consumer thread.

        Iterates through the assigned carts and operations, adds/removes products,
        and finally places the order.
        """
        # A new cart is created in the marketplace for the consumer.
        id_cart = self.marketplace.new_cart()
        # The consumer processes each cart assigned to it.
        for cart in self.carts:
            # Each operation within the cart is processed.
            for operation in cart:


                qty = operation['quantity']
                type_op = operation['type']
                product = operation['product']
                # The operation is retried until the desired quantity is fulfilled.
                while qty > 0:
                    # If the operation is to add a product.
                    if type_op == 'add':
                        # Attempt to add the product to the cart.
                        ret_value = self.marketplace.add_to_cart(id_cart, product)
                        # If the product was added successfully, decrement the quantity.
                        if ret_value:
                            qty = qty - 1
                        # If the product could not be added, wait before retrying.
                        else:
                            sleep(self.retry_wait_time)


                    # If the operation is to remove a product.
                    if type_op == 'remove':
                        # Remove the product from the cart.
                        self.marketplace.remove_from_cart(id_cart, product)
                        # Decrement the quantity for the operation.
                        qty = qty - 1
        # After all operations, the order is placed.
        cart_list = self.marketplace.place_order(id_cart)
        # The consumer prints out the products they have bought.
        for product in cart_list:
            print(self.kwargs['name'] + " bought " + str(product))

from threading import Lock


class Marketplace:
    """
    Manages the inventory and interactions between producers and consumers.

    This class is thread-safe, using locks to protect shared data structures
    representing producer inventories and consumer shopping carts.
    """
    def __init__(self, queue_size_per_producer):
        """
        Initializes the Marketplace.

        Args:
            queue_size_per_producer (int): The maximum number of products a single
                                           producer can have listed at one time.
        """
        self.queue_size = queue_size_per_producer
        self.producers = []
        self.carts = []
        self.lock_producers = Lock()
        self.lock_consumer = Lock()
        self.producers_locks = []

    def register_producer(self):
        """
        Registers a new producer with the marketplace.

        This method is thread-safe.

        Returns:
            int: The unique ID assigned to the new producer.
        """
        # This section is locked to prevent race conditions when multiple producers register.
        with self.lock_producers:
            id_new_producer = len(self.producers)
            self.producers.append(list())
            self.producers_locks.append(Lock())

        return id_new_producer

    def publish(self, producer_id, product):
        """
        Adds a product to a producer's inventory.

        Args:
            producer_id (int): The ID of the producer publishing the product.
            product (Product): The product to be published.

        Returns:
            bool: True if the product was successfully published, False if the
                  producer's inventory is full.
        """
        # The producer's inventory is checked to see if it is full.
        if len(self.producers[producer_id]) == self.queue_size:
            return False

        # The product is added to the producer's inventory.
        self.producers[producer_id].append(product)
        return True

    def new_cart(self):
        """
        Creates a new, empty shopping cart for a consumer.

        This method is thread-safe.

        Returns:
            int: The unique ID assigned to the new cart.
        """
        # This section is locked to ensure that each cart gets a unique ID.
        with self.lock_consumer:
            id_new_cart = len(self.carts)
            self.carts.append(list())

        return id_new_cart

    def add_to_cart(self, cart_id, product):
        """
        Adds a product to a consumer's shopping cart.

        Searches all producer inventories for the requested product and moves it
        to the cart if found. This operation is thread-safe.

        Args:
            cart_id (int): The ID of the cart to add the product to.
            product (Product): The product to add.

        Returns:
            bool: True if the product was found and added, False otherwise.
        """
        # Iterate through all producers to find the requested product.
        for id_producer in range(len(self.producers)):
            # Check if the product is in the current producer's inventory.
            if product in self.producers[id_producer]:
                # Lock the specific producer's inventory to prevent race conditions.
                with self.producers_locks[id_producer]:
                    # Double-check if the product is still available after acquiring the lock.
                    if product in self.producers[id_producer]:
                        # Atomically remove the product from the producer and add it to the cart.
                        self.producers[id_producer].remove(product)
                        self.carts[cart_id].append((product, id_producer))
                        return True
        return False

    def remove_from_cart(self, cart_id, product):
        """
        Removes a product from a shopping cart and returns it to the producer.

        Args:
            cart_id (int): The ID of the cart.
            product (Product): The product to remove.
        """
        # Find the product in the cart.
        for (current_product, id_producer) in self.carts[cart_id]:
            if current_product == product:
                # Return the product to the original producer's inventory.
                self.producers[id_producer].append(product)
                # Remove the product from the cart.
                self.carts[cart_id].remove((current_product, id_producer))
                break

    def place_order(self, cart_id):
        """
        Finalizes the purchase of all items in a cart.

        Args:
            cart_id (int): The ID of the cart to be ordered.

        Returns:
            list: A list of the products that were in the cart.
        """
        list_cart = list()
        # Extracts only the product from the (product, producer_id) tuple.
        for element in self.carts[cart_id]:
            list_cart.append(element[0])
        # The cart is cleared after the order is placed.
        self.carts[cart_id].clear()
        return list_cart


from threading import Thread
from time import sleep


class Producer(Thread):
    """
    Represents a producer that publishes products to the marketplace.
    
    Each producer runs in its own thread, simulating a supplier adding
    products to their inventory.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        Initializes a Producer thread.

        Args:
            products (list): A list of products the producer will attempt to publish.
                             Each element is a tuple (product, quantity, publication_interval).
            marketplace (Marketplace): The marketplace instance to publish to.
            republish_wait_time (float): Time in seconds to wait before retrying to
                                         publish if the inventory is full.
            **kwargs: Additional keyword arguments for the thread.
        """
        Thread.__init__(self, group=None, target=None, name=kwargs['name'], daemon=kwargs['daemon'])
        self.marketplace = marketplace
        self.products = products
        self.republish_wait_time = republish_wait_time


        self.kwargs = kwargs

    def run(self):
        """
        The main execution logic for the producer thread.

        Continuously attempts to publish its list of products to the marketplace.
        """
        # The producer registers with the marketplace to get a unique ID.
        id_producer = self.marketplace.register_producer()
        # The producer will continuously try to publish products.
        while True:
            # Iterates through the list of products it is supposed to produce.
            for (product, qty, time) in self.products:
                # Publishes the specified quantity of each product.
                while qty > 0:
                    # Attempts to publish a single unit of the product.
                    ret_value = self.marketplace.publish(id_producer, product)
                    # If successful, waits for the specified time and decrements quantity.
                    if ret_value:
                        sleep(time)
                        qty = qty - 1
                    # If the marketplace inventory is full, waits before retrying.
                    else:
                        sleep(self.republish_wait_time)


from dataclasses import dataclass


@dataclass(init=True, repr=True, order=False, frozen=True)
class Product:
    """A simple dataclass representing a generic product with a name and price."""
    name: str
    price: int


@dataclass(init=True, repr=True, order=False, frozen=True)
class Tea(Product):
    """A dataclass representing Tea, inheriting from Product and adding a 'type'."""
    type: str


@dataclass(init=True, repr=True, order=False, frozen=True)
class Coffee(Product):
    """A dataclass representing Coffee, inheriting from Product and adding acidity and roast level."""
    acidity: str
    roast_level: str
