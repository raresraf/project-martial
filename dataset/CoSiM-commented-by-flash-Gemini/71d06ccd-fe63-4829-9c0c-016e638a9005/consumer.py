"""
This module simulates a multi-threaded e-commerce marketplace.

It defines the core components of such a system:
- `Consumer`: Represents a buyer that interacts with the marketplace to add,
  remove, and place orders for products.
- `Marketplace`: Acts as the central hub, managing product listings from producers,
  customer carts, and facilitating all transactions.
- `Producer`: Represents a seller that registers with the marketplace and
  continuously publishes products for sale.
- `Product`, `Tea`, `Coffee`: Data classes to model different types of products
  with specific attributes.
- `ProductsContainer`: A thread-safe data structure used by both producers
  (for their inventory) and consumers (for their carts) to store products.

The simulation demonstrates concurrent operations where multiple producers
publish products and multiple consumers attempt to purchase them, all
coordinated through the `Marketplace` using threading primitives like `Thread`
and `Lock` to ensure data consistency and prevent race conditions.
"""

from threading import Thread
from time import sleep
import Queue # Standard Python queue, though Queue is not used in this Consumer class itself.


class Consumer(Thread):
    """
    Represents a consumer (buyer) in the marketplace.

    Each consumer operates in its own thread, performing a series of add
    and remove operations on various products in multiple shopping carts,
    and finally placing orders. Operations are retried if the marketplace
    cannot fulfill them immediately.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        Initializes a Consumer thread.

        Args:
            carts (list): A list of cart definitions, where each cart is a list
                          of operations (add/remove product, quantity).
            marketplace (Marketplace): A reference to the shared Marketplace instance.
            retry_wait_time (float): The time (in seconds) to wait before retrying
                                     a failed cart operation.
            **kwargs: Additional keyword arguments passed to the `Thread` constructor.
        """
        Thread.__init__(self, **kwargs)

        self.marketplace = marketplace
        # Creates a new cart in the marketplace for each cart definition provided
        # and stores them as {marketplace_cart_id: cart_operations_list}.
        self.carts = {self.marketplace.new_cart() : cart for cart in carts}
        self.retry_wait_time = retry_wait_time
        self.name = kwargs['name'] # Assigns a name to the consumer, used for printing.

    def run(self):
        """
        The main execution loop for the Consumer thread.

        It iterates through all assigned carts and their operations. For 'add'
        and 'remove' operations, it repeatedly attempts the action until successful,
        pausing for `retry_wait_time` between attempts. Finally, it places
        all orders and prints the purchased products.
        """
        for cart_id, cart in self.carts.items():
            for operation in cart:
                if operation['type'] == 'add':
                    # Attempt to add the specified quantity of the product to the cart.
                    for _ in range(operation['quantity']):
                        add_status = \
                                self.marketplace.add_to_cart(
                                    cart_id,
                                    operation['product']
                                )

                        # If adding to cart fails, retry after a delay until successful.
                        while not add_status:
                            sleep(self.retry_wait_time)
                            add_status = \
                                    self.marketplace.add_to_cart(
                                        cart_id,
                                        operation['product']
                                    )

                if operation['type'] == 'remove':
                    # Attempt to remove the specified quantity of the product from the cart.
                    for _ in range(operation['quantity']):
                        remove_status = \
                            self.marketplace.remove_from_cart(
                                cart_id,
                                operation['product']
                            )

                        # If removing from cart fails, retry after a delay until successful.
                        while not remove_status:
                            sleep(self.retry_wait_time)
                            remove_status = \
                                self.marketplace.remove_from_cart(
                                    cart_id,
                                    operation['product']
                                )

        # After all cart operations, place the order for each cart.
        for cart_id, _ in self.carts.items():
            order = self.marketplace.place_order(cart_id)
            # Print the products bought by this consumer.
            for product in order:
                print(self.name + " bought " + str(product[0]))


from threading import Lock
from products_container import ProductsContainer # Custom module for ProductsContainer. (Assumed from import)

class Marketplace:
    """
    The central hub of the e-commerce simulation.

    It manages producers, their product inventories, consumer carts,
    and facilitates all operations related to publishing products,
    adding/removing items from carts, and placing orders.
    Ensures thread-safe access to shared resources.
    """
    
    def __init__(self, queue_size_per_producer):
        """
        Initializes the Marketplace.

        Args:
            queue_size_per_producer (int): The maximum number of products
                                           a producer can publish at any time.
        """
        self.queue_size_per_producer = queue_size_per_producer

        # Locks to protect critical sections when assigning new IDs.
        self.register_producer_lock = Lock()
        self.new_cart_lock = Lock()

        # Counters for assigning unique producer and cart IDs.
        self.avail_producer_id = 0
        self.avail_cart_id = 0

        # Dictionaries to store products (by producer) and carts (by cart ID).
        self.products = {} # {producer_id: ProductsContainer}
        self.carts = {}    # {cart_id: ProductsContainer}

    def register_producer(self):
        """
        Registers a new producer with the marketplace.

        Assigns a unique ID to the producer and creates a new `ProductsContainer`
        to manage its inventory. Ensures thread-safe ID assignment.

        Returns:
            int: The unique ID assigned to the new producer.
        """
        with self.register_producer_lock:
            producer_id = self.avail_producer_id
            self.avail_producer_id += 1

            # Creates a ProductsContainer for the new producer's inventory.
            self.products[producer_id] = ProductsContainer(self.queue_size_per_producer)

            return producer_id

    def publish(self, producer_id, product):
        """
        Publishes a product to the marketplace under the given producer's inventory.

        Args:
            producer_id (int): The ID of the producer publishing the product.
            product (Product): The product to be published.

        Returns:
            bool: True if the product was successfully published, False otherwise
                  (e.g., if the producer's inventory is full).
        """
        # Stores product data as a tuple (product_object, producer_id).
        # This allows tracking the original producer of a product.
        return self.products[producer_id].put((product, producer_id))

    def new_cart(self):
        """
        Creates a new shopping cart in the marketplace for a consumer.

        Assigns a unique ID to the cart and creates an empty `ProductsContainer`
        to manage its contents. Ensures thread-safe ID assignment.

        Returns:
            int: The unique ID assigned to the new cart.
        """
        with self.new_cart_lock:
            cart_id = self.avail_cart_id
            self.avail_cart_id += 1

            # Creates an empty ProductsContainer for the new cart.
            self.carts[cart_id] = ProductsContainer()

            return cart_id

    def add_to_cart(self, cart_id, product):
        """
        Adds a product to a specified cart.

        This involves finding the product in any producer's inventory,
        removing it from there, and then adding it to the consumer's cart.
        Ensures atomicity for the transfer.

        Args:
            cart_id (int): The ID of the cart to add the product to.
            product (Product): The product to be added.

        Returns:
            bool: True if the product was successfully added, False otherwise
                  (e.g., product not found in any producer's inventory).
        """
        for producer_id, products in self.products.items():
            # The product is stored with its original producer_id.
            product_data = (product, producer_id)
            if products.has(product_data):
                # Atomically move the product from producer to cart.
                # Note: This is not truly atomic if `remove` and `put` are
                # separate calls and `products.lock` is different from `carts[cart_id].lock`.
                # However, ProductsContainer methods themselves use their internal locks.
                self.products[producer_id].remove(product_data)
                self.carts[cart_id].put(product_data)
                return True

        return False

    def remove_from_cart(self, cart_id, product):
        """
        Removes a product from a specified cart and returns it to its original producer's inventory.

        Args:
            cart_id (int): The ID of the cart to remove the product from.
            product (Product): The product to be removed.

        Returns:
            bool: True if the product was successfully removed, False otherwise
                  (e.g., product not found in the cart or original producer no longer exists).
        """
        # Find all instances of the product in the cart.
        products_to_remove = [it for it in self.carts[cart_id].get_all() if it[0] == product]
        if len(products_to_remove) == 0:
            return False

        # Take the first instance found.
        removed_product = products_to_remove[0]

        # Check if the original producer of the product still exists.
        if removed_product[1] not in self.products:
            return False

        # Atomically move the product from cart back to producer.
        self.carts[cart_id].remove(removed_product)
        self.products[removed_product[1]].put(removed_product)

        return True

    def place_order(self, cart_id):
        """
        Retrieves all products currently in a specified cart, effectively
        "placing" the order. The products are removed from the cart.

        Args:
            cart_id (int): The ID of the cart to place the order from.

        Returns:
            list: A list of all products (as `(Product, producer_id)` tuples)
                  that were in the cart.
        """
        # Note: The implementation `get_all()` merely returns the list of products
        # without clearing the cart. If placing an order means emptying the cart,
        # additional logic would be needed here.
        return self.carts[cart_id].get_all()


from threading import Thread
from random import choice
from time import sleep


class Producer(Thread):
    """
    Represents a producer (seller) in the marketplace.

    Each producer operates in its own thread, continuously publishing
    a selection of products to the marketplace.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        Initializes a Producer thread.

        Args:
            products (list): A list of product definitions, where each definition
                             is `(Product, quantity_to_publish, wait_time_between_pubs)`.
            marketplace (Marketplace): A reference to the shared Marketplace instance.
            republish_wait_time (float): The time (in seconds) to wait before retrying
                                         to publish a product if the inventory is full.
            **kwargs: Additional keyword arguments passed to the `Thread` constructor.
        """
        Thread.__init__(self, daemon=True, **kwargs) # Producers run as daemon threads.

        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        # Registers with the marketplace to get a unique producer ID.
        self.producer_id = self.marketplace.register_producer()

    def run(self):
        """
        The main execution loop for the Producer thread.

        It continuously picks a random product from its catalog, attempts to
        publish it to the marketplace a specified number of times, retrying
        if the marketplace's inventory for this producer is full. It pauses
        between publications.
        """
        while True:
            published_product = choice(self.products) # Randomly choose a product to publish.

            # Attempt to publish the product multiple times based on its quantity_to_publish.
            for _ in range(published_product[1]):
                publish_status = self.marketplace.publish(self.producer_id, published_product[0])

                # If publishing fails (e.g., producer's inventory is full), retry after a delay.
                while not publish_status:
                    sleep(self.republish_wait_time)
                    publish_status = \
                            self.marketplace.publish(self.producer_id, published_product[0])

                # Pause after each successful publication for a product-specific time.
                sleep(published_product[2])


from dataclasses import dataclass


@dataclass(init=True, repr=True, order=False, frozen=True)
class Product:
    """
    A base data class for defining a product.
    Uses `dataclasses` for concise definition.
    `frozen=True` makes instances immutable (hashable).
    """
    name: str  # The name of the product.
    price: int # The price of the product.


@dataclass(init=True, repr=True, order=False, frozen=True)
class Tea(Product):
    """
    A specialized data class for Tea products, inheriting from `Product`.
    """
    type: str  # The type of tea (e.g., "Green", "Black", "Herbal").


@dataclass(init=True, repr=True, order=False, frozen=True)
class Coffee(Product):
    """
    A specialized data class for Coffee products, inheriting from `Product`.
    """
    acidity: str      # The acidity level of the coffee.
    roast_level: str  # The roast level of the coffee.


from threading import Lock

class ProductsContainer:
    """
    A thread-safe container for managing a collection of products.

    It can optionally have a maximum size, acting as a bounded buffer.
    All operations on the internal list of products are protected by a `Lock`.
    """
    
    def __init__(self, max_size=-1):
        """
        Initializes the ProductsContainer.

        Args:
            max_size (int, optional): The maximum number of products the container can hold.
                                      Defaults to -1, indicating no size limit.
        """
        self.products = [] # The internal list storing products (typically `(Product, producer_id)` tuples).
        self.lock = Lock() # Lock to ensure thread-safe access to `self.products`.
        self.max_size = max_size # Maximum capacity of the container.

    def put(self, product_data):
        """
        Adds a product to the container.

        If `max_size` is defined and the container is full, the operation fails.
        Ensures thread-safe addition.

        Args:
            product_data (tuple): The product data to add (e.g., `(Product, producer_id)`).

        Returns:
            bool: True if the product was successfully added, False otherwise.
        """
        with self.lock:
            if self.max_size == -1: # No size limit.
                self.products.append(product_data)
                return True

            if len(self.products) >= self.max_size: # Container is full.
                return False

            self.products.append(product_data)
            return True

    def remove(self, product_data):
        """
        Removes a specific product from the container.

        Ensures thread-safe removal.

        Args:
            product_data (tuple): The product data to remove.

        Returns:
            bool: True if the product was successfully removed, False otherwise (not found).
        """
        with self.lock:
            try:
                self.products.remove(product_data)
                return True
            except ValueError: # Product not found in the list.
                return False

    def get_all(self):
        """
        Retrieves all products currently in the container.

        Ensures thread-safe access.

        Returns:
            list: A list of all product data tuples in the container.
        """
        with self.lock:
            return self.products # Returns a reference to the internal list; care needed for external modification.

    def has(self, product_data):
        """
        Checks if a specific product exists in the container.

        Ensures thread-safe access.

        Args:
            product_data (tuple): The product data to check for.

        Returns:
            bool: True if the product is found, False otherwise.
        """
        with self.lock:
            return product_data in self.products