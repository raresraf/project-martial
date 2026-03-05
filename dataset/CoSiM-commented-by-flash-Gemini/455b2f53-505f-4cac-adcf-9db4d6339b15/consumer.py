"""
This module simulates a multi-threaded producer-consumer marketplace.

It defines classes for:
- Consumers: threads that interact with the marketplace to add/remove items from carts and place orders.
- Marketplace: manages product inventory, shopping carts, and producer registration.
- Producers: threads that publish products to the marketplace.
- Product, Tea, Coffee: data classes representing different types of products.
"""


import time
from threading import Thread


class Consumer(Thread):
    """
    Represents a consumer thread that simulates shopping activity in the marketplace.
    Consumers create carts, add/remove products, and place orders.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        Initializes a Consumer thread.

        Args:
            carts (list): A list of shopping cart operations (e.g., add, remove).
            marketplace (Marketplace): The marketplace instance to interact with.
            retry_wait_time (int): The time to wait before retrying an operation.
            **kwargs: Keyword arguments passed to the Thread.__init__ method.
        """


        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.operations = {"add": marketplace.add_to_cart,
                           "remove": marketplace.remove_from_cart}

    def run(self):
        """
        Executes the consumer's shopping behavior.

        For each predefined cart, the consumer creates a new cart in the marketplace,
        performs add/remove operations for products, retrying if the marketplace is
        temporarily unable to fulfill the request, and finally places the order.
        """
        # Iterate through each cart defined for this consumer.
        # Invariant: Each 'cart' represents a sequence of desired operations for a single shopping session.
        for cart in self.carts:
            # Pre-condition: A new, empty shopping cart is requested from the marketplace.
            cart_id = self.marketplace.new_cart()

            # Iterate through each operation within the current cart.
            # Invariant: Each 'operation' dictates adding or removing a specific product quantity.
            for operation in cart:
                quantity = operation["quantity"]

                # Pre-condition: The desired quantity for the current operation has not been fully processed.
                # Invariant: 'quantity' decreases with each successful product transaction.
                while quantity > 0:
                    operation_type = operation["type"]
                    product = operation["product"]

                    # Attempt to perform the cart operation (add or remove product).
                    # If successful, decrement the remaining quantity. Otherwise, wait and retry.
                    if self.operations[operation_type](cart_id, product) is not False:
                        quantity -= 1
                    else:
                        # Block Logic: Delays further attempts to prevent busy-waiting
                        # and to allow other threads to operate or marketplace state to change.
                        time.sleep(self.retry_wait_time)

            # Pre-condition: All operations for the current cart are completed or retries exhausted.
            # The consumer places the final order.
            self.marketplace.place_order(cart_id)

import sys
from threading import Lock, currentThread


class Marketplace:
    """
    Manages product inventory and shopping carts in a thread-safe manner.

    This class provides mechanisms for producers to publish products and for
    consumers to create carts, add/remove products, and place orders.
    It uses locks to ensure data consistency across multiple threads.
    """

    def __init__(self, queue_size_per_producer):
        """
        Initializes the Marketplace.

        Args:
            queue_size_per_producer (int): The maximum number of products
                                           a single producer can have published
                                           in the marketplace at any given time.
        """
        self.carts_lock = Lock()
        self.carts = []

        self.producers_lock = Lock()
        self.producers_capacity = queue_size_per_producer
        self.producers_sizes = []
        self.products = []

    def register_producer(self):
        """
        Registers a new producer with the marketplace.

        Returns:
            int: A unique producer ID.
        """
        with self.producers_lock:
            self.producers_sizes.append(0)
            return len(self.producers_sizes) - 1

    def publish(self, producer_id, product):
        """
        Publishes a product to the marketplace by a specific producer.

        Args:
            producer_id (int): The ID of the producer publishing the product.
            product: The product to be published.

        Returns:
            bool: True if the product was successfully published, False otherwise
                  (e.g., if the producer has reached its capacity limit).
        """
        with self.producers_lock:
            if self.producers_sizes[producer_id] == self.producers_capacity:
                return False

            self.producers_sizes[producer_id] += 1
            self.products.append((product, producer_id))
            return True

    def new_cart(self):
        """
        Creates a new, empty shopping cart in the marketplace.

        Returns:
            int: The ID of the newly created cart.
        """
        with self.carts_lock:
            self.carts.append([])
            return len(self.carts) - 1

    def add_to_cart(self, cart_id, product):
        """
        Adds a product to a specific cart.

        This method attempts to move a product from the marketplace's available
        products to the specified shopping cart. It is a blocking operation
        that acquires a lock on the producers to ensure atomicity during the
        product transfer.

        Args:
            cart_id (int): The ID of the cart to add the product to.
            product: The product to be added.

        Returns:
            bool: True if the product was successfully added, False if the
                  product was not found or could not be added.
        """
        # Acquire a lock to ensure exclusive access to the producers' state
        # during the product search and transfer to prevent race conditions.
        self.producers_lock.acquire()
        # Invariant: The producers_lock is held, preventing other threads from modifying
        # the product inventory or producer sizes.
        
        # Iterate through the list of available products in the marketplace.
        # Pre-condition: The marketplace contains products published by various producers.
        for (prod, prod_id) in self.products:
            # If the requested product is found:
            if prod == product:
                # Decrement the count of products for the publishing producer.
                self.producers_sizes[prod_id] -= 1
                # Remove the product instance from the marketplace's global product list.
                self.products.remove((prod, prod_id))
                # Release the lock as the critical section is complete.
                self.producers_lock.release()
                # Add the product to the consumer's cart.
                self.carts[cart_id].append((prod, prod_id))
                return True

        # If the product was not found after iterating through all available products,
        # release the lock and indicate failure.
        self.producers_lock.release()
        return False

    def remove_from_cart(self, cart_id, product):
        """
        Removes a product from a specific cart and returns it to the marketplace.

        This method iterates through the items in the specified cart to find the
        product. If found, it removes the product from the cart and publishes it
        back to the marketplace, incrementing the producer's count. This operation
        acquires a lock on the producers to ensure thread-safe product re-publishing.

        Args:
            cart_id (int): The ID of the cart from which to remove the product.
            product: The product to be removed.
        """
        # Iterate through the products currently in the specified cart.
        # Pre-condition: The cart exists and may contain products.
        for (prod, prod_id) in self.carts[cart_id]:
            # If the product to be removed is found in the cart:
            if prod == product:
                # Remove the product from the cart.
                self.carts[cart_id].remove((prod, prod_id))
                # Acquire a lock to ensure exclusive access to the producers' state
                # for re-publishing the product.
                self.producers_lock.acquire()
                # Return the product to the marketplace's available products.
                self.products.append((prod, prod_id))
                # Increment the count of products for the original publishing producer.
                self.producers_sizes[prod_id] += 1
                # Release the lock as the critical section is complete.
                self.producers_lock.release()
                return

    def place_order(self, cart_id):
        """
        Places an order for the items in the specified cart.

        This method processes the items in the given cart, simulates printing
        an order summary, and returns the list of products that were part of the order.

        Args:
            cart_id (int): The ID of the cart to place the order for.

        Returns:
            list: A list of (product, producer_id) tuples that were in the ordered cart.
        """


        order = ""
        for (product, _) in self.carts[cart_id]:
            order += "{} bought {}\n".format(
                currentThread().getName(), product)
        sys.stdout.write(order)
        return self.carts[cart_id]


import time
from threading import Thread


class Producer(Thread):
    """
    Represents a producer thread that continuously publishes products to the marketplace.

    Producers are responsible for making products available for consumers to purchase.
    They handle their own inventory and re-publish products if the marketplace
    is temporarily full.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        Initializes a Producer thread.

        Args:
            products (list): A list of (product, initial_quantity, wait_time) tuples
                             representing the products this producer will supply.
            marketplace (Marketplace): The marketplace instance to interact with.
            republish_wait_time (int): The time to wait before attempting to
                                       re-publish a product if the marketplace is full.
            **kwargs: Keyword arguments passed to the Thread.__init__ method.
        """
        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time


        self.producer_id = marketplace.register_producer()

    def run(self):
        """
        Executes the producer's product publishing behavior.

        The producer continuously attempts to publish its defined products to the
        marketplace. If the marketplace is at capacity for this producer, it waits
        before retrying.
        """
        # The producer's main loop, ensuring continuous operation.
        # Invariant: The producer attempts to keep its products available in the marketplace.
        while True:
            # Iterate through each product type this producer is responsible for.
            # Invariant: Each (product, quantity, wait_time) tuple represents a specific product to publish.
            for (product, quantity, wait_time) in self.products:
                # Pre-condition: There are still units of this product to publish.
                # Invariant: 'quantity' decreases with each successful publication.
                while quantity > 0:
                    # Attempt to publish the product to the marketplace.
                    if self.marketplace.publish(self.producer_id, product):
                        quantity -= 1
                        # Block Logic: Simulates the time taken to produce/prepare the next unit of product.
                        time.sleep(wait_time)
                    else:
                        # Block Logic: If the marketplace cannot accept more products from this producer,
                        # wait for a defined period before retrying to avoid busy-waiting.
                        time.sleep(self.republish_wait_time)


from dataclasses import dataclass


@dataclass(init=True, repr=True, order=False, frozen=True)
class Product:
    """
    Base data class for all products in the marketplace.
    Attributes:
        name (str): The name of the product.
        price (int): The price of the product.
    """
    name: str
    price: int


@dataclass(init=True, repr=True, order=False, frozen=True)
class Tea(Product):
    """
    Data class representing a type of tea, inheriting from Product.
    Attributes:
        type (str): The type of tea (e.g., "Green", "Black", "Herbal").
    """
    type: str


@dataclass(init=True, repr=True, order=False, frozen=True)
class Coffee(Product):
    """
    Data class representing a type of coffee, inheriting from Product.
    Attributes:
        acidity (str): The acidity level of the coffee.
        roast_level (str): The roast level of the coffee.
    """
    acidity: str
    roast_level: str
