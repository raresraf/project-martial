

"""
This module implements a simulation of a multi-threaded producer-consumer marketplace.
It defines classes for `Consumer` and `Producer` threads, a thread-safe `Marketplace`
for managing products and orders, and `Product` data classes (`Product`, `Tea`, `Coffee`).
"""

from threading import Thread
from time import sleep


class Consumer(Thread):
    """
    Simulates a consumer entity that interacts with a Marketplace.

    A Consumer creates shopping carts, adds or removes products to/from them,
    and finally places orders. It includes a retry mechanism for operations
    that might initially fail.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        Initializes a Consumer thread.

        Args:
            carts (list): A list of shopping cart definitions. Each definition
                          is a list of operations (add/remove) for products.
            marketplace (Marketplace): The shared marketplace instance to interact with.
            retry_wait_time (float): The time in seconds to wait before retrying
                                     a failed add/remove operation.
            **kwargs: Arbitrary keyword arguments passed to the Thread constructor.
        """
        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        # Functional Utility: Caches marketplace methods for convenience.
        self.new_cart = self.marketplace.new_cart
        self.add_to_cart = self.marketplace.add_to_cart
        self.remove_from_cart = self.marketplace.remove_from_cart
        self.place_order = self.marketplace.place_order


    def run(self):
        """
        Main execution loop for the Consumer thread.

        Architectural Intent: Iterates through predefined shopping carts,
        executing add/remove operations for products, retrying on failure,
        and finally placing the order.
        """
        # Block Logic: Iterates through each shopping cart definition provided to this consumer.
        for current_cart in self.carts:

            # Functional Utility: Creates a new, unique shopping cart in the marketplace.
            id_cart = self.new_cart()

            # Block Logic: Iterates through each item/operation defined for the current shopping cart.
            for cart_item in current_cart:

                quantity = cart_item['quantity']
                product = cart_item['product']
                op_type = cart_item['type']
                step = 1 # Counter for the number of successful operations for the current cart item.

                # Block Logic: Continues to attempt adding/removing the product until the desired quantity is reached.
                # Invariant: `step` increments only upon successful operation or if `success` is None (handled by marketplace).
                while step <= quantity:
                    success = False # Flag to track if the current operation was successful.
                    # Conditional Logic: Performs either an 'add' or 'remove' operation based on `op_type`.
                    if op_type == "add":
                        success = self.add_to_cart(id_cart, product)
                    elif op_type == "remove":
                        success = self.remove_from_cart(id_cart, product)

                    # Conditional Logic: If the operation was successful (or returned None indicating success),
                    # increment `step` and continue to the next iteration.
                    if success is None or success:
                        step += 1
                        continue

                    # Block Logic: If the operation failed, pause the thread for `retry_wait_time` before retrying.
                    sleep(self.retry_wait_time)

            # Functional Utility: Places the order for the fully populated/modified cart.
            self.place_order(id_cart)

from __future__ import print_function
from threading import Lock, currentThread


class Marketplace:
    """
    Acts as the central hub for producers and consumers to interact in a thread-safe manner.

    It manages the inventory of products, tracks active producers, and handles shopping carts.
    All critical operations that modify shared state are protected by locks to ensure atomicity
    and prevent race conditions in a multi-threaded environment.
    """
    
    def __init__(self, queue_size_per_producer):
        """
        Initializes the Marketplace.

        Args:
            queue_size_per_producer (int): The maximum number of products each producer
                                           can have available in the marketplace at any given time.
        """
        self.queue_size_per_producer = queue_size_per_producer

        # Locks for protecting shared resources and ensuring thread safety.
        self.lock_register_prod = Lock() # Protects producer registration.
        self.lock_q_size = Lock()      # Protects product availability and producer queue sizes.
        self.lock_carts = Lock()        # Protects shopping cart operations.
        self.lock_printing = Lock()     # Protects console output during order placement.

        self.carts_total = 0 # Counter for generating unique cart IDs.
        self.producer_index = 0 # Counter for generating unique producer IDs.
        self.prod_sizes = [] # List to track the current number of products published by each producer.
        self.all_products = [] # List of all products currently available for purchase in the marketplace.
        self.all_producers = {} # Maps product names to their producer IDs.
        self.all_carts = {} # Dictionary to store active shopping carts.


    def register_producer(self):
        """
        Registers a new producer with the marketplace, assigning it a unique ID.

        Returns:
            int: The unique ID assigned to the new producer.
        """
        self.lock_register_prod.acquire() # Block Logic: Acquires lock to ensure atomic registration.
        self.producer_index += 1 # Increments global producer counter.
        self.prod_sizes.append(0) # Initializes product count for the new producer.
        self.lock_register_prod.release() # Block Logic: Releases lock.

        return self.producer_index - 1 # Returns the newly assigned producer ID.

    def publish(self, producer_id, product):
        """
        Publishes a product from a producer to the marketplace, if capacity allows.

        Args:
            producer_id (str): The ID of the producer publishing the product.
            product (Product): The product object to publish.

        Returns:
            bool: True if the product was successfully published, False otherwise
                  (e.g., if the producer's queue is full).
        """
        id_prod = int(producer_id)

        # Conditional Logic: Checks if the producer has reached its product publishing limit.
        if self.queue_size_per_producer > self.prod_sizes[id_prod]:

            self.all_products.append(product) # Adds product to the globally available list.
            self.all_producers[product] = id_prod # Records which producer published this product.
            self.prod_sizes[id_prod] += 1 # Increments count of products by this producer.
            return True

        return False # Product not published due to queue size limit.

    def new_cart(self):
        """
        Creates a new, empty shopping cart and assigns it a unique ID.

        Returns:
            int: The unique ID of the newly created cart.
        """
        self.lock_carts.acquire() # Block Logic: Acquires lock to ensure atomic cart ID generation and creation.

        self.carts_total += 1 # Increments global cart counter.
        id_cart = self.carts_total # Assigns new cart ID.

        self.all_carts[id_cart] = [] # Initializes an empty list for the new cart.

        self.lock_carts.release() # Block Logic: Releases lock.

        return id_cart


    def add_to_cart(self, cart_id, product):
        """
        Adds a product to a specified shopping cart.

        This operation involves acquiring locks to ensure thread safety when
        modifying shared product inventory and producer-specific counts.

        Args:
            cart_id (int): The ID of the cart to which the product should be added.
            product (Product): The product object to add.

        Returns:
            bool: True if the product was successfully added, False if the product
                  is no longer available in the marketplace.
        """
        self.lock_q_size.acquire() # Block Logic: Acquires lock to protect shared product and producer queue size data.

        # Conditional Logic: Checks if the product is still available in the marketplace.
        if product in self.all_products:
            ignore = False
        else:
            ignore = True

        # Block Logic: If the product is available, remove it from the global inventory
        # and update the producer's product count.
        if ignore is False:
            self.all_products.remove(product)
            self.prod_sizes[self.all_producers[product]] -= 1

        self.lock_q_size.release() # Block Logic: Releases lock.

        if ignore is True:
            return False # Product was not available.

        self.all_carts[cart_id].append(product) # Adds the product to the consumer's cart.

        return True

    def remove_from_cart(self, cart_id, product):
        """
        Removes a product from a specified shopping cart.

        This operation makes the product available again in the marketplace
        and updates producer-specific product counts.

        Args:
            cart_id (int): The ID of the cart from which the product should be removed.
            product (Product): The product object to remove.
        """
        self.lock_q_size.acquire() # Block Logic: Acquires lock to protect shared product and producer queue size data.
        index = self.all_producers[product] # Gets the producer ID for the product.
        self.prod_sizes[index] += 1 # Increments the count of products by this producer (as it's now available again).
        self.lock_q_size.release() # Block Logic: Releases lock.

        self.all_carts[cart_id].remove(product) # Removes the product from the consumer's cart.
        self.all_products.append(product) # Adds the product back to the globally available list.


    def place_order(self, cart_id):
        """
        Finalizes an order for a given shopping cart, effectively "buying" the products.

        The products in the cart are then removed from the marketplace's cart tracking
        and a message is printed to indicate the purchase.

        Args:
            cart_id (int): The ID of the cart for which to place the order.

        Returns:
            list: A list of products that were in the placed order.
        """
        # Functional Utility: Atomically removes the cart from `all_carts` to prevent
        # other operations on this cart and retrieves its contents.
        prods = self.all_carts.pop(cart_id)

        # Block Logic: Iterates through each product in the placed order and prints a purchase message.
        for prod in prods:
            self.lock_printing.acquire() # Block Logic: Acquires lock to ensure atomic printing to console.
            thread_name = currentThread().getName() # Gets the name of the current thread (consumer).
            print('{0} bought {1}'.format(thread_name, prod))
            self.lock_printing.release() # Block Logic: Releases lock.

        return prods


from threading import Thread
from time import sleep


class Producer(Thread):
    """
    Simulates a producer entity that continuously publishes products to the Marketplace.

    Producers are registered with the marketplace and then repeatedly attempt to
    publish a predefined set of products, respecting marketplace capacity limits.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        Initializes a Producer thread.

        Args:
            products (list): A list of product definitions to publish. Each definition
                             is typically a tuple like (product_object, quantity, waiting_time).
            marketplace (Marketplace): The shared marketplace instance to interact with.
            republish_wait_time (float): The time in seconds to wait before retrying
                                         to publish a product if the marketplace is full.
            **kwargs: Arbitrary keyword arguments passed to the Thread constructor.
        """
        Thread.__init__(self, **kwargs)

        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time

        # Functional Utility: Registers the producer with the marketplace to get a unique ID.
        self.producer_id = self.marketplace.register_producer()

    def run(self):
        """
        Main execution loop for the Producer thread.

        Architectural Intent: Continuously attempts to publish its designated products
        to the marketplace, handling capacity limits and introducing delays.
        """
        while True:
            # Block Logic: Iterates through each product definition that this producer is responsible for.
            for current_product in self.products:
                step = 0 # Counter for how many units of the current product have been published.
                product = current_product[0]
                products_no = current_product[1] # Total quantity of this product to publish.
                waiting_time = current_product[2] # Time to wait after a successful publication.

                # Block Logic: Continues to attempt publishing the current product until `products_no` units are published.
                while True:
                    # Functional Utility: Attempts to publish a single unit of the product to the marketplace.
                    published = self.marketplace.publish(str(self.producer_id), product)

                    # Conditional Logic: If publication was successful, increment the published count and wait.
                    if published is True:
                        step += 1
                        sleep(waiting_time)
                    # Conditional Logic: If publication failed (e.g., marketplace full for this producer),
                    # wait for `republish_wait_time` before retrying.
                    else:
                        sleep(self.republish_wait_time)

                    # Termination Condition: If the desired quantity of the current product has been published, break this inner loop.
                    if step == products_no:
                        break


from dataclasses import dataclass


@dataclass(init=True, repr=True, order=False, frozen=True)
class Product:
    """
    Base data class representing a generic product in the marketplace.
    Attributes are name (string) and price (integer).
    Uses dataclass for concise definition of attributes and standard methods.
    """
    name: str
    price: int


@dataclass(init=True, repr=True, order=False, frozen=True)
class Tea(Product):
    """
    Data class representing a specific type of product: Tea.
    Inherits from `Product` and adds a 'type' attribute (e.g., "green", "black").
    """
    type: str


@dataclass(init=True, repr=True, order=False, frozen=True)
class Coffee(Product):
    """
    Data class representing a specific type of product: Coffee.
    Inherits from `Product` and adds 'acidity' and 'roast_level' attributes.
    """
    acidity: str
    roast_level: str
