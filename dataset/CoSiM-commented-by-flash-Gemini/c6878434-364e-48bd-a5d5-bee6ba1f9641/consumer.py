
"""
This module implements a simulation of a multi-threaded producer-consumer marketplace
with logging capabilities. It defines classes for `Consumer` and `Producer` threads,
a central `Marketplace` for managing products and orders, and `Product` data classes.

Note: The `Marketplace` implementation uses local locks within its methods which
      does not provide effective thread-safe access to shared state. This is a
      critical concurrency flaw in the current design. Documentation will reflect
      the intended functionality rather than correcting the implementation.
"""

import time
from threading import Thread


class Consumer(Thread):
    """
    Simulates a consumer entity that interacts with a Marketplace.

    A Consumer creates shopping carts, repeatedly attempts to add or remove
    products based on predefined operations, and finally places orders.
    It incorporates a retry mechanism with a wait time for failed operations.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        Initializes a Consumer thread.

        Args:
            carts (list): A list of shopping cart definitions. Each definition
                          is a list of operations (add/remove) for products and quantities.
            marketplace (Marketplace): The shared marketplace instance to interact with.
            retry_wait_time (float): The time in seconds to wait before retrying
                                     a failed `add_to_cart` operation.
            **kwargs: Arbitrary keyword arguments passed to the Thread constructor.
        """
        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.id_cart = 0 # Stores the ID of the currently active shopping cart.

    def wait(self):
        """
        Functional Utility: Pauses the consumer thread for `retry_wait_time` duration.
        Used when an operation (e.g., adding to cart) needs to be retried.
        """
        time.sleep(self.retry_wait_time)

    def print_output(self):
        """
        Functional Utility: Places the current cart's order and prints a message
        for each product bought, including the consumer's thread name.
        """
        # Block Logic: Places the order for the cart identified by `self.id_cart`.
        cart = self.marketplace.place_order(self.id_cart)
        # Block Logic: Iterates through the products in the placed order and prints a purchase confirmation.
        for product in cart:
            # Inline: `self.name` refers to the name of the current thread.
            print(self.name + ' bought ' + str(product))

    def run(self):
        """
        Main execution loop for the Consumer thread.

        Architectural Intent: Iterates through a series of predefined shopping carts.
        For each cart, it performs a sequence of add/remove operations for products,
        handling retries for additions, and finally places the order.
        """
        # Block Logic: Iterates through each shopping cart definition provided to this consumer.
        for cart_definition in self.carts:
            # Functional Utility: Creates a new, unique shopping cart in the marketplace and stores its ID.
            self.id_cart = self.marketplace.new_cart()

            # Block Logic: Iterates through each operation (add/remove product) defined for the current cart.
            for operation in cart_definition:
                quantity = operation['quantity']
                product = operation['product']

                # Conditional Logic: Handles "add" operations.
                # It repeatedly tries to add the product until the desired quantity is reached,
                # waiting if the marketplace indicates a failure (e.g., product not available).
                if operation['type'] == "add":
                    while quantity > 0:
                        # Functional Utility: Attempts to add a single unit of the product to the cart.
                        if self.marketplace.add_to_cart(self.id_cart, product) is False:
                            self.wait() # Waits if the product could not be added.
                        else:
                            quantity -= 1 # Decrements quantity upon successful addition.

                # Conditional Logic: Handles "remove" operations.
                # It removes the specified quantity of products from the cart without retry logic.
                if operation['type'] == "remove":
                    while quantity > 0:
                        # Functional Utility: Removes a single unit of the product from the cart.
                        self.marketplace.remove_from_cart(self.id_cart, product)
                        quantity -= 1 # Decrements quantity upon removal.

            # Functional Utility: Prints the final order details after all operations for the cart are complete.
                        self.print_output()
            
            
            import logging
            from logging.handlers import RotatingFileHandler
            import time
            import threading
            from threading import currentThread
            
            # Configures the logging system to write to a rotating file.
            # Functional Utility: Provides a centralized mechanism for recording marketplace events
            #                     and thread activities for debugging and monitoring.
            logging.basicConfig(handlers=
                                [RotatingFileHandler(filename='./marketplace.log', maxBytes=400000,
                                                     backupCount=10)],
                                level=logging.INFO, # Sets the minimum logging level to INFO.
                                format="[%(asctime)s]::%(levelname)s::%(message)s") # Defines the log message format.
            # Functional Utility: Configures the time converter for logging to use GMT (Greenwich Mean Time).
            logging.Formatter.converter = time.gmtime
            
            
            class Marketplace:
                """
                Acts as the central hub for producers and consumers to interact.
            
                It manages product inventory, tracks registered producers, and handles shopping carts.
                The design aims for thread-safe operations on shared state, though some locking
                mechanisms are flawed in their implementation (using local locks instead of shared).
                """
            
                def __init__(self, queue_size_per_producer):
                    """
                    Initializes the Marketplace.
            
                    Args:
                        queue_size_per_producer (int): The maximum number of products each producer
                                                       can have available in the marketplace at any given time.
                    """
                    self.queue_size_per_producer = queue_size_per_producer
                    self.list_of_carts = [] # Stores lists of products for each cart.
                    self.list_of_producers = [] # Stores lists of products for each producer.
                    self.id_producer = -1 # Counter for generating unique producer IDs.
                    self.id_cart = -1 # Counter for generating unique cart IDs.
            
                def register_producer(self):
                    """
                    Registers a new producer with the marketplace, assigning it a unique ID.
            
                    Note: The `register_lock` is instantiated locally within this method,
                          which means it does not provide thread-safe access to shared
                          `self.id_producer` or `self.list_of_producers` across multiple calls.
                          The intent was likely to use a shared lock defined in `__init__`.
            
                    Returns:
                        str: The unique ID assigned to the new producer.
                    """
                    logging.info("register_producer() called by Thread %s",
                                 currentThread().getName())
            
                    # Flawed Locking: This lock is local to this method call and does not protect
                    # shared state across concurrent calls to `register_producer`.
                    register_lock = threading.Lock()
            
                    producers = [] # A new list to hold products for this producer.
            
                    # Block Logic: Increments the global producer ID counter.
                    self.id_producer += 1
            
                    # Intended Thread Safety: The `with register_lock` block is intended to protect
                    # modifications to `self.list_of_producers`. However, due to local lock, it's ineffective.
                    with register_lock:
                        self.list_of_producers.append(producers)
            
                    logging.info("Thread %s exited register_producer()",
                                 currentThread().getName())
            
                    return str(self.id_producer)
            
                def publish(self, producer_id, product):
                    """
                    Publishes a product from a producer to the marketplace, if capacity allows.
            
                    Note: The `publish_lock` is instantiated locally within this method,
                          which means it does not provide thread-safe access to shared state
                          (`self.list_of_producers`, `self.queue_size_per_producer`) across
                          multiple concurrent calls.
            
                    Args:
                        producer_id (str): The ID of the producer publishing the product.
                        product (tuple): A tuple containing product details, typically (name, quantity, sleep_time).
            
                    Returns:
                        bool: True if the product was successfully published, False otherwise
                              (e.g., if the producer's queue is full or for other conditions).
                    """
                    logging.info("publish() called by Thread %s with producer_id %s to register product %s",
                                 currentThread().getName(), str(producer_id), str(product))
            
                    quantity_to_publish = product[1] # Quantity of products to publish in this call.
                    sleep_time = product[2] # Sleep time associated with publishing this product.
                    id_prod = int(producer_id)
            
                    # Flawed Locking: This lock is local to this method call.
                    publish_lock = threading.Lock()
            
                    publish_check = False # Flag to indicate success or failure of publication.
            
                    # Block Logic: Checks if the producer's current queue is full.
                    if len(self.list_of_producers[id_prod]) == self.queue_size_per_producer:
                        logging.info("Thread %s with producer_id %s exited publish() with %s",
                                     currentThread().getName(), str(producer_id), str(publish_check))
                        return publish_check
            
                    # Block Logic: Checks if adding the requested quantity would exceed the producer's queue size.
                    if len(self.list_of_producers[id_prod]) + quantity_to_publish < self.queue_size_per_producer:
                        # Intended Thread Safety: The `with publish_lock` block is intended to protect
                        # modifications to `self.list_of_producers[id_prod]`. Ineffective due to local lock.
                        with publish_lock:
                            while quantity_to_publish > 0:
                                time.sleep(sleep_time) # Simulates time taken to publish a product.
                                self.list_of_producers[id_prod].append(product[0]) # Adds a product unit to producer's list.
                                quantity_to_publish -= 1
                    else:
                        logging.info("Thread %s with producer_id %s exited publish() with %s",
                                     currentThread().getName(), str(producer_id), str(publish_check))
                        return publish_check
            
                    publish_check = True # Marks publication as successful.
            
                    logging.info("Thread %s with producer_id %s exited publish() with %s",
                                 currentThread().getName(), str(producer_id), str(publish_check))
            
                    return publish_check
            
                def new_cart(self):
                    """
                    Creates a new, empty shopping cart and assigns it a unique ID.
            
                    Note: The `new_cart_lock` is instantiated locally within this method,
                          which means it does not provide thread-safe access to shared
                          `self.list_of_carts` or `self.id_cart` across multiple concurrent calls.
            
                    Returns:
                        int: The unique ID of the newly created cart.
                    """
                    logging.info("new_cart() called by Thread %s",
                                 currentThread().getName())
            
                    cart = [] # A new list to represent the shopping cart.
                    # Flawed Locking: This lock is local to this method call.
                    new_cart_lock = threading.Lock()
            
                    # Intended Thread Safety: The `with new_cart_lock` block is intended to protect
                    # modifications to `self.list_of_carts` and `self.id_cart`. Ineffective due to local lock.
                    with new_cart_lock:
                        self.list_of_carts.append(cart) # Adds the new cart to the list of all carts.
                        self.id_cart += 1 # Increments global cart ID counter.
            
                    return self.id_cart
            
                def add_to_cart(self, cart_id, product):
                    """
                    Adds a product to a specified shopping cart.
            
                    Note: The `add_to_cart_lock` is instantiated locally within this method,
                          which means it does not provide thread-safe access to shared state
                          (`self.list_of_producers`, `self.list_of_carts`) across multiple concurrent calls.
                          Also, the product is removed from `prod_list` (a producer's product list)
                          without acquiring a lock specific to that producer's list, leading to race conditions.
            
                    Args:
                        cart_id (int): The ID of the cart to which the product should be added.
                        product (Product): The product object to add.
            
                    Returns:
                        bool: True if the product was successfully added (and was available), False otherwise.
                    """
                    logging.info("add_to_cart() called by Thread %s for the cart %s to add product %s",
                                 currentThread().getName(), str(cart_id), str(product))
            
                    product_existence = False # Flag to track if the product was found and added.
            
                    prod_list = [] # Placeholder for the producer's product list that contains the product.
            
                    # Flawed Locking: This lock is local to this method call.
                    add_to_cart_lock = threading.Lock()
            
                    # Block Logic: Searches for the product across all producers' inventories.
                    # This part is susceptible to race conditions as `self.list_of_producers` is not locked.
                    for producer_products_list in self.list_of_producers:
                        if product in producer_products_list:
                            product_existence = True
                            prod_list = producer_products_list # Stores reference to the producer's list where product was found.
                            break
            
                    # Conditional Logic: If the product was found and available.
                    if product_existence is True:
                        # Intended Thread Safety: The `with add_to_cart_lock` block is intended to protect
                        # modifications to `self.list_of_carts` and `prod_list`. Ineffective due to local lock.
                        with add_to_cart_lock:
                            self.list_of_carts[cart_id].append(product) # Adds product to the cart.
                            # Removes product from the producer's inventory. This modification is NOT thread-safe
                            # with concurrent `publish` or `add_to_cart` calls for the same producer.
                            prod_list.remove(product)
            
                    logging.info("Thread %s exited add_to_cart() with %s",
                                 currentThread().getName(), str(product_existence))
            
                    return product_existence
            
                def remove_from_cart(self, cart_id, product):
                    """
                    Removes a product from a specified shopping cart and returns it to the marketplace.
            
                    Note: The `remove_from_cart_lock` is instantiated locally within this method,
                          which means it does not provide thread-safe access to shared state
                          (`self.list_of_carts`, `self.list_of_producers`) across multiple concurrent calls.
                          The product is returned to `self.list_of_producers[0]` without locking.
            
                    Args:
                        cart_id (int): The ID of the cart from which the product should be removed.
                        product (Product): The product object to remove.
                    """
                    logging.info("remove_from_cart() called by Thread %s for the cart %s to remove product %s",
                                 currentThread().getName(), str(cart_id), str(product))
            
                    # Flawed Locking: This lock is local to this method call.
                    remove_from_cart_lock = threading.Lock()
            
                    # Block Logic: Checks if the product is in the specified cart.
                    if product in self.list_of_carts[cart_id]:
                        # Intended Thread Safety: The `with remove_from_cart_lock` block is intended to protect
                        # modifications to `self.list_of_carts` and `self.list_of_producers`. Ineffective due to local lock.
                        with remove_from_cart_lock:
                            self.list_of_carts[cart_id].remove(product) # Removes product from the cart.
                            # Returns the product to the inventory of producer with ID 0.
                            # This operation is NOT thread-safe for `self.list_of_producers[0]`.
                            self.list_of_producers[0].append(product)
            
                    logging.info("Thread %s exited remove_from_cart()",
                                 currentThread().getName())
            
                def place_order(self, cart_id):
                    """
                    Finalizes an order for a given shopping cart, effectively "buying" the products.
            
                    Note: This method retrieves the products from `self.list_of_carts` but does not
                          remove the cart itself from `self.list_of_carts`, nor does it update
                          any inventory counts, which could lead to inconsistencies.
            
                    Args:
                        cart_id (int): The ID of the cart for which to place the order.
            
                    Returns:
                        list: A list of products that were in the placed order.
                    """
                    logging.info("place_order() called by Thread %s for the cart %s",
                                 currentThread().getName(), str(cart_id))
            
                    return_list = self.list_of_carts[cart_id] # Retrieves products from the specified cart.
            
                    logging.info("Thread %s exited place_order()",
                                 currentThread().getName())
            
                    return return_list


        return return_list


import time
from threading import Thread


class Producer(Thread):
    """
    Simulates a producer entity that continuously publishes products to the Marketplace.

    Producers are registered with the marketplace and then repeatedly attempt to
    publish a predefined set of products, handling potential failures (e.g., full queue)
    by waiting and retrying.
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
        # Functional Utility: Registers the producer with the marketplace to get a unique ID.
        self.id_producer = self.marketplace.register_producer()
        self.republish_wait_time = republish_wait_time

    def wait(self):
        """
        Functional Utility: Pauses the producer thread for `republish_wait_time` duration.
        Used when a product could not be published and needs to be retried.
        """
        time.sleep(self.republish_wait_time)

    def run(self):
        """
        Main execution loop for the Producer thread.

        Architectural Intent: Continuously attempts to publish its designated products
        to the marketplace, handling capacity limits and introducing delays.
        """
        while True:
            # Block Logic: Iterates through each product definition that this producer is responsible for.
            for product_definition in self.products:
                # Block Logic: Continuously attempts to publish the current product until it succeeds.
                # It repeatedly calls `marketplace.publish` and waits if the attempt fails.
                while self.marketplace.publish(self.id_producer, product_definition) is False:
                    self.wait() # Waits if the product could not be published.


from dataclasses import dataclass


@dataclass(init=True, repr=True, order=False, frozen=True)
class Product:
    """
    Base data class representing a generic product in the marketplace.
    Attributes are `name` (string) and `price` (integer).
    Uses dataclass for concise definition of attributes and standard methods.
    `frozen=True` makes instances immutable.
    """
    name: str
    price: int


@dataclass(init=True, repr=True, order=False, frozen=True)
class Tea(Product):
    """
    Data class representing a specific type of product: Tea.
    Inherits from `Product` and adds a `type` attribute (e.g., "green", "black").
    `frozen=True` makes instances immutable.
    """
    type: str


@dataclass(init=True, repr=True, order=False, frozen=True)
class Coffee(Product):
    """
    Data class representing a specific type of product: Coffee.
    Inherits from `Product` and adds `acidity` and `roast_level` attributes.
    `frozen=True` makes instances immutable.
    """
    acidity: str
    roast_level: str
