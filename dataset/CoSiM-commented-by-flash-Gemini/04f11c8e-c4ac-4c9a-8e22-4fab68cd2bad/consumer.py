
"""
This module simulates a multi-threaded marketplace with producers and consumers.
Producers add products to the marketplace, and consumers attempt to purchase them,
managing their carts and placing orders. The Marketplace class ensures thread-safe
operations using synchronization primitives.

Algorithm:
- Producer-Consumer Pattern: Producers publish products, and consumers try to acquire them.
- Retry Mechanism: Both producers and consumers implement a retry mechanism with a wait time
  if an operation (publishing or adding to cart) initially fails due to resource limitations.
- Thread-Safe Operations: The Marketplace uses semaphores and locks to ensure that
  product queues and shopping carts are accessed and modified safely by multiple threads.
"""

import time
from threading import Thread


class Consumer(Thread):
    """
    Simulates a consumer in a marketplace.

    A consumer attempts to add and remove products from a shopping cart
    managed by the marketplace. It retries operations if they fail due
    to product unavailability and ultimately places an order.

    Attributes:
        name (str): The name of the consumer thread.
        retry_wait_time (int): The time in seconds to wait before retrying an operation.
        id_cart (int): The unique identifier for the consumer's shopping cart.
        carts (list): A list of shopping cart command lists, where each command
                      specifies an action (add/remove), product, and quantity.
        marketplace (Marketplace): A reference to the shared marketplace instance.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        Initializes a new Consumer instance.

        Args:
            carts (list): A list of lists, where each inner list represents a cart
                          with commands (add/remove product, quantity).
            marketplace (Marketplace): The marketplace instance to interact with.
            retry_wait_time (int): The time to wait before retrying a failed operation.
            **kwargs: Arbitrary keyword arguments, used here to set the thread's name.
        """
        # Call the constructor of the base Thread class.
        super().__init__()
        self.name = kwargs["name"]
        self.retry_wait_time = retry_wait_time
        # Initialize cart ID, to be assigned by the marketplace.
        self.id_cart = -1
        self.carts = carts
        self.marketplace = marketplace

    def run(self):
        """
        The main execution method for the consumer thread.

        Invariant: Iterates through each cart's commands, performing add/remove
                   operations, retrying if necessary, and finally placing the order.
        """
        # Block Logic: Process each predefined shopping cart for this consumer.
        for cart in self.carts:
            # Create a new shopping cart in the marketplace for this consumer.
            self.id_cart = self.marketplace.new_cart()
            # Block Logic: Execute each command within the current cart.
            for command in cart:
                comm_type = command["type"]
                product = command["product"]
                quantity = command["quantity"]

                # Block Logic: Handle "add" commands.
                if comm_type == "add":
                    # Attempt to add the product to the cart the specified number of times.
                    for i in range(quantity):
                        add_result = self.marketplace.add_to_cart(self.id_cart, product)
                        # Retry adding the product until successful.
                        while True:
                            # Pre-condition: `add_result` is False, indicating a failed attempt.
                            if not add_result:
                                # Wait before retrying to avoid busy-waiting.
                                time.sleep(self.retry_wait_time)
                                add_result = self.marketplace.add_to_cart(self.id_cart, product)
                            else:
                                # Invariant: Product successfully added, exit retry loop.
                                break
                # Block Logic: Handle "remove" commands.
                elif comm_type == "remove":
                    # Attempt to remove the product from the cart the specified number of times.
                    for i in range(quantity):
                        remove_result = self.marketplace.remove_from_cart(self.id_cart, product)
                        # Functional Utility: Error handling for failed removal, which should not happen
                        # if the cart is managed correctly.
                        if not remove_result:
                            print("INVALID OPERATION RESULT! REMOVED FAILED! EXITING THREAD")
                            return
                # Functional Utility: Error handling for invalid command types.
                else:
                    print("INVALID OUTPUT! EXITING THREAD")
                    return
            # Place the final order for the current cart.
            cart_list = self.marketplace.place_order(self.id_cart)
            # Print the items bought by the consumer.
            for item in cart_list:
                print(f"{self.name} bought {item}")


import threading


class Marketplace(object):
    """
    Manages products from producers and shopping carts for consumers in a thread-safe manner.

    It provides functionalities for producers to register and publish products,
    and for consumers to create carts, add/remove products, and place orders.
    Synchronization is handled using semaphores to ensure data consistency.

    Attributes:
        queues (dict): Stores product queues for each producer, using producer IDs as keys.
                       Each entry contains a list of products and a semaphore for that queue.
        capacity (int): The maximum number of products a producer's queue can hold.
        id_producer (int): A counter for assigning unique producer IDs.
        id_cart (int): A counter for assigning unique shopping cart IDs.
        general_semaphore (threading.Semaphore): A general semaphore to protect
                                                 producer registration.
        carts (dict): Stores shopping carts, using cart IDs as keys. Each entry
                      is a list of (producer_id, product) tuples.
    """

    def __init__(self, queue_size_per_producer):
        """
        Initializes a new Marketplace instance.

        Args:
            queue_size_per_producer (int): The maximum number of products
                                           a single producer can have in its queue.
        """
        # Dictionary to hold product queues for each producer.
        # Format: {producer_id: {"products": [product1, product2], "semaphore": Semaphore_obj}}
        self.queues = {}
        self.capacity = queue_size_per_producer
        self.id_producer = -1
        self.id_cart = -1
        # Semaphore to protect modifications to the `queues` dictionary during producer registration.
        self.general_semaphore = threading.Semaphore(1)
        # Dictionary to hold shopping carts. Format: {cart_id: [(producer_id, product), ...]}
        self.carts = {}

    def register_producer(self):
        """
        Registers a new producer with the marketplace, creating a dedicated product queue for it.

        Returns:
            int: The unique identifier assigned to the new producer.
        """
        # Acquire a general semaphore to ensure atomic registration of producers.
        self.general_semaphore.acquire()

        # Increment producer ID and create a new queue with its own semaphore.
        self.id_producer += 1
        self.queues[self.id_producer] = {}
        self.queues[self.id_producer]["products"] = []
        self.queues[self.id_producer]["semaphore"] = threading.Semaphore(1)
        # Release the general semaphore.
        self.general_semaphore.release()
        return self.id_producer

    def publish(self, producer_id, product):
        """
        Attempts to publish a product to the specified producer's queue.

        Pre-condition: The producer's queue must not be at its maximum capacity.
        Invariant: If the queue has space, the product is added; otherwise, it's not.

        Args:
            producer_id (int): The ID of the producer publishing the product.
            product (any): The product to be published.

        Returns:
            bool: True if the product was successfully published, False otherwise.
        """
        # Acquire the semaphore specific to this producer's queue to ensure thread safety.
        self.queues[producer_id]["semaphore"].acquire()
        # Check if the producer's queue has space.
        if len(self.queues[producer_id]["products"]) < self.capacity:
            # Add the product to the queue.
            self.queues[producer_id]["products"].append(product)
            # Release the producer's queue semaphore.
            self.queues[producer_id]["semaphore"].release()
            return True
        # If the queue is full, release the semaphore and return False.
        self.queues[producer_id]["semaphore"].release()
        return False

    def new_cart(self):
        """
        Creates a new, empty shopping cart and assigns it a unique ID.

        Returns:
            int: The unique identifier for the newly created cart.
        """
        # Increment cart ID and create a new empty list for the cart.
        self.id_cart += 1
        self.carts[self.id_cart] = []
        return self.id_cart

    def add_to_cart(self, cart_id, product):
        """
        Attempts to add a product from a producer's queue to a consumer's cart.

        It searches through all producer queues. If the product is found, it's
        removed from the producer's queue and added to the specified cart.

        Args:
            cart_id (int): The ID of the cart to add the product to.
            product (any): The product to add.

        Returns:
            bool: True if the product was found and added to the cart, False otherwise.
        """
        # Block Logic: Iterate through all producer queues to find the desired product.
        for id_producer, queue_producer in self.queues.items():
            # Acquire the semaphore for the current producer's queue.
            queue_producer["semaphore"].acquire()
            # Block Logic: Iterate through products in the current producer's queue.
            for queue_product in queue_producer["products"]:
                # If the product is found:
                if product == queue_product:
                    # Remove the product from the producer's queue.
                    queue_producer["products"].remove(queue_product)
                    # Add the product along with its producer's ID to the consumer's cart.
                    self.carts[cart_id].append((id_producer, product))
                    # Release the producer's queue semaphore.
                    queue_producer["semaphore"].release()
                    return True
            # If product not found in this queue, release semaphore before checking next queue.
            queue_producer["semaphore"].release()

        # If the product was not found in any producer's queue.
        return False

    def remove_from_cart(self, cart_id, product):
        """
        Removes a product from a specified shopping cart and returns it to the
        original producer's queue.

        Args:
            cart_id (int): The ID of the cart to remove the product from.
            product (any): The product to remove.

        Returns:
            bool: True if the product was successfully removed and returned, False otherwise.
        """
        # Block Logic: Iterate through items in the specified cart.
        for id_producer, cart_product in self.carts[cart_id]:
            # If the product to remove is found in the cart:
            if product == cart_product:
                # Remove the product from the cart.
                self.carts[cart_id].remove((id_producer, cart_product))
                # Acquire the semaphore for the original producer's queue.
                self.queues[id_producer]["semaphore"].acquire()
                # Return the product to the producer's queue.
                self.queues[id_producer]["products"].append(product)
                # Release the producer's queue semaphore.
                self.queues[id_producer]["semaphore"].release()
                return True
        # If the product was not found in the cart.
        return False

    def place_order(self, cart_id):
        """
        Retrieves the list of products in a specified cart for final order placement.

        Args:
            cart_id (int): The ID of the cart to place the order for.

        Returns:
            list: A list of products (without producer IDs) in the ordered cart.
        """
        result = []
        # Extract just the product names from the (producer_id, product) tuples in the cart.
        for _, product in self.carts[cart_id]:
            result.append(product)

        return result


from threading import Thread
import time


class Producer(Thread):
    """
    Simulates a producer that continuously publishes products to the marketplace.

    Each producer has a set of products with specified quantities and publication delays.
    It retries publishing if the marketplace's queue is full.

    Attributes:
        products (list): A list of product structures, where each structure
                         is (product_name, quantity_to_produce, sleep_time_after_publish).
        marketplace (Marketplace): A reference to the shared marketplace instance.
        name (str): The name of the producer thread.
        republish_wait_time (int): The time in seconds to wait before retrying
                                   to publish a product if the queue is full.
        id_producer (int): The unique identifier for this producer.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        Initializes a new Producer instance.

        Args:
            products (list): A list of products with their quantities and sleep times.
            marketplace (Marketplace): The marketplace instance to publish to.
            republish_wait_time (int): Time to wait before retrying a publish operation.
            **kwargs: Arbitrary keyword arguments, used here to set the thread's name.
        """
        # Call the constructor of the base Thread class, setting it as a daemon thread.
        super().__init__(daemon=True)
        self.products = products
        self.marketplace = marketplace
        self.name = kwargs["name"]
        self.republish_wait_time = republish_wait_time
        # Initialize producer ID, to be assigned by the marketplace upon registration.
        self.id_producer = -1

    def run(self):
        """
        The main execution method for the producer thread.

        Invariant: Continuously attempts to publish its predefined products to the
                   marketplace, retrying if the marketplace queue is full,
                   and pausing between publications.
        """
        # Register this producer with the marketplace to get a unique ID.
        self.id_producer = self.marketplace.register_producer()

        # Infinite loop for continuous production.
        while True:
            # Block Logic: Iterate through each product type this producer is responsible for.
            for product_struct in self.products:
                product = product_struct[0]
                quantity = product_struct[1]
                sleep_time = product_struct[2]

                # Block Logic: Publish the specified quantity of the current product.
                for _ in range(quantity):
                    publish_result = self.marketplace.publish(self.id_producer, product)

                    # Pre-condition: `publish_result` is False, indicating the queue is full.
                    if not publish_result:
                        # Retry publishing until successful.
                        while True:
                            # Wait before retrying.
                            time.sleep(self.republish_wait_time)
                            publish_result = self.marketplace.publish(self.id_producer, product)
                            # Invariant: Product successfully published, exit retry loop.
                            if publish_result:
                                # Functional Utility: Pause after successful publication to simulate work.
                                time.sleep(sleep_time)
                                break
                    else:
                        # Functional Utility: Pause after successful publication.
                        time.sleep(sleep_time)
