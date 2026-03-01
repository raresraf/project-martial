"""
@file consumer.py
@brief A multi-threaded, producer-consumer simulation of a marketplace.

This script models a simple e-commerce environment where multiple Producer threads
create products and add them to a central Marketplace, and multiple Consumer
threads purchase them. The simulation highlights concurrent operations, resource
sharing, and synchronization using Python's threading module.

- Marketplace: The central class that manages product inventories and shopping carts.
- Producer: A thread that simulates creating products and publishing them to the marketplace.
- Consumer: A thread that simulates adding products to a cart and purchasing them.
- Cart: A data structure representing a consumer's shopping cart.
"""

from threading import Lock, Thread
from time import sleep

def print_products(consumer_name, products):
    """
    Prints the products a consumer has bought in a thread-safe manner.

    @param consumer_name The name of the consumer.
    @param products A list of products purchased.
    """
    for product in products:
        print("{} bought {}".format(consumer_name, product))

class Consumer(Thread):
    """
    Represents a consumer that interacts with the marketplace to buy products.
    Each consumer runs in its own thread and processes a predefined set of
    shopping operations.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        Initializes a Consumer thread.

        @param carts A list of shopping sessions. Each session is a list of
                     cart operations (add/remove).
        @param marketplace The shared Marketplace instance.
        @param retry_wait_time The time in seconds to wait before retrying to
                               add a product if it's out of stock.
        @param kwargs Additional keyword arguments, expects 'name'.
        """
        name = kwargs["name"]
        super().__init__()

        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.name = name

        # A lock to ensure that print statements from different threads do not interleave.
        self.print_lock = Lock()

    def run(self):
        """
        The main execution logic for the consumer thread.
        It iterates through its assigned shopping carts, performs the operations,
        and places an order for each.
        """
        # Invariant: Each loop processes one full shopping session (a single cart).
        for cart_operations in self.carts:
            # Pre-condition: Start with a new, empty cart for the session.
            cart_id = self.marketplace.new_cart()

            # Block Logic: Process all add/remove operations for the current cart.
            for cart_operation in cart_operations:
                operation_type = cart_operation["type"]
                operation_product = cart_operation["product"]
                operation_cnt = cart_operation["quantity"]

                # Perform the operation 'operation_cnt' times.
                for _ in range(operation_cnt):
                    if operation_type == "add":
                        added = False

                        # Block Logic: Retry adding the product until successful.
                        # This handles cases where the product is temporarily out of stock.
                        # Invariant: The loop continues until 'added' is True.
                        while True:
                            added = self.marketplace.add_to_cart(cart_id, operation_product)

                            if not added:
                                # Wait before retrying to avoid busy-waiting.
                                sleep(self.retry_wait_time)
                            else:
                                break
                    elif operation_type == "remove":
                        self.marketplace.remove_from_cart(cart_id, operation_product)
                    else:
                        raise Exception("Unknown op: cart {}, cons {}".format(cart_id, self.name))

            # Finalize the purchase for the current cart.
            ordered_products = self.marketplace.place_order(cart_id)

            # Block Logic: Use a lock to ensure console output is not corrupted by other threads.
            with self.print_lock:
                print_products(self.name, ordered_products)


from threading import Lock

class Marketplace:
    """
    The central hub of the simulation, managing producers, inventory, and carts.
    This class orchestrates all interactions between producers and consumers,
    ensuring thread-safe access to shared resources like product queues and carts.
    """

    def __init__(self, queue_size_per_producer):
        """
        Initializes the Marketplace.

        @param queue_size_per_producer The maximum number of products a single
                                       producer can have in their inventory queue.
        """
        self.queue_size_per_producer = queue_size_per_producer
        
        # Holds the inventory for each registered producer.
        self.producer_queues = {}
        # A lock to protect access to the shared producer queues.
        self.producer_queue_lock = Lock()
        
        # A counter for generating unique producer IDs.
        self.producer_next_id = 0
        # A lock to ensure thread-safe generation of producer IDs.
        self.producer_id_generator_lock = Lock()

        # Holds all active shopping carts.
        self.carts = {}
        # A counter for generating unique cart IDs.
        self.cart_next_id = 0
        # A lock to ensure thread-safe generation of cart IDs.
        self.cart_id_generator_lock = Lock()

    def register_producer(self):
        """
        Registers a new producer, allocating an inventory queue for them.

        @return A unique ID for the newly registered producer.
        """
        # Block Logic: Atomically generate a new ID and create the producer's queue.
        with self.producer_id_generator_lock:
            producer_id = self.producer_next_id
            self.producer_queues[producer_id] = []
            self.producer_next_id += 1

        return producer_id

    def publish(self, producer_id, product):
        """
        Allows a producer to add a product to their inventory.
        The operation fails if the producer's inventory queue is full.

        @param producer_id The ID of the producer publishing the product.
        @param product The product to be added.
        @return True if the product was successfully published, False otherwise.
        """
        producer_queue = self.producer_queues[producer_id]

        # Block Logic: Atomically check queue capacity and add the product.
        with self.producer_queue_lock:
            if len(producer_queue) < self.queue_size_per_producer:
                producer_queue.append(product)
                return True

        return False


    def new_cart(self):
        """
        Creates a new, empty shopping cart and returns its unique ID.

        @return A unique ID for the new cart.
        """
        # Block Logic: Atomically generate a new cart ID and initialize the cart.
        with self.cart_id_generator_lock:
            cart_id = self.cart_next_id
            self.carts[cart_id] = Cart()
            self.cart_next_id += 1
            return cart_id

    def add_to_cart(self, cart_id, product):
        """
        Adds a product to a shopping cart by taking it from a producer's inventory.
        It searches through all producer inventories to find the product.

        @param cart_id The ID of the cart to add the product to.
        @param product The product to be added.
        @return True if the product was found and added, False otherwise.
        """
        # Get the total number of producers in a thread-safe way.
        with self.producer_id_generator_lock:
            no_producers = self.producer_next_id

        # Invariant: Iterate through all producers to find the requested product.
        for producer_id in range(no_producers):
            producer_stock = self.producer_queues[producer_id]

            # Pre-condition: Check if the product exists in the current producer's stock.
            if product in producer_stock:
                # Block Logic: Atomically remove the product from the producer's
                # stock to prevent race conditions with other consumers.
                with self.producer_queue_lock:
                    producer_stock.remove(product)

                # Add the product to the consumer's cart, tracking its origin.
                self.carts[cart_id].add_product(product, producer_id)
                return True
        
        return False

    def remove_from_cart(self, cart_id, product):
        """
        Removes a product from a shopping cart and returns it to the
        original producer's inventory.

        @param cart_id The ID of the cart.
        @param product The product to remove.
        """
        # Identify which producer the product originally came from.
        producer_id = self.carts[cart_id].remove_product(product)

        # Block Logic: Atomically return the product to the producer's queue
        # if there is space available.
        with self.producer_queue_lock:
            producer_queue = self.producer_queues[producer_id]
            if len(producer_queue) < self.queue_size_per_producer:
                producer_queue.append(product)


    def place_order(self, cart_id):
        """
        Finalizes the order and returns the list of products in the cart.

        @param cart_id The ID of the cart to order.
        @return A list of products that were in the cart.
        """
        return self.carts[cart_id].get_products()

class Cart:
    """
    A data structure representing a consumer's shopping cart. It holds items
    before they are purchased and tracks their original producer.
    """

    def __init__(self):
        """Initializes an empty cart."""
        self.products = []

    def add_product(self, product, producer_id):
        """
        Adds a product to the cart.

        @param product The product identifier.
        @param producer_id The ID of the producer who supplied the product.
        """
        self.products.append({"product": product, "producer_id": producer_id})

    def remove_product(self, product):
        """
        Removes a product from the cart.

        @param product The product to be removed.
        @return The ID of the producer who originally supplied the product, or None.
        """
        # Invariant: Search for the first matching product to remove.
        for prod in self.products:
            if prod["product"] == product:
                producer_id = prod["producer_id"]
                self.products.remove(prod)
                # Return the producer_id so the item can be returned to the correct stock.
                return producer_id
        return None

    def get_products(self):
        """
        Returns a simple list of all products in the cart.

        @return A list of product identifiers.
        """
        product_list = []
        for product_item in self.products:
            product_list.append(product_item["product"])
        return product_list


from threading import Thread
from time import sleep

class Producer(Thread):
    """
    Represents a producer that creates products and publishes them to the marketplace.
    Each producer runs in its own thread, simulating production time and handling
    cases where the marketplace inventory is full.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        Initializes a Producer thread.

        @param products A list of products this producer can create. Each item is a
                        tuple of (product_id, quantity, production_time).
        @param marketplace The shared Marketplace instance.
        @param republish_wait_time The time to wait before retrying to publish if
                                   the inventory queue is full.
        @param kwargs Additional keyword arguments, expects 'daemon' and 'name'.
        """
        Thread.__init__(self, daemon=kwargs["daemon"])

        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        self.name = kwargs["name"]

    def run(self):
        """

        The main execution logic for the producer thread. It continuously
        produces items from its product list and tries to publish them.
        """
        producer_id = self.marketplace.register_producer()

        # Invariant: The producer runs in an infinite loop to continuously supply products.
        while True:
            for product in self.products:
                product_id = product[0]
                product_quantity = product[1]
                product_production_time = product[2]

                # Simulate the time it takes to produce the item.
                sleep(product_production_time)
                
                # Produce the specified quantity of the product.
                for _ in range(product_quantity):
                    produced = False

                    # Block Logic: Retry publishing until successful.
                    # This handles cases where the producer's queue in the marketplace is full.
                    # Invariant: Loop continues until 'produced' is True.
                    while True:
                        produced = self.marketplace.publish(producer_id, product_id)

                        if not produced:
                            # Wait before retrying to avoid busy-waiting.
                            sleep(self.republish_wait_time)
                        else:
                            break
