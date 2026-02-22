"""
@6025a43b-a5bd-46a9-bccf-f533ca406b31/consumer.py
@brief Implements a multi-threaded producer-consumer system with a marketplace, cart, and producer entities.

This module sets up a simulation for an e-commerce-like system where producers supply products
to a shared marketplace, and consumers attempt to add/remove items from carts and place orders.
It utilizes threading for concurrency and locks for managing access to shared resources.
"""

from threading import Lock, Thread
from time import sleep

def print_products(consumer_name, products):
    """
    @brief Prints a formatted message indicating what a consumer has bought.

    Args:
        consumer_name (str): The name of the consumer.
        products (list): A list of products bought by the consumer.
    """
    # Block Logic: Iterates through the list of products and prints a message for each.
    # Pre-condition: 'products' is a list of items bought by the consumer.
    # Invariant: Each product will be printed with the consumer's name.
    for product in products:
        print("{} bought {}".format(consumer_name, product))

class Consumer(Thread):
    """
    @brief Represents a consumer thread that interacts with the marketplace to make purchases.

    Each consumer processes a list of cart operations (add/remove products) and then
    places an order through the Marketplace.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        @brief Initializes a Consumer thread.

        Args:
            carts (list): A list of lists, where each inner list represents a sequence of cart operations.
            marketplace (Marketplace): The shared marketplace instance.
            retry_wait_time (float): Time in seconds to wait before retrying an add operation if product is unavailable.
            **kwargs: Arbitrary keyword arguments, expecting 'name' for the consumer.
        """
        # Functional Utility: Extracts the consumer's name from keyword arguments.
        name = kwargs["name"]

        # Functional Utility: Calls the constructor of the parent Thread class.
        super().__init__()

        # Functional Utility: Stores the sequence of cart operations for this consumer.
        self.carts = carts
        # Functional Utility: Stores a reference to the shared Marketplace instance.
        self.marketplace = marketplace
        # Functional Utility: Stores the retry wait time for product unavailability.
        self.retry_wait_time = retry_wait_time
        # Functional Utility: Stores the name of the consumer.
        self.name = name

        # Functional Utility: A lock to ensure exclusive access to the print statement,
        # preventing interleaved output from multiple consumers.
        self.print_lock = Lock()

    def run(self):
        """
        @brief The main execution loop for the Consumer thread.

        Iterates through all assigned carts, performs add/remove operations,
        places orders, and prints the purchased products.
        """
        # Block Logic: Iterates through each sequence of cart operations (effectively, each shopping session).
        # Pre-condition: 'self.carts' contains lists of cart operations.
        # Invariant: Each cart operation sequence will be processed.
        for cart_operations in self.carts:
            # Functional Utility: Obtains a new, unique cart ID from the marketplace.
            cart_id = self.marketplace.new_cart()

            # Block Logic: Processes each individual operation within the current cart sequence.
            # Pre-condition: 'cart_operations' is a list of dictionaries detailing operations.
            # Invariant: Each operation (add/remove) will be attempted for the current cart.
            for cart_operation in cart_operations:
                operation_type = cart_operation["type"]
                operation_product = cart_operation["product"]
                operation_cnt = cart_operation["quantity"]

                # Block Logic: Repeats the current operation 'operation_cnt' times.
                # Pre-condition: 'operation_cnt' is the number of times the operation needs to be performed.
                # Invariant: The specified operation will be attempted 'operation_cnt' times.
                for _ in range(operation_cnt):
                    # Conditional Logic: Handles "add" operations.
                    if operation_type == "add":
                        added = False

                        # Block Logic: Continuously attempts to add the product until successful.
                        # Invariant: The loop retries adding the product if it's not available.
                        while True:
                            added = self.marketplace.add_to_cart(cart_id, operation_product)

                            # Conditional Logic: If the product was not added, waits and retries.
                            if not added:
                                sleep(self.retry_wait_time)
                            else:
                                break # Exit retry loop on successful addition.
                    # Conditional Logic: Handles "remove" operations.
                    elif operation_type == "remove":
                        self.marketplace.remove_from_cart(cart_id, operation_product)
                    # Conditional Logic: Raises an error for unsupported operation types.
                    else:
                        raise Exception("Unknown op: cart {}, cons {}".format(cart_id, self.name))

            # Functional Utility: Places the order for all items currently in the cart.
            ordered_products = self.marketplace.place_order(cart_id)

            # Block Logic: Acquires a lock before printing to ensure clean output.
            # Functional Utility: Calls a helper function to print the purchased products.
            with self.print_lock:
                print_products(self.name, ordered_products)


class Marketplace:
    """
    @brief Manages products from producers, consumer carts, and ensures thread-safe operations.

    It handles producer registration, product publishing, cart creation, and product transfers
    between producers, marketplace stock, and consumer carts.
    """

    def __init__(self, queue_size_per_producer):
        """
        @brief Initializes the Marketplace.

        Args:
            queue_size_per_producer (int): The maximum number of products a single
                                           producer can have waiting in the marketplace.
        """
        # Functional Utility: Stores the maximum number of items a producer can have in its queue.
        self.queue_size_per_producer = queue_size_per_producer
        # Functional Utility: Dictionary mapping producer IDs to their respective product queues (lists).
        self.producer_queues = {}
        # Functional Utility: A lock to protect access to 'producer_queues' and their contents.
        self.producer_queue_lock = Lock()
        
        # Functional Utility: Counter for assigning unique producer IDs.
        self.producer_next_id = 0
        # Functional Utility: A lock to protect access to the 'producer_next_id' counter.
        self.producer_id_generator_lock = Lock()

        # Functional Utility: Dictionary mapping cart IDs to Cart objects.
        self.carts = {}
        # Functional Utility: Counter for assigning unique cart IDs.
        self.cart_next_id = 0
        # Functional Utility: A lock to protect access to the 'cart_next_id' counter.
        self.cart_id_generator_lock = Lock()

    def register_producer(self):
        """
        @brief Registers a new producer with the marketplace.

        Assigns a unique ID to the producer and initializes an empty product queue for them.

        Returns:
            int: The unique ID assigned to the registered producer.
        """
        # Block Logic: Ensures exclusive access when generating a new producer ID and initializing its queue.
        with self.producer_id_generator_lock:
            # Functional Utility: Retrieves the next available producer ID.
            producer_id = self.producer_next_id

            # Functional Utility: Initializes an empty list as the product queue for the new producer.
            self.producer_queues[producer_id] = []

            # Functional Utility: Increments the producer ID counter for the next registration.
            self.producer_next_id += 1

        return producer_id

    def publish(self, producer_id, product):
        """
        @brief Publishes a product from a producer to their queue in the marketplace.

        The product is added only if the producer's queue size does not exceed the limit.

        Args:
            producer_id (int): The ID of the producer publishing the product.
            product (any): The product to be published.

        Returns:
            bool: True if the product was successfully published, False otherwise.
        """
        # Functional Utility: Retrieves the product queue for the given producer.
        producer_queue = self.producer_queues[producer_id]

        # Block Logic: Ensures exclusive access to the producer's queue for checking size and appending.
        with self.producer_queue_lock:
            # Conditional Logic: Checks if the producer's queue has space.
            if len(producer_queue) < self.queue_size_per_producer:
                # Functional Utility: Adds the product to the producer's queue.
                producer_queue.append(product)
                # Functional Utility: Indicates successful publishing.
                return True

        # Functional Utility: Indicates that the product could not be published (queue was full).
        return False


    def new_cart(self):
        """
        @brief Creates a new shopping cart and assigns it a unique ID.

        Returns:
            int: The unique ID of the newly created cart.
        """
        # Block Logic: Ensures exclusive access when generating a new cart ID and creating a new Cart object.
        with self.cart_id_generator_lock:
            # Functional Utility: Retrieves the next available cart ID.
            cart_id = self.cart_next_id

            # Functional Utility: Creates a new Cart object and associates it with the generated ID.
            self.carts[cart_id] = Cart()

            # Functional Utility: Increments the cart ID counter for the next new cart.
            self.cart_next_id += 1

            return cart_id

    def add_to_cart(self, cart_id, product):
        """
        @brief Adds a product from a producer's queue to a consumer's cart.

        Searches through all producer queues for the specified product. If found,
        it's removed from the producer's queue and added to the consumer's cart.

        Args:
            cart_id (int): The ID of the consumer's cart.
            product (any): The product to add to the cart.

        Returns:
            bool: True if the product was successfully added, False otherwise.
        """
        # Functional Utility: Stores the total number of registered producers.
        no_producers = 0

        # Block Logic: Ensures exclusive access when determining the number of producers.
        with self.producer_id_generator_lock:
            no_producers = self.producer_next_id

        # Block Logic: Iterates through each producer's queue to find the desired product.
        # Pre-condition: 'no_producers' is the total count of registered producers.
        # Invariant: Each producer's stock will be checked for the product.
        for producer_id in range(no_producers):
            producer_stock = self.producer_queues[producer_id]

            # Conditional Logic: Checks if the product is present in the current producer's stock.
            if product in producer_stock:
                # Block Logic: Ensures exclusive access to the producer's queue for removal.
                with self.producer_queue_lock:
                    # Functional Utility: Removes the product from the producer's queue.
                    producer_stock.remove(product)

                # Functional Utility: Adds the product to the consumer's cart, along with the producer ID.
                self.carts[cart_id].add_product(product, producer_id)

                # Functional Utility: Indicates successful addition to the cart.
                return True

        # Functional Utility: Indicates that the product was not found in any producer's queue.
        return False

    def remove_from_cart(self, cart_id, product):
        """
        @brief Removes a product from a consumer's cart and returns it to the originating producer's queue.

        Args:
            cart_id (int): The ID of the consumer's cart.
            product (any): The product to remove from the cart.
        """
        # Functional Utility: Removes the product from the specified cart and gets the original producer's ID.
        producer_id = self.carts[cart_id].remove_product(product)

        # Block Logic: Ensures exclusive access to the producer's queue when returning the product.
        with self.producer_queue_lock:
            # Functional Utility: Retrieves the product queue for the original producer.
            producer_queue = self.producer_queues[producer_id]

            # Conditional Logic: Checks if the producer's queue has space to re-add the product.
            if len(producer_queue) < self.queue_size_per_producer:
                # Functional Utility: Re-adds the product to the producer's queue.
                producer_queue.append(product)


    def place_order(self, cart_id):
        """
        @brief Finalizes a cart by retrieving all products within it.

        Args:
            cart_id (int): The ID of the cart to place the order for.

        Returns:
            list: A list of products that were in the ordered cart.
        """
        # Functional Utility: Retrieves the list of products from the specified cart.
        return self.carts[cart_id].get_products()

class Cart:
    """
    @brief Represents a shopping cart for a consumer.

    Stores products added to it, along with the ID of the producer who supplied each product.
    """

    def __init__(self):
        """
        @brief Initializes an empty shopping cart.
        """
        # Functional Utility: A list to store products in the cart, each as a dictionary
        # containing the product and its originating producer_id.
        self.products = []

    def add_product(self, product, producer_id):
        """
        @brief Adds a product to the cart, noting its originating producer.

        Args:
            product (any): The product to add.
            producer_id (int): The ID of the producer who supplied this product.
        """
        self.products.append({"product": product, "producer_id": producer_id})

    def remove_product(self, product):
        """
        @brief Removes a specific product from the cart.

        Searches for the product in the cart, removes the first occurrence found,
        and returns the ID of the producer who originally supplied it.

        Args:
            product (any): The product to remove.

        Returns:
            int or None: The producer ID if the product was found and removed, otherwise None.
        """
        # Block Logic: Iterates through the products in the cart to find the specified product.
        # Pre-condition: 'self.products' is a list of product dictionaries.
        # Invariant: The first matching product will be removed.
        for prod in self.products:

            # Conditional Logic: Checks if the current product in the cart matches the product to be removed.
            if prod["product"] == product:
                # Functional Utility: Stores the producer ID of the product being removed.
                producer_id = prod["producer_id"]

                # Functional Utility: Removes the product dictionary from the list.
                self.products.remove(prod)

                # Functional Utility: Returns the producer ID so the product can be returned to their stock.
                return producer_id

        # Functional Utility: Returns None if the product was not found in the cart.
        return None

    def get_products(self):
        """
        @brief Retrieves a list of all products (without producer information) currently in the cart.

        Returns:
            list: A list of the actual product objects in the cart.
        """
        product_list = []

        # Block Logic: Extracts only the product objects from the internal representation.
        # Pre-condition: 'self.products' contains dictionaries with a "product" key.
        # Invariant: 'product_list' will contain all product objects from the cart.
        for product_item in self.products:
            product_list.append(product_item["product"])

        return product_list


class Producer(Thread):
    """
    @brief Represents a producer thread that supplies products to the marketplace.

    Producers continuously produce and publish products to the Marketplace,
    waiting if their designated queue space in the Marketplace is full.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        @brief Initializes a Producer thread.

        Args:
            products (list): A list of product specifications (product_id, quantity, production_time).
            marketplace (Marketplace): The shared marketplace instance.
            republish_wait_time (float): Time in seconds to wait before retrying if marketplace is full.
            **kwargs: Arbitrary keyword arguments, expecting 'daemon' for thread status and 'name'.
        """
        # Functional Utility: Calls the parent Thread class constructor, setting daemon status.
        Thread.__init__(self, daemon=kwargs["daemon"])

        # Functional Utility: Stores the list of products this producer will generate.
        self.products = products
        # Functional Utility: Stores a reference to the shared Marketplace instance.
        self.marketplace = marketplace
        # Functional Utility: Stores the wait time if the marketplace queue is full.
        self.republish_wait_time = republish_wait_time
        # Functional Utility: Stores the name of the producer.
        self.name = kwargs["name"]

    def run(self):
        """
        @brief The main execution loop for the Producer thread.

        Registers with the marketplace, then continuously produces items and attempts
        to publish them, waiting if the marketplace queue is full.
        """
        # Functional Utility: Registers this producer with the marketplace and obtains a unique ID.
        producer_id = self.marketplace.register_producer()

        # Invariant: The producer continuously attempts to produce and publish products.
        while True:
            # Block Logic: Iterates through the list of products this producer is configured to make.
            # Pre-condition: 'self.products' contains tuples of product specifications.
            # Invariant: Each product type will be produced in its specified quantity.
            for product in self.products:
                # Functional Utility: Unpacks product details for the current item.
                product_id = product[0]
                product_quantity = product[1]
                product_production_time = product[2]

                # Functional Utility: Simulates the time taken to produce a batch of this product.
                sleep(product_production_time)

                # Block Logic: Attempts to publish the product 'product_quantity' times.
                # Pre-condition: 'product_quantity' is the number of units to publish.
                # Invariant: The loop continues until all units are published or the producer stops.
                for _ in range(product_quantity):
                    produced = False

                    # Block Logic: Continuously attempts to publish the single unit of product.
                    # Invariant: Retries publishing if the marketplace queue is full.
                    while True:
                        # Functional Utility: Attempts to publish one unit of the product to the marketplace.
                        produced = self.marketplace.publish(producer_id, product_id)

                        # Conditional Logic: If publishing failed (marketplace queue full), waits and retries.
                        if not produced:
                            sleep(self.republish_wait_time)
                        else:
                            break # Exit retry loop on successful publishing.
