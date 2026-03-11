"""
@file consumer.py
@brief Implements a multi-threaded producer-consumer marketplace simulation.

This module defines three key components:
- `Consumer`: Represents entities that purchase products from the marketplace.
- `Marketplace`: Acts as a central exchange where producers publish products and consumers create carts to add/remove items.
- `Producer`: Represents entities that generate and publish products to the marketplace.

The simulation models concurrent operations using Python's `threading` module,
managing shared resources like product inventory and shopping carts with locks
to ensure thread safety.
"""
from threading import Lock, currentThread, Thread
import time


class Consumer(Thread):
    """
    @brief Simulates a consumer entity that interacts with the marketplace
           to add and remove products from its shopping cart.

    Consumers operate as separate threads, processing a list of predefined
    cart operations (add/remove products). They handle retries for failed
    operations and ultimately place orders for their accumulated products.
    """
    

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        @brief Initializes a Consumer thread.

        @param carts: A list of cart operations (dictionaries) to be performed by this consumer.
                      Each operation specifies quantity, type (add/remove), and product.
        @param marketplace: The Marketplace instance to interact with.
        @param retry_wait_time: Time in seconds to wait before retrying a failed cart operation.
        @param kwargs: Additional keyword arguments passed to the Thread constructor.
        """
        
        Thread.__init__(self, **kwargs)

        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time

    def run(self):
        """
        @brief The main execution method for the Consumer thread.

        Iterates through the assigned carts, creates a new cart in the marketplace,
        and then processes each operation (add/remove product) for that cart.
        Includes a retry mechanism for failed operations. Once all operations
        for a cart are attempted, the order is placed.
        """
        for cart in self.carts:
            cart_id = self.marketplace.new_cart()

            # Block Logic: Iterate through each operation within the current cart.
            # Pre-condition: 'cart' contains a list of operations to be executed.
            for operation in cart:
                no_ops = 0

                # Extract operation details for clarity.
                qty = operation["quantity"]
                op_type = operation["type"]
                prod = operation["product"]

                # Block Logic: Attempt to execute the current operation 'qty' times.
                # Invariant: 'no_ops' tracks the number of successfully executed operations.
                while no_ops < qty:
                    result = self.execute_operation(cart_id, op_type, prod)

                    # Conditional Logic: Check if the operation was successful.
                    # If successful, increment the count; otherwise, wait and retry.
                    if result is None or result: # 'None' could imply a successful removal that returns nothing
                        no_ops += 1
                    else:
                        # Functional Utility: Introduce a delay before retrying a failed operation
                        # to prevent busy-waiting and reduce contention.
                        time.sleep(self.retry_wait_time)

            # Block Logic: After all operations for a cart are attempted, place the final order.
            # This concludes the processing for the current shopping cart.
            self.marketplace.place_order(cart_id)

    def execute_operation(self, cart_id, operation_type, product) -> bool:
        """
        @brief Executes a single cart operation (add or remove) by delegating to the marketplace.

        @param cart_id: The ID of the consumer's cart.
        @param operation_type: A string indicating the type of operation ("add" or "remove").
        @param product: The product to which the operation applies.
        @return: True if the operation was successful, False otherwise.
        """
        
        if operation_type == "add":
            return self.marketplace.add_to_cart(cart_id, product)

        if operation_type == "remove":
            return self.marketplace.remove_from_cart(cart_id, product)

        return False


class Marketplace:
    """
    @brief Manages products, producers, and consumer carts in a thread-safe manner.

    The Marketplace acts as a central hub for the simulation. It provides functionalities
    for producers to publish products, consumers to create carts, add/remove products,
    and place orders. It uses threading locks to ensure atomicity and consistency
    of shared data structures during concurrent access.
    """
    
    def __init__(self, queue_size_per_producer):
        """
        @brief Initializes the Marketplace with specified queue size and sets up internal data structures.

        @param queue_size_per_producer: The maximum number of products a single producer can have
                                        in the marketplace's available products list at any given time.
        """

        self.queue_size_per_producer = queue_size_per_producer
        self.products_mapping = {}  # Maps product to its producer_id
        self.producers_queues = []  # Tracks current queue size for each producer
        self.consumers_carts = {}  # Stores products in each consumer's cart

        self.available_products = []  # List of products currently available in the marketplace

        self.no_carts = 0 # Counter for unique cart IDs

        # Locks for ensuring thread safety during concurrent operations
        self.consumer_cart_creation_lock = Lock()
        self.cart_operation_lock = Lock()

    def register_producer(self):
        """
        @brief Registers a new producer with the marketplace, assigning it a unique ID.

        Each producer is allocated an entry to track its current stock in the marketplace.
        @return: The unique integer ID assigned to the new producer.
        """
        
        new_producer_id = len(self.producers_queues)

        self.producers_queues.append(0)

        return new_producer_id

    def publish(self, producer_id, product):
        """
        @brief Publishes a product to the marketplace, making it available for consumers.

        The product is added only if the producer's queue for available products
        does not exceed the `queue_size_per_producer` limit.
        @param producer_id: The ID of the producer publishing the product.
        @param product: The product to be published.
        @return: True if the product was successfully published, False otherwise (e.g., queue full).
        """
        
        if self.producers_queues[producer_id] >= self.queue_size_per_producer:
            return False

        self.producers_queues[producer_id] += 1
        self.available_products.append(product)

        self.products_mapping[product] = producer_id

        return True

    def new_cart(self):
        """
        @brief Creates a new, empty shopping cart for a consumer and returns its unique ID.

        This operation is protected by a lock to ensure thread-safe cart ID generation
        and initialization of the cart.
        @return: The unique integer ID of the newly created cart.
        """
        
        with self.consumer_cart_creation_lock:
            self.no_carts += 1

            self.consumers_carts[self.no_carts] = []

            return self.no_carts

    def add_to_cart(self, cart_id, product):
        """
        @brief Adds a specified product to a consumer's cart.

        This operation is thread-safe using a lock. It checks product availability,
        updates producer queues, and transfers the product from available items to the cart.
        @param cart_id: The ID of the consumer's cart.
        @param product: The product to add to the cart.
        @return: True if the product was successfully added, False if the product is not available.
        """
        
        with self.cart_operation_lock:
            if product not in self.available_products:
                return False

            producer_id = self.products_mapping[product]
            self.producers_queues[producer_id] -= 1

            self.available_products.remove(product)

            self.consumers_carts[cart_id].append(product)

            return True

    def remove_from_cart(self, cart_id, product):
        """
        @brief Removes a specified product from a consumer's cart and returns it to the marketplace.

        This operation is thread-safe using a lock. It removes the product from the cart,
        makes it available again in the marketplace, and updates the producer's queue.
        @param cart_id: The ID of the consumer's cart.
        @param product: The product to remove from the cart.
        """
        with self.cart_operation_lock:
            self.consumers_carts[cart_id].remove(product)
            self.available_products.append(product)

            producer_id = self.products_mapping[product]
            self.producers_queues[producer_id] += 1

    def place_order(self, cart_id):
        """
        @brief Finalizes the order for a given cart, removing it from the active carts.

        The products in the cart are considered "bought" and a message is printed
        to indicate the purchase.
        @param cart_id: The ID of the cart to place the order for.
        @return: A list of products that were in the placed order.
        """
        
        products = self.consumers_carts.pop(cart_id, None)

        for product in products:
            print(currentThread().getName() + " bought " + str(product))

        return products


class Producer(Thread):
    """
    @brief Simulates a producer entity that continuously generates and publishes products
           to the marketplace.

    Producers operate as separate threads, each responsible for a specific set of products.
    They attempt to publish products to the marketplace, respecting marketplace limits,
    and include a mechanism for retrying publication if the marketplace is full.
    """
    

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        @brief Initializes a Producer thread.

        @param products: A list of products this producer will publish. Each item is a tuple
                         (product_name, number_of_products_to_publish, publish_wait_time_after_success).
        @param marketplace: The Marketplace instance to interact with.
        @param republish_wait_time: Time in seconds to wait before retrying to publish a product
                                    if the marketplace's queue is full.
        @param kwargs: Additional keyword arguments passed to the Thread constructor.
        """
        
        Thread.__init__(self, **kwargs)

        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time

        # Register the producer with the marketplace to get a unique ID
        self.producer_id = self.marketplace.register_producer()

    def run(self):
        """
        @brief The main execution method for the Producer thread.

        Continuously attempts to publish its predefined list of products to the marketplace.
        It iterates through each product specification and tries to publish the required
        number of units, handling delays as specified.
        """
        
        # Block Logic: Main production loop, ensuring continuous publishing of products.
        while True:
            # Block Logic: Iterate through each type of product this producer is responsible for.
            for (product, no_products, publish_wait_time) in self.products:
                no_prod = 0

                # Block Logic: Publish the specified quantity of the current product.
                # Invariant: 'no_prod' tracks the number of successfully published products of this type.
                while no_prod < no_products:
                    result = self.publish_product(product, publish_wait_time)

                    # Conditional Logic: If publishing was successful, increment counter.
                    if result:
                        no_prod += 1

    def publish_product(self, product, publish_wait_time) -> bool:
        """
        @brief Attempts to publish a single unit of a product to the marketplace.

        If successful, it waits for a specified `publish_wait_time`. If unsuccessful
        (e.g., marketplace queue is full), it waits for `republish_wait_time` before
        the next retry.
        @param product: The product to publish.
        @param publish_wait_time: The time to wait (in seconds) after a successful publication.
        @return: True if the product was successfully published, False otherwise.
        """
        
        result = self.marketplace.publish(self.producer_id, product)

        # Conditional Logic: If publishing was successful, wait before publishing the next product.
        if result:
            time.sleep(publish_wait_time)
            return True

        # Functional Utility: If publishing failed (e.g., marketplace full), wait
        # for a defined retry period to avoid busy-waiting.
        time.sleep(self.republish_wait_time)
        return False