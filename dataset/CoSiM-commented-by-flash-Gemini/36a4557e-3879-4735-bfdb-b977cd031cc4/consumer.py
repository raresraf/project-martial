"""
@file consumer.py
@brief Implements a simulated producer-consumer system with a shared marketplace.

This module defines the `Consumer`, `Marketplace`, and `Producer` classes
to simulate a basic e-commerce or resource management system. Producers
publish products to a marketplace, and Consumers add/remove items from
their carts and place orders, all while handling concurrency and resource
availability.
"""

from threading import Thread, Lock
import time

class Consumer(Thread):
    """
    @class Consumer
    @brief Represents a consumer in the marketplace system.

    A `Consumer` thread interacts with the `Marketplace` to add and remove
    products from its shopping cart and ultimately place an order. It handles
    retrying operations if products are not immediately available.

    @attribute marketplace (Marketplace): A reference to the shared marketplace instance.
    @attribute carts (list): A list of cart configurations, specifying products and quantities.
    @attribute consumer_id (int): Unique identifier for this consumer's cart.
    @attribute retry_wait_time (float): Time to wait before retrying an add-to-cart operation.
    @attribute name (str): The name of the consumer thread.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        @brief Initializes a new Consumer thread.

        @param carts (list): A list of dictionaries, each describing products and quantities for a cart.
        @param marketplace (Marketplace): The shared marketplace instance.
        @param retry_wait_time (float): The time in seconds to wait before retrying an add-to-cart operation.
        @param kwargs: Arbitrary keyword arguments passed to the Thread constructor (e.g., 'name').
        """
        Thread.__init__(self, **kwargs)
        self.marketplace = marketplace
        self.carts = carts

        self.consumer_id = marketplace.new_cart()
        self.retry_wait_time = retry_wait_time
        self.name = kwargs['name']

    def run(self):
        """
        @brief The main execution loop for the Consumer thread.

        This method simulates the consumer's shopping activity. It iterates
        through predefined carts, performing add and remove operations for
        specified products. It handles retries for adding items to the cart
        if the product is not immediately available in the marketplace.
        Finally, it places an order and prints the bought items.
        """
        # Block Logic: Iterates through each predefined cart for this consumer.
        # Invariant: Each 'cart' represents a set of operations for a shopping scenario.
        for cart_idx in range(len(self.carts)):
            # Block Logic: Iterates through each operation (add/remove) within the current cart.
            # Invariant: Each 'operation' specifies a product, type of action, and quantity.
            for operation_idx in range(len(self.carts[cart_idx])):
                op_type = self.carts[cart_idx][operation_idx]['type']
                product = self.carts[cart_idx][operation_idx]['product']
                quantity = self.carts[cart_idx][operation_idx]['quantity']

                # Block Logic: Handles 'add' operations to the cart.
                # Precondition: The operation type is "add".
                if op_type == "add":
                    # Block Logic: Attempts to add the specified quantity of the product.
                    # Invariant: Continues until the desired quantity has been added.
                    while quantity > 0:
                        # Block Logic: Continuously retries adding to cart until successful.
                        while True:
                            # Functional Utility: Attempts to add the product to the consumer's cart.
                            verdict = self.marketplace.add_to_cart(self.consumer_id, product)

                            # Block Logic: If adding to cart is successful, break from retry loop.
                            if verdict:
                                break
                            # Functional Utility: Waits for a short period before retrying the add-to-cart operation.
                            time.sleep(self.retry_wait_time)
                        quantity -= 1 # Inline: Decrements the remaining quantity to be added.
                else:  # Block Logic: Handles 'remove' operations from the cart.
                    # Block Logic: Attempts to remove the specified quantity of the product.
                    # Invariant: Continues until the desired quantity has been removed.
                    while quantity > 0:
                        # Functional Utility: Removes the product from the consumer's cart.
                        self.marketplace.remove_from_cart(self.consumer_id, product)
                        quantity -= 1 # Inline: Decrements the remaining quantity to be removed.

        # Block Logic: Places the final order with all items currently in the cart.
        products = self.marketplace.place_order(self.consumer_id)
        # Block Logic: Prints a confirmation message for each item bought by the consumer.
        for item in products:
            print(self.name + " bought " + str(item))


class Marketplace:
    """
    @class Marketplace
    @brief Manages products and shopping carts in a simulated e-commerce system.

    The `Marketplace` acts as a central hub where producers publish products
    and consumers manage their shopping carts. It handles concurrency using locks
    to ensure thread-safe operations on products and carts.

    @attribute queue_size_per_producer (int): Maximum number of items a producer can have in the marketplace queue.
    @attribute carts (list): A list of lists, where each inner list represents a consumer's cart.
    @attribute products (list): A list of (product, producer_id) tuples representing available products.
    @attribute no_of_products (list): A list where each index corresponds to a producer_id, storing the count of products currently published by that producer.
    @attribute producer_id (int): Counter for assigning unique producer IDs.
    @attribute consumer_id (int): Counter for assigning unique consumer (cart) IDs.
    @attribute producer_lock (Lock): Lock to protect `producer_id` during registration.
    @attribute consumer_lock (Lock): Lock to protect `consumer_id` and `carts` during new cart creation.
    @attribute buffer_lock (Lock): Lock to protect `products` and `no_of_products` during publish/add/remove operations.
    """

    def __init__(self, queue_size_per_producer):
        """
        @brief Initializes the Marketplace.

        Sets up the product queues, cart storage, and all necessary locks
        for thread-safe operation.

        @param queue_size_per_producer (int): The maximum number of products each producer can have in the marketplace.
        """
        self.queue_size_per_producer = queue_size_per_producer
        self.carts = []
        self.products = []
        self.no_of_products = []
        self.producer_id = -1
        self.consumer_id = -1
        self.producer_lock = Lock()
        self.consumer_lock = Lock()
        self.buffer_lock = Lock()

    def register_producer(self):
        """
        @brief Registers a new producer with the marketplace.

        Assigns a unique ID to the new producer and initializes its product
        count in the marketplace. This operation is thread-safe.

        @return (int): The unique ID assigned to the new producer.
        """
        # Block Logic: Acquires a lock to ensure exclusive access to `producer_id`.
        self.producer_lock.acquire()

        self.producer_id += 1 # Inline: Increments the producer ID counter.
        ids = self.producer_id # Inline: Stores the newly assigned ID.
        self.no_of_products.append(0) # Inline: Initializes the product count for the new producer.

        self.producer_lock.release() # Inline: Releases the producer lock.

        return ids

    def publish(self, producer_id, product):
        """
        @brief Publishes a product to the marketplace.

        A producer calls this method to add a product to the available inventory.
        It checks against the `queue_size_per_producer` limit and ensures
        thread-safe updates to the product list.

        @param producer_id (int): The ID of the producer publishing the product.
        @param product (any): The product being published.
        @return (bool): True if the product was successfully published, False otherwise (e.g., queue full).
        """
        verdict = True

        # Block Logic: Acquires a lock to ensure exclusive access to the shared product buffer.
        self.buffer_lock.acquire()

        # Block Logic: Checks if the producer has reached its product queue limit.
        if self.no_of_products[producer_id] >= self.queue_size_per_producer:
            verdict = False # Inline: Cannot publish, queue is full.
        else:
            # Inline: Adds the product and its producer ID to the shared product list.
            self.products.append((product, producer_id))
            self.no_of_products[producer_id] += 1 # Inline: Increments the count of products by this producer.

        self.buffer_lock.release() # Inline: Releases the buffer lock.

        return verdict

    def new_cart(self):
        """
        @brief Creates a new shopping cart for a consumer.

        Assigns a unique cart ID to a new consumer and initializes an empty
        cart for them. This operation is thread-safe.

        @return (int): The unique ID assigned to the new cart.
        """
        # Block Logic: Acquires a lock to ensure exclusive access to `consumer_id` and `carts`.
        self.consumer_lock.acquire()

        self.consumer_id += 1 # Inline: Increments the consumer ID counter.
        ids = self.consumer_id # Inline: Stores the newly assigned ID.
        self.carts.append([]) # Inline: Creates an empty cart for the new consumer.

        self.consumer_lock.release() # Inline: Releases the consumer lock.

        return ids

    def add_to_cart(self, cart_id, product):
        """
        @brief Adds a product to a consumer's shopping cart.

        Searches for the product in the marketplace. If found, it moves the
        product from the marketplace inventory to the consumer's cart. This
        operation is thread-safe.

        @param cart_id (int): The ID of the consumer's cart.
        @param product (any): The product to be added.
        @return (bool): True if the product was successfully added, False if not found.
        """
        verdict = False

        # Block Logic: Acquires a lock to ensure exclusive access to the shared product buffer.
        self.buffer_lock.acquire()

        # Block Logic: Iterates through available products to find the one to add to the cart.
        for prod, producer_id in self.products:
            # Precondition: Checks if the current product matches the requested product.
            if prod == product:
                self.no_of_products[producer_id] -= 1 # Inline: Decrements the producer's product count in the marketplace.
                self.carts[cart_id].append((product, producer_id)) # Inline: Adds the product to the consumer's cart.
                self.products.remove((product, producer_id)) # Inline: Removes the product from the marketplace.
                verdict = True # Inline: Marks the operation as successful.
                break # Inline: Exits the loop once the product is found and moved.

        self.buffer_lock.release() # Inline: Releases the buffer lock.

        return verdict

    def remove_from_cart(self, cart_id, product):
        """
        @brief Removes a product from a consumer's shopping cart.

        Searches for the product in the consumer's cart. If found, it moves the
        product back to the marketplace inventory. This operation is thread-safe.

        @param cart_id (int): The ID of the consumer's cart.
        @param product (any): The product to be removed.
        """
        # Block Logic: Iterates through items in the consumer's cart.
        for prod, producer_id in self.carts[cart_id]:
            # Precondition: Checks if the current product in the cart matches the product to be removed.
            if prod == product:
                self.carts[cart_id].remove((product, producer_id)) # Inline: Removes the product from the cart.

                # Block Logic: Acquires a lock to ensure exclusive access to the shared product buffer.
                self.buffer_lock.acquire()

                self.products.append((product, producer_id)) # Inline: Adds the product back to the marketplace.
                self.no_of_products[producer_id] += 1 # Inline: Increments the producer's product count in the marketplace.

                self.buffer_lock.release() # Inline: Releases the buffer lock.

                break # Inline: Exits the loop once the product is found and moved.

    def place_order(self, cart_id):
        """
        @brief Finalizes a consumer's order.

        Collects all products currently in the specified cart and returns them
        as a list, effectively completing the purchase for that cart.

        @param cart_id (int): The ID of the consumer's cart.
        @return (list): A list of products that were in the cart at the time of placing the order.
        """
        products = []

        # Block Logic: Iterates through the products in the specified cart.
        for prod, _ in self.carts[cart_id]:
            products.append(prod) # Inline: Adds each product to the list of ordered products.

        return products


class Producer(Thread):
    """
    @class Producer
    @brief Represents a producer in the marketplace system.

    A `Producer` thread continuously publishes products to the `Marketplace`.
    It attempts to republish products if the marketplace queue is full,
    waiting for a specified time before retrying.

    @attribute marketplace (Marketplace): A reference to the shared marketplace instance.
    @attribute products (list): A list of products (product_name, quantity, wait_time) to be produced.
    @attribute producer_id (int): Unique identifier for this producer.
    @attribute republish_wait_time (float): Time to wait before retrying a publish operation.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        @brief Initializes a new Producer thread.

        @param products (list): A list of tuples, each containing (product_name, quantity_to_produce, time_to_wait_after_publish).
        @param marketplace (Marketplace): The shared marketplace instance.
        @param republish_wait_time (float): The time in seconds to wait before retrying a publish operation.
        @param kwargs: Arbitrary keyword arguments passed to the Thread constructor (e.g., 'name').
        """
        Thread.__init__(self, **kwargs)
        self.marketplace = marketplace
        self.products = products
        self.producer_id = marketplace.register_producer()
        self.republish_wait_time = republish_wait_time

    def run(self):
        """
        @brief The main execution loop for the Producer thread.

        This method continuously publishes products to the marketplace. It
        iterates through its list of products, attempting to publish each
        one. If the marketplace queue is full, it waits and retries the
        publishing operation.
        """
        # Block Logic: Main loop for continuous product publishing.
        # Invariant: The producer continuously attempts to publish its products.
        while True:
            # Block Logic: Iterates through each product in the producer's list.
            # Each product is a tuple: (product_name, initial_quantity, wait_time_after_publish).
            for product_details in self.products:
                product_name = product_details[0]
                initial_quantity = product_details[1]
                wait_time_after_publish = product_details[2]

                quantity_to_publish = initial_quantity
                # Block Logic: Attempts to publish the specified quantity of the current product.
                # Invariant: Continues until the desired quantity of the product has been published.
                while quantity_to_publish > 0:
                    # Block Logic: Continuously retries publishing the product until successful.
                    while True:
                        # Functional Utility: Attempts to publish the product to the marketplace.
                        verdict = self.marketplace.publish(self.producer_id, product_name)
                        # Block Logic: If publishing is successful, wait and break from retry loop.
                        if verdict:
                            # Functional Utility: Simulates work or time taken to produce the next item.
                            time.sleep(wait_time_after_publish)
                            break
                        # Functional Utility: Waits for a short period before retrying the publish operation.
                        time.sleep(self.republish_wait_time)

                    quantity_to_publish -= 1 # Inline: Decrements the remaining quantity to be published.
"""
