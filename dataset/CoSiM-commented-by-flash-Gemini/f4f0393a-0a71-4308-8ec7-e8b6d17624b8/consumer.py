"""
This module implements a simplified producer-consumer marketplace simulation
using Python threads. It defines `Consumer`, `Marketplace`, and `Producer` classes
to demonstrate basic interactions such as adding/removing items from a cart,
publishing products, and placing orders.
"""

from threading import Thread
import time


class Consumer(Thread):
    """
    Manages the actions of a single consumer in the marketplace simulation.
    Consumers attempt to add and remove products from their carts and ultimately place orders.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        Initializes a Consumer thread.

        Args:
            carts (list): A list of shopping carts, each containing a sequence of operations
                          (add/remove product, quantity).
            marketplace (Marketplace): The shared marketplace instance where operations occur.
            retry_wait_time (float): The time in seconds to wait before retrying a failed operation.
            **kwargs: Additional keyword arguments to pass to the Thread constructor.
        """

        Thread.__init__(self, **kwargs)



        self.carts = carts  # Stores the list of carts for this consumer to process.
        self.marketplace = marketplace  # Reference to the shared marketplace.
        self.retry_wait_time = retry_wait_time  # Time to wait on failed operations.

        # Inline: Maps operation types to corresponding marketplace methods for dynamic dispatch.
        self.actions = {
            'add': self.marketplace.add_to_cart,
            'remove': self.marketplace.remove_from_cart
        }

    def run(self):
        """
        Executes the consumer's behavior: iterating through predefined carts,
        performing add/remove operations, and placing orders.
        """

        # Block Logic: Iterates through each cart assigned to this consumer.
        # Invariant: Each 'cart' in 'self.carts' represents a list of operations for a single shopping cart.
        for cart in self.carts:
            # Functional Utility: Creates a new unique cart in the marketplace for the consumer.
            cart_id = self.marketplace.new_cart()

            # Block Logic: Processes each operation (e.g., add/remove a product) within the current cart.
            # Invariant: 'operation' contains 'type', 'quantity', and 'product' keys.
            for operation in cart:
                iters = 0

                # Block Logic: Attempts to perform the operation for the specified quantity.
                # Invariant: 'iters' tracks the number of successful operations for the current item.
                while iters < operation['quantity']:
                    # Functional Utility: Dynamically calls the appropriate marketplace action.
                    ret = self.actions[operation['type']](
                        cart_id, operation['product'])

                    # Block Logic: Checks the result of the marketplace operation.
                    # Pre-condition: 'ret' is a boolean indicating success or failure.
                    # If the operation was successful or resulted in `None` (implying a specific success case),
                    # increment the iteration count. Otherwise, wait and retry.
                    if ret or ret is None:
                        iters += 1
                    else:
                        time.sleep(self.retry_wait_time)

            # Functional Utility: Places the order for the completed cart.
            self.marketplace.place_order(cart_id)


from threading import currentThread, Lock


class Marketplace:
    """
    Manages the central logic for product publishing, cart creation, and order processing.
    It acts as the shared resource between producers and consumers.
    """

    def __init__(self, queue_size_per_producer):
        """
        Initializes the Marketplace with core data structures and synchronization primitives.

        Args:
            queue_size_per_producer (int): The maximum number of products a single producer
                                           can have in the marketplace's available stock.
        """

        self.queue_size_per_producer = queue_size_per_producer  # Max products a producer can have.
        self.producer_queues = []  # Tracks current product count for each producer.
        self.all_products = []  # Global list of all available products.
        self.producted_by = dict()  # Maps a product to its producing producer_id.
        self.no_carts = 0  # Counter for generating unique cart IDs.
        self.carts = dict()  # Stores all active shopping carts, mapped by cart_id.

        self.producer_lock = Lock()  # Lock for synchronizing producer registration and queue updates.
        self.consumer_lock = Lock()  # Lock for synchronizing new cart creation.
        self.cart_lock = Lock()  # Lock for synchronizing cart modifications (add/remove items).

    def register_producer(self):
        """
        Registers a new producer with the marketplace, assigning it a unique ID
        and initializing its product queue counter.

        Returns:
            int: The unique ID assigned to the registered producer.
        """

        # Block Logic: Ensures exclusive access to producer-related data during registration.
        # Invariant: Only one thread can register a producer at a time.
        with self.producer_lock:
            # Functional Utility: Assigns a new producer ID based on the current number of producers.
            producer_id = len(self.producer_queues)
            self.producer_queues.append(0)

            return producer_id

    def publish(self, producer_id, product):
        """
        Attempts to publish a product from a specific producer to the marketplace.

        Args:
            producer_id (int): The ID of the producer publishing the product.
            product (str): The name of the product to publish.

        Returns:
            bool: True if the product was successfully published, False otherwise
                  (e.g., if the producer's queue is full).
        """

        # Block Logic: Checks if the producer has reached its maximum allowed products in the marketplace.
        # Pre-condition: 'producer_id' is a valid, registered producer ID.
        if self.producer_queues[producer_id] >= self.queue_size_per_producer:
            return False

        # Functional Utility: Increments the producer's product count and records the product's origin.
        self.producer_queues[producer_id] += 1
        self.producted_by[product] = producer_id

        # Functional Utility: Adds the product to the global list of available products.
        self.all_products.append(product)

        return True

    def new_cart(self):
        """
        Creates and registers a new, empty shopping cart in the marketplace.

        Returns:
            int: The unique ID of the newly created cart.
        """

        # Block Logic: Ensures exclusive access when assigning a new cart ID.
        # Invariant: 'self.no_carts' is incremented atomically.
        with self.consumer_lock:
            # Functional Utility: Assigns a unique cart ID.
            cart_id = self.no_carts
            self.no_carts += 1

        # Functional Utility: Initializes an empty list for the new cart.
        self.carts.setdefault(cart_id, [])

        return cart_id

    def add_to_cart(self, cart_id, product):
        """
        Attempts to add a product to a specified shopping cart.

        Args:
            cart_id (int): The ID of the cart to add the product to.
            product (str): The name of the product to add.

        Returns:
            bool: True if the product was successfully added, False if the product
                  is not available in the marketplace.
        """

        # Block Logic: Ensures atomicity of operations involving product availability and cart modification.
        # Invariant: Product removal from 'all_products' and cart addition are synchronized.
        with self.cart_lock:
            # Block Logic: Checks if the product is currently available in the marketplace.
            # Pre-condition: 'product' is a string identifying a product.
            if product not in self.all_products:
                return False

            # Functional Utility: Decrements the product count for the producer of the product.
            self.producer_queues[self.producted_by[product]] -= 1

            # Functional Utility: Removes the product from the global list of available products.
            self.all_products.remove(product)

        # Functional Utility: Adds the product to the consumer's specific cart.
        self.carts[cart_id].append(product)

        return True

    def remove_from_cart(self, cart_id, product):
        """
        Removes a product from a specified shopping cart and returns it to the marketplace.

        Args:
            cart_id (int): The ID of the cart to remove the product from.
            product (str): The name of the product to remove.
        """

        # Functional Utility: Removes the product from the consumer's cart.
        self.carts[cart_id].remove(product)

        # Functional Utility: Returns the product to the global list of available products.
        self.all_products.append(product)

        # Functional Utility: Increments the product count for the producer of the product.
        self.producer_queues[self.producted_by[product]] += 1

    def place_order(self, cart_id):
        """
        Places an order for a given cart, removing it from the marketplace's active carts
        and printing the purchased items.

        Args:
            cart_id (int): The ID of the cart to place the order for.

        Returns:
            list: The list of products that were in the ordered cart.
        """

        # Functional Utility: Atomically removes the cart from the active carts.
        # Invariant: 'products' contains the list of items from the now-removed cart.
        products = self.carts.pop(cart_id, None)

        # Block Logic: Prints each product that was bought.
        # Invariant: 'product' is a string representing an item.
        for product in products:
            print(f'{currentThread().getName()} bought {product}')

        return products


class Producer(Thread):
    """
    Manages the actions of a single producer in the marketplace simulation.
    Producers continuously publish products to the marketplace.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        Initializes a Producer thread.

        Args:
            products (list): A list of tuples, each defining a product, its quantity,
                             and the wait time between publishing instances.
            marketplace (Marketplace): The shared marketplace instance to publish to.
            republish_wait_time (float): The time in seconds to wait before retrying
                                         to publish a product if the marketplace queue is full.
            **kwargs: Additional keyword arguments to pass to the Thread constructor.
        """

        Thread.__init__(self, **kwargs)

        self.products = products  # List of products to be published by this producer.
        self.marketplace = marketplace  # Reference to the shared marketplace.
        self.republish_wait_time = republish_wait_time  # Time to wait on failed publish attempts.

        # Functional Utility: Registers the producer with the marketplace and obtains a unique ID.
        self.own_id = marketplace.register_producer()  # Unique ID for this producer.

    def run(self):
        """
        Executes the producer's continuous behavior: attempting to publish products
        to the marketplace and waiting if the queue is full.
        """
        # Block Logic: Main loop for continuous product publishing.
        # Invariant: The producer attempts to publish all its defined products repeatedly.
        while True:
            # Block Logic: Iterates through each type of product this producer is responsible for.
            # Invariant: Each tuple contains (product_name, quantity, wait_time).
            for (product, no_products, wait_time) in self.products:
                i = 0
                # Block Logic: Attempts to publish a specific product 'no_products' times.
                # Invariant: 'i' tracks the number of successfully published instances of the current product.
                while i < no_products:
                    # Functional Utility: Attempts to publish a single instance of the product.
                    if self.marketplace.publish(self.own_id, product):
                        time.sleep(wait_time)
                        i += 1
                    else:
                        # Block Logic: If publishing fails (marketplace queue is full), waits and retries.
                        # Pre-condition: Marketplace's queue for this producer is at its limit.
                        time.sleep(self.republish_wait_time)
