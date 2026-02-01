"""
This module simulates a multi-threaded marketplace system where producers publish products
and consumers purchase them. It incorporates thread-safe operations using locks
and includes logging for tracking marketplace activities.

Algorithm:
- Producer-Consumer Pattern: Producers add products to a marketplace's queues,
  and consumers retrieve them.
- Retry Mechanism: Both producers and consumers implement retry logic with a wait time
  if an operation (publishing a product or adding to cart) fails due to resource
  constraints (e.g., full queue, product unavailability).
- Thread-Safe Operations: The `Marketplace` class uses multiple `threading.Lock` objects
  to ensure atomic operations on shared data structures such as producer queues,
  product inventory, and shopping carts.
- Logging: All significant marketplace operations are logged to a file for auditing
  and debugging purposes.
"""

from threading import Thread
import time


class Consumer(Thread):
    """
    Represents a consumer in the marketplace simulation.

    A consumer interacts with the `Marketplace` to create a shopping cart,
    add and remove products, and finally place an order. It includes retry
    logic for adding products if they are not immediately available.

    Attributes:
        carts (list): A list of shopping cart definitions, where each definition
                      is a list of actions (add/remove product and quantity).
        marketplace (Marketplace): The shared marketplace instance.
        retry_wait_time (int): The duration (in seconds) to wait before retrying
                                a failed 'add to cart' operation.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        Initializes a new Consumer thread.

        Args:
            carts (list): A list of cart plans for this consumer.
            marketplace (Marketplace): The marketplace instance to interact with.
            retry_wait_time (int): Time to wait before retrying a cart operation.
            **kwargs: Keyword arguments passed to the Thread constructor, e.g., 'name'.
        """
        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time

    def run(self):
        """
        The main execution method for the Consumer thread.

        Invariant: Iterates through each predefined cart plan, performing add/remove
                   operations as specified. 'Add' operations include a retry mechanism.
                   Finally, an order is placed for each processed cart.
        """
        # Block Logic: Process each shopping cart defined for this consumer.
        for cart in self.carts:
            # Create a new cart in the marketplace and get its unique ID.
            cart_id = self.marketplace.new_cart()
            # Block Logic: Execute each action (add or remove) specified in the current cart plan.
            for action in cart:
                # Handle 'add' actions.
                if action["type"] == 'add':
                    qty = action["quantity"]
                    # Loop until the desired quantity of the product is added.
                    while qty != 0:
                        # Attempt to add the product to the cart.
                        flag = self.marketplace.add_to_cart(cart_id, action["product"])

                        # Block Logic: If adding fails, retry after a delay.
                        while not flag:
                            time.sleep(self.retry_wait_time)
                            flag = self.marketplace.add_to_cart(cart_id, action["product"])
                        qty -= 1  # Decrement remaining quantity after successful addition.
                # Handle 'remove' actions.
                elif action["type"] == 'remove':
                    qty = action["quantity"]
                    # Loop until the desired quantity of the product is removed.
                    while qty != 0:
                        # Remove the product from the cart.
                        self.marketplace.remove_from_cart(cart_id, action["product"])
                        qty -= 1  # Decrement remaining quantity after removal.

            # Place the order for the fully processed cart.
            self.marketplace.place_order(cart_id)

import time
from threading import Lock, currentThread
import logging
import logging.handlers as lh


class Marketplace:
    """
    Manages the central logic for product exchange between producers and consumers.

    This class handles product queues for producers, shopping carts for consumers,
    and maintains a global inventory of products. It ensures thread safety for
    all operations using various locks and provides detailed logging.

    Attributes:
        queue_size_per_producer (int): The maximum number of products a single producer can queue.
        producer_id (int): A counter for assigning unique producer IDs.
        producer_id_lock (threading.Lock): Protects `producer_id` and `producer_queue` during registration.
        producer_queue (list): A list of lists, where each inner list is a product queue for a producer.
        current_cart_id (int): A counter for assigning unique shopping cart IDs.
        consumer_cart (list): A list of lists, where each inner list represents a consumer's cart.
        consumer_cart_lock (threading.Lock): Protects `current_cart_id` and `consumer_cart` during cart creation.
        add_to_cart_lock (threading.Lock): Protects `consumer_cart` and `products` during 'add to cart' operations.
        remove_from_cart_lock (threading.Lock): Protects `consumer_cart` and `products` during 'remove from cart' operations.
        add_products_lock (threading.Lock): Protects `products` during product publishing.
        products (dict): A global inventory of products and their current quantities.
        print_lock (threading.Lock): Protects print statements to ensure clean console output.
    """

    def __init__(self, queue_size_per_producer):
        """
        Initializes the Marketplace with a specified queue size for producers.

        Sets up various locks and initializes logging to a rotating file.

        Args:
            queue_size_per_producer (int): The maximum number of items a producer can
                                           have in its queue at any given time.
        """
        self.queue_size_per_producer = queue_size_per_producer
        self.producer_id = 0
        self.producer_id_lock = Lock()
        self.producer_queue = []

        self.current_cart_id = 0
        self.consumer_cart = []
        self.consumer_cart_lock = Lock()

        self.add_to_cart_lock = Lock()
        self.remove_from_cart_lock = Lock()

        self.add_products_lock = Lock()
        self.products = {}  # Global inventory: {product_name: quantity}

        self.print_lock = Lock()

        # Set up rotating file logging for marketplace activities.
        log_formatter = logging.Formatter('%(asctime)s - %(message)s')
        log_formatter.converter = time.gmtime  # Use GMT time for logs.

        rf_handler = lh.RotatingFileHandler("marketplace.log", maxBytes=1000000, backupCount=20)
        rf_handler.setFormatter(log_formatter)

        logging.basicConfig(
            level=logging.INFO,
            handlers=[rf_handler]
        )

    def register_producer(self):
        """
        Registers a new producer with the marketplace, assigning a unique ID
        and initializing an empty product queue for it.

        Returns:
            int: The unique ID assigned to the newly registered producer.
        """
        logging.info('Register producer start (no args)')
        # Acquire lock to ensure atomic update of producer ID and queue list.
        self.producer_id_lock.acquire()
        id_return = self.producer_id
        self.producer_queue.insert(id_return, [])  # Insert an empty queue for the new producer.
        self.producer_id += 1
        self.producer_id_lock.release()
        logging.info('Register producer end: {}'.format(id_return))

        return id_return

    def publish(self, producer_id, product):
        """
        Publishes a product to the specified producer's queue and updates global inventory.

        Pre-condition: The producer's queue must not exceed its `queue_size_per_producer` capacity.
        Invariant: If successful, the product is added to the producer's queue and the
                   global product count is incremented.

        Args:
            producer_id (int): The ID of the producer.
            product (tuple): The product to publish (e.g., ("product_name", quantity, sleep_time)).

        Returns:
            bool: True if the product was successfully published, False otherwise (if queue is full).
        """
        logging.info('Publish start: producer_id: {}, product: {}'.format(producer_id, product))
        # Check if the producer's queue is full.
        if len(self.producer_queue[producer_id]) >= self.queue_size_per_producer:
            logging.info('Publish end: False')
            return False

        # Acquire lock to ensure atomic update of the global products dictionary.
        self.add_products_lock.acquire()
        # Update global inventory count for the product.
        if product[0] in self.products:
            size = self.products[product[0]]
            self.products[product[0]] = size + 1
        else:
            self.products[product[0]] = 1

        self.add_products_lock.release()

        # Add the product to the specific producer's queue.
        self.producer_queue[producer_id].append(product)
        logging.info('Publish end: True')
        return True

    def new_cart(self):
        """
        Creates a new empty shopping cart and returns its unique ID.

        Returns:
            int: The unique ID of the newly created cart.
        """
        logging.info('New_cart start (no args)')
        # Acquire lock to ensure atomic cart ID assignment and cart list modification.
        self.consumer_cart_lock.acquire()
        self.consumer_cart.append([])  # Add an empty list for the new cart.
        id_return = self.current_cart_id
        self.current_cart_id += 1
        self.consumer_cart_lock.release()
        logging.info('New_cart end: {}'.format(id_return))

        return id_return

    def add_to_cart(self, cart_id, product):
        """
        Adds a product to a specified shopping cart and decrements its count in the global inventory.

        Pre-condition: The product must be available in the global inventory (`self.products`).
        Invariant: If successful, the product is added to the cart, and its quantity in `self.products`
                   is decremented.

        Args:
            cart_id (int): The ID of the cart to add the product to.
            product (str): The name of the product to add.

        Returns:
            bool: True if the product was added, False if not available.
        """
        logging.info('Add_to_cart start: cart_id: {}, product: {}'.format(cart_id, product))
        # Acquire lock to ensure atomic updates to carts and global products.
        self.add_to_cart_lock.acquire()
        # Check if the product is in stock.
        if product in self.products and self.products[product] > 0:
            self.consumer_cart[cart_id].append(product)
            size = self.products[product]
            self.products[product] = size - 1
            self.add_to_cart_lock.release()
            logging.info('Add_to_cart end: True')
            return True

        self.add_to_cart_lock.release()
        logging.info('Add_to_cart end: False')
        return False

    def remove_from_cart(self, cart_id, product):
        """
        Removes a product from a specified shopping cart and increments its count
        in the global inventory.

        Invariant: If the product is found in the cart, it is removed, and its quantity
                   in `self.products` is incremented.

        Args:
            cart_id (int): The ID of the cart to remove the product from.
            product (str): The name of the product to remove.
        """
        logging.info('Remove_from_cart start: cart_id: {}, product: {}'.format(cart_id, product))

        # Check if the product is actually in the cart.
        if product in self.consumer_cart[cart_id]:
            self.consumer_cart[cart_id].remove(product)
        else:
            # If product not in cart, nothing to remove.
            return

        # Acquire lock to ensure atomic updates to global products.
        self.remove_from_cart_lock.acquire()
        size = self.products[product]
        self.products[product] = size + 1
        self.remove_from_cart_lock.release()
        logging.info('Remove_from_cart end (no return)')

    def place_order(self, cart_id):
        """
        Places an order for the specified cart, printing the items bought by the consumer.

        Args:
            cart_id (int): The ID of the cart for which to place the order.

        Returns:
            list: The list of products that were in the placed order.
        """
        logging.info('Place_order start: cart_id: {}'.format(cart_id))
        # Use a lock to ensure print statements from different threads don't interleave.
        for item in self.consumer_cart[cart_id]:
            self.print_lock.acquire()
            print("{} bought {}".format(currentThread().name, item))
            self.print_lock.release()

        logging.info('Place_order end: {}'.format(self.consumer_cart[cart_id]))
        return self.consumer_cart[cart_id]


from threading import Thread
import time


class Producer(Thread):
    """
    Represents a producer in the marketplace simulation.

    A producer continuously attempts to publish products to the `Marketplace`.
    It includes retry logic if publishing fails (e.g., if the marketplace queue is full)
    and pauses for a specified duration between publications.

    Attributes:
        products (list): A list of product definitions. Each definition is a tuple:
                         (product_name, quantity_to_produce, sleep_time_after_publish).
        marketplace (Marketplace): The shared marketplace instance.
        republish_wait_time (int): The duration (in seconds) to wait before retrying
                                   a failed publish operation.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        Initializes a new Producer thread.

        Args:
            products (list): A list of products with their quantities and sleep times.
            marketplace (Marketplace): The marketplace instance to publish to.
            republish_wait_time (int): Time to wait before retrying a publish operation.
            **kwargs: Keyword arguments passed to the Thread constructor, e.g., 'name'.
        """
        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time

    def run(self):
        """
        The main execution method for the Producer thread.

        Invariant: Continuously attempts to publish its predefined products to the
                   marketplace, respecting specified quantities and sleep times.
                   If publishing fails, it retries after a `republish_wait_time` delay.
        """
        # Register with the marketplace to obtain a unique producer ID.
        producer_id = self.marketplace.register_producer()

        # Infinite loop for continuous production.
        while True:
            # Block Logic: Iterate through each product type that this producer supplies.
            for product_info in self.products:
                # Unpack product information: product_name, initial_quantity, sleep_time.
                product_name = product_info[0]
                qty = product_info[1]
                sleep_time = product_info[2]

                # Loop to publish the specified quantity of the current product.
                while qty != 0:
                    # Attempt to publish the product.
                    flag = self.marketplace.publish(producer_id, product_info)

                    # Block Logic: If publishing is successful, pause for the specified sleep time.
                    if flag:
                        time.sleep(sleep_time)
                    # Block Logic: If publishing fails, retry after a delay.
                    else:
                        # Loop to retry publishing until successful.
                        while not flag:
                            time.sleep(self.republish_wait_time)
                            flag = self.marketplace.publish(producer_id, product_info)

                    qty -= 1  # Decrement the quantity after a successful publish (or retry success).
