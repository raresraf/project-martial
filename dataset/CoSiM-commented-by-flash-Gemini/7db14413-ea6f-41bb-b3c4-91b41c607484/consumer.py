


from time import sleep
from threading import Thread


class Consumer(Thread):
    """
    A Consumer represents a buyer in the marketplace. It operates as a separate thread,
    managing multiple shopping carts, adding and removing products, and ultimately
    placing orders. It handles retries for 'add' operations if products are
    temporarily unavailable.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        Initializes a Consumer thread.

        Args:
            carts (List[List[Dict]]): A list of shopping cart definitions. Each cart
                                      is a list of dictionaries, where each dictionary
                                      represents an operation ('type', 'product', 'quantity').
            marketplace (Marketplace): The shared marketplace instance to interact with.
            retry_wait_time (float): The time in seconds to wait before retrying
                                     an 'add' operation if a product is unavailable.
            **kwargs: Arbitrary keyword arguments, including 'name' for the thread.
        """
        Thread.__init__(self, name=kwargs['name'])
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time

    def run(self):
        """
        The main execution method for the Consumer thread.
        It iterates through each defined cart, performs the specified operations
        (adding/removing products with retries), and then places the order.
        """
        # Block Logic: Iterates through each pre-defined list of operations,
        # each representing a shopping cart's actions.
        for cart_operations in self.carts:
            # Pre-condition: Consumer is ready to start a new cart.
            # Post-condition: A new cart_id is obtained from the marketplace.
            cart_id = self.marketplace.new_cart()
            # Block Logic: Processes all operations within the current cart.
            for operation in cart_operations:
                # Conditional Logic: Handles 'add' operations.
                if operation['type'] == 'add':
                    # Block Logic: Attempts to add the specified quantity of a product.
                    # It retries if the product is not immediately available.
                    for _ in range(operation['quantity']):
                        while not self.marketplace.add_to_cart(
                                cart_id, operation['product']):
                            sleep(self.retry_wait_time) # Wait before retrying.

                # Conditional Logic: Handles 'remove' operations.
                if operation['type'] == 'remove':
                    # Block Logic: Removes the specified quantity of a product from the cart.
                    for _ in range(operation['quantity']):
                        self.marketplace.remove_from_cart(
                            cart_id, operation['product'])

            # Block Logic: Places the final order for the current cart.
            # Pre-condition: All add/remove operations for the cart have been attempted.
            # Post-condition: `products` contains the list of products successfully ordered.
            products = self.marketplace.place_order(cart_id)
            # Block Logic: Prints the purchased products, protected by a printing lock
            # to prevent interleaved output from multiple consumers.
            for prod in products:
                with self.marketplace.printing_lock:
                    print(f'{self.name} bought {prod}')


from uuid import uuid4
import unittest


import logging
from logging.handlers import RotatingFileHandler

from threading import Lock
import time

from .product import Product


def logger_set_up():
    """
    Configures the logging system for the marketplace.
    It sets up a rotating file handler to log messages to 'marketplace.log',
    with a maximum file size and backup count. The log level is set to DEBUG,
    and a specific format is applied for log entries, including timestamp,
    level, module, function, line number, and message.
    Timestamps are converted to GM time.
    """
    logging.basicConfig(
        handlers=[RotatingFileHandler(
            'marketplace.log', maxBytes=10000, backupCount=10)],
        level=logging.DEBUG,
        format="[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s",
        datefmt='%Y-%m-%dT%H:%M:%S')

    logging.Formatter.converter = time.gmtime


class Marketplace:
    """
    The central marketplace where producers publish products and consumers create carts,
    add/remove products, and place orders. It manages product inventory across producers
    and handles cart operations, ensuring thread safety with various locks.
    It also integrates logging for operational insights.
    """

    def __init__(self, queue_size_per_producer: int):
        """
        Initializes the Marketplace.

        Args:
            queue_size_per_producer (int): The maximum number of items a single producer
                                           can have in the marketplace at any given time.
        """
        logger_set_up() # Configure logging when marketplace is initialized.
        self.queue_size_per_producer = queue_size_per_producer

        # Dictionary to store products published by each producer, keyed by producer ID.
        self.producers_queues: dict[str, list[Product]] = {}
        # Dictionary to store the count of available products, keyed by Product object.
        self.available_products: dict[Product, int] = {}
        # Dictionary to store active shopping carts, keyed by cart ID.
        self.carts: dict[int, list] = {}

        # Lock to ensure thread-safe access to customer-related operations (carts, available products).
        self.customer_lock = Lock()
        # Lock to ensure thread-safe access to producer-related operations (registering, publishing).
        self.producer_lock = Lock()
        # Lock to ensure thread-safe printing to console from multiple threads.
        self.printing_lock = Lock()

    def register_producer(self):
        """
        Registers a new producer with the marketplace, assigning it a unique ID
        and initializing its product queue.

        Returns:
            str: The unique ID assigned to the registered producer.
        """
        logging.info('register producer started.')
        with self.producer_lock: # Ensure thread-safe registration.
            p_id = uuid4().hex # Generate a unique hexadecimal ID for the producer.
            self.producers_queues[p_id] = [] # Initialize an empty list for this producer's products.
        logging.info('register producer finished. Returned %s.', p_id)
        return p_id



    def publish(self, producer_id: str, product: Product) -> bool:
        """
        Publishes a product from a producer to the marketplace.

        Args:
            producer_id (str): The ID of the producer publishing the product.
            product (Product): The Product object to publish.

        Returns:
            bool: True if the product was successfully published, False otherwise
                  (e.g., if the producer's queue is full).
        """
        logging.info(
            'publish started. Parameters: producer_id = %s, product = %s.', producer_id, product)
        
        # Conditional Logic: Checks if the producer's current inventory exceeds the maximum allowed.
        if len(self.producers_queues[producer_id]) == self.queue_size_per_producer:
            logging.info('publish finished. Returned False.')
            return False

        # If queue is not full, add the product to the producer's queue.
        self.producers_queues[producer_id].append(product)

        # Update the global count of available products.
        # This part is not protected by a lock, which could lead to race conditions
        # if multiple producers publish the same product concurrently.
        if product not in self.available_products:
            self.available_products[product] = 1
        else:
            self.available_products[product] += 1

        logging.info('publish finished. Returned True.')
        return True

    def new_cart(self) -> int:
        """
        Creates a new shopping cart and registers it with the marketplace, assigning a unique ID.

        Returns:
            int: The unique ID assigned to the new cart.
        """
        logging.info('new_cart started.')
        with self.customer_lock: # Ensure thread-safe cart creation.
            cart_id = uuid4().int # Generate a unique integer ID for the cart.
            self.carts[cart_id] = [] # Initialize an empty list for this cart's products.
        logging.info('new_cart finished. Returned %i', cart_id)


        return cart_id

    def add_to_cart(self, cart_id: int, product: Product) -> bool:
        """
        Attempts to add a specified product to a consumer's cart.
        It checks for product availability and decrements the available count if successful.

        Args:
            cart_id (int): The ID of the cart to add the product to.
            product (Product): The Product object to add.

        Returns:
            bool: True if the product was successfully added, False if the product
                  is not available.
        """
        logging.info(
            'add_to_cart started. Parameters: cart_id=%i, product=%s.', cart_id, product)

        with self.customer_lock: # Protects access to available_products and carts.

            # Conditional Logic: Checks if the product is available in the marketplace.
            if product not in self.available_products or self.available_products[product] == 0:
                logging.info('add_to_cart finished. Returned False.')
                return False

            # If available, decrement the global count.
            self.available_products[product] -= 1

            # Add the product to the specific cart.
            self.carts[cart_id].append(product)

        logging.info('add_to_cart finished. Returned True.')
        logging.debug('added')


        return True

    def remove_from_cart(self, cart_id: int, product: Product):
        """
        Removes a product from a consumer's cart and increments its available count.

        Args:
            cart_id (int): The ID of the cart from which to remove the product.
            product (Product): The Product object to remove.
        """
        logging.info(
            'remove_from_cart started. Parameters: cart_id=%i, product=%s.', cart_id, product)
        with self.customer_lock: # Protects access to carts and available_products.
            # Remove the product from the specific cart.
            self.carts[cart_id].remove(product)

            # Increment the global count of available products.
            self.available_products[product] += 1
        logging.info('remove_from_cart finished.')



    def place_order(self, cart_id: int) -> list[Product]:
        """
        Places an order for all items currently in the specified cart.
        This operation finalizes the purchase and removes the bought products
        from the marketplace's inventory (specifically from the producer's queue).

        Args:
            cart_id (int): The ID of the cart for which to place the order.

        Returns:
            List[Product]: A list of Product objects that were successfully purchased.
        """
        logging.info('place_order started. Parameters: cart_id=%i.', cart_id)
        
        bought_items = []

        # Block Logic: Iterates through products in the cart to finalize their purchase.
        for product in self.carts[cart_id]:
            # Block Logic: Finds the producer that has this product and removes it from their queue.
            # This simulates the product being taken from the stock.
            for producer_queue in self.producers_queues.values():
                if product in producer_queue:
                    bought_items.append(product) # Add to the list of successfully bought items.
                    producer_queue.remove(product) # Remove from the producer's stock.
                    break # Move to the next product in the cart.

        logging.info('place_order finished. Returned %s.', bought_items)
        return bought_items



class TestMarketplace(unittest.TestCase):
    """
    A test suite for the Marketplace class, using Python's unittest framework.
    It verifies the core functionalities of the marketplace such as producer
    registration, cart creation, product publishing, adding/removing from cart,
    and placing orders.
    """

    def setUp(self):
        """
        Sets up the test environment before each test method is executed.
        Initializes a fresh Marketplace instance with a queue size of 1 per producer.
        """
        self.marketplace = Marketplace(1)

    def test_register_producer_return_str(self):
        """
        Tests that registering a producer returns a string ID.
        """
        p_id = self.marketplace.register_producer()
        self.assertEqual(type(p_id), str)

    def test_new_cart_return_int(self):
        """
        Tests that creating a new cart returns an integer ID.
        """
        c_id = self.marketplace.new_cart()
        self.assertEqual(type(c_id), int)

    def test_publish_if_queue_not_full_then_return_true(self):
        """
        Tests that publishing a product succeeds (returns True) when the
        producer's queue is not full.
        """
        p_id = self.marketplace.register_producer()
        self.assertEqual(self.marketplace.publish(p_id, Product('Tea', 11)),
                         True)

    def test_publish_if_queue_full_then_return_true(self):
        """
        Tests that publishing a product fails (returns False) when the
        producer's queue is already full.
        """
        p_id = self.marketplace.register_producer()
        self.marketplace.publish(p_id, Product('Tea', 11)) # Fill the queue
        self.assertEqual(self.marketplace.publish(
            p_id, Product('Tea', 11)), False) # Attempt to publish again when full

    def test_add_to_cart_if_product_not_available_return_false(self):
        """
        Tests that adding a product to a cart fails (returns False) if the product
        is not available in the marketplace.
        """
        c_id = self.marketplace.new_cart()
        self.assertEqual(self.marketplace.add_to_cart(
            c_id, Product('Tea', 11)), False)

    def test_add_to_cart_if_product_available_return_true(self):
        """
        Tests that adding a product to a cart succeeds (returns True) if the product
        is available in the marketplace.
        """
        c_id = self.marketplace.new_cart()
        p_id = self.marketplace.register_producer()
        self.marketplace.publish(p_id, Product('Tea', 11)) # Make product available

        self.assertEqual(self.marketplace.add_to_cart(
            c_id, Product('Tea', 11)), True)

    def test_remove_from_cart(self):
        """
        Tests that a product can be successfully removed from a cart.
        Verifies that the cart becomes empty after removal.
        """
        c_id = self.marketplace.new_cart()
        p_id = self.marketplace.register_producer()
        self.marketplace.publish(p_id, Product('Tea', 11))
        self.marketplace.add_to_cart(
            c_id, Product('Tea', 11))
        self.marketplace.remove_from_cart(c_id, Product('Tea', 11))
        self.assertEqual(len(self.marketplace.carts[c_id]), 0)

    def test_place_order(self):
        """
        Tests the `place_order` functionality.
        Verifies that after placing an order, the purchased products are removed
        from the producer's queue in the marketplace.
        """
        p_id = self.marketplace.register_producer()
        c_id = self.marketplace.new_cart()
        self.marketplace.publish(p_id, Product('Tea', 11))
        self.marketplace.add_to_cart(
            c_id, Product('Tea', 11))
        self.marketplace.place_order(c_id)
        self.assertEqual(len(self.marketplace.producers_queues[p_id]), 0)


class Producer(Thread):
    """
    A Producer represents a seller in the marketplace. It operates as a separate thread,
    continuously producing products and publishing them to the marketplace according
    to a defined production schedule.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        Initializes a Producer thread.

        Args:
            products (List[Tuple[Product, int, float]]): A list of production definitions.
                                                          Each definition is a tuple containing
                                                          (Product, quantity, wait_time).
            marketplace (Marketplace): The shared marketplace instance to interact with.
            republish_wait_time (float): The time in seconds to wait before attempting
                                         to republish a product if the marketplace's
                                         queue for this producer is full.
            **kwargs: Arbitrary keyword arguments passed to the Thread constructor.
        """
        Thread.__init__(self, **kwargs)

        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        self.p_id = '' # Stores the producer's ID assigned by the marketplace.

    def run(self):
        """
        The main execution loop for the Producer thread.
        It registers with the marketplace, then continuously produces and publishes
        products. It iterates through its list of products, publishing each product
        'quantity' times, with a `wait_time` between each production. If publishing
        fails (e.g., marketplace queue is full), it waits `republish_wait_time`
        and retries.
        """
        self.p_id = self.marketplace.register_producer() # Register with the marketplace.
        # Block Logic: The producer's main loop, running indefinitely to simulate
        # continuous production.
        while True:
            # Block Logic: Iterates through each product definition in the producer's list.
            for prod_info in self.products:
                # Block Logic: Produces and publishes the specified quantity of the current product.
                for _ in range(prod_info[1]): # prod_info[1] is the quantity to produce.
                    sleep(prod_info[2]) # prod_info[2] is the wait_time after producing one unit.
                    # Block Logic: Continuously attempts to publish the product until successful.
                    # It waits `republish_wait_time` if the marketplace queue is full.
                    while not self.marketplace.publish(self.p_id, prod_info[0]): # prod_info[0] is the Product object.
                        sleep(self.republish_wait_time)

