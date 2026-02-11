"""
This module implements a producer-consumer simulation with a central marketplace.

It defines three main classes:
- Consumer: A thread that simulates a consumer buying products.
- Marketplace: A thread-safe central hub for managing products, producers, and consumer carts.
- Producer: A thread that simulates a producer creating and publishing products.
"""

from threading import Thread
import time


class Consumer(Thread):
    """
    Represents a consumer thread that interacts with the marketplace.

    Each consumer has a list of carts, and for each cart, it performs a series of
    add or remove actions before placing an order.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        Initializes a Consumer thread.

        Args:
            carts (list): A list of shopping carts, where each cart is a list of actions.
            marketplace (Marketplace): The marketplace instance to interact with.
            retry_wait_time (float): The time to wait before retrying to add a product to the cart.
            **kwargs: Additional keyword arguments for the Thread constructor.
        """
        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time

    def run(self):
        """
        The main execution logic for the consumer thread.

        Iterates through each cart, processes the actions (add/remove products),
        and places the order.
        """
        for cart in self.carts:
            cart_id = self.marketplace.new_cart()
            for action in cart:
                if action["type"] == 'add':
                    qty = action["quantity"]
                    while qty != 0:
                        flag = self.marketplace.add_to_cart(cart_id, action["product"])

                        while not flag:
                            time.sleep(self.retry_wait_time)
                            flag = self.marketplace.add_to_cart(cart_id, action["product"])
                        qty -= 1
                elif action["type"] == 'remove':
                    qty = action["quantity"]
                    while qty != 0:
                        self.marketplace.remove_from_cart(cart_id, action["product"])
                        qty -= 1

            self.marketplace.place_order(cart_id)

import time
from threading import Lock, currentThread
import logging
import logging.handlers as lh


class Marketplace:
    """
    A thread-safe marketplace for producers and consumers.

    This class manages the inventory of products, producer registrations, and consumer shopping carts.
    It uses locks to ensure that concurrent operations from multiple producer and consumer threads
    do not lead to race conditions.
    """

    def __init__(self, queue_size_per_producer):
        """
        Initializes the Marketplace.

        Args:
            queue_size_per_producer (int): The maximum number of products a single producer can have in the queue.
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
        self.products = {}

        self.print_lock = Lock()

        log_formatter = logging.Formatter('%(asctime)s - %(message)s')
        log_formatter.converter = time.gmtime

        rf_handler = lh.RotatingFileHandler("marketplace.log", maxBytes=1000000, backupCount=20)
        rf_handler.setFormatter(log_formatter)

        logging.basicConfig(
            level=logging.INFO,
            handlers=[rf_handler]
        )

    def register_producer(self):
        """
        Registers a new producer with the marketplace.

        Returns:
            int: The unique ID assigned to the new producer.
        """
        logging.info('Register producer start (no args)')
        self.producer_id_lock.acquire()
        id_return = self.producer_id
        self.producer_queue.insert(id_return, [])
        self.producer_id += 1
        self.producer_id_lock.release()
        logging.info('Register producer end: {}'.format(id_return))

        return id_return

    def publish(self, producer_id, product):
        """
        Allows a producer to publish a product to the marketplace.

        Args:
            producer_id (int): The ID of the producer publishing the product.
            product (any): The product to be published.

        Returns:
            bool: True if the product was published successfully, False otherwise (e.g., if the producer's queue is full).
        """
        logging.info('Publish start: producer_id: {}, product: {}'.format(producer_id, product))
        if len(self.producer_queue[producer_id]) >= self.queue_size_per_producer:
            logging.info('Publish end: False')
            return False

        self.add_products_lock.acquire()
        if product[0] in self.products:
            size = self.products[product[0]]
            self.products[product[0]] = size + 1
        else:
            self.products[product[0]] = 1

        self.add_products_lock.release()

        
        self.producer_queue[producer_id].append(product)
        logging.info('Publish end: True')
        return True

    def new_cart(self):
        """
        Creates a new, empty shopping cart for a consumer.

        Returns:
            int: The unique ID for the newly created cart.
        """
        logging.info('New_cart start (no args)')
        self.consumer_cart_lock.acquire()
        self.consumer_cart.append([])
        id_return = self.current_cart_id
        self.current_cart_id += 1
        self.consumer_cart_lock.release()
        logging.info('New_cart end: {}'.format(id_return))

        return id_return

    def add_to_cart(self, cart_id, product):
        """
        Adds a product to a consumer's shopping cart.

        Args:
            cart_id (int): The ID of the cart to add the product to.
            product (any): The product to add.

        Returns:
            bool: True if the product was added successfully, False if the product is not available.
        """
        logging.info('Add_to_cart start: cart_id: {}, product: {}'.format(cart_id, product))
        self.add_to_cart_lock.acquire()
        if product in self.products:
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
        Removes a product from a consumer's shopping cart.

        Args:
            cart_id (int): The ID of the cart to remove the product from.
            product (any): The product to remove.
        """
        logging.info('Remove_from_cart start: cart_id: {}, product: {}'.format(cart_id, product))
        
        if product in self.consumer_cart[cart_id]:
            self.consumer_cart[cart_id].remove(product)
        else:
            return

        self.remove_from_cart_lock.acquire()
        size = self.products[product]
        self.products[product] = size + 1
        self.remove_from_cart_lock.release()
        logging.info('Remove_from_cart end (no return')

    def place_order(self, cart_id):
        """
        Places an order for the items in a shopping cart.

        This method simulates the final purchase by printing the items bought.

        Args:
            cart_id (int): The ID of the cart to place an order for.

        Returns:
            list: The list of items in the cart that were "bought".
        """
        logging.info('Place_order start: cart_id: {}'.format(cart_id))
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
    Represents a producer thread that creates and publishes products to the marketplace.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        Initializes a Producer thread.

        Args:
            products (list): A list of products that the producer can create.
            marketplace (Marketplace): The marketplace instance to publish products to.
            republish_wait_time (float): The time to wait before retrying to publish a product.
            **kwargs: Additional keyword arguments for the Thread constructor.
        """
        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time

    def run(self):
        """
        The main execution logic for the producer thread.

        Registers with the marketplace and then enters an infinite loop to produce
        and publish products.
        """
        producer_id = self.marketplace.register_producer()

        while True:
            for product in self.products:
                qty = product[1]
                while qty != 0:
                    flag = self.marketplace.publish(producer_id, product)

                    if flag:
                        time.sleep(product[2])
                    else:
                        while not flag:
                            time.sleep(self.republish_wait_time)
                            flag = self.marketplace.publish(producer_id, product)

                    qty -= 1

