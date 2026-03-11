"""
This module implements a Producer-Consumer simulation using multithreading.
It defines three classes: Producer, Consumer, and Marketplace, which model the
interactions in a simple e-commerce environment.
"""

import time
from threading import Thread, Lock


class Consumer(Thread):
    """
    Represents a consumer that buys products from the marketplace.
    Each consumer runs in its own thread.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        Initializes a Consumer thread.

        :param carts: A list of carts, where each cart is a list of operations
                      (add/remove products).
        :param marketplace: The Marketplace object where the consumer will shop.
        :param retry_wait_time: Time in seconds to wait before retrying to get a
                                product that is out of stock.
        :param kwargs: Additional keyword arguments for the Thread constructor.
        """
        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.print_locked = Lock()

    def run(self):
        """
        The main execution logic for the consumer thread.
        It processes each cart, adds/removes products, and places the order.
        """
        for cart in self.carts:
            # Each consumer gets a new cart from the marketplace.
            my_cart = self.marketplace.new_cart()
            for to_do in cart:
                repeat = to_do['quantity']
                while repeat > 0:
                    # Check if the product is available in the marketplace stock.
                    if to_do['product'] in self.marketplace.market_stock:
                        self.execute_task(to_do['type'], my_cart, to_do['product'])
                        repeat -= 1
                    else:
                        # If the product is not in stock, wait and retry.
                        time.sleep(self.retry_wait_time)

            # Place the order after processing all operations in the cart.
            order = self.marketplace.place_order(my_cart)
            # Use a lock for printing to prevent interleaved output from different threads.
            with self.print_locked:
                for product in order:
                    print(self.getName(), "bought", product)

    def execute_task(self, task_type, cart_id, product):
        """
        Executes a single task (add or remove a product from the cart).

        :param task_type: The type of task, either 'add' or 'remove'.
        :param cart_id: The ID of the cart to modify.
        :param product: The product to add or remove.
        """
        if task_type == 'add':
            self.marketplace.add_to_cart(cart_id, product)
        elif task_type == 'remove':
            self.marketplace.remove_from_cart(cart_id, product)


class Marketplace:
    """
    Represents the marketplace where producers publish products and consumers buy them.
    This class is thread-safe, using locks to protect shared data.
    """

    def __init__(self, queue_size_per_producer):
        """
        Initializes the Marketplace.

        :param queue_size_per_producer: The maximum number of products a single
                                        producer can have in the market at one time.
        """
        self.queue_size_per_producer = queue_size_per_producer
        self.num_producers = -1
        self.register_locked = Lock()
        self.market_stock = []  # List of products currently available for sale.
        self.product_counter = []  # Tracks number of products per producer.
        self.product_owner = {}  # Maps a product to its producer owner.
        self.num_consumers = -1
        self.cart = [[]]  # A list of lists, where each inner list is a consumer's cart.
        self.cart_locked = Lock()
        self.add_locked = Lock()
        self.remove_locked = Lock()
        self.publish_locked = Lock()
        self.market_locked = Lock()

    def register_producer(self):
        """
        Registers a new producer, assigning it a unique ID.
        :return: The new producer's ID.
        """
        with self.register_locked:
            self.num_producers += 1
            new_producer_id = self.num_producers
        self.product_counter.append(0)
        return new_producer_id

    def publish(self, producer_id, product):
        """
        Allows a producer to publish a product to the marketplace.

        :param producer_id: The ID of the producer publishing the product.
        :param product: The product to be published.
        :return: True if the product was published successfully, False otherwise
                 (e.g., if the producer's queue is full).
        """
        # A producer cannot publish more products than its queue size allows.
        if self.product_counter[producer_id] >= self.queue_size_per_producer:
            return False
        self.market_stock.append(product)
        with self.publish_locked:
            self.product_counter[producer_id] += 1
            self.product_owner[product] = producer_id
        return True

    def new_cart(self):
        """
        Creates a new, empty cart for a consumer.
        :return: The ID of the new cart.
        """
        with self.cart_locked:
            self.num_consumers += 1
            new_consumer_cart_id = self.num_consumers
        self.cart.append([])
        return new_consumer_cart_id

    def add_to_cart(self, cart_id, product):
        """
        Adds a product from the market stock to a consumer's cart.

        :param cart_id: The ID of the cart to add the product to.
        :param product: The product to add.
        :return: True if the product was added successfully, False if it was not in stock.
        """
        if product not in self.market_stock:
            return False
        self.cart[cart_id].append(product)
        # Decrement the producer's product counter.
        with self.add_locked:
            self.product_counter[self.product_owner[product]] -= 1
        # Remove the product from the central market stock.
        with self.market_locked:
            if product in self.market_stock:
                element_index = self.market_stock.index(product)
                del self.market_stock[element_index]
        return True

    def remove_from_cart(self, cart_id, product):
        """
        Removes a product from a consumer's cart and returns it to the market stock.

        :param cart_id: The ID of the cart to remove the product from.
        :param product: The product to remove.
        """
        if product in self.cart[cart_id]:
            # Increment the producer's product counter as the item is returned.
            with self.remove_locked:
                self.product_counter[self.product_owner[product]] += 1
            self.cart[cart_id].remove(product)
            self.market_stock.append(product)

    def place_order(self, cart_id):
        """
        Finalizes the shopping process and returns the items in the cart.
        
        :param cart_id: The ID of the cart to place the order from.
        :return: A list of products in the final order.
        """
        return self.cart[cart_id]


class Producer(Thread):
    """
    Represents a producer that creates products and publishes them to the marketplace.
    Each producer runs in its own thread.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        Initializes a Producer thread.

        :param products: A list of products that the producer will create. Each product
                         is a tuple of (name, quantity, production_time).
        :param marketplace: The Marketplace object to publish products to.
        :param republish_wait_time: Time in seconds to wait before trying to
                                    re-publish a product if the queue is full.
        :param kwargs: Additional keyword arguments for the Thread constructor.
        """
        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        self.my_id = self.marketplace.register_producer()

    def run(self):
        """
        The main execution logic for the producer thread.
        It continuously produces and publishes products to the marketplace.
        """
        while True:
            for (product, quantity, seconds) in self.products:
                repeat = quantity
                while repeat > 0:
                    # Attempt to publish the product.
                    wait = self.marketplace.publish(self.my_id, product)
                    if wait:
                        # If successful, wait for the production time.
                        time.sleep(seconds)
                        repeat -= 1
                    else:
                        # If the producer's queue in the marketplace is full, wait and retry.
                        time.sleep(self.republish_wait_time)
