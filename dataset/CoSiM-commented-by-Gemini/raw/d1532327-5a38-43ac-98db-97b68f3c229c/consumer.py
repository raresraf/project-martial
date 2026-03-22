"""
@file consumer.py
@brief Implements a multi-threaded producer-consumer simulation for a marketplace.
@details This file defines three classes: Marketplace, Producer, and Consumer, which together model a
basic e-commerce environment. Producers add products, Consumers purchase them, and the Marketplace
coordinates these activities in a thread-safe manner.
"""

from threading import Thread, Lock
import time

class Consumer(Thread):
    """
    Represents a consumer thread that simulates a user shopping in the marketplace.
    Each consumer processes a predefined list of shopping carts, where each cart contains
    a series of actions (add or remove products).
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        Initializes a Consumer instance.
        @param carts A list of carts, where each cart is a list of operations.
        @param marketplace The central Marketplace instance to interact with.
        @param retry_wait_time The time in seconds to wait before retrying a failed operation.
        @param kwargs Additional keyword arguments for the Thread constructor.
        """
        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.cart_id = -1

    def run(self):
        """
        The main execution logic for the consumer thread.
        It iterates through its assigned carts, processes all operations within each,
        and finally places the order.
        """
        # Invariant: Processes one shopping session (cart) at a time.
        for c in self.carts:
            self.cart_id = self.marketplace.new_cart()
            # Invariant: Executes all operations for the current cart.
            for op in c:
                op_type = op['type']
                # Block Logic: Handles the "add" operation for a product.
                if op_type == "add":
                    i = 0
                    # Pre-condition: 'quantity' items of 'product' need to be added.
                    # Invariant: Loop continues until the desired quantity is successfully added.
                    while i < op['quantity']:
                        ret = self.marketplace.add_to_cart(self.cart_id, op['product'])
                        if ret is True:
                            i = i + 1
                        else:
                            # Inline: If adding to cart fails (e.g., product unavailable), wait and retry.
                            time.sleep(self.retry_wait_time)
                # Block Logic: Handles the "remove" operation for a product.
                elif op_type == "remove":
                    i = 0
                    # Pre-condition: 'quantity' items of 'product' need to be removed.
                    # Invariant: Loop continues until the desired quantity is successfully removed.
                    while i < op['quantity']:
                        ret = self.marketplace.remove_from_cart(self.cart_id, op['product'])
                        if ret is True:
                            i = i + 1
                        else:
                            # Inline: If removal fails, wait and retry. This could happen if the item
                            # was never in the cart, which indicates a logic issue in the input.
                            time.sleep(self.retry_wait_time)
            # Finalizes the transaction for the current cart.
            my_cart = self.marketplace.place_order(self.cart_id)
            for p in my_cart:
                print(self.name + ' bought ' + str(p))

class Marketplace:
    """
    A thread-safe marketplace that manages producers, products, and consumer carts.
    It uses locks to ensure that concurrent operations from multiple threads do not
    corrupt its internal state.
    """

    def __init__(self, queue_size_per_producer):
        """
        Initializes the Marketplace.
        @param queue_size_per_producer The maximum number of products a single producer can publish.
        """
        self.queue_size_per_producer = queue_size_per_producer
        self.producersList = []
        self.cartsList = []
        self.available_products_pairs = []
        # Lock to synchronize adding/removing from carts and creating new carts.
        self.add_remove_lock = Lock()
        # Lock to synchronize producer registration and publishing.
        self.producer_lock = Lock()

    def register_producer(self):
        """
        Registers a new producer with the marketplace.
        @return An integer ID for the newly registered producer.
        """
        new_producer = []
        self.producersList.append(new_producer)
        return len(self.producersList) - 1

    def publish(self, producer_id, product):
        """
        Allows a producer to publish a product to the marketplace.
        This operation is thread-safe.
        @param producer_id The ID of the producer publishing the product.
        @param product The product to be published.
        @return True if the product was published successfully, False if the producer's queue is full.
        """
        self.producer_lock.acquire()
        # Block Logic: Enforces the queue size limit for each producer.
        if len(self.producersList[producer_id]) == self.queue_size_per_producer:
            self.producer_lock.release()
            return False
        
        self.producersList[producer_id].append(product)
        self.available_products_pairs.append((product, producer_id))
        self.producer_lock.release()
        return True

    def new_cart(self):
        """
        Creates a new, empty shopping cart for a consumer.
        This operation is thread-safe.
        @return An integer ID for the newly created cart.
        """
        self.add_remove_lock.acquire()
        new_c = []
        self.cartsList.append(new_c)
        self.add_remove_lock.release()
        return len(self.cartsList) - 1

    def add_to_cart(self, cart_id, product):
        """
        Adds a product to a consumer's shopping cart.
        It finds an available product and moves it from the marketplace's general pool
        to the specified cart. This operation is thread-safe.
        @param cart_id The ID of the cart to add the product to.
        @param product The product to add.
        @return True if the product was added successfully, False if the product is not available.
        """
        self.add_remove_lock.acquire()
        # Block Logic: Searches for the requested product among all available products.
        for pair in self.available_products_pairs:
            if pair[0] == product:
                # State Change: Moves the product from available pool to the consumer's cart.
                self.cartsList[cart_id].append(pair)
                self.available_products_pairs.remove(pair)
                self.add_remove_lock.release()
                return True
        self.add_remove_lock.release()
        return False

    def remove_from_cart(self, cart_id, product):
        """
        Removes a product from a consumer's shopping cart.
        It moves the product from the cart back to the marketplace's general pool.
        This operation is thread-safe.
        @param cart_id The ID of the cart to remove the product from.
        @param product The product to remove.
        @return True if the product was removed successfully, False otherwise.
        """
        self.add_remove_lock.acquire()
        # Block Logic: Searches for the product within the specific consumer's cart.
        for pair in self.cartsList[cart_id]:
            if pair[0] == product:
                # State Change: Moves the product from the cart back to the available pool.
                self.available_products_pairs.append(pair)
                self.cartsList[cart_id].remove(pair)
                self.add_remove_lock.release()
                return True
        self.add_remove_lock.release()
        return False

    def place_order(self, cart_id):
        """
        Finalizes an order for a given cart.
        This 'consumes' the products, removing them from the producer's inventory.
        This operation uses a producer-level lock to ensure atomicity.
        @param cart_id The ID of the cart for which the order is placed.
        @return A list of products that were successfully purchased.
        """
        prod_list = []
        self.producer_lock.acquire()
        # Block Logic: Iterates through all items in the finalized cart.
        # For each item, it's added to the final purchase list and removed
        # from the original producer's inventory, completing the transaction.
        for pair in self.cartsList[cart_id]:
            prod_list.append(pair[0])
            self.producersList[pair[1]].remove(pair[0])
        self.producer_lock.release()
        return prod_list

class Producer(Thread):
    """
    Represents a producer thread that continuously publishes products to the marketplace.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        Initializes a Producer instance.
        @param products A list of products that this producer will generate.
        @param marketplace The central Marketplace instance to publish to.
        @param republish_wait_time Time to wait before retrying to publish if the queue is full.
        @param kwargs Additional keyword arguments for the Thread constructor.
        """
        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        # Functional Utility: Registers itself with the marketplace to get a unique ID.
        self.producer_id = marketplace.register_producer()

    def run(self):
        """
        The main execution logic for the producer thread.
        It runs in an infinite loop, continuously publishing its products.
        """
        # Block Logic: An infinite loop to ensure the producer continuously stocks products.
        while True:
            # Invariant: Iterates through all product types the producer is responsible for.
            for p in self.products:
                i = 0
                # Pre-condition: A specific quantity of a product must be published.
                # Invariant: Loop continues until the desired quantity is published.
                while i < p[1]:
                    ret = self.marketplace.publish(self.producer_id, p[0])
                    if ret is True:
                        i = i + 1
                        # Inline: Simulates the time taken to produce one unit of the product.
                        time.sleep(float(p[2]))
                    else:
                        # Inline: If publishing fails (queue is full), wait before retrying.
                        time.sleep(float(self.republish_wait_time))
