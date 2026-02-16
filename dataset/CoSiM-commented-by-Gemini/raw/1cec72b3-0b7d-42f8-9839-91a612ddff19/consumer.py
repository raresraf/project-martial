"""
This module implements a robust producer-consumer simulation of an e-commerce
marketplace, with a focus on thread-safe interactions.

It defines `Producer` and `Consumer` threads that interact with a central
`Marketplace` class. This implementation correctly uses fine-grained locking
to ensure data consistency during concurrent operations.
"""

import time
from threading import Thread, Lock
from tema.marketplace import Marketplace
from tema.product import Product


class Consumer(Thread):
    """
    Represents a consumer thread that shops in the marketplace.

    Each consumer processes a list of carts, with each cart containing a sequence
    of 'add' or 'remove' actions for specific products.
    """

    def __init__(self,
                 carts: list,
                 marketplace: Marketplace,
                 retry_wait_time: int,
                 **kwargs) \
    :
        """
        Initializes a Consumer thread.

        Args:
            carts (list): A list of shopping actions to be performed.
            marketplace (Marketplace): The shared marketplace instance.
            retry_wait_time (float): Time to wait before retrying to add a
                                     product that is currently unavailable.
        """
        super().__init__(**kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time

    def run(self):
        """
        The main execution loop for the consumer.
        It processes each cart, executes its actions, and prints the final order.
        """
        for cart in self.carts:
            cart_id = self.marketplace.new_cart()

            for action in cart:
                type_ = action['type']
                product = action['product']
                qty = action['quantity']

                # Perform the 'add' or 'remove' action 'qty' times.
                for _ in range(qty):
                    if type_ == 'add':
                        # Invariant: If adding to cart fails (product is unavailable),
                        # the consumer must wait and retry. This handles contention
                        # with other consumers.
                        while not self.marketplace.add_to_cart(cart_id, product):
                            time.sleep(self.retry_wait_time)
                    elif type_ == 'remove':
                        self.marketplace.remove_from_cart(cart_id, product)

            order = self.marketplace.place_order(cart_id)

            for product in order:
                print(f'{self.name} bought {product}')


class Marketplace:
    """
    Manages the inventory and carts in a thread-safe manner.

    This class uses a fine-grained locking strategy where each producer's
    product queue has its own dedicated lock. This prevents race conditions
    when multiple consumers try to access products simultaneously.
    """

    def __init__(self, queue_size_per_producer: int):
        """
        Initializes the Marketplace.

        Args:
            queue_size_per_producer (int): The max number of products a single
                                           producer can list at one time.
        """
        self.queue_size_per_producer = queue_size_per_producer
        # Each element is a tuple: (list_of_products, Lock). Index is producer_id.
        self.producer_queues = []
        # Each element is a list of products in a cart. Index is cart_id.
        self.consumer_carts = []
        # Lock to ensure thread-safe producer registration.
        self.register_producer_lock = Lock()
        # Lock to ensure thread-safe cart creation.
        self.new_cart_lock = Lock()

    def register_producer(self) -> int:
        """
        Registers a new producer, allocating a dedicated queue and lock for them.

        Returns:
            int: The new producer's unique ID (their index in the list).
        """
        with self.register_producer_lock:
            producer_id = len(self.producer_queues)
            # Each producer gets their own product list and a dedicated lock.
            self.producer_queues.append(([], Lock()))
        return producer_id

    def publish(self, producer_id: int, product: Product) -> bool:
        """
        Allows a producer to list a product. This operation is thread-safe.
        """
        queue, lock = self.producer_queues[producer_id]
        # Acquire the specific lock for this producer's queue.
        with lock:
            if len(queue) >= self.queue_size_per_producer:
                return False
            queue.append(product)
        return True

    def new_cart(self) -> int:
        """
        Creates a new, empty cart for a consumer in a thread-safe way.
        """
        with self.new_cart_lock:
            cart_id = len(self.consumer_carts)
            self.consumer_carts.append([])
        return cart_id

    def add_to_cart(self, cart_id: int, product: Product) -> bool:
        """
        Atomically finds a product in any producer's queue, removes it,
        and adds it to the consumer's cart.

        Returns:
            bool: True if the product was found and moved, False otherwise.
        """
        cart = self.consumer_carts[cart_id]

        # Block Logic: Iterate through all producer queues to find the product.
        for producer_id, (queue, lock) in enumerate(self.producer_queues):
            # Acquire the lock for the current producer's queue.
            with lock:
                try:
                    # The check and removal are performed atomically inside the locked block.
                    queue.remove(product)
                except ValueError:
                    # If product is not in this queue, release lock and continue.
                    continue
            
            # If successful, add the product and its original producer to the cart.
            cart.append((product, producer_id))
            return True
        # Return False if the product was not found in any queue.
        return False

    def remove_from_cart(self, cart_id: int, product: Product) -> bool:
        """
        Atomically removes a product from a cart and returns it to the
        original producer's queue.
        """
        cart = self.consumer_carts[cart_id]

        for i, (prod, producer_id) in enumerate(cart):
            if prod == product:
                del cart[i]
                # Get the original producer's queue and lock.
                queue, lock = self.producer_queues[producer_id]
                # Atomically add the product back to the producer's queue.
                with lock:
                    queue.append(prod)
                return True
        return False

    def place_order(self, cart_id) -> list:
        """
        Retrieves the list of products from a cart for final checkout.
        Note: This implementation does not clear the cart after placing the order.
        """
        cart = self.consumer_carts[cart_id]
        return [product for product, producer_id in cart]


class Producer(Thread):
    """
    Represents a producer that publishes a list of products to the marketplace.
    """

    def __init__(self,
                 products: list,
                 marketplace: Marketplace,
                 republish_wait_time: int,
                 **kwargs) \
    :
        """
        Initializes a Producer thread.
        """
        super().__init__(**kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        # Register with the marketplace upon creation to get a unique ID.
        self.id_ = self.marketplace.register_producer()

    def run(self):
        """
        The main execution loop for the producer.
        Continuously publishes its products to the marketplace.
        """
        while True:
            for product, qty, wait_time in self.products:
                for _ in range(qty):
                    time.sleep(wait_time)
                    
                    # Invariant: If the producer's queue is full, wait and retry.
                    while not self.marketplace.publish(self.id_, product):
                        time.sleep(self.republish_wait_time)
