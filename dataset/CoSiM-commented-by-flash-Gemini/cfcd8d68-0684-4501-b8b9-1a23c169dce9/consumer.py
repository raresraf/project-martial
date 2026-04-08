"""
This module simulates a marketplace with producers and consumers running in
separate threads.

It demonstrates a producer-consumer pattern with a more advanced concurrency
model using fine-grained locking for each producer's inventory.
"""
import time
from threading import Thread
# NOTE: The following import seems to be a remnant, as Marketplace is redefined below.
from tema.marketplace import Marketplace

class Consumer(Thread):
    """
    Represents a consumer thread that simulates purchasing items from the marketplace.

    The consumer processes a list of cart actions, adding or removing products,
    and finally "buys" the items by placing an order.
    """

    def __init__(self,
                 carts: list,
                 marketplace: 'Marketplace',
                 retry_wait_time: int,
                 **kwargs) \
    :
        """
        Initializes the Consumer thread.

        Args:
            carts (list): A list of carts, where each cart is a list of actions
                          (add/remove) to perform.
            marketplace (Marketplace): The shared marketplace instance.
            retry_wait_time (int): Time in seconds to wait before retrying to add
                                 a product if it's unavailable.
            **kwargs: Forwarded to the Thread constructor.
        """
        super().__init__(**kwargs)

        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time

    def run(self):
        """
        The main logic for the consumer thread.

        Iterates through its assigned carts, executes add/remove actions, and
        places an order for each cart.
        """
        for cart in self.carts:
            cart_id = self.marketplace.new_cart()

            for action in cart:
                type_ = action['type']
                product = action['product']
                qty = action['quantity']

                for _ in range(qty):
                    if type_ == 'add':
                        # Busy-wait until the product can be added to the cart.
                        while not self.marketplace.add_to_cart(cart_id, product):
                            time.sleep(self.retry_wait_time)
                    elif type_ == 'remove':
                        self.marketplace.remove_from_cart(cart_id, product)

            order = self.marketplace.place_order(cart_id)

            for product in order:
                print(f'{self.name} bought {product}')


from threading import Lock
# NOTE: The following import seems to be a remnant.
from tema.product import Product

class Marketplace:
    """
    Manages the interaction between producers and consumers.

    This class provides a thread-safe environment for publishing products,
    managing shopping carts, and processing orders. It uses a fine-grained
    locking mechanism where each producer's product queue is protected by its
    own lock, allowing for higher concurrency.
    """

    def __init__(self, queue_size_per_producer: int):
        """
        Initializes the Marketplace.

        Args:
            queue_size_per_producer (int): The maximum number of products that
                                           can be in a single producer's queue.
        """
        self.queue_size_per_producer = queue_size_per_producer

        self.producer_queues = []
        self.consumer_carts = []

        self.register_producer_lock = Lock()
        self.new_cart_lock = Lock()

    def register_producer(self) -> int:
        """
        Registers a new producer, giving them a dedicated product queue and lock.

        Returns:
            int: The new producer's unique ID.
        """
        with self.register_producer_lock:
            producer_id = len(self.producer_queues)
            # Each producer gets a tuple: (list_of_products, lock_for_that_list)
            self.producer_queues.append(([], Lock()))

        return producer_id

    def publish(self, producer_id: int, product: 'Product') -> bool:
        """
        Publishes a product to a specific producer's queue.

        Args:
            producer_id (int): The ID of the producer publishing the product.
            product (Product): The product to publish.

        Returns:
            bool: True if the product was published successfully, False if the
                  producer's queue was full.
        """
        queue, lock = self.producer_queues[producer_id]

        with lock:
            if len(queue) >= self.queue_size_per_producer:
                return False
            queue.append(product)

        return True

    def new_cart(self) -> int:
        """
        Creates a new, empty shopping cart for a consumer.

        Returns:
            int: The new cart's unique ID.
        """
        with self.new_cart_lock:
            cart_id = len(self.consumer_carts)
            self.consumer_carts.append([])

        return cart_id

    def add_to_cart(self, cart_id: int, product: 'Product') -> bool:
        """
        Adds a product to a consumer's cart.

        It iterates through all producer queues, acquiring each lock individually,
        to find and claim the requested product.

        Args:
            cart_id (int): The ID of the cart to add the product to.
            product (Product): The product to add.

        Returns:
            bool: True if the product was found and added, False otherwise.
        """
        cart = self.consumer_carts[cart_id]

        for producer_id, (queue, lock) in enumerate(self.producer_queues):
            with lock:
                try:
                    # Atomically remove the product from the producer's inventory.
                    queue.remove(product)
                except ValueError:
                    continue # Product not in this queue, try the next one.
            
            # If successful, add product and its origin producer to the cart.
            cart.append((product, producer_id))
            return True

        return False

    def remove_from_cart(self, cart_id: int, product: 'Product') -> bool:
        """
        Removes a product from a consumer's cart, returning it to the producer.

        Args:
            cart_id (int): The ID of the cart.
            product (Product): The product to remove.

        Returns:
            bool: True if the product was found and removed, False otherwise.
        """
        cart = self.consumer_carts[cart_id]

        for i, (prod, producer_id) in enumerate(cart):
            if prod == product:
                del cart[i]

                # Return the product to the original producer's queue.
                queue, lock = self.producer_queues[producer_id]
                with lock:
                    queue.append(prod)
                return True

        return False

    def place_order(self, cart_id) -> list:
        """
        Retrieves all products from a cart to finalize an order.

        Args:
            cart_id (int): The ID of the cart.

        Returns:
            list: A list of the products that were in the cart.
        """
        cart = self.consumer_carts[cart_id]
        return [product for product, producer_id in cart]


import time
from threading import Thread
# NOTE: The following import seems to be a remnant.
from tema.marketplace import Marketplace

class Producer(Thread):
    """
    Represents a producer thread that continuously creates products.

    The producer publishes products to its dedicated queue in the marketplace.
    """

    def __init__(self,
                 products: list,
                 marketplace: Marketplace,
                 republish_wait_time: int,
                 **kwargs) \
    :
        """
        Initializes the Producer thread.

        Args:
            products (list): A list of products to produce, containing the
                             product, quantity, and production time.
            marketplace (Marketplace): The shared marketplace instance.
            republish_wait_time (int): Time to wait before retrying to publish
                                     if the queue is full.
            **kwargs: Forwarded to the Thread constructor.
        """
        super().__init__(**kwargs)

        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time

        self.id_ = self.marketplace.register_producer()

    def run(self):
        """
        The main logic for the producer thread.

        Continuously loops, producing items and attempting to publish them
        to the marketplace.
        """
        while True:
            for product, qty, wait_time in self.products:
                for _ in range(qty):
                    time.sleep(wait_time)

                    # Busy-wait until the product can be published.
                    while not self.marketplace.publish(self.id_, product):
                        time.sleep(self.republish_wait_time)
