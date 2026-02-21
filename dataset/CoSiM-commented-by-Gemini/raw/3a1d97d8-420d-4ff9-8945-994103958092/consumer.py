"""A producer-consumer marketplace simulation with per-producer queues.

This module models an e-commerce marketplace where multiple Producer threads
and Consumer threads interact concurrently. The Marketplace is designed with a
separate, locked queue for each producer, which is a key architectural choice.

NOTE: The file contains multiple class definitions and local package imports
(e.g., `from tema.marketplace import Marketplace`), suggesting it was intended
to be part of a larger `tema` package. It is documented here as a single file.
"""

import time
from threading import Thread

# The following import suggests a circular dependency or project structure issue.
from tema.marketplace import Marketplace

class Consumer(Thread):
    """Represents a consumer thread that processes a list of shopping actions."""

    def __init__(self,
                 carts: list,
                 marketplace: Marketplace,
                 retry_wait_time: int,
                 **kwargs) \
    :
        """Initializes the consumer."""
        super().__init__(**kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time

    def run(self):
        """Executes the consumer's shopping simulation.
        
        For each assigned "cart" (list of actions), it adds or removes items
        from the marketplace, using a busy-wait loop to retry failed "add"
        operations. Finally, it places the order and prints the items bought.
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
# The following import suggests a circular dependency or project structure issue.
from tema.product import Product

class Marketplace:
    """The central marketplace managing inventory via per-producer queues.

    This implementation gives each producer its own queue with a dedicated lock,
    avoiding contention between producers when they publish products. However,
    consumers must search through all producer queues to find an item, and the
    logic for this search contains concurrency flaws.
    """

    def __init__(self, queue_size_per_producer: int):
        """Initializes the marketplace."""
        self.queue_size_per_producer = queue_size_per_producer
        # A list where each element is a tuple: (list_of_products, lock)
        self.producer_queues = []
        self.consumer_carts = []
        self.register_producer_lock = Lock()
        self.new_cart_lock = Lock()

    def register_producer(self) -> int:
        """Registers a new producer, creating a dedicated queue and lock for it."""
        with self.register_producer_lock:
            producer_id = len(self.producer_queues)
            self.producer_queues.append(([], Lock()))
        return producer_id

    def publish(self, producer_id: int, product: Product) -> bool:
        """Publishes a product on behalf of a producer.
        
        This operation is thread-safe with respect to other producers, as it
        only locks the specific queue for the given producer_id.
        """
        queue, lock = self.producer_queues[producer_id]
        with lock:
            if len(queue) >= self.queue_size_per_producer:
                return False
            queue.append(product)
        return True

    def new_cart(self) -> int:
        """Creates a new, empty cart for a consumer."""
        with self.new_cart_lock:
            cart_id = len(self.consumer_carts)
            self.consumer_carts.append([])
        return cart_id

    def add_to_cart(self, cart_id: int, product: Product) -> bool:
        """Searches all producer queues and moves a product to the consumer's cart.

        WARNING: This method has a critical race condition. It acquires a
        producer's lock, removes the item from the producer's queue, and then
        releases the lock *before* adding the item to the consumer's cart.
        The state is inconsistent between these two operations, and the item
        is in limbo, belonging to neither the producer nor the consumer.
        """
        cart = self.consumer_carts[cart_id]

        for producer_id, (queue, lock) in enumerate(self.producer_queues):
            with lock:
                try:
                    # Item is removed from producer within the lock...
                    queue.remove(product)
                except ValueError:
                    continue  # Product not in this queue, try next one.
            
            # ...but added to consumer cart outside the lock, creating a race condition.
            cart.append((product, producer_id))
            return True

        return False # Product not found in any producer queue.

    def remove_from_cart(self, cart_id: int, product: Product) -> bool:
        """Removes a product from a cart and returns it to its original producer."""
        cart = self.consumer_carts[cart_id]

        for i, (prod, producer_id) in enumerate(cart):
            if prod == product:
                del cart[i]
                queue, lock = self.producer_queues[producer_id]
                with lock:
                    queue.append(prod)
                return True
        return False

    def place_order(self, cart_id) -> list:
        """Returns the list of products in the cart.
        
        NOTE: This method does not clear or consume the cart, which may be
        unintended. It simply returns the current contents.
        """
        cart = self.consumer_carts[cart_id]
        return [product for product, producer_id in cart]


import time
from threading import Thread
# The following import suggests a circular dependency or project structure issue.
from tema.marketplace import Marketplace

class Producer(Thread):
    """Represents a producer thread that publishes products to the marketplace."""

    def __init__(self,
                 products: list,
                 marketplace: Marketplace,
                 republish_wait_time: int,
                 **kwargs) \
    :
        """Initializes the producer and registers it with the marketplace."""
        super().__init__(**kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        self.id_ = self.marketplace.register_producer()

    def run(self):
        """The main loop for the producer.
        
        Continuously attempts to publish its products. If a producer's queue in
        the marketplace is full, it uses a busy-wait loop until space is available.
        """
        while True:
            for product, qty, wait_time in self.products:
                for _ in range(qty):
                    time.sleep(wait_time)
                    # Busy-wait until the product can be published.
                    while not self.marketplace.publish(self.id_, product):
                        time.sleep(self.republish_wait_time)
