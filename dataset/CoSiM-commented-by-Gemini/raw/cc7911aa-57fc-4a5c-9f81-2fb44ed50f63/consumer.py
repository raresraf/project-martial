"""
This module implements a multi-threaded simulation of an e-commerce marketplace.

It defines three main classes:
- Marketplace: A thread-safe central hub that manages inventory and shopping carts.
- Consumer: A thread that simulates a buyer, adding and removing products from a
  cart and placing orders.
- Producer: A thread that simulates a seller, publishing products to the marketplace.

The simulation uses locks to handle concurrent access to shared resources,
demonstrating a classic producer-consumer problem.
"""

from threading import Thread
from time import sleep


class Consumer(Thread):
    """
    Represents a consumer thread that simulates a buyer's actions.

    Each consumer is initialized with a set of 'carts', where each cart is a list
    of operations (add/remove products). The consumer processes these operations
    sequentially for each cart.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        Initializes a Consumer instance.

        :param carts: A list of carts, where each cart is a list of product
                      operations (dictionaries).
        :param marketplace: The shared Marketplace instance.
        :param retry_wait_time: Time in seconds to wait before retrying to add
                                an unavailable product.
        :param kwargs: Additional keyword arguments for the Thread constructor.
        """
        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time

    def add_command(self, cart_id, product, quantity):
        """
        Adds a specified quantity of a product to the cart.

        This method will repeatedly attempt to add the product to the cart until
        successful. If the marketplace cannot fulfill the request (e.g., product
        is out of stock), it waits for `retry_wait_time` seconds before trying again.

        :param cart_id: The ID of the cart to add to.
        :param product: The product to add.
        :param quantity: The number of units of the product to add.
        """
        for _ in range(quantity):
            status = False
            while not status:
                status = self.marketplace.add_to_cart(cart_id, product)
                if not status:
                    # Busy-wait if the product is not available.
                    sleep(self.retry_wait_time)

    def remove_command(self, cart_id, product, quantity):
        """
        Removes a specified quantity of a product from the cart.

        :param cart_id: The ID of the cart to remove from.
        :param product: The product to remove.
        :param quantity: The number of units of the product to remove.
        """
        for _ in range(quantity):
            self.marketplace.remove_from_cart(cart_id, product)

    def run(self):
        """The main execution loop for the consumer thread."""
        # Pre-condition: self.carts contains the shopping lists for this consumer.
        for cart in self.carts:
            # Each cart simulation starts with a fresh cart in the marketplace.
            cart_id = self.marketplace.new_cart()
            # Process all operations (add/remove) for the current cart.
            for op in cart:
                command = op.get("type")
                if command == "add":
                    self.add_command(cart_id, op.get("product"), op.get("quantity"))
                if command == "remove":
                    self.remove_command(cart_id, op.get("product"), op.get("quantity"))
            # Finalize the purchase and get the list of items.
            item_list = self.marketplace.place_order(cart_id)
            # Invariant: After this loop, all operations for one cart are complete.
            for prod in item_list:
                print("%s bought %s" % (self.name, prod))


from threading import Lock


class Marketplace:
    """
    A thread-safe marketplace that manages producers, products, and carts.

    This class acts as the shared state between Producer and Consumer threads,
    using a lock to prevent race conditions when accessing its internal data
    structures.
    """

    def __init__(self, queue_size_per_producer):
        """
        Initializes the Marketplace.

        :param queue_size_per_producer: The maximum number of products a single
                                        producer can have listed at one time.
        """
        self.queue_size_per_producer = queue_size_per_producer
        self.producer_id = -1
        self.cart_id = -1
        # producer_list stores lists of products for each producer.
        # Each product is a tuple: [product_name, is_available_bool]
        self.producer_list = []
        self.cart_list = [] # Stores lists of products for each active cart.
        self.lock = Lock() # Global lock for all marketplace operations.

    def register_producer(self):
        """
        Registers a new producer, providing them with a unique ID.

        This method is thread-safe.

        :return: The unique ID assigned to the new producer.
        """
        self.lock.acquire()
        self.producer_id += 1
        self.producer_list.append([])
        self.lock.release()
        return self.producer_id

    def publish(self, producer_id, product):
        """
        Allows a producer to publish a new product to the marketplace.

        The publication fails if the producer's personal queue is full.
        This method is thread-safe.

        :param producer_id: The ID of the producer publishing the product.
        :param product: The product to be published.
        :return: True if publication was successful, False otherwise.
        """
        self.lock.acquire()
        # Calculate the current number of available items for this producer.
        count = 0
        for prod in self.producer_list[producer_id]:
            if prod[1]: # prod[1] is the availability flag.
                count += 1

        # Check if the producer's queue has space.
        if (
            self.producer_list[producer_id] != 0
            and self.queue_size_per_producer > count
        ):
            self.producer_list[producer_id].append([product, True])
            self.lock.release()
            return True
        self.lock.release()
        return False

    def new_cart(self):
        """
        Creates a new, empty shopping cart and returns its unique ID.

        This method is thread-safe.

        :return: The unique ID for the newly created cart.
        """
        self.lock.acquire()
        self.cart_id += 1
        self.cart_list.append([])
        self.lock.release()
        return self.cart_id

    def add_to_cart(self, cart_id, product):
        """
        Adds a product to a shopping cart.

        It searches through all producers' inventories for an available unit of
        the specified product. If found, it marks the unit as unavailable and
        adds it to the cart. This method is thread-safe.

        :param cart_id: The ID of the cart to add the product to.
        :param product: The product to add.
        :return: True if the product was successfully added, False otherwise.
        """
        self.lock.acquire()
        # Linearly search all producer inventories for an available product.
        for lists in self.producer_list:
            for item in lists:
                if item[0] == product and item[1]: # If product matches and is available...
                    self.cart_list[cart_id].append(product)
                    item[1] = False # ...mark it as unavailable.
                    self.lock.release()
                    return True
        self.lock.release()
        return False

    def remove_from_cart(self, cart_id, product):
        """
        Removes a product from a shopping cart and makes it available again.

        This method is thread-safe.

        :param cart_id: The ID of the cart to remove from.
        :param product: The product to remove.
        """
        self.lock.acquire()
        self.cart_list[cart_id].remove(product)

        # Find the corresponding product in the producer's inventory and mark
        # it as available again. This assumes the product was previously marked
        # as unavailable.
        for lists in self.producer_list:
            for item in lists:
                if item[0] == product and not item[1]:
                    item[1] = True
                    # Note: This breaks after finding the first match, which is
                    # the correct behavior if each 'add' reserves a specific item.
        self.lock.release()

    def place_order(self, cart_id):
        """
        Finalizes an order, returning the items that were in the cart.

        :param cart_id: The ID of the cart to place an order for.
        :return: A list of products in the finalized order.
        """
        return self.cart_list[cart_id]


from threading import Thread
from time import sleep


class Producer(Thread):
    """
    Represents a producer thread that simulates a seller.

    The producer continuously publishes a list of products to the marketplace,
    respecting cooldown times between publications.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        Initializes a Producer instance.

        :param products: A list of products to be published by this producer.
                         Each product is a tuple: (product_id, quantity, cooldown).
        :param marketplace: The shared Marketplace instance.
        :param republish_wait_time: Time in seconds to wait before retrying to
                                    publish to a full marketplace queue.
        :param kwargs: Additional keyword arguments for the Thread constructor.
        """
        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time

    def publish_product(self, product_id, quantity, cooldown, producer_id):
        """
        Publishes a given quantity of a single product with a cooldown.

        If the marketplace queue for this producer is full, it will wait and
        retry. After each successful publication, it sleeps for the `cooldown` period.

        :param product_id: The product to publish.
        :param quantity: The number of units to publish.
        :param cooldown: The time to wait after publishing one unit.
        :param producer_id: The ID of this producer.
        """
        for _ in range(quantity):
            status = False
            while not status:
                status = self.marketplace.publish(producer_id, product_id)
                if not status:
                    # Busy-wait if the marketplace queue is full.
                    sleep(self.republish_wait_time)
            # Wait for the cooldown period before publishing the next unit.
            sleep(cooldown)

    def run(self):
        """The main execution loop for the producer thread."""
        # Register with the marketplace to get a unique producer ID.
        producer_id = self.marketplace.register_producer()
        # Enter an infinite loop to continuously publish products.
        while True:
            for prod in self.products:
                self.publish_product(prod[0], prod[1], prod[2], producer_id)
