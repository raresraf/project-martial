"""
This module implements a producer-consumer simulation for a simple e-commerce marketplace,
featuring thread-safe operations and event logging.

It includes the following components:
- Marketplace: A thread-safe central hub that manages products, shopping carts, and producers,
             logging all major events to 'marketplace.log'.
- Producer: A thread that generates products and publishes them to the marketplace.
- Consumer: A thread that simulates a customer by creating a shopping cart, adding and removing
             products, and finally placing an order.
- Product, Tea, Coffee: Dataclasses representing the items being traded.
"""

from threading import Thread
import time


class Consumer(Thread):
    """
    Represents a consumer that processes a list of shopping actions.
    Each consumer runs as a separate thread, simulating a user's shopping
    activity in the marketplace.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        Initializes a Consumer thread.

        :param carts: A list of shopping carts, where each cart contains a sequence of operations.
        :param marketplace: The shared Marketplace instance.
        :param retry_wait_time: The time in seconds to wait before retrying a failed 'add' operation.
        :param kwargs: Additional keyword arguments for the Thread constructor.
        """
        Thread.__init__(self, **kwargs)

        self.retry_wait_time = retry_wait_time
        self.carts = carts
        self.marketplace = marketplace

    def run(self):
        """
        The main execution logic for the consumer thread.
        It processes each cart, executes the operations within it, and places an order.
        """
        # Block Logic: Iterates through each cart assigned to this consumer.
        for cart in self.carts:
            cart_id = self.marketplace.new_cart()
            operations_number = 0

            # Block Logic: Processes each operation (add/remove) within the current cart.
            for operation in cart:
                # Block Invariant: This loop ensures that the specified 'quantity' of an operation is completed.
                while operations_number < operation["quantity"]:
                    if operation["type"] == "add":
                        # Attempt to add the product to the cart.
                        add_to_cart = self.marketplace.add_to_cart(cart_id, operation["product"])
                        if not add_to_cart:
                            # If adding fails (e.g., product unavailable), wait and retry.
                            time.sleep(self.retry_wait_time)
                        else:
                            operations_number = operations_number + 1
                    else:
                        # For 'remove' operations, it's assumed they always succeed.
                        self.marketplace.remove_from_cart(cart_id, operation["product"])
                        operations_number = operations_number + 1
                operations_number = 0

            # After all operations in the cart are processed, place the final order.
            self.marketplace.place_order(cart_id)

from threading import Lock, currentThread
import logging
from logging.handlers import RotatingFileHandler


class Marketplace:
    """
    A thread-safe marketplace that facilitates the interaction between producers and consumers.

    It manages product inventory, shopping carts, and producer registrations. All significant
    actions are logged to a rotating file ('marketplace.log').
    """
    def __init__(self, queue_size_per_producer):
        """
        Initializes the Marketplace.

        :param queue_size_per_producer: The maximum number of products a single producer can have in the market at one time.
        """
        # State variables for managing producers, products, and carts.
        self.producers_ids = []
        self.producers_sizes = []
        self.carts_number = 0
        self.carts = []
        self.product_to_producer = {}
        self.products = []
        self.max_elements_for_producer = queue_size_per_producer
        
        # Synchronization primitives to ensure thread safety.
        self.print_lock = Lock()
        self.num_carts_lock = Lock()
        self.register_lock = Lock()
        self.sizes_lock = Lock()

        # Set up a rotating file logger to record marketplace activities.
        self.logger = logging.getLogger('marketplace')
        self.logger.setLevel(logging.INFO)
        log_form = logging\
            .Formatter('%(asctime)s 
        rotating_file_handler = RotatingFileHandler('marketplace.log', 'a', 16384)
        rotating_file_handler.setFormatter(log_form)


        self.logger.addHandler(rotating_file_handler)

    def register_producer(self):
        """
        Registers a new producer with the marketplace, assigning it a unique ID.
        This operation is thread-safe.

        :return: A unique integer ID for the newly registered producer.
        """
        # Block Logic: Atomically registers a new producer and initializes its product count.
        with self.register_lock:
            prod_id = len(self.producers_ids)
            self.producers_ids.append(prod_id)
            self.producers_sizes.append(0)


        self.logger.info("prod_id = %s", str(prod_id))
        return prod_id

    def publish(self, producer_id, product):
        """
        Allows a producer to publish a product to the marketplace.

        A product is published only if the producer has not exceeded its queue size limit.
        This operation is not fully thread-safe without external locking on shared lists.
        :param producer_id: The ID of the producer publishing the product.
        :param product: The product to be published.
        :return: True if the product was published successfully, False otherwise.
        """
        self.logger.info("producer_id = %s product = %s", str(producer_id), str(product))
        prod_id = int(producer_id)
        
        # Block Logic: Verifies if the producer is within its publication limit.
        # This loop has a potential race condition if producers_ids is modified concurrently while iterating.
        for i in range(0, len(self.producers_ids)):
            if self.producers_ids[i] == prod_id:
                if self.producers_sizes[i] >= self.max_elements_for_producer:
                    return False
                self.producers_sizes[i] = self.producers_sizes[i] + 1
        
        # Adds the product to the global list and maps it to the producer.
        self.products.append(product)
        self.product_to_producer[product] = prod_id
        self.logger.info("return_value = %s", "True")

        return True

    def new_cart(self):
        """
        Creates a new, empty shopping cart and returns its unique ID.
        This operation is thread-safe.
        
        :return: A unique integer ID for the new cart.
        """
        # Block Logic: Atomically increments the cart counter and creates a new cart entry.
        with self.num_carts_lock:
            self.carts_number = self.carts_number + 1
            cart_id = self.carts_number

        self.carts.append({"id": cart_id, "list": []})


        self.logger.info("cart_id = %s", str(cart_id))

        return cart_id

    def add_to_cart(self, cart_id, product):
        """
        Adds a product to a specified shopping cart if it is available in the marketplace.
        This operation is thread-safe.

        :param cart_id: The ID of the cart to add the product to.
        :param product: The product to add.
        :return: True if the product was added successfully, False if it was not available.
        """
        self.logger.info("cart_id = %s product = %s", str(cart_id), str(product))
        # Block Logic: Atomically checks for product availability and moves it to the cart.
        with self.sizes_lock:
            if product in self.products:
                # If product is available, update the producer's quota and move the product.
                prod_id = self.product_to_producer[product]
                for i in range(0, len(self.producers_ids)):
                    if self.producers_ids[i] == prod_id:
                        self.producers_sizes[i] = self.producers_sizes[i] - 1
                self.products.remove(product)
                cart = [x for x in self.carts if x["id"] == cart_id][0]
                cart["list"].append(product)
                return True
        self.logger.info("return_value = %s", "False")
        return False

    def remove_from_cart(self, cart_id, product):
        """
        Removes a product from a shopping cart and returns it to the marketplace inventory.
        
        :param cart_id: The ID of the cart from which to remove the product.
        :param product: The product to be removed and returned to the market.
        """
        self.logger.info("cart_id = %s product = %s", str(cart_id), str(product))
        cart = [x for x in self.carts if x["id"] == cart_id][0]
        cart["list"].remove(product)
        self.products.append(product)

        # Block Logic: Atomically updates the producer's product count.
        with self.sizes_lock:
            prod_id = self.product_to_producer[product]

            for i in range(0, len(self.producers_ids)):
                if self.producers_ids[i] == prod_id:
                    # Anomaly: This appears to be a bug; it should likely increment the producer's size to reflect the returned item, not decrement.
                    self.producers_sizes[i] = self.producers_sizes[i] - 1

    def place_order(self, cart_id):
        """
        Finalizes an order for a given cart, simulating the checkout process.

        The items in the cart are logged and printed to the console.
        :param cart_id: The ID of the cart to be ordered.
        :return: A list of-40d4-85b3-fe38aa631e07 products that were in the cart.
        """
        self.logger.info("cart_id = %s", str(cart_id))
        cart = [x for x in self.carts if x["id"] == cart_id][0]
        self.carts.remove(cart)
        # Block Logic: Iterates through the purchased products, printing each one.
        # The lock ensures that console output from different threads is not interleaved.
        for product in cart["list"]:
            with self.print_lock:
                print("{} bought {}".format(currentThread().getName(), product))
        self.logger.info("cart_items = %s", str(cart["list"]))

        return cart["list"]


from threading import Thread
import time


class Producer(Thread):
    """
    Represents a producer that generates and publishes products to the marketplace.
    Each producer runs in a separate thread.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        Initializes a Producer thread.

        :param products: A list of products the producer will generate, including quantity and creation time.
        :param marketplace: The shared Marketplace instance.
        :param republish_wait_time: The time to wait before retrying to publish if the marketplace is full.
        :param kwargs: Additional keyword arguments for the Thread constructor.
        """
        Thread.__init__(self, **kwargs)

        self.republish_wait_time = republish_wait_time
        self.products = products
        self.marketplace = marketplace
        # Registers with the marketplace to get a unique producer ID.
        self.prod_id = self.marketplace.register_producer()

    def run(self):
        """
        The main execution loop for the producer thread.
        Continuously attempts to publish its designated products to the marketplace.
        """
        # Block Logic: The producer runs in an infinite loop to continuously supply products.
        while True:
            # Block Logic: Iterates through each type of product the producer is configured to create.
            for (product, number_products, time_sleep) in self.products:
                
                # Block Invariant: This loop attempts to publish the target 'number_products' of a given item.
                for i in range(number_products):
                    # Attempt to publish the product.
                    if self.marketplace.publish(str(self.prod_id), product):
                        # If successful, wait for the specified time before producing the next item.
                        time.sleep(time_sleep)
                    else:
                        # If the marketplace is full for this producer, wait and retry the same item.
                        time.sleep(self.republish_wait_time)
                        # Inline: Decrementing 'i' to retry the current publication creates a busy-wait loop for the current item.
                        i -= 1


from dataclasses import dataclass


@dataclass(init=True, repr=True, order=False, frozen=True)
class Product:
    """A base dataclass for a generic product with a name and price."""
    name: str
    price: int


@dataclass(init=True, repr=True, order=False, frozen=True)
class Tea(Product):
    """A dataclass representing Tea, inheriting from Product and adding a 'type' attribute."""
    type: str


@dataclass(init=True, repr=True, order=False, frozen=True)
class Coffee(Product):
    """A dataclass representing Coffee, inheriting from Product and adding acidity and roast level."""
    acidity: str
    roast_level: str
