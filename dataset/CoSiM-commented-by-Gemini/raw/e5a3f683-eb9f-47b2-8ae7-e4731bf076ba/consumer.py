"""
This module implements a producer-consumer simulation for a simple e-commerce marketplace.

It includes the following components:
- Marketplace: A thread-safe central hub that manages products and shopping carts.
- Producer: A thread that generates products and publishes them to the marketplace.
- Consumer: A thread that simulates a customer by creating a shopping cart, adding and removing
             products, and finally placing an order.
- Product, Tea, Coffee: Dataclasses representing the items being traded.
"""

from threading import Thread
from time import sleep


class Consumer(Thread):
    """
    Represents a consumer that processes a list of shopping carts.

    Each consumer is a thread that simulates a user's shopping activity,
    including creating a cart, adding/removing items, and placing an order.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        Initializes a Consumer thread.

        :param carts: A list of carts, where each cart is a list of actions (add/remove).
        :param marketplace: The central marketplace instance.
        :param retry_wait_time: Time to wait before retrying an operation (e.g., add to cart).
        :param kwargs: Additional keyword arguments for the Thread constructor.
        """
        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        # Functional Utility: Binds marketplace methods to the consumer instance for convenience.
        self.new_cart = self.marketplace.new_cart
        self.add_to_cart = self.marketplace.add_to_cart
        self.remove_from_cart = self.marketplace.remove_from_cart
        self.place_order = self.marketplace.place_order


    def run(self):
        """
        The main execution loop for the consumer thread.

        Iterates through the assigned carts and processes the actions in each cart.
        """
        # Block Logic: Process each shopping cart assigned to this consumer.
        for current_cart in self.carts:

            # Creates a new empty cart in the marketplace for the current shopping session.
            id_cart = self.new_cart()

            # Block Logic: Process each action (add/remove) within the current cart.
            for cart in current_cart:

                quantity = cart['quantity']
                product = cart['product']
                op_type = cart['type']
                step = 1

                # Block Logic: Repeats the add/remove operation for the specified quantity.
                # Invariant: The loop continues until the desired quantity of the product
                #            is successfully added or removed.
                while step <= quantity:
                    success = False
                    # Block Logic: Determines whether to add or remove a product based on the operation type.
                    if op_type == "add":
                        success = self.add_to_cart(id_cart, product)
                    elif op_type == "remove":
                        success = self.remove_from_cart(id_cart, product)

                    # If the operation was successful (or if the marketplace indicates it should be ignored),
                    # move to the next item in the quantity.
                    if success is None or success:
                        step += 1
                        continue

                    # If the operation failed (e.g., product not available), wait before retrying.
                    sleep(self.retry_wait_time)

            # Finalizes the shopping session by placing the order for the current cart.
            self.place_order(id_cart)

from __future__ import print_function
from threading import Lock, currentThread


class Marketplace:
    """
    A thread-safe marketplace that facilitates the interaction between producers and consumers.

    It manages the inventory of products, active shopping carts, and ensures that
    operations are synchronized to prevent race conditions.
    """
    def __init__(self, queue_size_per_producer):
        """
        Initializes the Marketplace.

        :param queue_size_per_producer: The maximum number of products a single producer can have in the market at one time.
        """
        self.queue_size_per_producer = queue_size_per_producer

        # Synchronization Primitives: Locks to protect shared data structures.
        self.lock_register_prod = Lock() # Protects producer registration (producer_index, prod_sizes).
        self.lock_q_size = Lock()        # Protects access to product lists and producer size quotas.
        self.lock_carts = Lock()         # Protects cart creation and access to the main cart dictionary.
        self.lock_printing = Lock()      # Ensures that print statements from different threads are not interleaved.

        # State Variables
        self.carts_total = 0             # A counter to generate unique cart IDs.
        self.producer_index = 0          # A counter to generate unique producer IDs.
        self.prod_sizes = []             # A list to track the number of products each producer has on the market.
        self.all_products = []           # A global list of all available products.
        self.all_producers = {}          # A dictionary mapping a product to its producer ID.
        self.all_carts = {}              # A dictionary storing the contents of active shopping carts.


    def register_producer(self):
        """
        Registers a new producer with the marketplace, providing a unique ID.
        
        :return: A unique integer ID for the new producer.
        """
        # Block Logic: Atomically registers a new producer.
        self.lock_register_prod.acquire()
        self.producer_index += 1
        self.prod_sizes.append(0)
        self.lock_register_prod.release()

        return self.producer_index - 1

    def publish(self, producer_id, product):
        """
        Allows a producer to publish a product to the marketplace.

        The product is only published if the producer has not exceeded its queue size limit.
        :param producer_id: The ID of the producer publishing the product.
        :param product: The product to be published.
        :return: True if the product was published successfully, False otherwise.
        """
        id_prod = int(producer_id)

        # Block Logic: Enforces the producer's queue size limit.
        # Pre-condition: Checks if the producer has space to publish a new product.
        if self.queue_size_per_producer > self.prod_sizes[id_prod]:

            self.all_products.append(product) 
            self.all_producers[product] = id_prod
            self.prod_sizes[id_prod] += 1
            return True

        return False

    def new_cart(self):
        """
        Creates a new, empty shopping cart and returns its unique ID.
        
        :return: A unique integer ID for the new cart.
        """
        # Block Logic: Atomically creates a new cart.
        self.lock_carts.acquire()

        self.carts_total += 1
        id_cart = self.carts_total

        self.all_carts[id_cart] = []

        self.lock_carts.release()

        return id_cart


    def add_to_cart(self, cart_id, product):
        """
        Adds a product to a specified shopping cart.

        This involves removing the product from the global product list and decrementing
        the producer's product count.
        :param cart_id: The ID of the cart to add the product to.
        :param product: The product to add.
        :return: True if the product was added successfully, False if the product was not available.
        """
        # Block Logic: Atomically checks for product availability and updates producer quotas.
        self.lock_q_size.acquire()

        if product in self.all_products:
            ignore = False
        else:
            ignore = True

        if ignore is False:
            # The product is available; remove it from the market and update the producer's count.
            self.all_products.remove(product)
            self.prod_sizes[self.all_producers[product]] -= 1

        self.lock_q_size.release()

        # If the product was not available, the operation fails.
        if ignore is True:
            return False

        # Add the product to the specified cart.
        self.all_carts[cart_id].append(product) 

        return True

    def remove_from_cart(self, cart_id, product):
        """
        Removes a product from a shopping cart, returning it to the marketplace.
        
        This is effectively the reverse of add_to_cart.
        :param cart_id: The ID of the cart from which to remove the product.
        :param product: The product to remove.
        """
        # Block Logic: Atomically returns the product to the marketplace and updates producer quotas.
        self.lock_q_size.acquire()
        index = self.all_producers[product]
        self.prod_sizes[index] += 1
        self.lock_q_size.release()

        # Remove the product from the cart and add it back to the global product list.
        self.all_carts[cart_id].remove(product)
        self.all_products.append(product)


    def place_order(self, cart_id):
        """
        Finalizes an order for a given cart.

        This simulates the "checkout" process, printing each product that was "bought".
        :param cart_id: The ID of the cart for which to place the order.
        :return: A list of products that were in the cart.
        """
        # Removes the cart from the active list, returning its contents.
        prods = self.all_carts.pop(cart_id)

        # Block Logic: Iterate through the purchased products and print them.
        # The lock ensures that output from different consumer threads is not mixed.
        for prod in prods:
            self.lock_printing.acquire()
            thread_name = currentThread().getName()
            print('{0} bought {1}'.format(thread_name, prod))
            self.lock_printing.release()

        return prods


from threading import Thread
from time import sleep


class Producer(Thread):
    """
    Represents a producer that generates and publishes products to the marketplace.

    Each producer is a thread that continuously tries to publish its assigned products.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        Initializes a Producer thread.

        :param products: A list of products that this producer is responsible for.
        :param marketplace: The central marketplace instance.
        :param republish_wait_time: Time to wait before retrying to publish a product if the queue is full.
        :param kwargs: Additional keyword arguments for the Thread constructor.
        """
        Thread.__init__(self, **kwargs)

        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time

        # Registers this producer with the marketplace to get a unique ID.
        self.producer_id = self.marketplace.register_producer()

    def run(self):
        """
        The main execution loop for the producer thread.

        Continuously attempts to publish products to the marketplace.
        """
        # This loop runs indefinitely, making the producer a long-running service.
        while True:
            # Block Logic: Iterates through each type of product the producer can create.
            for current_product in self.products:
                step = 0
                product = current_product[0]
                products_no = current_product[1]
                waiting_time = current_product[2]

                # Block Logic: Publishes a specific product 'products_no' times.
                # Invariant: Loop continues until the target number of products of this type have been published.
                while True:
                    published = self.marketplace.publish(str(self.producer_id), product)
                    
                    # If publication was successful, wait for a bit before publishing the next unit.
                    if published is True:
                        step += 1
                        sleep(waiting_time)
                    else:
                        # If publication failed (queue full), wait before retrying.
                        sleep(self.republish_wait_time)

                    # Once the desired number of this product has been published, break to the next product type.
                    if step == products_no:
                        break


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
