"""
This module implements a multi-threaded producer-consumer simulation for a marketplace.
It defines classes for Consumer, Marketplace, Producer, and product types (Product, Tea, Coffee).
The simulation involves producers adding products to a shared marketplace and consumers
retrieving them to place orders, demonstrating concurrent access and resource management.
"""


from threading import Thread
import time

ADD_COMMAND = "add"
REMOVE_COMMAND = "remove"
COMMAND_TYPE = "type"
ITEM_QUANTITY = "quantity"
PRODUCT = "product"
NAME = "name"

class Consumer(Thread):
    """
    The Consumer class represents a buyer in the marketplace.
    Each consumer runs as a separate thread, simulating the process of
    adding and removing items from a cart, and finally placing an order.
    Consumers handle retry logic for adding items to the cart if the marketplace
    is temporarily unable to fulfill the request.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        Initializes a Consumer thread.

        :param carts: A list of carts, where each cart is a list of commands (add/remove product).
                      Each command specifies the type, quantity, and product.
        :param marketplace: The shared marketplace instance to interact with.
        :param retry_wait_time: The time in seconds to wait before retrying an operation
                                 (e.g., adding a product to a full cart).
        :param kwargs: Additional keyword arguments passed to the Thread constructor,
                       e.g., 'name' for the consumer's identifier.
        """
        Thread.__init__(self, **kwargs)

        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.consumer_name = kwargs[NAME]

    
    def run(self):
        """
        Executes the consumer's main logic.
        This method is called when the thread starts. It simulates the consumer's
        journey through the marketplace: creating a new cart, processing a series
        of add/remove commands for products, and finally placing the order.
        It handles retries for 'add' operations if the marketplace's cart
        addition fails.
        """
        id_cart = self.marketplace.new_cart()

        # Block Logic: Iterates through each predefined cart for this consumer.
        # Each 'cart' here represents a sequence of operations (add/remove products).
        for item in self.carts:
            # Block Logic: Processes each command within the current cart's sequence.
            # Commands dictate whether to add or remove products from the shopping cart.
            for command in item:

                # Conditional Logic: Checks if the current command is to add a product.
                if command[COMMAND_TYPE] == ADD_COMMAND:

                    # Block Logic: Attempts to add the specified quantity of a product to the cart.
                    # This loop ensures that all instances of a product are added,
                    # retrying if the marketplace temporarily fails to add the product.
                    for _ in range(command[ITEM_QUANTITY]):
                        # Invariant: Continues to loop until the product is successfully added to the cart.
                        # Pre-condition: The marketplace's add_to_cart method might return False, indicating failure.
                        while self.marketplace.add_to_cart(id_cart, command[PRODUCT]) is False:
                            time.sleep(self.retry_wait_time)

                # Conditional Logic: Checks if the current command is to remove a product.
                elif command[COMMAND_TYPE] == REMOVE_COMMAND:

                    # Block Logic: Attempts to remove the specified quantity of a product from the cart.
                    # This loop ensures that all instances of a product are removed.
                    for _ in range(command[ITEM_QUANTITY]):
                        self.marketplace.remove_from_cart(id_cart, command[PRODUCT])

        # Functional Utility: Places the final order with all accumulated items in the cart.
        order_result = self.marketplace.place_order(id_cart)

        # Block Logic: Iterates through the items successfully ordered and prints a confirmation message.
        for item in order_result:
            # Synchronization: Acquires a lock to ensure exclusive access to the print statement
            # to prevent interleaved output from multiple consumers.
            self.marketplace.lock.acquire()
            print(self.consumer_name + " bought " + str(item[1]))
            # Synchronization: Releases the lock after printing.
            self.marketplace.lock.release()






from logging.handlers import RotatingFileHandler
from threading import Lock
import logging

class Marketplace:
    """
    The Marketplace class simulates a central hub where producers publish products
    and consumers can add/remove products from their carts to place orders.
    It manages product inventory, producer and consumer registration, and cart operations,
    ensuring thread-safe access to shared resources using a lock.
    It also logs significant operations to a rotating file.
    """

    
    def __init__(self, queue_size_per_producer):
        """
        Initializes the Marketplace with a specified queue size per producer.

        :param queue_size_per_producer: The maximum number of products a single producer
                                        can have available in the marketplace at any given time.
        """

        self.queue_size_per_producer = queue_size_per_producer

        self.producer_id = 0 # Initializes a counter for unique producer IDs.
        self.consumer_id = 0 # Initializes a counter for unique consumer IDs.

        self.products = [] # Stores all products currently available in the marketplace. Format: [(producer_id, product_object)]
        self.producers = [] # Stores references to registered Producer objects.
        self.carts = [] # Stores consumer carts. Each cart is a list of product items. Format: [[(producer_id, product_object)]]

        self.lock = Lock() # A threading lock to ensure thread-safe access to shared resources.

        # Functional Utility: Sets up logging for the Marketplace operations.
        # Logs are written to 'marketplace.log' with a rotating file handler.
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        file_handler = RotatingFileHandler("marketplace.log")
        self.logger.addHandler(file_handler)

    
    def register_producer(self):
        """
        Registers a new producer with the marketplace and assigns a unique ID.
        This method is thread-safe.

        :return: A unique integer ID for the newly registered producer.
        """

        self.logger.info("Entered method: register_producer")
        # Synchronization: Acquires a lock to safely increment producer_id.
        self.lock.acquire()

        self.producer_id += 1
        producer_id = self.producer_id

        self.lock.release() # Synchronization: Releases the lock.

        self.logger.info("Exited method: register_producer")
        return producer_id


    
    def publish(self, producer_id, product):
        """
        Publishes a product from a given producer to the marketplace.
        The product is added only if the producer has not exceeded its maximum
        allowed products in the marketplace. This method is thread-safe.

        :param producer_id: The ID of the producer publishing the product.
        :param product: The product object to be published.
        :return: True if the product was successfully published, False otherwise.
        """

        self.logger.info("Entered method: publish")
        self.logger.info("Params: producer_id: " + str(producer_id)
        + ", product: " + str(product.name))
        # Synchronization: Acquires a lock to safely modify shared product lists and producer counts.
        self.lock.acquire()

        # Conditional Logic: Checks if the producer has reached its product limit.
        if self.producers[producer_id - 1].nr_products < self.queue_size_per_producer:

            self.products.append((producer_id, product)) # Adds the product to the marketplace's inventory.
            self.producers[producer_id - 1].nr_products += 1 # Increments the producer's active product count.
            self.lock.release() # Synchronization: Releases the lock.
            self.logger.info("Exited method: publish")
            return True

        self.lock.release() # Synchronization: Releases the lock.
        self.logger.info("Exited method: publish")
        return False

    
    def new_cart(self):
        """
        Creates a new, empty shopping cart for a consumer and assigns a unique ID.
        This method is thread-safe.

        :return: A unique integer ID for the new cart.
        """

        self.logger.info("Entered method: new_cart")
        # Synchronization: Acquires a lock to safely increment consumer_id and add a new cart.
        self.lock.acquire()

        self.consumer_id += 1
        consumer_id = self.consumer_id

        self.carts.append([]) # Adds a new empty list to represent the new cart.

        self.lock.release() # Synchronization: Releases the lock.

        self.logger.info("Exited method: new_cart")
        return consumer_id

    
    def add_to_cart(self, cart_id, product):
        """
        Adds a specified product to a consumer's cart from the marketplace's available products.
        This operation is atomic; if the product is found, it's moved from the marketplace
        to the cart. This method is thread-safe.

        :param cart_id: The ID of the consumer's cart.
        :param product: The product object to add to the cart.
        :return: True if the product was successfully added, False if not found or unavailable.
        """

        self.logger.info("Entered method: add_to_cart")
        self.logger.info("Params: cart_id: " + str(cart_id) + ", product: " + str(product.name))
        # Synchronization: Acquires a lock to safely modify shared product lists and cart contents.
        self.lock.acquire()
        # Block Logic: Iterates through the currently available products in the marketplace.
        for item in self.products:
            # Conditional Logic: Checks if the desired product matches an available item.
            if product == item[1]:

                self.carts[cart_id - 1].append(item) # Adds the product to the consumer's cart.
                self.products.remove(item) # Removes the product from the marketplace's inventory.
                self.producers[item[0] - 1].nr_products -= 1 # Decrements the producer's active product count.
                self.lock.release() # Synchronization: Releases the lock.
                self.logger.info("Exited method: add_to_cart")
                return True

        self.lock.release() # Synchronization: Releases the lock.
        self.logger.info("Exited method: add_to_cart")
        return False


    
    def remove_from_cart(self, cart_id, product):
        """
        Removes a specified product from a consumer's cart and returns it to the marketplace.
        This method is thread-safe.

        :param cart_id: The ID of the consumer's cart.
        :param product: The product object to remove from the cart.
        """

        self.logger.info("Entered method: remove_from_cart")
        self.logger.info("Params: cart_id: " + str(cart_id) + ", product: " + str(product.name))
        # Synchronization: Acquires a lock to safely modify shared product lists and cart contents.
        self.lock.acquire()
        # Block Logic: Iterates through the items currently in the consumer's cart.
        for item in self.carts[cart_id - 1]:
            # Conditional Logic: Checks if the desired product matches an item in the cart.
            if product == item[1]:

                self.carts[cart_id - 1].remove(item) # Removes the product from the consumer's cart.
                self.products.append(item) # Returns the product to the marketplace's inventory.
                self.producers[item[0] - 1].nr_products += 1 # Increments the producer's active product count.
                self.lock.release() # Synchronization: Releases the lock.
                self.logger.info("Exited method: remove_from_cart")
                return

        self.lock.release() # Synchronization: Releases the lock.
        self.logger.info("Exited method: remove_from_cart")

    
    def place_order(self, cart_id):
        """
        Finalizes the order for a given cart.
        In this simulation, placing an order simply means returning the contents
        of the specified cart. No further processing (e.g., payment, shipping)
        is simulated.

        :param cart_id: The ID of the cart to place the order for.
        :return: A list of product items in the placed order.
        """
        self.logger.info("Entered method: place_order")
        self.logger.info("Exited method: place_order")
        return self.carts[cart_id - 1]



from threading import Thread
import time


class Producer(Thread):
    """
    The Producer class represents a seller in the marketplace.
    Each producer runs as a separate thread, continuously publishing products
    to the marketplace based on its predefined inventory and timing.
    Producers will retry publishing if the marketplace's capacity for them is full.
    """

    
    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        Initializes a Producer thread.

        :param products: A list of products this producer will offer. Each item is a tuple
                         (product_object, quantity, time_to_republish).
        :param marketplace: The shared marketplace instance to interact with.
        :param republish_wait_time: The time in seconds to wait before retrying to publish
                                    a product if the marketplace is full for this producer.
        :param kwargs: Additional keyword arguments passed to the Thread constructor.
        """

        Thread.__init__(self, **kwargs)

        self.nr_products = 0 # Tracks the number of products currently published by this producer in the marketplace.
        self.products = products # The inventory of products this producer has.
        self.marketplace = marketplace # Reference to the shared marketplace.
        self.republish_wait_time = republish_wait_time # Time to wait before retrying to publish.

        # Functional Utility: Adds this producer instance to the marketplace's list of producers.
        self.marketplace.producers.append(self)
        # Functional Utility: Registers this producer with the marketplace to get a unique ID.
        self.producer_id = self.marketplace.register_producer()

    
    def run(self):
        """
        Executes the producer's main logic.
        This method is called when the thread starts. It continuously iterates
        through its product list, attempting to publish each product to the marketplace.
        It includes retry logic if the marketplace refuses publication (e.g., due to capacity limits).
        """
        # Invariant: The producer continuously attempts to publish products.
        while True:
            # Block Logic: Iterates through each type of product in the producer's inventory.
            for item in self.products:
                # Block Logic: Publishes the specified quantity of the current product type.
                for _ in range(item[1]):

                    # Invariant: Continues to loop until the product is successfully published.
                    # Pre-condition: The marketplace's publish method might return False, indicating failure.
                    while self.marketplace.publish(self.producer_id, item[0]) is False:
                        time.sleep(self.republish_wait_time)

                    # Functional Utility: Pauses for a specified time after publishing a product,
                    # simulating production time or inventory replenishment.
                    time.sleep(item[2])


from dataclasses import dataclass


@dataclass(init=True, repr=True, order=False, frozen=True)
class Product:
    """
    Represents a generic product with a name and price.
    This is a frozen dataclass, meaning its instances are immutable.
    """
    name: str
    price: int


@dataclass(init=True, repr=True, order=False, frozen=True)
class Tea(Product):
    """
    Represents a specific type of product: Tea.
    It inherits from Product and adds a 'type' attribute.
    This is a frozen dataclass, meaning its instances are immutable.
    """
    type: str


@dataclass(init=True, repr=True, order=False, frozen=True)
class Coffee(Product):
    """
    Represents a specific type of product: Coffee.
    It inherits from Product and adds 'acidity' and 'roast_level' attributes.
    This is a frozen dataclass, meaning its instances are immutable.
    """
    acidity: str
    roast_level: str
