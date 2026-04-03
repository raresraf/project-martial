"""
@8a9c36ec-36af-4dff-8e02-452646b6e1d9/consumer.py
@brief This script simulates a marketplace with producers and consumers.
It defines classes for managing products, carts, and the marketplace logic,
including thread-safe operations. This version includes implementations
for Consumer, Marketplace, Producer, and Product related classes.
Domain: Concurrency, Object-Oriented Programming, Simulation.
"""

from threading import Thread, Lock
from time import sleep
from dataclasses import dataclass

class Consumer(Thread):
    """
    @brief Represents a consumer agent in the marketplace simulation.
    Consumers create carts, add/remove products, and place orders.
    Algorithm: Iterative cart operation and order placement.
    Time Complexity: Depends on the number of carts and operations per cart.
    Space Complexity: O(1) for consumer state, O(N) for cart contents within the marketplace.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        @brief Initializes a Consumer thread.
        @param carts: A list of cart operations (add/remove product, quantity).
        @param marketplace: The marketplace instance to interact with.
        @param retry_wait_time: Time to wait before retrying an operation.
        @param kwargs: Additional keyword arguments, including 'name' for the consumer.
        """
        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.name = kwargs["name"]

    def run(self):
        """
        @brief The main execution method for the consumer thread.
        It iterates through predefined cart operations and places an order.
        """
        # Block Logic: Process each predefined cart for the consumer.
        # Invariant: Each cart is processed sequentially.
        for cart in self.carts:
            # Pre-condition: A new cart is created in the marketplace.
            cart_id = self.marketplace.new_cart()
            # Block Logic: Execute each operation within the current cart.
            # Invariant: Operations are processed as defined in the 'cart' list.
            for opp in cart:
                # Block Logic: Perform the specified quantity of add or remove operations.
                # Invariant: The loop runs 'quantity' times for each operation.
                for i in range(0, opp["quantity"]):
                    # Block Logic: Handle 'add' operations.
                    # Pre-condition: Product must be available in the marketplace.
                    # Invariant: Retries adding the product until successful.
                    if opp["type"] == "add":
                        while self.marketplace.add_to_cart(cart_id, opp["product"]) == False:
                            sleep(self.retry_wait_time)
                    # Block Logic: Handle 'remove' operations.
                    # Pre-condition: Product must be in the cart to be removed.
                    elif opp["type"] == "remove":
                        self.marketplace.remove_from_cart(cart_id, opp["product"])

            # Post-condition: All products in the cart are ordered.
            prod_list = self.marketplace.place_order(cart_id)

            # Block Logic: Print the details of the purchased products.
            # Invariant: Each product successfully placed in the order is reported.
            for product in prod_list:
                print(str(self.name) + " bought " + str(product))


class Marketplace:
    """
    @brief Manages producers, product queues, and consumer carts in a thread-safe manner.
    It acts as the central hub for all product and order transactions.
    Domain: Concurrency, Resource Management, Producer-Consumer Pattern.
    """

    def __init__(self, queue_size_per_producer):
        """
        @brief Initializes the Marketplace.
        @param queue_size_per_producer: The maximum number of products a producer can have in its queue.
        """
        self.queue_size_per_producer = queue_size_per_producer
        # Inline: Tracks the next available producer ID.
        self.producer_id = 0
        # Inline: Tracks the next available consumer/cart ID.
        self.consumer_id = 0
        # Inline: Dictionary to store product queues for each producer, keyed by producer ID.
        self.prod_dict = {}
        # Inline: Dictionary to store carts for each consumer, keyed by cart ID.
        self.cart_dict = {}
        # Inline: Lock to ensure thread-safe operations when adding to cart.
        self.lock_add_cart = Lock()
        # Inline: Lock to ensure thread-safe operations when publishing products.
        self.lock_publish = Lock()

        pass

    def register_producer(self):
        """
        @brief Registers a new producer and assigns it a unique ID and an empty product queue.
        @return: The unique ID assigned to the new producer.
        """
        # Block Logic: Increment producer ID and initialize an empty list for its products.
        self.producer_id += 1
        self.prod_dict[self.producer_id] = []
        return self.producer_id
        pass

    def publish(self, producer_id, product):
        """
        @brief Attempts to publish a product from a producer to its queue.
        @param producer_id: The ID of the producer publishing the product.
        @param product: The product to publish.
        @return: True if the product was published, False if the queue is full.
        """
        # Block Logic: Acquire lock to ensure thread-safe access to the producer's dictionary and queue.
        self.lock_publish.acquire()
        # Pre-condition: Check if the producer's queue has space.
        if len(self.prod_dict[producer_id]) < self.queue_size_per_producer:
            self.prod_dict[producer_id].append(product)
            self.lock_publish.release()
            return True
        # Post-condition: Release lock and return False if queue is full.
        self.lock_publish.release()
        return False
        pass

    def new_cart(self):
        """
        @brief Creates a new shopping cart and assigns it a unique ID.
        @return: The unique ID assigned to the new cart.
        """
        # Block Logic: Increment consumer ID and initialize an empty list for its cart.
        self.consumer_id += 1
        self.cart_dict[self.consumer_id] = []
        return self.consumer_id
        pass

    def add_to_cart(self, cart_id, product):
        """
        @brief Adds a product to a specified cart by taking it from a producer's queue.
        @param cart_id: The ID of the cart to add the product to.
        @param product: The product to add.
        @return: True if the product was successfully added, False otherwise.
        """
        # Block Logic: Acquire lock to ensure thread-safe operations on carts and producer queues.
        self.lock_add_cart.acquire()

        # Block Logic: Iterate through all producer queues to find the product.
        # Invariant: The loop continues until the product is found and added or all queues are checked.
        for prod_id in self.prod_dict.keys():
            # Block Logic: Iterate through products in the current producer's queue.
            for p in self.prod_dict[prod_id]:
                # Pre-condition: If the product is found in a producer's queue.
                if p == product:
                    self.prod_dict[prod_id].remove(product)
                    # Inline: Store the product along with the producer ID for later return to inventory.
                    self.cart_dict[cart_id].append([product, prod_id])
                    self.lock_add_cart.release()
                    return True
        # Post-condition: Release lock and return False if product not found.
        self.lock_add_cart.release()
        return False
        pass

    def remove_from_cart(self, cart_id, product):
        """
        @brief Removes a product from a specified cart and returns it to its originating producer's queue.
        @param cart_id: The ID of the cart to remove the product from.
        @param product: The product to remove.
        """
        # Block Logic: Iterate through the items in the specified cart.
        # Invariant: The loop continues until the product is found and returned to the producer.
        for prod in self.cart_dict[cart_id]:
            # Pre-condition: If the product is found in the cart.
            if prod[0] == product:
                self.cart_dict[cart_id].remove(prod)
                # Inline: Return the product to the original producer's queue.
                self.prod_dict[prod[1]].append(prod[0])
                break

    def place_order(self, cart_id):
        """
        @brief Retrieves all products from a specified cart, effectively placing an order.
        @param cart_id: The ID of the cart to place the order from.
        @return: A list of products from the cart.
        """
        prod_list = []
        # Block Logic: Collect all product items from the specified cart.
        for prod in self.cart_dict[cart_id]:
           prod_list.append(prod[0])
        return prod_list


class Producer(Thread):
    """
    @brief Represents a producer agent in the marketplace simulation.
    Producers continuously produce and publish products to the marketplace.
    Algorithm: Continuous production and publishing with retry mechanism.
    Time Complexity: Runs indefinitely, production time depends on 'production_time' and 'republish_wait_time'.
    Space Complexity: O(1) for producer state, O(N) for product list.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        @brief Initializes a Producer thread.
        @param products: A list of tuples, each containing (product, quantity, production_time).
        @param marketplace: The marketplace instance to interact with.
        @param republish_wait_time: Time to wait before retrying publishing a product.
        @param kwargs: Additional keyword arguments, including 'daemon' and 'name'.
        """
        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        self.name = kwargs

    def run(self):
        """
        @brief The main execution method for the producer thread.
        It registers with the marketplace and continuously publishes products.
        """
        # Pre-condition: Producer registers itself with the marketplace to get a unique ID.
        producer_id = self.marketplace.register_producer()
        # Block Logic: Main production loop, runs indefinitely.
        # Invariant: Producer continuously attempts to produce and publish products.
        while True:
            # Block Logic: Iterate through the predefined list of products to produce.
            # Invariant: Each product is produced for its specified quantity after a production delay.
            for product in self.products:
                # Pre-condition: Wait for the specified production time for the current product.
                sleep(product[2])
                # Block Logic: Attempt to publish the product 'quantity' times.
                # Invariant: Each product instance is published individually.
                for i in range(0, product[1]):
                    # Block Logic: Publish the product, retrying if the marketplace queue is full.
                    # Pre-condition: The producer's queue in the marketplace must not be full.
                    # Invariant: Retries publishing until successful.
                    while self.marketplace.publish(producer_id, product[0]) == False:
                        sleep(self.republish_wait_time)


@dataclass(init=True, repr=True, order=False, frozen=True)
class Product:
    """
    @brief Base class for products in the marketplace.
    Uses dataclass for automatic __init__, __repr__, etc.
    """
    name: str
    price: int


@dataclass(init=True, repr=True, order=False, frozen=True)
class Tea(Product):
    """
    @brief Represents a Tea product, inheriting from Product.
    Adds a 'type' attribute specific to tea.
    """
    type: str


@dataclass(init=True, repr=True, order=False, frozen=True)
class Coffee(Product):
    """
    @brief Represents a Coffee product, inheriting from Product.
    Adds 'acidity' and 'roast_level' attributes specific to coffee.
    """
    acidity: str
    roast_level: str
