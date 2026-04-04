

"""
@95d4cb28-35ea-4ce6-8cb7-d9603f3b895b/consumer.py
@brief Implements a multi-threaded marketplace simulation with producers and consumers.

This module defines the core components for simulating an e-commerce marketplace
where `Producer` threads publish products and `Consumer` threads add/remove
products from carts and place orders. The central `Marketplace` class manages
product inventory, carts, and thread-safe operations using `Lock` objects.
Custom `dataclass` definitions are used for `Product` and its specialized types (`Tea`, `Coffee`).

The simulation models concurrent interactions in a shared marketplace, highlighting
challenges and solutions related to resource management and synchronization in
a multi-threaded environment.

Classes:
- Consumer: Represents a customer agent that interacts with the marketplace.
- Marketplace: The central hub managing products, carts, and producer/consumer interactions.
- Producer: Represents a supplier agent that publishes products to the marketplace.
- Product: Base dataclass for all products.
- Tea, Coffee: Specialized product dataclasses.

Domain: Concurrent Programming, Producer-Consumer Problem, Multi-threading, Marketplace Simulation.
"""

from threading import Thread, currentThread
import time

class Consumer(Thread):
    """
    @brief Represents a consumer agent in the marketplace simulation.

    Each `Consumer` thread simulates a customer's shopping journey, including
    creating carts, adding/removing products, and placing orders. It interacts
    with the `Marketplace` and incorporates a retry mechanism for failed operations.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        @brief Initializes a new Consumer thread.

        @param carts: A list of shopping cart operations (e.g., add, remove) for this consumer.
        @param marketplace: A reference to the shared Marketplace instance.
        @param retry_wait_time: The time in seconds to wait before retrying a failed operation.
        @param kwargs: Additional keyword arguments for the `Thread` constructor.
        """
        Thread.__init__(self, **kwargs)
        # List of cart operations to perform.
        self.carts = carts
        # Reference to the shared marketplace instance.
        self.marketplace = marketplace
        # Time to wait before retrying an operation.
        self.retry_wait_time = retry_wait_time

    def run(self):
        """
        @brief The main execution logic for the Consumer thread.

        Pre-condition: `carts` contains a list of operations to perform.
        Invariant: The consumer attempts to process all operations in its carts,
                   retrying failed additions, and eventually places orders.
        """
        # Block Logic: Iterates through each defined shopping cart sequence for this consumer.
        for cart in self.carts:
            # Creates a new shopping cart in the marketplace and gets its unique ID.
            id_cart = self.marketplace.new_cart()
            # Block Logic: Processes each operation (add/remove product) within the current cart.
            for operation in cart:
                op_count = 0
                # Block Logic: Attempts to fulfill the desired quantity for the current operation.
                while op_count < operation['quantity']:
                    if operation['type'] == 'add':
                        # Block Logic: Attempts to add a product to the cart.
                        if self.marketplace.add_to_cart(id_cart, operation['product']) is False:
                            # Inline: If adding to cart fails (e.g., product out of stock), wait and retry.
                            time.sleep(self.retry_wait_time)        
                        else:
                            # Inline: If successful, increment the count of fulfilled operations.
                            op_count += 1
                    elif operation['type'] == 'remove':
                        # Block Logic: Attempts to remove a product from the cart.
                        self.marketplace.remove_from_cart(id_cart, operation['product'])
                        op_count += 1 # Assume remove always succeeds, so increment count.

            # Places the final order for the current cart and retrieves the list of purchased products.
            products_in_cart = self.marketplace.place_order(id_cart)
            # Block Logic: Prints a confirmation message for each product successfully bought.
            for product in products_in_cart:                               
                print(currentThread().getName() + " bought " + str(product))

from threading import Lock

class Marketplace:
    """
    @brief The central marketplace managing products, producers, consumers, and carts.

    This class handles inventory management, producer registration, product publishing,
    and all cart-related operations in a thread-safe manner using various locks.
    """
    
    def __init__(self, queue_size_per_producer):
        """
        @brief Initializes the Marketplace.

        @param queue_size_per_producer: The maximum number of products a single producer can have published in the marketplace at any time.
        """
        # Maximum number of products a single producer can have in the marketplace.
        self.queue_size_per_producer = queue_size_per_producer
        # Dictionary to store all active shopping carts, keyed by cart ID.
        self.all_carts = {}
        # Lock to protect `id_cart` when generating new cart IDs.
        self.id_carts_lock = Lock()
        # Counter for generating unique cart IDs.
        self.id_cart = -1
        # Counter for generating unique producer IDs.
        self.id_producer = -1
        # Lock to protect `id_producer` when registering new producers.
        self.id_producer_lock = Lock()
        # A global list of all products currently available in the marketplace.
        self.products_in_marketplace = []
        # Dictionary to track the number of products each producer has published.
        self.producers_queues = {}
        # Dictionary mapping producer IDs to the list of products they have published.
        self.producers_products = {}
        # Lock to protect shared resources during `add_to_cart` and `remove_from_cart` operations.
        self.add_remove_lock = Lock()

    def register_producer(self):
        """
        @brief Registers a new producer with the marketplace and assigns a unique ID.

        @return: The unique ID assigned to the new producer.
        """
        # Critical Section: Acquire lock to ensure atomic increment of `id_producer`.
        self.id_producer_lock.acquire()
        self.id_producer += 1               
        self.id_producer_lock.release()

        # Initializes tracking for the new producer's products and queue size.
        self.producers_products[self.id_producer] = []     
        self.producers_queues[self.id_producer] = 0        

        return self.id_producer

    def publish(self, producer_id, product):
        """
        @brief Attempts to publish a product from a producer to the marketplace.

        The product is published only if the producer's queue size limit is not exceeded.

        @param producer_id: The ID of the producer publishing the product.
        @param product: The product object to be published.
        @return: True if the product was published successfully, False otherwise.
        """
        # Block Logic: Checks if the producer has reached its maximum queue size for published products.
        if not self.producers_queues[int(producer_id)] < self.queue_size_per_producer:
            return False

        # If limit not exceeded, add product to marketplace and update producer's queue.
        self.producers_queues[int(producer_id)] += 1
        self.products_in_marketplace.append(product)                   
        self.producers_products[int(producer_id)].append(product)

        return True

    def new_cart(self):
        """
        @brief Creates a new empty shopping cart in the marketplace and assigns a unique ID.

        @return: The unique ID assigned to the new cart.
        """
        # Critical Section: Acquire lock to ensure atomic increment of `id_cart`.
        self.id_carts_lock.acquire()
        self.id_cart += 1                       
        self.id_carts_lock.release()
        # Initialize an empty list for the new cart.
        self.all_carts[self.id_cart] = []

        return self.id_cart

    def add_to_cart(self, cart_id, product):
        """
        @brief Attempts to add a product to a specific shopping cart.

        This operation is thread-safe. If the product is available, it's moved
        from the marketplace inventory to the cart, and producer's queue is updated.

        @param cart_id: The ID of the cart to which the product should be added.
        @param product: The product object to add.
        @return: True if the product was successfully added, False if not available.
        """
        # Critical Section: Use a reentrant lock to protect marketplace inventory and producer states.
        with self.add_remove_lock:
            # Block Logic: Check if the product is currently available in the marketplace.
            if product not in self.products_in_marketplace:
                return False

            # Remove product from global marketplace inventory.
            self.products_in_marketplace.remove(product)
            # Block Logic: Find which producer originally published this product and update its queue.
            for producer in self.producers_products:
                if product in self.producers_products[producer]:            
                    self.producers_queues[producer] -= 1                    
                    self.producers_products[producer].remove(product)       
                    break

        # Add the product to the consumer's cart.
        self.all_carts[cart_id].append(product)
        return True

    def remove_from_cart(self, cart_id, product):
        """
        @brief Removes a product from a specific shopping cart and returns it to the marketplace.

        This operation is thread-safe.

        @param cart_id: The ID of the cart from which the product should be removed.
        @param product: The product object to remove.
        """
        # Remove product from the consumer's cart.
        self.all_carts[cart_id].remove(product)

        # Critical Section: Use a reentrant lock to protect marketplace inventory and producer states.
        with self.add_remove_lock:
            # Add the product back to the global marketplace inventory.
            self.products_in_marketplace.append(product)
            # Block Logic: Find which producer originally published this product and update its queue.
            for producer in self.producers_products:
                if product in self.producers_products[producer]:        
                    self.producers_queues[producer] += 1                
                    self.producers_products[producer].append(product)   
                    break


    def place_order(self, cart_id):
        """
        @brief Finalizes the shopping cart, effectively placing an order.

        @param cart_id: The ID of the cart to finalize.
        @return: A list of products that were in the finalized cart.
        """
        return self.all_carts[cart_id]                             


from threading import Thread
import time

class Producer(Thread):
    """
    @brief Represents a producer agent in the marketplace simulation.

    Each `Producer` thread continuously attempts to publish products to the
    `Marketplace`, adhering to a specified republish wait time and
    producer-specific queue size limits.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        @brief Initializes a new Producer thread.

        @param products: A list of products this producer will offer, with quantity and publish interval.
                       Format: `[(product_obj, quantity, publish_interval), ...]`.
        @param marketplace: A reference to the shared Marketplace instance.
        @param republish_wait_time: The time in seconds to wait before retrying publishing if the marketplace is full for this producer.
        @param kwargs: Additional keyword arguments for the `Thread` constructor.
        """
        Thread.__init__(self, **kwargs)
        # List of products this producer will publish.
        self.products = products
        # Reference to the shared marketplace instance.
        self.marketplace = marketplace
        # Time to wait before retrying to publish.
        self.republish_wait_time = republish_wait_time
        # Registers with the marketplace and gets a unique producer ID.
        self.producerID = self.marketplace.register_producer()


    def run(self):
        """
        @brief The main execution logic for the Producer thread.

        Pre-condition: `products` contains definitions of products to publish.
        Invariant: The producer continuously attempts to publish its products,
                   respecting quantity limits and retry intervals.
        """
        # Block Logic: Main loop for continuous product publishing.
        while True:
            # Block Logic: Iterates through each product defined for this producer.
            for product in self.products:
                quantity = 0
                # Block Logic: Attempts to publish the desired quantity of the current product.
                while quantity < product[1]:
                    # Inline: Attempts to publish a product. If successful, wait and increment quantity.
                    if self.marketplace.publish(str(self.producerID), product[0]):
                        time.sleep(product[2])                     
                        quantity += 1                              
                    else:
                        # Inline: If publishing fails (e.g., producer's queue is full), wait and retry.
                        time.sleep(self.republish_wait_time)


from dataclasses import dataclass


@dataclass(init=True, repr=True, order=False, frozen=True)
class Product:
    """
    @brief Base dataclass representing a generic product in the marketplace.

    Uses `@dataclass` for convenience, providing automatic `__init__`, `__repr__`,
    and `__eq__` methods. It is frozen, meaning instances are immutable.
    """
    # Name of the product (e.g., "coffee beans").
    name: str
    # Price of the product.
    price: int


@dataclass(init=True, repr=True, order=False, frozen=True)
class Tea(Product):
    """
    @brief Specialized dataclass for Tea products, inheriting from `Product`.

    Adds a specific `type` attribute for different tea varieties.
    """
    # Type of tea (e.g., "Green", "Black", "Herbal").
    type: str


@dataclass(init=True, repr=True, order=False, frozen=True)
class Coffee(Product):
    """
    @brief Specialized dataclass for Coffee products, inheriting from `Product`.

    Adds attributes specific to coffee characteristics like acidity and roast level.
    """
    # Acidity level of the coffee.
    acidity: str
    # Roast level of the coffee (e.g., "Light", "Medium", "Dark").
    roast_level: str
