"""
@file consumer.py
@brief This module implements a classic producer-consumer problem simulation using threads and a shared marketplace.
It defines `Consumer` and `Producer` entities that interact with a central `Marketplace` for
publishing and buying products. Synchronization is managed using threading primitives to ensure
thread-safe access to shared resources.

Algorithm:
- `Marketplace`: Acts as the central shared resource. It manages product availability, producer quotas,
  and shopping carts. It uses a single `threading.Lock` to protect critical sections, ensuring
  atomicity of operations like product publishing, adding/removing from carts, and generating new cart IDs.
- `Producer`: Continuously tries to publish products to the `Marketplace`. If the marketplace is full
  for that producer, it waits and retries.
- `Consumer`: Iterates through a list of predefined shopping carts, adding and removing products
  from the `Marketplace`. It handles cases where a product is not immediately available by retrying
  after a delay. Finally, it places the order.
- `Product`, `Tea`, `Coffee`: Simple data classes to represent different types of products.

Time Complexity:
- `Marketplace` methods: Most operations involve list manipulations or dictionary lookups, typically O(1) or O(N) where N is the number of products. Locking overhead is present.
- `Producer.run`: O(P * Q * M) where P is the number of products per producer, Q is the quantity, and M is the average number of retries for publishing.
- `Consumer.run`: O(C * O * M) where C is the number of carts, O is the number of operations per cart, and M is the average number of retries for adding to cart.
Space Complexity:
- `Marketplace`: O(P_total) for `products`, O(C_total) for `carts`, O(N_producers) for `producer_items` and `producers` where P_total, C_total, N_producers are total products, carts and producers.
- `Consumer`, `Producer`: O(C_ops) or O(P_ops) for storing their respective operation lists.
"""

from threading import Thread
import time


class Consumer(Thread):
    """
    @class Consumer
    @brief Represents a consumer entity in the marketplace simulation.
    Functional Utility: Each consumer runs as a separate thread, simulating a user
    who creates shopping carts, adds/removes products to/from them, and finally
    places orders through the shared `Marketplace`. It handles retries if products
    are not immediately available.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        @brief Initializes a new Consumer thread.
        @param carts (list): A list of shopping cart definitions. Each cart is a list of operations
                              (e.g., add/remove product, quantity).
        @param marketplace (Marketplace): The shared marketplace instance that consumers interact with.
        @param retry_wait_time (float): The time (in seconds) to wait before retrying an operation
                                        if a product is not available.
        @param kwargs: Arbitrary keyword arguments passed to the `Thread` constructor.
        """
        Thread.__init__(self, **kwargs) # Call the base Thread constructor.
        self.carts = carts # List of cart operations to perform.
        self.marketplace = marketplace # Reference to the shared Marketplace.
        self.retry_wait_time = retry_wait_time # Time to wait on retry.

    def run(self):
        """
        @brief The main execution loop for the Consumer thread.
        Functional Utility: Iterates through each predefined shopping cart, performs the specified
        add and remove operations, handling retries for 'add' operations if a product is not
        available. Finally, it places the order for each cart.
        """
        for cart in self.carts: # Block Logic: Process each cart definition sequentially.
            cart_id = self.marketplace.new_cart() # Functional Utility: Create a new cart in the marketplace.

            for operation in cart: # Block Logic: Perform operations (add/remove) for the current cart.
                quantity = operation["quantity"] # Get the quantity for the current operation.
                if operation["type"] == "add": # Case 1: Add product to cart.
                    for _ in range(quantity): # Block Logic: Attempt to add product multiple times if needed.
                        while self.marketplace.add_to_cart(cart_id, operation["product"]) is False:
                            time.sleep(self.retry_wait_time) # Wait and retry if product is unavailable.

                if operation["type"] == "remove": # Case 2: Remove product from cart.
                    for _ in range(quantity): # Block Logic: Remove product multiple times.
                        self.marketplace.remove_from_cart(cart_id, operation["product"])

            self.marketplace.place_order(cart_id) # Functional Utility: Place the order for the completed cart.


from threading import Lock, currentThread

class Marketplace:
    """
    @class Marketplace
    @brief Central shared resource in the producer-consumer simulation.
    Functional Utility: Manages product inventory, producer quotas, shopping carts,
    and handles all interactions between `Producer` and `Consumer` threads.
    It employs a `threading.Lock` to ensure all its internal operations are thread-safe.
    """

    def __init__(self, queue_size_per_producer):
        """
        @brief Initializes the Marketplace.
        @param queue_size_per_producer (int): The maximum number of items a single producer
                                              can have in the marketplace at any given time.
        """
        self.queue_size_per_producer = queue_size_per_producer # Max items per producer.
        self.lock = Lock() # Global lock to protect shared resources within the marketplace.
        self.cid = 0 # Counter for generating unique cart IDs.
        self.producer_items = [] # List tracking current item count for each producer.
        self.products = [] # List of all products currently available in the marketplace.
        self.carts = {} # Dictionary mapping cart IDs to lists of products in each cart.
        self.producers = {} # Dictionary mapping product to the producer_id that published it.

    def register_producer(self):
        """
        @brief Registers a new producer with the marketplace.
        @return (int): A unique ID assigned to the new producer.
        Functional Utility: Assigns a unique ID to a producer and initializes its item count.
        """
        self.lock.acquire() # Acquire lock for thread-safe access.
        prod_id = len(self.producer_items) # Assign a new producer ID.
        self.producer_items.append(0) # Initialize item count for this producer to 0.
        self.lock.release() # Release lock.
        return prod_id

    def publish(self, producer_id, product):
        """
        @brief Publishes a product to the marketplace.
        @param producer_id (int): The ID of the producer publishing the product.
        @param product (object): The product object to publish.
        @return (bool): `True` if the product was published successfully, `False` otherwise
                        (e.g., if the producer's queue is full).
        Functional Utility: Adds a product to the marketplace's inventory if the producer
        has not exceeded its quota.
        """
        producer_id = int(producer_id) # Ensure producer_id is an integer.

        if self.producer_items[producer_id] >= self.queue_size_per_producer: # Block Logic: Check if producer's quota is full.
            return False # Cannot publish if quota exceeded.

        self.producer_items[producer_id] += 1 # Increment producer's item count.
        self.products.append(product) # Add product to global product list.
        self.producers[product] = producer_id # Record which producer published this product.

        return True # Product published successfully.

    def new_cart(self):
        """
        @brief Creates a new shopping cart.
        @return (int): A unique ID for the new cart.
        Functional Utility: Generates a unique cart ID and initializes an empty cart for it.
        """
        self.lock.acquire() # Acquire lock for thread-safe access.
        self.cid += 1 # Increment cart ID counter.
        cart_id = self.cid # Assign new cart ID.
        self.lock.release() # Release lock.

        self.carts[cart_id] = [] # Initialize an empty list for the new cart.
        return cart_id

    def add_to_cart(self, cart_id, product):
        """
        @brief Adds a product to a specific shopping cart.
        @param cart_id (int): The ID of the cart to add the product to.
        @param product (object): The product object to add.
        @return (bool): `True` if the product was added successfully, `False` otherwise
                        (e.g., if the product is not available).
        Functional Utility: Moves a product from the marketplace's general inventory to a
        specific cart, updating producer quotas and product availability.
        """
        self.lock.acquire() # Acquire lock for thread-safe access.
        if product not in self.products: # Block Logic: Check if the product is available in the marketplace.
            self.lock.release() # Release lock.
            return False # Product not found, cannot add.

        self.producer_items[self.producers[product]] -= 1 # Decrement producer's item count.
        self.products.remove(product) # Remove product from global product list.

        self.carts[cart_id].append(product) # Add product to the specified cart.
        self.lock.release() # Release lock.

        return True # Product added successfully.

    def remove_from_cart(self, cart_id, product):
        """
        @brief Removes a product from a specific shopping cart and returns it to the marketplace.
        @param cart_id (int): The ID of the cart to remove the product from.
        @param product (object): The product object to remove.
        Functional Utility: Moves a product from a specific cart back to the marketplace's
        general inventory, updating producer quotas and product availability.
        """
        self.carts[cart_id].remove(product) # Remove product from the cart.
        self.products.append(product) # Add product back to global product list.

        self.lock.acquire() # Acquire lock for thread-safe access.
        self.producer_items[self.producers[product]] += 1 # Increment producer's item count.
        self.lock.release() # Release lock.

    def place_order(self, cart_id):
        """
        @brief Finalizes an order for a given cart.
        @param cart_id (int): The ID of the cart to place the order for.
        @return (list): A list of products that were part of the order.
        Functional Utility: Simulates the process of placing an order by iterating
        through the products in the cart and printing a message for each item bought.
        """
        products_list = self.carts.get(cart_id) # Retrieve the list of products in the cart.
        for product in products_list: # Block Logic: Iterate through each product in the order.
            self.lock.acquire() # Acquire lock for thread-safe printing.

            print("{} bought {}".format(currentThread().getName(), product)) # Print the purchase message.
            self.lock.release() # Release lock.

        return products_list


from threading import Thread
import time


class Producer(Thread):
    """
    @class Producer
    @brief Represents a producer entity in the marketplace simulation.
    Functional Utility: Each producer runs as a separate thread, simulating a supplier
    that continuously publishes products to the shared `Marketplace`. It handles
    situations where the marketplace is temporarily full for its products.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        @brief Initializes a new Producer thread.
        @param products (list): A list of product definitions, where each definition includes
                                (product_object, quantity, wait_time_after_publish).
        @param marketplace (Marketplace): The shared marketplace instance that producers interact with.
        @param republish_wait_time (float): The time (in seconds) to wait before retrying to publish
                                            if the marketplace is full for this producer.
        @param kwargs: Arbitrary keyword arguments passed to the `Thread` constructor.
        """
        Thread.__init__(self, **kwargs) # Call the base Thread constructor.
        self.products = products # List of products to publish.
        self.marketplace = marketplace # Reference to the shared Marketplace.
        self.republish_wait_time = republish_wait_time # Time to wait on republish retry.
        self.prod_id = self.marketplace.register_producer() # Functional Utility: Register with marketplace to get a unique producer ID.

    def run(self):
        """
        @brief The main execution loop for the Producer thread.
        Functional Utility: Continuously attempts to publish its predefined products to the
        marketplace. If the marketplace's quota for this producer is full, it waits and retries.
        It also introduces a delay after successfully publishing each item.
        """
        while True: # Block Logic: Infinite loop for continuous production.
            for product, quantity, wait_time in self.products: # Block Logic: Iterate through each product definition.
                for _ in range(quantity): # Block Logic: Publish the specified quantity of each product.
                    # Functional Utility: Repeatedly attempt to publish until successful.
                    while self.marketplace.publish(str(self.prod_id), product) is False:
                        time.sleep(self.republish_wait_time) # Wait and retry if marketplace is full.

                    time.sleep(wait_time) # Wait after publishing each item.


from dataclasses import dataclass


@dataclass(init=True, repr=True, order=False, frozen=True)
class Product:
    """
    @dataclass Product
    @brief Base data class representing a generic product.
    Attributes:
        name (str): The name of the product.
        price (int): The price of the product.
    Functional Utility: Provides a immutable, lightweight data structure for product information.
    """
    name: str
    price: int

@dataclass(init=True, repr=True, order=False, frozen=True)
class Tea(Product):
    """
    @dataclass Tea
    @brief Data class representing a type of tea, inheriting from `Product`.
    Attributes:
        type (str): The type of tea (e.g., 'Green', 'Black').
    Functional Utility: Extends the generic `Product` with tea-specific attributes.
    """
    type: str

@dataclass(init=True, repr=True, order=False, frozen=True)
class Coffee(Product):
    """
    @dataclass Coffee
    @brief Data class representing a type of coffee, inheriting from `Product`.
    Attributes:
        acidity (str): The acidity level of the coffee.
        roast_level (str): The roast level of the coffee.
    Functional Utility: Extends the generic `Product` with coffee-specific attributes.
    """
    acidity: str
    roast_level: str