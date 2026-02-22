"""
@6c55464a-97e6-4eea-bcfe-bba9684e2be0/consumer.py
@brief Implements a multi-threaded producer-consumer system with a marketplace and product types.

This module simulates a marketplace where producers provide items and consumers attempt to purchase them.
It features thread-safe operations for managing product availability, registering participants,
and handling shopping carts, ensuring concurrency control through threading primitives and locks.
"""

from threading import Thread, Lock, currentThread
import time
from dataclasses import dataclass

class Consumer(Thread):
    """
    @brief Represents a consumer thread that interacts with the marketplace to buy products.

    Each Consumer processes a series of operations to add or remove products from a cart,
    and then places an order through the shared Marketplace.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        @brief Initializes a Consumer thread.

        Args:
            carts (list): A list of shopping carts, each containing a list of operations to perform.
            marketplace (Marketplace): The shared marketplace instance to interact with.
            retry_wait_time (float): The time in seconds to wait before retrying an operation
                                     if a product is not immediately available.
            **kwargs: Arbitrary keyword arguments, including 'name' for the consumer thread.
        """
        # Functional Utility: Calls the constructor of the parent Thread class, passing kwargs.
        Thread.__init__(self, **kwargs)
        # Functional Utility: Stores the list of cart operations for this consumer.
        self.carts = carts
        # Functional Utility: Stores a reference to the shared Marketplace instance.
        self.marketplace = marketplace
        # Functional Utility: Stores the retry wait time for product unavailability.
        self.retry_wait_time = retry_wait_time
        # Functional Utility: Stores the name of the consumer, primarily for logging.
        self.name = kwargs['name']

    def get_name(self):
        """
        @brief Returns the name of the consumer.

        Returns:
            str: The name of this consumer thread.
        """
        return self.name

    def run(self):
        """
        @brief Executes the consumer's purchasing logic.

        Iterates through each cart's operations, adding or removing products,
        and handles retries if products are not available. Finally, it places the order.
        """
        # Block Logic: Iterates through each defined shopping cart's operations.
        # Pre-condition: 'self.carts' is a list of lists, where each inner list contains cart operations.
        # Invariant: Each cart will be processed from start to end.
        for cart in self.carts:
            # Functional Utility: Creates a new cart in the marketplace and gets its unique ID.
            cart_id = self.marketplace.new_cart()
            # Block Logic: Processes each operation within the current shopping cart.
            # Pre-condition: 'cart' contains dictionaries detailing add/remove operations.
            # Invariant: Each operation is attempted the specified number of times.
            for operation in cart:
                quan_nr = 0
                # Block Logic: Ensures the operation is performed 'quantity' times, with retries if needed.
                # Pre-condition: 'operation['quantity']' specifies how many units to process.
                # Invariant: 'quan_nr' counts successfully processed units for the current operation.
                while quan_nr < operation['quantity']:
                    res = None
                    # Conditional Logic: Handles "add" operations for the product.
                    if operation['type'] == 'add':
                        res = self.marketplace.add_to_cart(cart_id,
                                                           operation
                                                           ['product'])
                    # Conditional Logic: Handles "remove" operations for the product.
                    if operation['type'] == 'remove':
                        res = self.marketplace.remove_from_cart(cart_id,
                                                                operation
                                                                ['product'])
                    
                    # Conditional Logic: If the operation was successful (product added/removed).
                    if res is None or res is True:
                        quan_nr = quan_nr + 1 # Functional Utility: Increments count of successful operations.
                    else:
                        # Functional Utility: If product was not available (for 'add') or some other issue, waits.
                        time.sleep(self.retry_wait_time)
            
            # Functional Utility: Places the final order for the current cart.
            self.marketplace.place_order(cart_id)

class Marketplace:
    """
    @brief Central hub for managing product availability, producers, and consumer carts.

    This class orchestrates the interaction between producers and consumers,
    handling product publishing, cart management, and order placement with
    thread-safe mechanisms.
    """
    
    def __init__(self, queue_size_per_producer):
        """
        @brief Initializes the Marketplace with specified producer queue size.

        Args:
            queue_size_per_producer (int): The maximum number of products a single
                                           producer can have in the marketplace's inventory.
        """
        # Functional Utility: Maximum number of items a single producer can have in stock.
        self.queue_size_per_producer = queue_size_per_producer
        # Functional Utility: Counter for assigning unique producer IDs.
        self.nr_of_producers = 0
        # Functional Utility: Counter for assigning unique cart IDs.
        self.nr_of_carts = 0
        # Functional Utility: List tracking the current number of items each producer has published.
        self.nr_of_items = []

        # Functional Utility: Dictionary mapping cart IDs to lists of (product, producer_id) pairs.
        self.carts = {}
        # Functional Utility: Dictionary mapping products to the ID of the producer currently supplying them.
        self.producers = {}

        # Functional Utility: A global lock to protect critical sections involving shared marketplace state.
        self.lock = Lock()

    def register_producer(self):
        """
        @brief Registers a new producer with the marketplace.

        Assigns a unique ID to the producer and initializes their item count.

        Returns:
            int: The unique ID assigned to the registered producer.
        """
        # Block Logic: Ensures thread-safe registration of a new producer.
        with self.lock:
            producer_id = self.nr_of_producers # Functional Utility: Assigns the current number of producers as the new ID.
            # Functional Utility: Increments the total count of registered producers.
            self.nr_of_producers = self.nr_of_producers + 1
            # Functional Utility: Initializes the item count for the new producer to 0.
            self.nr_of_items.insert(producer_id, 0)

        return producer_id

    def publish(self, producer_id, product):
        """
        @brief Publishes a product from a producer to the marketplace.

        The product is published only if the producer has not exceeded their
        allotted queue size.

        Args:
            producer_id (str): The ID of the producer (converted to int internally).
            product (any): The product to be published.

        Returns:
            bool: True if the product was successfully published, False otherwise.
        """
        # Conditional Logic: Checks if the producer's current item count exceeds the allowed queue size.
        if self.nr_of_items[int(producer_id)] >= self.queue_size_per_producer:
            return False # Functional Utility: Publishing fails if queue is full.
        
        # Functional Utility: Increments the item count for the publishing producer.
        self.nr_of_items[int(producer_id)] += 1
        # Functional Utility: Records which producer currently has this product available.
        self.producers[product] = int(producer_id)
        return True

    def new_cart(self):
        """
        @brief Creates a new shopping cart and assigns it a unique ID.

        Returns:
            int: The unique ID of the newly created cart.
        """
        # Block Logic: Ensures thread-safe creation of a new cart.
        with self.lock:
            # Functional Utility: Increments the total count of carts.
            self.nr_of_carts = self.nr_of_carts + 1
            cart_id = self.nr_of_carts # Functional Utility: Assigns the current number of carts as the new ID.
        
        self.carts[cart_id] = [] # Functional Utility: Initializes an empty list to represent the new cart.
        return cart_id

    def add_to_cart(self, cart_id, product):
        """
        @brief Adds a product from the marketplace to a specific consumer's cart.

        If the product is available, it is moved from the marketplace inventory
        to the cart, and the producing producer's item count is updated.

        Args:
            cart_id (int): The ID of the consumer's shopping cart.
            product (any): The product to add.

        Returns:
            bool: True if the product was successfully added, False if not available.
        """
        # Block Logic: Ensures thread-safe access to marketplace state for adding to cart.
        with self.lock:
            # Conditional Logic: Checks if the product is currently available in the marketplace (i.e., has a producer).
            if self.producers.get(product) is None:
                return False # Functional Utility: Product is not available.
            
            # Functional Utility: Decrements the item count for the producer who supplied this product.
            self.nr_of_items[self.producers[product]] -= 1
            # Functional Utility: Removes the product from the marketplace's general availability.
            producers_id = self.producers.pop(product)

        # Functional Utility: Appends the product to the cart.
        self.carts[cart_id].append(product)
        # Functional Utility: Also stores the producer's ID in the cart for later tracking (e.g., removal).
        self.carts[cart_id].append(producers_id)
        return True

    def remove_from_cart(self, cart_id, product):
        """
        @brief Removes a product from a specific cart and returns it to the producer's queue.

        Args:
            cart_id (int): The ID of the cart from which to remove the product.
            product (any): The product to remove.
        """
        # Conditional Logic: Checks if the product is actually in the cart.
        if product in self.carts[cart_id]:
            index = self.carts[cart_id].index(product) # Functional Utility: Finds the index of the product.
            
            self.carts[cart_id].remove(product) # Functional Utility: Removes the product from the cart list.
            # Functional Utility: Removes the corresponding producer ID from the cart list.
            producers_id = self.carts[cart_id].pop(index)
            
            # Functional Utility: Returns the product to the producer's available stock in the marketplace.
            self.producers[product] = producers_id
            # Block Logic: Ensures thread-safe update of the producer's item count.
            with self.lock:
                # Functional Utility: Increments the item count for the producer whose product was returned.
                self.nr_of_items[int(producers_id)] += 1


    def place_order(self, cart_id):
        """
        @brief Finalizes an order for a given cart, removing it from the marketplace's active carts
               and printing what was bought.

        Args:
            cart_id (int): The ID of the cart to place the order for.

        Returns:
            list: A list of products that were part of the order.
        """
        # Functional Utility: Retrieves and removes the entire cart's contents.
        product_list = self.carts.pop(cart_id)
        # Block Logic: Iterates through the ordered products to print and update producer counts.
        # Pre-condition: 'product_list' contains alternating product and producer_id.
        # Invariant: Each product will be printed, and its producer's item count will be decremented.
        for i in range(0, len(product_list), 2): # Inline: Iterates by 2 to get product and its associated producer_id.
            # Block Logic: Ensures thread-safe printing and update of producer item counts.
            with self.lock:
                # Functional Utility: Prints the consumer's name and the product bought.
                print(currentThread().get_name() +" bought " +
                      str(product_list[i]))
                # Functional Utility: Decrements the item count for the producer of the bought product.
                self.nr_of_items[product_list[i + 1]] -= 1
        return product_list


class Producer(Thread):
    """
    @brief Represents a producer thread that continuously supplies products to the marketplace.

    Producers generate products according to a predefined list and attempt to publish
    them to the Marketplace, pausing if their designated queue space is full.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        @brief Initializes a Producer thread.

        Args:
            products (list): A list of product specifications (product_id, quantity, production_time).
            marketplace (Marketplace): The shared marketplace instance.
            republish_wait_time (float): Time in seconds to wait before retrying if marketplace is full.
            **kwargs: Arbitrary keyword arguments, including 'daemon' status for the thread.
        """
        # Functional Utility: Calls the parent Thread class constructor, setting daemon status.
        Thread.__init__(self, **kwargs)

        # Functional Utility: Stores the list of products this producer will generate.
        self.products = products
        # Functional Utility: Stores a reference to the shared Marketplace instance.
        self.marketplace = marketplace
        # Functional Utility: Stores the wait time if the marketplace queue is full.
        self.republish_wait_time = republish_wait_time
        # Functional Utility: Will store the unique ID assigned to this producer by the marketplace.
        self.producer_id = None


    def run(self):
        """
        @brief The main execution loop for the Producer thread.

        Registers with the marketplace, then continuously produces items and attempts
        to publish them, waiting if the marketplace queue is full.
        """
        # Functional Utility: Registers this producer with the marketplace and obtains a unique ID.
        self.producer_id = self.marketplace.register_producer()
        # Invariant: The producer continuously attempts to produce and publish products.
        while True:
            # Block Logic: Iterates through the list of products this producer is configured to make.
            # Pre-condition: 'self.products' contains tuples of product specifications.
            # Invariant: Each product type will be produced in its specified quantity.
            for product in self.products:
                # Functional Utility: Unpacks product details for the current item.
                product_id = product[0]
                product_quantity = product[1]
                product_production_time = product[2]

                produced = 0
                # Block Logic: Attempts to publish the product 'product_quantity' times.
                # Pre-condition: 'produced' tracks the number of units successfully published.
                # Invariant: The loop continues until all units are published for this product batch.
                while produced < product_quantity:
                    # Functional Utility: Attempts to publish one unit of the product to the marketplace.
                    # Inline: Converts producer_id to string, potentially due to API expectation or type mismatch.
                    res = self.marketplace.publish(str(self.producer_id),
                                                   product_id)
                    # Conditional Logic: If publishing was successful.
                    if res:
                        # Functional Utility: Simulates the time taken to produce one unit of the product.
                        time.sleep(product_production_time)
                        produced = produced + 1 # Functional Utility: Increments count of successfully published units.
                    else:
                        # Functional Utility: If publishing failed (marketplace queue full), waits and retries.
                        time.sleep(self.republish_wait_time)


@dataclass(init=True, repr=True, order=False, frozen=True)
class Product:
    """
    @brief Represents a generic product with a name and price.

    This is a frozen dataclass, meaning instances are immutable after creation.
    It defines custom hash and equality methods based solely on the product name.
    """
    name: str
    price: int

    # Functional Utility: A custom hash function is often required for dataclasses
    # used as dictionary keys or in sets, especially if __eq__ is customized.
    def __hash__(self) -> int:
        return hash(self.name)

    # Functional Utility: A custom equality check based solely on the product name.
    def __eq__(self, other):
        if not isinstance(other, Product):
            return NotImplemented
        return self.name == other.name


@dataclass(init=True, repr=True, order=False, frozen=True)
class Tea(Product):
    """
    @brief Represents a specific type of Product: Tea.

    Extends the base Product with an additional attribute for 'type'.
    """
    type: str

@dataclass(init=True, repr=True, order=False, frozen=True)
class Coffee(Product):
    """
    @brief Represents a specific type of Product: Coffee.

    Extends the base Product with additional attributes for 'acidity' and 'roast_level'.
    """
    acidity: str
    roast_level: str
