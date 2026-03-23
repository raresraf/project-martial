"""
@file consumer.py
@brief This module implements a multi-threaded simulation of a marketplace system.
       It defines `Consumer` and `Producer` entities that interact with a central
       `Marketplace` to buy and sell products. The system uses threading to
       simulate concurrent operations and `logging` for tracking events.

       The marketplace manages product inventory, producer capacities, and shopping carts.
       It uses various locks to ensure thread safety during concurrent access to shared data.

Algorithm:
- **Marketplace Setup:** Initializes product queues, producer tracking, cart storage,
  and a suite of `threading.Lock` objects to protect shared resources.
- **Producer Lifecycle:**
    1. Registers itself with the `Marketplace`, receiving a unique ID.
    2. Enters a continuous loop, attempting to `publish` its products to the `Marketplace`.
    3. If publishing fails (e.g., due to marketplace capacity), it waits and retries.
    4. Waits for a specified `republish_wait_time` or `product_wait_time` between publications.
- **Consumer Lifecycle:**
    1. Obtains a new shopping `cart` from the `Marketplace`.
    2. Iterates through a predefined list of desired products (`self.carts`).
    3. For each product, it attempts to `add_to_cart` or `remove_from_cart`.
    4. If `add_to_cart` fails (e.g., product unavailable), it waits (`retry_wait_time`) and retries.
    5. After processing all desired products, it `place_order` for its cart.
- **Marketplace Operations (Thread-Safe):**
    - `register_producer`: Assigns an ID and sets up initial capacity.
    - `publish`: Adds product to producer's inventory if capacity allows.
    - `new_cart`: Creates an empty cart and assigns an ID.
    - `add_to_cart`: Searches for product across producers, moves it to cart, and frees producer capacity.
    - `remove_from_cart`: Removes product from cart and returns it to a designated producer (producer 0 in this case).
    - `place_order`: Transfers cart contents to a final order list and clears the cart.
    - `print_cons`: Logs the consumption action.

Threading & Synchronization:
- `threading.Thread`: Used for `Consumer` and `Producer` to enable concurrent execution.
- `threading.Lock`: Extensively used within `Marketplace` methods to ensure atomic operations
  on shared data structures like `producers`, `products`, and `carts`, preventing race conditions.

Time Complexity:
- Operations like `add_to_cart` and `remove_from_cart` involve iterating through lists of
  producers and products. In the worst case, this can be O(N_producers * N_products_per_producer).
- The overall system performance is heavily influenced by contention for `Marketplace` locks,
  the number of producers/consumers, and the `retry_wait_time`/`republish_wait_time`.

Space Complexity:
- O(N_producers * P_avg + N_carts * C_avg) where P_avg is average products per producer
  and C_avg is average products per cart. Additional space for thread objects, locks, and logging.
"""

from threading import Thread # Import Thread for multi-threading.
import time                  # Import time for sleep functionality.


class Consumer(Thread):
    """
    @class Consumer
    @brief Represents a consumer (buyer) in the marketplace simulation.
           Each consumer runs as a separate thread, creating a cart,
           adding/removing products, and eventually placing an order.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        @brief Initializes a new Consumer instance.
        @param carts: A list of products the consumer wishes to interact with (buy/remove).
                      Format: [[{"product":"apple", "quantity":1, "type":"add"}], ...]
        @param marketplace: The shared Marketplace instance this consumer interacts with.
        @param retry_wait_time: Time in seconds to wait before retrying an add_to_cart operation.
        @param kwargs: Arbitrary keyword arguments passed to the Thread constructor.
        """
        self.carts = carts # List of product interactions.
        self.marketplace = marketplace # Reference to the shared marketplace.
        self.retry_wait_time: int = retry_wait_time # Wait time for retries.
        Thread.__init__(self, **kwargs) # Initialize the Thread base class.

    def print_carts(self, id_cart):
        """
        @brief Places the order for a given cart and prints the details of bought products.
        @param id_cart: The ID of the cart to place the order for.
        Functional Utility: Finalizes a shopping session by processing the cart contents.
        """
        # Block Logic: Retrieves the list of ordered products from the marketplace.
        list_order = self.marketplace.place_order(id_cart)
        # Block Logic: Iterates through each product in the placed order and prints consumption details.
        for product in list_order:
            self.marketplace.print_cons(self.name, product)

    def add_product_to_cart(self, id_cart, prod):
        """
        @brief Attempts to add a product to a specified cart. Retries if the operation fails.
        @param id_cart: The ID of the cart to add the product to.
        @param prod: The product dictionary to add (e.g., {"product":"apple", "quantity":1, "type":"add"}).
        Functional Utility: Ensures a product is eventually added to the cart, handling temporary unavailability.
        """
        # Attempts to add the product to the cart via the marketplace.
        go_next = self.marketplace.add_to_cart(id_cart,prod)
        # Block Logic: If adding fails (product not available), wait and retry.
        if go_next is False:
            time.sleep(self.retry_wait_time) # Wait before retrying.
            self.add_product_to_cart(id_cart, prod) # Recursive retry.

    def run(self):
        """
        @brief The main execution loop for the Consumer thread.
        Functional Utility: Simulates the entire shopping process for a consumer.
        """
        id_cart = self.marketplace.new_cart() # Obtains a new cart ID from the marketplace.
        # Block Logic: Iterates through each list of products the consumer wants to interact with.
        for products in self.carts:
            # Block Logic: Iterates through each individual product within the current list.
            for produs in products:
                # Block Logic: Performs the add or remove operation for the specified quantity of the product.
                for _ in range(produs["quantity"]):
                    if produs["type"] == "remove":
                        self.marketplace.remove_from_cart(id_cart, produs)
                    else:
                        self.add_product_to_cart(id_cart, produs) # Calls retry-enabled add function.
        self.print_carts(id_cart) # Places the order and prints the items.

# Logging configuration for the marketplace.
from threading import Lock # Import Lock for thread synchronization.
import logging             # Import logging module.
from logging.handlers import RotatingFileHandler # For rotating log files.
import time                # Import time for time-related functions.


logger = logging.getLogger('loggerOne') # Get a logger instance.
logger.setLevel(logging.INFO)           # Set logging level to INFO.


handler = RotatingFileHandler('file.log', maxBytes=500000, backupCount=10) # Configure file handler.


formatter = logging.Formatter('%(asctime)s %(levelname)8s: %(message)s') # Define log message format.
handler.setFormatter(formatter) # Set formatter for the handler.
logger.addHandler(handler)      # Add handler to the logger.


logging.Formatter.converter = time.gmtime # Use GMT for timestamps in logs.


class Marketplace:
    """
    @class Marketplace
    @brief Represents the central marketplace where producers publish products
           and consumers manage shopping carts. It ensures thread safety for
           all concurrent operations using various locks.
    """

    def __init__(self, queue_size_per_producer):
        """
        @brief Initializes the Marketplace.
        @param queue_size_per_producer: The maximum number of products each producer can have in its queue.
        """
        logger.info("init Marketplace, argument qsise: %d", queue_size_per_producer) # Log initialization.
        self.queue_size_per_producer = queue_size_per_producer # Capacity for each producer.
        self.id_prod = 0          # Counter for assigning producer IDs.
        self.id_cart = 0          # Counter for assigning cart IDs.
        self.producers = []       # List to track producer capacities (index corresponds to producer ID).
        self.producers.append(0)  # Dummy entry at index 0 or for a special producer.
        self.products = []        # List of lists, where each inner list stores products for a producer.
        self.products.append([])  # Dummy entry at index 0.
        self.carts = []           # List of lists, where each inner list stores products in a cart.
        
        # Locks to ensure thread safety for critical sections.
        self.add_to_cart_lock = Lock()        # Protects add_to_cart operation.
        self.publish_lock = Lock()            # Protects publish operation.
        self.print_lock = Lock()              # Protects console print output.
        self.new_cart_lock = Lock()           # Protects new_cart operation.
        self.register_producer_lock = Lock()  # Protects register_producer operation.
        self.remove_from_cart_lock = Lock()   # Protects remove_from_cart operation.

        logger.info("init Marketplace, all") # Log completion of initialization.

    def register_producer(self):
        """
        @brief Registers a new producer with the marketplace and assigns it a unique ID.
        @return The unique ID assigned to the new producer.
        Functional Utility: Integrates a new producer into the marketplace system.
        """
        # Block Logic: Acquires a lock to ensure thread-safe registration.
        self.register_producer_lock.acquire()
        logger.info("register_producer, id_prod =%d", self.id_prod) # Log producer registration.
        self.producers.append(self.queue_size_per_producer) # Add producer with initial queue size.
        self.id_prod = self.id_prod + 1 # Increment producer ID counter.
        self.products.append([]) # Add an empty list for this producer's products.

        self.register_producer_lock.release() # Releases the lock.
        logger.info("register_producer, id_prod-exit =%d", self.id_prod) # Log exit.
        return self.id_prod # Return the new producer's ID.

    def publish(self, producer_id, product):
        """
        @brief Publishes a product from a specific producer to the marketplace.
        @param producer_id: The ID of the producer publishing the product.
        @param product: A tuple or list containing product details, where product[0] is the item name.
        @return True if the product was successfully published, False otherwise (e.g., no capacity).
        Functional Utility: Adds a product to the marketplace's available inventory for a producer.
        """
        it_produced = False # Flag to indicate if product was published.
        
        # Block Logic: Acquires a lock to ensure thread-safe publishing.
        self.publish_lock.acquire()
        logger.info("publish; id_producer =%d", producer_id) # Log publish attempt.
        # Block Logic: Checks if the producer has capacity to publish more products.
        if self.producers[producer_id] > 0:
            self.products[producer_id].append(product[0]) # Add product to producer's inventory.
            self.producers[producer_id] = self.producers[producer_id] -1 # Decrement producer's capacity.
            it_produced = True # Mark as successfully produced.

        logger.info("publish; exit =%d", producer_id) # Log exit.
        self.publish_lock.release() # Releases the lock.
        return it_produced # Return publishing status.

    def new_cart(self):
        """
        @brief Creates a new shopping cart and assigns it a unique ID.
        @return The unique ID of the newly created cart.
        Functional Utility: Provides a container for consumers to collect products.
        """
        # Block Logic: Acquires a lock to ensure thread-safe cart creation.
        self.new_cart_lock.acquire()
        logger.info("new_cart;") # Log new cart creation.

        self.carts.append([]) # Add an empty list for the new cart.
        current_cart_id = self.id_cart # Get the current cart ID.
        self.id_cart = self.id_cart+1 # Increment cart ID counter.

        logger.info("new_cart; iese") # Log exit.
        self.new_cart_lock.release() # Releases the lock.
        return current_cart_id # Return the new cart's ID.

    def add_to_cart(self, cart_id, product):
        """
        @brief Adds a product to a specified cart.
               It searches for the product among all producers. If found, it moves
               the product from the producer's inventory to the cart and frees up
               capacity for that producer.
        @param cart_id: The ID of the cart to add the product to.
        @param product: A dictionary containing product details (e.g., {"product":"apple"}).
        @return True if the product was successfully added, False if not found/available.
        Functional Utility: Facilitates purchasing by moving an available product to a cart.
        """
        producer_found_id = 0 # ID of the producer where the product was found.
        product_found = False # Flag to indicate if the product was found.
        available_product = None # Store the actual product object if found.

        # Block Logic: Acquires a lock to ensure thread-safe cart modification.
        self.add_to_cart_lock.acquire()
        logger.info("ad_to_cart %d;", cart_id) # Log add to cart attempt.

        # Block Logic: Iterates through all producers' product lists to find the desired product.
        for producer_idx, producer_products in enumerate(self.products):
            for p_idx, p_item in enumerate(producer_products):
                if product["product"] == p_item: # Check if product name matches.
                    product_found = True
                    producer_found_id = producer_idx
                    available_product = p_item
                    break # Break inner loop, product found.
            if product_found:
                break # Break outer loop, product found.

        # Block Logic: If the product was found and is available.
        if product_found:
            self.products[producer_found_id].remove(available_product) # Remove from producer's inventory.
            self.producers[producer_found_id] = self.producers[producer_found_id] + 1 # Increment producer's capacity.
            self.carts[cart_id].append(available_product) # Add to the consumer's cart.
            logger.info("ad_to_cart exit True;") # Log success.
            self.add_to_cart_lock.release() # Releases the lock.
            return True # Indicate success.

        logger.info("ad_to_cart exit False;") # Log failure.
        self.add_to_cart_lock.release() # Releases the lock.
        return False # Indicate failure.

    def remove_from_cart(self, cart_id, product):
        """
        @brief Removes a product from a specified cart.
               The removed product is returned to the inventory of producer 0.
               Note: This logic is somewhat simplified as typically a product would
               return to its original producer or a general pool.
        @param cart_id: The ID of the cart to remove the product from.
        @param product: A dictionary containing product details (e.g., {"product":"apple"}).
        Functional Utility: Reverts a purchase or corrects a cart entry.
        """
        product_found = False # Flag to indicate if the product was found in the cart.
        producer_found_id = 0 # Placeholder, not directly used for return to producer.
        available_product = None # Store the actual product object if found.

        # Block Logic: Acquires a lock to ensure thread-safe cart modification.
        self.remove_from_cart_lock.acquire()
        logger.info("reomce_from_cart start True;%d", cart_id) # Log remove attempt.
        
        # Block Logic: Iterates through the cart's products to find the desired product.
        for idx, item_in_cart in enumerate(self.carts[cart_id]):
            if product["product"] == item_in_cart: # Check if product name matches.
                product_found = True
                producer_found_id = idx # This is actually the index in the cart, not a producer ID.
                available_product = item_in_cart
                break # Break loop, product found.

        # Block Logic: If the product was found in the cart.
        if product_found:
            del self.carts[cart_id][producer_found_id] # Remove from the cart.
            self.products[0].append(available_product) # Add removed product to producer 0's inventory.
                                                     # This implicitly assumes producer 0 always has capacity.

        logger.info("ad_to_cart exit;") # Log exit.
        self.remove_from_cart_lock.release() # Releases the lock.

    def place_order(self, cart_id):
        """
        @brief Places an order for a specified cart, transferring its contents to a return list
               and clearing the cart.
        @param cart_id: The ID of the cart to place the order for.
        @return A list of products that were in the cart (the placed order).
        Functional Utility: Finalizes the transaction for a shopping cart.
        """
        # Note: This method does not appear to use a lock for `self.carts[cart_id]`
        #        access, which could lead to race conditions if `add_to_cart` or
        #        `remove_from_cart` are called concurrently on the same cart ID.
        logger.info("place_order start %d;", cart_id) # Log order placement.
        copie =self.carts[cart_id] # Copy cart contents.
        self.carts[cart_id] = [] # Clear the original cart.
        logger.info("place_order end;") # Log completion.
        return copie # Return the list of ordered products.

    def print_cons(self, name, product):
        """
        @brief Prints a message to the console indicating a consumer bought a product.
               Uses a lock to ensure atomic printing to avoid interleaved output.
        @param name: The name of the consumer thread.
        @param product: The product that was bought.
        Functional Utility: Provides visual feedback of marketplace transactions.
        """
        # Block Logic: Acquires a lock to ensure exclusive access to console printing.
        self.print_lock.acquire()
        logger.info("print_cons start; name=%s;", name) # Log print attempt.

        print(name, "bought", product) # Print to console.
        self.print_lock.release() # Releases the lock.


class Producer(Thread):
    """
    @class Producer
    @brief Represents a producer (seller) in the marketplace simulation.
           Each producer runs as a separate thread, registers with the marketplace,
           and continuously publishes its products.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        @brief Initializes a new Producer instance.
        @param products: A list of products this producer will publish.
                         Format: [[product_name, quantity, time_to_produce], ...]
        @param marketplace: The shared Marketplace instance this producer interacts with.
        @param republish_wait_time: Time in seconds to wait before retrying to publish a product.
        @param kwargs: Arbitrary keyword arguments passed to the Thread constructor.
        """
        Thread.__init__(self, **kwargs) # Initialize the Thread base class.
        self.products  = products # List of products to produce.

        self.marketplace = marketplace # Reference to the shared marketplace.
        self.republish_wait_time: int = republish_wait_time # Wait time for republishing.
        self.my_id = self.marketplace.register_producer() # Registers with the marketplace.

    def run(self):
        """
        @brief The main execution loop for the Producer thread.
        Functional Utility: Continuously publishes products to the marketplace,
                            managing production time and retry delays.
        """
        # Block Logic: The producer continuously tries to publish its products.
        while True:
            # Block Logic: Iterates through each product type the producer has.
            for prod in self.products:
                # Block Logic: Publishes the product for the specified quantity.
                for _ in range(prod[1]): # prod[1] is the quantity.
                    it_worked = self.marketplace.publish(self.my_id, prod) # Attempt to publish.
                    if it_worked:
                        time.sleep(prod[2]) # Wait for production time (prod[2]).
                    else:
                        time.sleep(self.republish_wait_time) # If publish failed, wait and retry.


from dataclasses import dataclass # Import dataclass decorator.


@dataclass(init=True, repr=True, order=False, frozen=True)
class Product:
    """
    @dataclass
    @brief Base class for products in the marketplace.
    @param name: The name of the product.
    @param price: The price of the product.
    """
    name: str
    price: int


@dataclass(init=True, repr=True, order=False, frozen=True)
class Tea(Product):
    """
    @dataclass
    @brief Represents a Tea product, inheriting from Product.
    @param type: The type of tea (e.g., "green", "black").
    """
    type: str


@dataclass(init=True, repr=True, order=False, frozen=True)
class Coffee(Product):
    """
    @dataclass
    @brief Represents a Coffee product, inheriting from Product.
    @param acidity: The acidity level of the coffee.
    @param roast_level: The roast level of the coffee.
    """
    acidity: str
    roast_level: str
