"""
This module implements a multi-threaded marketplace simulation. It defines classes
for Consumers (buyers), Producers (sellers), and the central Marketplace that
manages products and transactions. It also includes data classes for various
product types.

Domain: Concurrency, Multi-threading, Producer-Consumer Problem, Object-Oriented Design.
"""

from threading import Thread
import time

class Consumer(Thread):
    """
    @brief Simulates a consumer (buyer) in the marketplace.

    Each consumer thread creates a cart, performs a series of add/remove operations
    on products, and finally places an order. It handles retries for failed operations.
    """
    

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        @brief Initializes a new Consumer instance.

        @param carts: A list of cart operations, where each operation is a dictionary
                      specifying "type" (add/remove), "quantity", and "product".
        @param marketplace: The Marketplace instance to interact with.
        @param retry_wait_time: The time (in seconds) to wait before retrying a failed operation.
        @param kwargs: Additional keyword arguments passed to the Thread constructor (e.g., name).
        """
        Thread.__init__(self, **kwargs)
        self.carts = carts # @brief List of purchasing plans.
        self.marketplace = marketplace # @brief Reference to the shared marketplace.
        self.retry_wait_time = retry_wait_time # @brief Time to wait before retrying.
        self.kwargs = kwargs # @brief Stored kwargs for potential future use (e.g., thread name).

    def run(self):
        """
        @brief The main execution logic for the consumer thread.

        Iterates through a list of cart operations for this consumer. For each cart:
        1. Creates a new cart in the marketplace.
        2. Processes each "add" or "remove" operation for products in the cart.
        3. Retries failed operations after `retry_wait_time`.
        4. Places the final order and prints the purchased products.
        """
        for cart in self.carts:
            cart_id = self.marketplace.new_cart() # @brief Get a new cart ID from the marketplace.

            # Block Logic: Process each operation (add/remove) in the current cart.
            for operation in cart:
                type_opp = operation["type"] # @brief Type of operation: "add" or "remove".
                quantity = operation["quantity"] # @brief Number of times to perform the operation.
                product = operation["product"] # @brief The product involved in the operation.

                while quantity > 0:
                    result = False # @brief Flag to store the result of the marketplace operation.
                    if type_opp == "add":
                        result = self.marketplace.add_to_cart(cart_id, product)
                    elif type_opp == "remove":
                        result = self.marketplace.remove_from_cart(cart_id, product)

                    # Block Logic: Handle operation outcome.
                    # If successful (result is True or None for some operations), decrement quantity.
                    # Otherwise, wait and retry.
                    if result or result is None:
                        quantity -= 1
                    else:
                        time.sleep(self.retry_wait_time) # Inline: Wait before retrying a failed operation.

            order = self.marketplace.place_order(cart_id) # @brief Place the final order for the cart.
            # Block Logic: Print the details of the products bought by this consumer.
            for product in order:
                result = self.kwargs["name"] + " bought " + str(product)
                print(result)

from threading import Lock
from collections import defaultdict

class Marketplace:
    """
    @brief Central hub for managing products, producers, and consumer carts.

    This class provides thread-safe operations for producers to publish products,
    and for consumers to create carts, add/remove products, and place orders.
    It uses locks to protect shared data structures.
    """
    
    def __init__(self, queue_size_per_producer):
        """
        @brief Initializes the Marketplace.

        @param queue_size_per_producer: The maximum number of products a single producer
                                        can have listed in the marketplace at any time.
        """
        # @brief Maximum queue size for each producer.
        self.queue_size_per_producer = queue_size_per_producer

        # @brief Counter for assigning unique producer IDs.
        self.number_of_producers = 0

        # @brief Dictionary to track the number of products published by each producer.
        self.producers_queue_sizes = dict()

        # @brief List of all products currently available in the marketplace.
        self.products = []

        # @brief Counter for assigning unique cart IDs.
        self.number_of_carts = 0

        # @brief Dictionary to store products in each consumer's cart.
        self.carts = defaultdict()

        # @brief Lock to protect access during producer registration.
        self.lock_register = Lock()

        # @brief Lock to protect access during cart creation.
        self.lock_cart = Lock()

        # @brief Lock to protect access during product addition/removal from carts.
        self.lock_product = Lock()

    def register_producer(self):
        """
        @brief Registers a new producer with the marketplace.

        Assigns a unique ID to the producer and initializes its product queue size counter.
        Ensures thread-safe registration using `lock_register`.

        @return The unique ID assigned to the new producer.
        """
        self.lock_register.acquire() # @brief Acquire lock to protect shared producer registration data.

        # Block Logic: Assign a new unique producer ID.
        producer_id = self.number_of_producers

        self.number_of_producers += 1
        

        # Block Logic: Initialize the product count for the new producer.
        self.producers_queue_sizes[producer_id] = 0

        self.lock_register.release() # @brief Release lock.

        return producer_id

    def publish(self, producer_id, product):
        """
        @brief Publishes a product to the marketplace.

        The product is added only if the producer has not exceeded its queue size limit.
        Assigns ownership of the product to the publishing producer.

        @param producer_id: The ID of the producer publishing the product.
        @param product: The product object to be published.
        @return True if the product was successfully published, False otherwise.
        """
        id_producer = int(producer_id) # @brief Ensure producer_id is an integer.

        # Block Logic: Assign ownership of the product to the publishing producer.
        product.owner = id_producer

        # Block Logic: Check if the producer has reached its queue limit.
        if self.producers_queue_sizes[id_producer] < self.queue_size_per_producer:
            
            self.producers_queue_sizes[id_producer] += 1 # Inline: Increment producer's active product count.
            
            self.products.append(product) # Inline: Add product to the general marketplace inventory.
            return True

        return False

    def new_cart(self):
        """
        @brief Creates a new shopping cart for a consumer.

        Assigns a unique ID to the cart and initializes it as an empty list.
        Ensures thread-safe cart creation using `lock_cart`.

        @return The unique ID assigned to the new cart.
        """
        self.lock_cart.acquire() # @brief Acquire lock to protect shared cart creation data.

        # Block Logic: Assign a new unique cart ID.
        id_cart = self.number_of_carts

        self.number_of_carts += 1 # Inline: Increment the total number of carts.

        self.lock_cart.release() # @brief Release lock.

        # Block Logic: Initialize the new cart as an empty list.
        self.carts[id_cart] = []

        return id_cart

    def add_to_cart(self, cart_id, product):
        """
        @brief Adds a product to a specified shopping cart.

        Ensures thread-safe product operations using `lock_product`.
        Checks if the product is available in the marketplace before adding.
        If added, the product is removed from the marketplace and the producer's
        queue size is updated.

        @param cart_id: The ID of the cart to add the product to.
        @param product: The product object to add.
        @return True if the product was successfully added, False if not available.
        """
        self.lock_product.acquire() # @brief Acquire lock to protect shared product inventory and cart data.

        # Block Logic: Check if the product is currently available in the marketplace.
        if product not in self.products:
            self.lock_product.release() # Inline: Release lock before returning if product is not found.
            return False
        
        producer_id = product.owner # @brief Get the owner of the product.

        # Block Logic: Update the producer's queue size and remove the product from marketplace.
        self.producers_queue_sizes[producer_id] -= 1

        self.products.remove(product) # Inline: Remove product from the general marketplace inventory.

        # Block Logic: Add the product to the consumer's cart.
        self.carts[cart_id].append(product)

        self.lock_product.release() # @brief Release lock.

        return True

    def remove_from_cart(self, cart_id, product):
        """
        @brief Removes a product from a specified shopping cart.

        The product is returned to the marketplace and the producer's queue size is updated.
        This operation is not explicitly protected by a lock in the provided code,
        which could lead to race conditions if `remove_from_cart` and `add_to_cart`
        are called concurrently on the same product.

        @param cart_id: The ID of the cart to remove the product from.
        @param product: The product object to remove.
        """
        # Block Logic: Remove the product from the consumer's cart.
        self.carts[cart_id].remove(product)
        # Block Logic: Return the product to the general marketplace inventory.
        self.products.append(product)

        # Block Logic: Update the producer's queue size to reflect the returned product.
        producer_id = product.owner
        self.producers_queue_sizes[producer_id] += 1


    def place_order(self, cart_id):
        """
        @brief Finalizes a shopping cart and places an order.

        Transfers all products from the specified cart to a list representing the order
        and clears the cart. This operation is not explicitly protected by a lock in the
        provided code, which could lead to race conditions if multiple threads try to
        place orders for the same cart concurrently.

        @param cart_id: The ID of the cart to place an order for.
        @return A list of products that were in the placed order.
        """
        # Block Logic: Get the list of products from the specified cart.
        prod_list = self.carts[cart_id]
        
        # Block Logic: Clear the cart after placing the order.
        self.carts[cart_id] = []

        return prod_list


class Producer(Thread):
    """
    @brief Simulates a producer (seller) in the marketplace.

    Each producer thread registers with the marketplace, then continuously attempts
    to publish its products. It handles retries for failed publications.
    """
    

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        @brief Initializes a new Producer instance.

        @param products: A list of products to be published by this producer. Each product
                         is represented as a tuple: (product_type, quantity, wait_time).
        @param marketplace: The Marketplace instance to interact with.
        @param republish_wait_time: The time (in seconds) to wait before retrying a failed publication.
        @param kwargs: Additional keyword arguments passed to the Thread constructor (e.g., name).
        """
        Thread.__init__(self, **kwargs)
        self.products = products # @brief List of products this producer will offer.
        self.marketplace = marketplace # @brief Reference to the shared marketplace.
        self.republish_wait_time = republish_wait_time # @brief Time to wait before retrying a publication.
        self.kwargs = kwargs # @brief Stored kwargs for potential future use (e.g., thread name).

    def run(self):
        """
        @brief The main execution logic for the producer thread.

        Registers the producer with the marketplace, then continuously iterates
        through its list of products. For each product, it attempts to publish
        the specified quantity to the marketplace, retrying if publication fails.
        """
        producer_id = self.marketplace.register_producer() # @brief Register with the marketplace to get a unique ID.
        
        # Block Logic: Main loop for continuous product publishing.
        while True:
            # Block Logic: Iterate through each product this producer wants to sell.
            for product in self.products:
                type_prod = product[0] # @brief The product type.
                quantity = product[1] # @brief The quantity of this product to publish.
                wait_time = product[2] # @brief Time to wait after a successful publication.

                while quantity > 0:
                    ret = self.marketplace.publish(str(producer_id), type_prod) # @brief Attempt to publish the product.

                    # Block Logic: Handle publication outcome.
                    # If successful, decrement quantity and wait. Otherwise, wait and retry.
                    if ret:
                        time.sleep(wait_time) # Inline: Wait after successful publication.
                        quantity -= 1
                    else:
                        time.sleep(self.republish_wait_time) # Inline: Wait before retrying failed publication.


from dataclasses import dataclass


@dataclass(init=True, repr=True, order=False, frozen=False)
class Product:
    """
    @brief Base class for all products in the marketplace.

    Attributes:
        name (str): The name of the product.
        price (int): The price of the product.
        owner (int): The ID of the producer who owns this product (-1 by default if not owned).
    """
    name: str
    price: int
    owner = -1 # @brief The ID of the producer that owns this product.


@dataclass(init=True, repr=True, order=False, frozen=False)
class Tea(Product):
    """
    @brief Represents a specific type of tea product.

    Inherits from Product.

    Attributes:
        type (str): The type or variety of tea (e.g., "Green", "Black", "Herbal").
    """
    type: str
    owner = -1 # @brief Overrides owner from Product; the ID of the producer that owns this tea product.

@dataclass(init=True, repr=True, order=False, frozen=False)
class Coffee(Product):
    """
    @brief Represents a specific type of coffee product.

    Inherits from Product.

    Attributes:
        acidity (str): The acidity level of the coffee (e.g., "Low", "Medium", "High").
        roast_level (str): The roast level of the coffee (e.g., "Light", "Medium", "Dark").
    """
    acidity: str
    roast_level: str
    owner = -1 # @brief Overrides owner from Product; the ID of the producer that owns this coffee product.