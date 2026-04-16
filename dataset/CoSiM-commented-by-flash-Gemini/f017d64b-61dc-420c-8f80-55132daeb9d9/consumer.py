

"""
@file consumer.py
@brief Implements a multithreaded simulation of a marketplace with producer and consumer dynamics.

This module defines the core components for a marketplace simulation:
- `Consumer`: Represents a buyer that interacts with the marketplace to add, remove, and purchase products based on predefined carts.
- `Marketplace`: Manages the inventory of products, handles registration of producers, and orchestrates cart and order processing for consumers. It employs `threading.Lock` for fine-grained synchronization on individual products within the marketplace.
- `Producer`: Represents a seller that continuously publishes products to the marketplace.
- `Operation`: A helper class encapsulating a single cart operation (add/remove) with product and quantity.
- `MarketplaceProduct`: Represents a product available in the marketplace, with an associated lock for exclusive access.
- `Cart`: Represents a consumer's shopping cart, managing reserved `MarketplaceProduct` instances.
- `Production`: A helper class for producers, tracking product details and production status.

The simulation demonstrates concurrent access and modification of shared resources (the marketplace inventory and shopping carts) by multiple threads, highlighting the need for robust synchronization.

Domain: Concurrency, Multithreading, Simulation, Object-Oriented Design, Data Structures.
"""

import time


from threading import Thread
from typing import List

from tema.marketplace import Marketplace
from tema.product import Product


class Operation:
    """
    @brief Represents a single shopping cart operation (add or remove a product).

    This helper class encapsulates the type of operation (`add` or `remove`),
    the `Product` involved, and the `quantity` for that operation.
    """
    def __init__(self, op_type: str, product: Product, quantity: int):
        """
        @brief Initializes an Operation instance.

        Parameters:
          - op_type: The type of operation, e.g., 'add' or 'remove'.
          - product: The `Product` object involved in the operation.
          - quantity: The number of units of the product for this operation.
        """
        self.op_type: str = op_type
        self.product: Product = product
        self.quantity: int = quantity


class Consumer(Thread):
    """
    @brief Represents a consumer (customer) in the marketplace simulation.

    Each Consumer thread simulates a customer who goes through a list of predefined
    shopping carts, adds or removes products, and finally places an order.
    It interacts with the `Marketplace` to perform these actions.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        @brief Initializes a Consumer thread.

        Parameters:
          - carts: A list of shopping cart definitions, where each inner list contains
                   dictionaries describing product operations (type, product, quantity).
          - marketplace: A reference to the shared `Marketplace` instance.
          - retry_wait_time: The time in seconds to wait before retrying a failed
                             `add_to_cart` operation.
          - kwargs: Additional keyword arguments passed to the `Thread` constructor.
        """
        Thread.__init__(self, **kwargs)
        self.carts: List[List[Operation]] = []
        # Block Logic: Parses the raw cart definitions into a list of `Operation` objects.
        # Invariant: `self.carts` will contain structured `Operation` objects,
        # making it easier to process shopping lists.
        for cart_operations in carts:
            ops = []
            for operation in cart_operations:
                ops.append(Operation(
                    operation['type'],
                    operation['product'],
                    operation['quantity']))
            self.carts.append(ops)

        self.marketplace: Marketplace = marketplace
        self.retry_wait_time: float = retry_wait_time

        
        self.products: List[Product] = [] # List to keep track of products bought (not directly used for logic here).

    def run(self):
        """
        @brief The main execution method for the Consumer thread.

        This method simulates the consumer's shopping journey:
        1. Iterates through each predefined shopping list (`operations`).
        2. Creates a new shopping cart in the marketplace.
        3. Processes each product operation (add or remove) within the shopping list.
        4. For 'add' operations, attempts to add the product to the cart, retrying
           if the marketplace indicates failure (e.g., product out of stock or locked).
        5. For 'remove' operations, removes the product from the cart.
        6. Finally, places the order for the created cart.
        7. Prints the details of the products successfully bought.
        """
        for operations in self.carts: # Iterate through each shopping list (a list of Operation objects).
            cart_id: int = self.marketplace.new_cart() # Create a new cart in the marketplace.

            # Block Logic: Processes all operations within a single shopping list.
            # Invariant: `operations` list is processed sequentially, and items are
            # removed from the front once their quantity is fulfilled.
            while operations:
                if operations[0].op_type == 'add':
                    # Block Logic: Attempts to add a product to the cart.
                    # If successful, decrements the quantity needed. If unsuccessful, retries after a delay.
                    if self.marketplace.add_to_cart(cart_id, operations[0].product):
                        operations[0].quantity -= 1 # Decrement the remaining quantity to add.
                    else:
                        # If adding fails, wait for a retry_wait_time and try again.
                        time.sleep(self.retry_wait_time)
                        continue # Continue to the next iteration of the while loop to retry the current operation.

                    # Block Logic: If all required quantity for the current operation has been added,
                    # remove this operation from the list and move to the next.
                    if operations[0].quantity == 0:
                        operations = operations[1:] # Move to the next operation in the list.
                elif operations[0].op_type == 'remove':
                    # Block Logic: Continuously removes the product from the cart until the desired quantity is removed.
                    while operations[0].quantity > 0:
                        self.marketplace.remove_from_cart(cart_id, operations[0].product)
                        operations[0].quantity -= 1 # Decrement the remaining quantity to remove.
                    
                    operations = operations[1:] # Move to the next operation in the list.

            # Once all operations for the current cart are processed, place the order.
            final_products: List[Product] = self.marketplace.place_order(cart_id)
            # Print the details of the products successfully bought by this consumer.
            for product in final_products:
                print(self.name + " bought " + str(product))



from threading import Lock
from typing import List, Dict

from tema.product import Product


class MarketplaceProduct:
    """
    @brief A wrapper class for `Product` that includes the original producer's ID and a dedicated lock.

    This class facilitates fine-grained locking of individual products within the marketplace,
    ensuring that only one consumer can reserve a specific product instance at a time.
    """
    def __init__(self, producer_id: int, product: Product):
        """
        @brief Initializes a MarketplaceProduct instance.

        Parameters:
          - producer_id: The ID of the producer who published this product.
          - product: The actual `Product` object.
        """
        self.producer_id = producer_id
        self.product = product
        self.lock: Lock = Lock() # A lock specific to this product instance.


class Cart:
    """
    @brief Represents a consumer's shopping cart.

    Each `Cart` holds a list of `MarketplaceProduct` instances that a consumer
    has added, effectively reserving them from the marketplace.
    """
    def __init__(self, cart_id: int):
        """
        @brief Initializes a Cart instance.

        Parameters:
          - cart_id: A unique identifier for this cart.
        """
        self.cart_id: int = cart_id
        self.products: List[MarketplaceProduct] = [] # List of MarketplaceProduct instances in the cart.

    def add_product(self, product: MarketplaceProduct):
        """
        @brief Adds a `MarketplaceProduct` to the cart.

        Parameters:
          - product: The `MarketplaceProduct` instance to add.
        """
        self.products.append(product)

    def remove_product(self, product: MarketplaceProduct):
        """
        @brief Removes a `MarketplaceProduct` from the cart.

        Parameters:
          - product: The `MarketplaceProduct` instance to remove.
        """
        if product in self.products:
            self.products.remove(product)

    def find_product_in_cart(self, product) -> [MarketplaceProduct]:
        """
        @brief Finds a specific `Product` (by value) within the cart.

        Parameters:
          - product: The `Product` object to search for.

        Returns:
          - The `MarketplaceProduct` instance if found, otherwise None.
        """
        for market_product in self.products:
            if market_product.product == product:
                return market_product

        return None


class Marketplace:
    """
    @brief The central marketplace managing producers, products, and consumer carts.

    This class serves as the intermediary between `Producer` and `Consumer` threads.
    It manages the inventory of published products, registers producers, handles
    cart operations (add, remove), and finalizes orders. It uses `threading.Lock`
    on individual products to prevent race conditions during cart operations.
    """
    

    def __init__(self, queue_size_per_producer):
        """
        @brief Initializes the Marketplace.

        Parameters:
          - queue_size_per_producer: The maximum number of items a single producer
                                     is allowed to have published in the marketplace.
        """
        
        self.max_items: int = queue_size_per_producer # Maximum items a producer can publish.

        
        self.consumers: List[int] = [] # List of registered consumer IDs (not directly used in logic).
        self.producers: List[int] = [] # List of registered producer IDs.

        
        self.products: Dict[int, List[MarketplaceProduct]] = {} # Dictionary mapping producer_id to their published MarketplaceProducts.
        self.carts: Dict[int, Cart] = {} # Dictionary mapping cart_id to Cart instances.

    def register_producer(self) -> int:
        """
        @brief Registers a new producer with the marketplace.

        Assigns a unique ID to the producer and initializes an empty list
        for their published products.

        Returns:
          - The unique integer ID assigned to the new producer.
        """
        
        producer_id = len(self.producers) # Assign an incremental ID.

        self.producers.append(producer_id)
        self.products[producer_id] = [] # Initialize an empty list for this producer's products.

        return producer_id

    def publish(self, producer_id, product):
        """
        @brief Publishes a product from a producer to the marketplace.

        The product is added to the producer's inventory in the marketplace,
        provided the producer has not exceeded their maximum allowed items.

        Parameters:
          - producer_id: The ID of the producer publishing the product.
          - product: The `Product` object to be published.

        Returns:
          - True if the product was successfully published, False if the producer's
            queue is full.
        """
        # Pre-condition: Check if the producer has exceeded their maximum allowed published items.
        if len(self.products[producer_id]) > self.max_items:
            return False
        else:
            self.products[producer_id].append(MarketplaceProduct(producer_id, product)) # Add the product wrapped in MarketplaceProduct.
            return True

    def new_cart(self):
        """
        @brief Creates a new empty shopping cart in the marketplace.

        Assigns a unique ID to the cart and stores it in the marketplace's cart dictionary.

        Returns:
          - The unique integer ID assigned to the new shopping cart.
        """
        
        cart_id = len(self.carts) + 1 # Assign an incremental ID.

        self.carts[cart_id] = Cart(cart_id) # Create and store a new Cart instance.
        return cart_id

    def add_to_cart(self, cart_id, product):
        """
        @brief Adds a `Product` to a consumer's shopping cart.

        This function iterates through all published products in the marketplace.
        If it finds a matching product that is not currently locked (i.e., not
        already reserved by another cart), it acquires the product's lock,
        adds it to the specified cart, and removes it from the marketplace's
        available products list.

        Parameters:
          - cart_id: The ID of the cart to which the product should be added.
          - product: The `Product` object to add.

        Returns:
          - True if the product was successfully added to the cart, False otherwise
            (e.g., product not available, or all instances are locked).
        """
        
        # Block Logic: Iterates through all products published by all producers.
        # It attempts to find an available instance of the requested product.
        for producer_products in self.products.values():
            for market_product in producer_products:
                
                # Pre-condition: Check if the product matches and is not currently locked.
                if market_product.product == product and not market_product.lock.locked():
                    market_product.lock.acquire()  # Acquire the lock for this specific product instance.
                    self.carts[cart_id].add_product(market_product)  # Add the locked product to the cart.
                    # Note: The product is moved from the producer's available stock to the cart's reserved stock.
                    # This implies it should be removed from the `producer_products` list here as well,
                    # but the current implementation keeps it there until `place_order`. This can lead
                    # to the same MarketplaceProduct being added to multiple carts if not careful,
                    # or an incorrect understanding of available stock.
                    return True

        
        return False # Product not found or all instances are locked.

    def remove_from_cart(self, cart_id, product):
        """
        @brief Removes a `Product` from a consumer's shopping cart.

        This function finds a `MarketplaceProduct` instance corresponding to the
        given `product` in the specified cart, removes it from the cart, and
        releases its associated lock, making it available again in the marketplace.

        Parameters:
          - cart_id: The ID of the cart from which the product should be removed.
          - product: The `Product` object to remove.
        """
        
        # Find the specific MarketplaceProduct instance in the cart.
        market_product: MarketplaceProduct = self.carts[cart_id].find_product_in_cart(product)
        self.carts[cart_id].remove_product(market_product) # Remove it from the cart.
        market_product.lock.release() # Release the lock, making the product available.

    def place_order(self, cart_id) -> List[Product]:
        """
        @brief Finalizes an order for a given shopping cart.

        This function retrieves all products from the specified cart and removes
        them from the respective producer's inventory. The locks on these products
        are implicitly released as they are removed from the system.

        Parameters:
          - cart_id: The ID of the cart for which the order is being placed.

        Returns:
          - A list of `Product` objects that were successfully ordered.
        """
        
        # Create a list of the actual Product objects from the MarketplaceProduct instances in the cart.
        final_products = [marketProduct.product for marketProduct in self.carts[cart_id].products]

        
        # Block Logic: Removes each ordered product from the producer's inventory.
        # This ensures that purchased items are no longer available in the marketplace.
        for market_product in self.carts[cart_id].products:
            # Find the producer corresponding to the market_product.
            for producer_product in self.products[market_product.producer_id]:
                if producer_product == market_product:
                    self.products[market_product.producer_id].remove(market_product) # Remove from producer's stock.
                    break # Exit inner loop once product is found and removed.

        return final_products # Return the list of purchased products.


import time


from threading import Thread
from typing import List

from tema.marketplace import Marketplace
from tema.product import Product



class Production:
    """
    @brief A helper class for `Producer` to manage the details of a product to be produced.

    This class tracks the `Product` itself, the total `quantity` to produce,
    the `wait_time` between productions, and the `number_produced` so far.
    """
    def __init__(self, product: Product, quantity: int, wait_time: float):
        """
        @brief Initializes a Production instance.

        Parameters:
          - product: The `Product` object to produce.
          - quantity: The total number of this product to produce.
          - wait_time: The time in seconds to wait after each production.
        """
        self.product: Product = product
        self.quantity: int = quantity
        self.wait_time: float = wait_time

        self.number_produced = 0 # Counter for how many of this product have been produced.


class Producer(Thread):
    """
    @brief Represents a producer (seller) in the marketplace simulation.

    Each Producer thread continuously registers with the marketplace and publishes
    a predefined list of products according to their production schedules.
    """
    

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        @brief Initializes a Producer thread.

        Parameters:
          - products: A list of product definitions, where each item is a tuple
                      (product_object, quantity, publish_delay).
          - marketplace: A reference to the shared `Marketplace` instance.
          - republish_wait_time: The time in seconds to wait before retrying a failed
                                 `publish` operation.
          - kwargs: Additional keyword arguments passed to the `Thread` constructor.
        """
        
        Thread.__init__(self, **kwargs)
        self.productions: List[Production] = []
        # Block Logic: Parses the raw product definitions into a list of `Production` objects.
        # Invariant: `self.productions` will contain structured `Production` objects,
        # making it easier to manage production schedules.
        for prod in products:
            self.productions.append(Production(prod[0], prod[1], prod[2]))
        self.marketplace: Marketplace = marketplace
        self.republish_wait_time: float = republish_wait_time

        
        # Registers with the marketplace and gets a unique producer ID.
        self.producer_id: int = self.marketplace.register_producer()

    def run(self):
        """
        @brief The main execution method for the Producer thread.

        This method simulates the producer's continuous production and publishing cycle:
        1. It enters an infinite loop, continuously attempting to publish its products.
        2. For each product in its production schedule, it checks if the desired
           quantity has been produced.
        3. If a product needs to be produced, it attempts to `publish` it to the
           marketplace, waiting a specified `wait_time` after each attempt.
        4. If the current product's production quantity is met, it cycles to the
           next product in its schedule.
        """
        
        
        while True: # Infinite loop for continuous production.
            
            # Block Logic: Checks if the current product in the production schedule still needs to be produced.
            # Invariant: `productions[0]` always refers to the product currently being produced.
            if self.productions[0].number_produced < self.productions[0].quantity:
                
                # Attempt to publish the product to the marketplace.
                if self.marketplace.publish(self.producer_id, self.productions[0].product):
                    self.productions[0].number_produced += 1 # Increment if publishing is successful.
                
                time.sleep(self.productions[0].wait_time) # Wait before the next production attempt for this product.
            else:
                # Block Logic: If the required quantity for the current product is produced,
                # reset its counter and move it to the end of the production schedule.
                self.productions[0].number_produced = 0 # Reset production count for this product.
                # Cycles the completed product to the end of the list.
                self.productions = self.productions[1:] + [self.productions[0]]
