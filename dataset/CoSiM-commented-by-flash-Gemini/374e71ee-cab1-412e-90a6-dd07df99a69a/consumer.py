
from threading import Thread, Lock
import time

class Consumer(Thread):
    """
    Represents a consumer in a marketplace simulation.

    Each consumer operates as a separate thread, creating a shopping cart,
    adding and removing products based on a list of operations, and finally
    placing an order. It handles retries for adding products to the cart.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        Initializes a new Consumer instance.

        Args:
            carts (list): A list of lists of product operations (add/remove)
                          that this consumer will perform.
            marketplace (Marketplace): The marketplace object with which
                                       this consumer will interact.
            retry_wait_time (float): The time in seconds to wait before
                                     retrying a failed 'add to cart' operation.
            **kwargs: Arbitrary keyword arguments passed to the Thread constructor.
        """
        Thread.__init__(self, **kwargs)
        self._carts = carts
        self._market = marketplace
        self.retry_time = retry_wait_time
        self._lock = Lock() # A lock to synchronize printing output (optional, but good practice).

    def run(self):
        """
        The main execution method for the consumer thread.

        It creates a new shopping cart in the marketplace, then processes
        a series of product operations. For 'add' operations, it repeatedly
        tries to add the product, waiting if the operation fails. For 'remove'
        operations, it simply removes the product. After processing all
        operations, it places the order and prints the purchased items.
        """
        cart_id = self._market.new_cart()
        for op_list in self._carts:

            # Iterate through each operation in the current list of operations.
            for op in op_list:
                op_type = op["type"]
                prod = op["product"]
                quantity = op["quantity"]

                # Pre-condition: If the operation is to "add" a product.
                if op_type == "add":
                    # Continuously try to add the product until the desired quantity is met.
                    while quantity > 0:
                        ret = self._market.add_to_cart(cart_id, prod)

                        if ret == True:
                            # If successful, decrement the remaining quantity.
                            quantity -= 1
                        else:
                            # If failed (e.g., product not available), wait and retry.
                            time.sleep(self.retry_time)

                # Pre-condition: If the operation is to "remove" a product.
                if op_type == "remove":
                    # Remove the product from the cart the specified number of times.
                    while quantity > 0:
                        self._market.remove_from_cart(cart_id, prod)
                        quantity -= 1 # Decrement quantity after removal.

            # Acquire the lock before placing the order and printing to prevent mixed output.
            with self._lock:
                # Place the order and get the list of products bought.
                products_list = self._market.place_order(cart_id)
                # Print each product that was successfully bought by this consumer.
                for prod in products_list:
                    print("cons" + str(cart_id) + " bought " + str(prod))

class Marketplace:
    """
    @brief Represents a central marketplace where producers publish products and consumers manage carts.
    It handles product inventory, producer queues, and cart operations with thread-safe mechanisms.
    """
    def __init__(self, queue_size_per_producer):
        """
        @brief Initializes the Marketplace.
        @param queue_size_per_producer: The maximum number of products a single producer can have
                                         published in the marketplace at any given time.
        """
        self._producers_queue = {}  # Dictionary to track the number of products each producer has published.
        self._carts = {}            # Dictionary to store products in each consumer's cart.
        self._id_carts = 0          # Counter for unique cart IDs.

        self._id_producers = 0      # Counter for unique producer IDs.
        self._products = []         # List of all currently available products in the marketplace.
        self._product_producer = {} # Dictionary mapping product to its producer ID.
        self._queue_size = queue_size_per_producer # Max products per producer.
        self._lock0 = Lock() # Lock for _id_producers (register_producer).
        self._lock1 = Lock() # Lock for _id_carts (new_cart).
        self._lock2 = Lock() # Lock for _carts, _products, _producers_queue, _product_producer (add_to_cart).

    def register_producer(self):
        """
        @brief Registers a new producer with the marketplace and assigns a unique ID.
        @return: The unique ID assigned to the new producer.
        """
        with self._lock0: # Acquire lock to ensure atomic increment of producer ID.
            self._id_producers += 1
        return self._id_producers

    def publish(self, producer_id, product):
        """
        @brief Publishes a product to the marketplace by a producer.
        Products are added if the producer's queue limit has not been reached.
        @param producer_id: The ID of the producer publishing the product.
        @param product: The product object to publish.
        @return: True if the product was successfully published, False otherwise.
        """
        # Pre-condition: Initialize producer's queue count if it's a new producer.
        if producer_id not in self._producers_queue:
            self._producers_queue[producer_id] = 0

        # Invariant: Check if the producer has reached its maximum queue size.
        if self._producers_queue[producer_id] >= self._queue_size:
            return False # Cannot publish if queue is full.

        # Increment the producer's published product count.
        self._producers_queue[producer_id] += 1

        # Add the product to the global list of available products.
        self._products.append(product)

        # Map the product to its producer.
        self._product_producer[product] = producer_id

        return True

    def new_cart(self):
        """
        @brief Creates a new shopping cart and assigns a unique cart ID.
        @return: The unique ID assigned to the new cart.
        """
        with self._lock1: # Acquire lock to ensure atomic increment of cart ID.
            self._id_carts += 1
        return self._id_carts

    def add_to_cart(self, cart_id, product):
        """
        @brief Adds a product to a specific cart.
        This operation is thread-safe and updates inventory accordingly.
        @param cart_id: The ID of the cart to add the product to.
        @param product: The product object to add.
        @return: True if the product was successfully added, False if the product is not available.
        """
        with self._lock2: # Acquire lock to protect shared marketplace data structures.
            # Pre-condition: Initialize cart if it's new.
            if cart_id not in self._carts:
                self._carts[cart_id] = []
            
            # Invariant: Check if the product is currently available in the marketplace.
            if product not in self._products:
                return False # Product not found, cannot add to cart.
            
            # Remove the product from the marketplace's available products list.
            self._products.remove(product)

            # Decrement the count of products published by the corresponding producer.
            pid = self._product_producer[product]
            self._producers_queue[pid] -= 1

            # Add the product to the consumer's cart.
            self._carts[cart_id].append(product)
        
        return True # Product successfully added.

    def remove_from_cart(self, cart_id, product):
        """
        @brief Removes a product from a specific cart.
        This operation returns the product to the marketplace inventory and updates producer's queue.
        @param cart_id: The ID of the cart to remove the product from.
        @param product: The product object to remove.
        """
        # Remove the product from the consumer's cart.
        self._carts[cart_id].remove(product)

        # Increment the count of products published by the corresponding producer.
        pid = self._product_producer[product]
        self._producers_queue[pid] += 1

        # Add the product back to the marketplace's available products list.
        self._products.append(product)

    def place_order(self, cart_id):
        """
        @brief Places an order for the items in a specific cart.
        The products are effectively "bought" and the cart is emptied.
        @param cart_id: The ID of the cart to place the order for.
        @return: A copy of the list of products that were in the cart.
        """
        # Create a copy of the products in the cart before emptying it.
        cart_prods_copy = self._carts[cart_id].copy()
        self._carts[cart_id] = [] # Empty the cart.
        
        return cart_prods_copy # Return the list of bought products.

import time
from threading import Thread

class Producer(Thread):
    """
    @brief Represents a producer thread that continuously publishes products to the marketplace.
    Producers attempt to republish products after a specified wait time, respecting marketplace queue limits.
    """
    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        @brief Initializes a Producer thread.
        @param products: A list of products (with quantity and republish time) this producer will offer.
        @param marketplace: The shared Marketplace instance to interact with.
        @param republish_wait_time: The time to wait between successful product publications.
        @param kwargs: Additional keyword arguments for the Thread constructor.
        """
        Thread.__init__(self, **kwargs)
        self._prods = products
        self._market = marketplace
        self._id = marketplace.register_producer() # Register with marketplace to get a unique producer ID.
        self._rwait_time = republish_wait_time

    def run(self):
        """
        @brief The main execution loop for the producer thread.
        Continuously attempts to publish products to the marketplace.
        """
        # Invariant: Loop indefinitely to continuously publish products.
        while True:

            # Block Logic: Iterate through the list of products this producer offers.
            for product in self._prods:
                prod = product[0] # The product object.
                quantity = product[1] # The initial quantity to publish.
                repub_time = product[2] # Time to wait if publishing fails.

                # Invariant: Try to publish the specified quantity of the current product.
                while quantity > 0:
                    ret = self._market.publish(self._id, prod)
                    if ret is True:
                        # If publishing is successful, wait for the republish_wait_time.
                        time.sleep(self._rwait_time)
                        quantity -= 1 # Decrement quantity.
                    else:
                        # If publishing fails (e.g., marketplace queue full), wait for repub_time.
                        time.sleep(repub_time)


from dataclasses import dataclass

@dataclass(init=True, repr=True, order=False, frozen=True)
class Product:
    """
    @brief Base dataclass representing a generic product.
    * Data Structure: Immutable data class for product attributes.
    """
    name: str # The name of the product.
    price: int # The price of the product.

@dataclass(init=True, repr=True, order=False, frozen=True)
class Tea(Product):
    """
    @brief Dataclass representing a tea product, inheriting from Product.
    * Data Structure: Immutable data class for tea product attributes.
    """
    type: str # The type of tea (e.g., "Green", "Black").

@dataclass(init=True, repr=True, order=False, frozen=True)
class Coffee(Product):
    """
    @brief Dataclass representing a coffee product, inheriting from Product.
    * Data Structure: Immutable data class for coffee product attributes.
    """
    acidity: str # The acidity level of the coffee.
    roast_level: str # The roast level of the coffee.