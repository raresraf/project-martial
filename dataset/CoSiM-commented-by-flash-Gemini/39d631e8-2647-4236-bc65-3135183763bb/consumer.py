


from threading import Thread
import time

class Consumer(Thread):
    """
    Represents a consumer in a marketplace simulation.

    Each consumer operates as a separate thread, creating shopping carts,
    performing add and remove operations on products based on a list of
    predefined actions, and finally placing orders. It handles retries
    for operations that might temporarily fail (e.g., due to product unavailability).
    """
    

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        Initializes a new Consumer instance.

        Args:
            carts (list): A list of shopping carts, where each cart is a list of
                          product operations (dictionaries containing 'type', 'product', 'quantity').
            marketplace (Marketplace): The marketplace object with which
                                       this consumer will interact.
            retry_wait_time (float): The time in seconds to wait before
                                     retrying a failed 'add to cart' operation.
            **kwargs: Arbitrary keyword arguments passed to the Thread constructor.
        """
        Thread.__init__(self, **kwargs)

        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time

    def run(self):
        """
        The main execution method for the consumer thread.

        It iterates through each predefined shopping cart. For each cart, it
        creates a new cart in the marketplace, then processes each operation
        (add/remove products). Operations are retried if they fail (e.g.,
        product not available). After all operations for a cart are processed,
        the order is placed.
        """
        # Iterate through each shopping cart definition provided to this consumer.
        for crt_cart in self.carts:
            # Create a new cart in the marketplace and get its unique ID.
            cart_id = self.marketplace.new_cart()

            # Process each operation (add or remove product) within the current cart.
            for crt_operation in crt_cart:

                number_of_operations = 0
                # Loop until the desired quantity for the current operation is met.
                while number_of_operations < crt_operation["quantity"]:

                    op_product = crt_operation["product"]

                    # Perform either 'add' or 'remove' operation based on the operation type.
                    if crt_operation["type"] == "add":
                        return_code = self.marketplace.add_to_cart(cart_id, op_product)
                    elif crt_operation["type"] == "remove":
                        return_code = self.marketplace.remove_from_cart(cart_id, op_product)

                    # If the operation was successful (True) or if it's a remove operation
                    # that typically doesn't return a boolean status (None).
                    if return_code == True or return_code is None:
                        number_of_operations += 1 # Increment successful operations.
                    else:
                        # If the operation failed, wait before retrying.
                        time.sleep(self.retry_wait_time)

            # After processing all operations for the current cart, place the order.
            self.marketplace.place_order(cart_id)


from threading import Lock, currentThread


class Marketplace:
    """
    Simulates a central marketplace for producers and consumers to interact.

    It manages product inventory, producer queues, shopping carts for consumers,
    and provides thread-safe operations for registering producers, publishing
    products, creating carts, adding/removing products from carts, and
    placing orders.
    """
    
    def __init__(self, queue_size_per_producer):
        """
        Initializes the Marketplace with configuration for producer queues.

        Args:
            queue_size_per_producer (int): The maximum number of products
                                           a single producer can have in the marketplace
                                           at any given time.
        """
        self.queue_size_per_producer = queue_size_per_producer
        # List to track the current number of products for each registered producer.
        self.sizes_per_producer = [] 

        # Dictionary to store consumer carts, keyed by cart_id.
        self.carts = {} 

        # Counter for generating unique cart IDs.
        self.number_of_carts = 0

        # List of all products currently available in the marketplace.
        self.products = [] 
        # Dictionary mapping each product to its producer_id.
        self.producers = {} 

        # Lock to protect access to self.sizes_per_producer.
        self.lock_for_sizes = Lock() 
        # Lock to protect access to cart-related operations (new_cart, carts dict).
        self.lock_for_carts = Lock() 
        # Lock to protect producer registration.
        self.lock_for_register = Lock() 
        # Lock to protect print statements to prevent interleaved output.
        self.lock_for_print = Lock() 

        def register_producer(self):
            """
            Registers a new producer with the marketplace and assigns a unique ID.
    
            Returns:
                int: The unique producer ID.
            """
            with self.lock_for_register:
                # Assign a new producer ID based on the current number of registered producers.
                producer_id = len(self.sizes_per_producer)
            # Initialize the product count for this new producer.
            self.sizes_per_producer.append(0)
            return producer_id
    
        def publish(self, producer_id, product):
            """
            Allows a producer to publish a product to the marketplace.
    
            The product is added only if the producer has not exceeded its
            queue size limit.
    
            Args:
                producer_id (str): The ID of the producer.
                product (object): The product to be published.
    
            Returns:
                bool: True if the product was published successfully, False otherwise.
            """
            num_prod_id = int(producer_id) # Ensure producer_id is an integer.
    
            max_size = self.queue_size_per_producer
            crt_size = self.sizes_per_producer[num_prod_id]
    
            # Check if the producer has exceeded its allowed queue size.
            if crt_size >= max_size:
                return False
    
            # Increment the product count for this producer in a thread-safe manner.
            with self.lock_for_sizes:
                self.sizes_per_producer[num_prod_id] += 1
            # Add the product to the global list of available products.
            self.products.append(product)
            # Record which producer published this product.
            self.producers[product] = num_prod_id
    
            return True
    
        def new_cart(self):
            """
            Creates a new shopping cart and assigns a unique cart ID.
    
            Returns:
                int: The unique cart ID.
            """
            ret_id = 0
            # Increment the global cart counter in a thread-safe manner.
            with self.lock_for_carts:
                self.number_of_carts += 1
                ret_id = self.number_of_carts
    
            # Initialize an empty list for the new cart.
            self.carts[ret_id] = []
    
            return ret_id
    
        def add_to_cart(self, cart_id, product):
            """
            Adds a product to a specified shopping cart.
    
            This operation is atomic and ensures that a product is only added
            if it's available in the marketplace. It also updates producer counts.
    
            Args:
                cart_id (int): The ID of the cart to add the product to.
                product (object): The product to add.
    
            Returns:
                bool: True if the product was successfully added, False otherwise.
            """
            with self.lock_for_sizes:
                # Check if the product is available in the marketplace.
                if product not in self.products:
                    return False
    
                # Remove the product from the global list of available products.
                self.products.remove(product)
    
                # Decrement the product count for the producer who published this product.
                producer = self.producers[product]
                self.sizes_per_producer[producer] -= 1
    
            # Add the product to the consumer's cart.
            self.carts[cart_id].append(product)
            return True
    
        def remove_from_cart(self, cart_id, product):
            """
            Removes a product from a specified shopping cart and returns it to the marketplace.
    
            Args:
                cart_id (int): The ID of the cart to remove the product from.
                product (object): The product to remove.
            """
            # Remove the product from the consumer's cart.
            self.carts[cart_id].remove(product)
    
            # Increment the product count for the producer who published this product in a thread-safe manner.
            with self.lock_for_sizes:
                producer = self.producers[product]
                self.sizes_per_producer[producer] += 1
    
            # Add the product back to the global list of available products.
            self.products.append(product)
    
        def place_order(self, cart_id):
            """
            Processes a shopping cart, effectively "buying" the items and printing
            what was bought. The cart is then cleared.
    
            Args:
                cart_id (int): The ID of the cart to place the order for.
    
            Returns:
                list: A copy of the list of products that were in the cart.
            """
            # Retrieve the products from the cart and clear the cart.
            product_list = self.carts.pop(cart_id, None)
    
            # Print each product that was bought in a thread-safe manner.
            for prod in product_list:
                with self.lock_for_print:
                    print(str(currentThread().getName()) + " bought " + str(prod))
    
            return product_list


from threading import Thread
import time

class Producer(Thread):
    """
    Represents a producer in a marketplace simulation.

    Each producer operates as a separate thread, continuously attempting to
    publish products to the marketplace. It manages a list of products to
    publish, retrying if the marketplace's limits are reached, and waits
    for specified durations between publishing attempts.
    """
    

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        Initializes a new Producer instance.

        Args:
            products (list): A list of products this producer will offer,
                             each item being a tuple (product_object, quantity, wait_time).
            marketplace (Marketplace): The marketplace object with which
                                       this producer will interact.
            republish_wait_time (float): The time in seconds to wait before
                                         retrying to publish a product if the
                                         marketplace queue is full.
            **kwargs: Arbitrary keyword arguments passed to the Thread constructor.
        """
        Thread.__init__(self, **kwargs)

        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        # Register this producer with the marketplace and get a unique ID.
        self.producer_id = self.marketplace.register_producer()

    def run(self):
        """
        The main execution method for the producer thread.

        It continuously iterates through its list of products. For each product,
        it attempts to publish the specified quantity to the marketplace. If
        publishing fails (e.g., marketplace queue is full), it waits for
        `republish_wait_time` before retrying. If successful, it waits for
        `product_wait_time` before attempting to publish the next item.
        The loop `while 69 - 420 < 3` is a constant `while True` loop, ensuring
        continuous operation until the program terminates externally.
        """
        # The condition '69 - 420 < 3' simplifies to 'while True', meaning the producer runs indefinitely.
        while 69 - 420 < 3:

            # Iterate through the list of products this producer is offering.
            for (product, number_of_products, product_wait_time) in self.products:

                i = 0
                # Loop to publish the specified quantity of the current product.
                while i < number_of_products:
                    # Attempt to publish the product to the marketplace.
                    return_code = self.marketplace.publish(str(self.producer_id), product)

                    if not return_code: 
                        # If publishing failed (marketplace queue full), wait and retry.
                        time.sleep(self.republish_wait_time)
                    else:
                        # If successful, wait for the specified product wait time and increment count.
                        time.sleep(product_wait_time)
                        i += 1


from dataclasses import dataclass


@dataclass(init=True, repr=True, order=False, frozen=True)
class Product:
    """
    Represents a generic product with basic attributes.

    Attributes:
        name (str): The name of the product.
        price (int): The price of the product.
    """
    name: str
    price: int


@dataclass(init=True, repr=True, order=False, frozen=True)
class Tea(Product):
    """
    Represents a Tea product, inheriting from Product, with an additional 'type' attribute.

    Attributes:
        type (str): The type of tea (e.g., "Green", "Black", "Herbal").
    """
    type: str


@dataclass(init=True, repr=True, order=False, frozen=True)
class Coffee(Product):
    """
    Represents a Coffee product, inheriting from Product, with additional
    'acidity' and 'roast_level' attributes.

    Attributes:
        acidity (str): The acidity level of the coffee.
        roast_level (str): The roast level of the coffee.
    """
    acidity: str
    roast_level: str
