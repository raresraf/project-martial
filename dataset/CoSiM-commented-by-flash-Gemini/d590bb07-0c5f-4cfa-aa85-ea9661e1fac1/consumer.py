"""
This module defines classes for a simulated multi-threaded marketplace system, including:
- Consumer: Represents a buyer interacting with the marketplace, handling cart operations.
- Producer: Represents a seller supplying products to the marketplace.
- Marketplace: Manages product inventory, producer queues, consumer carts, and ensures thread safety.
- Product, Tea, Coffee: Data classes for defining various product types.

The system uses threading primitives like Locks and Threads for concurrent operations.
"""


from threading import Thread
import time


class Consumer(Thread):
    """
    Represents a consumer (buyer) in the marketplace. Inherits from `threading.Thread`.
    Consumers interact with the marketplace to add and remove products from their cart,
    and finally place an order.
    """
    

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        Initializes a Consumer thread.

        Args:
            carts (list): A list of products the consumer intends to buy.
            marketplace (Marketplace): The marketplace instance the consumer interacts with.
            retry_wait_time (float): Time in seconds to wait before retrying an action (e.g., adding to cart).
            **kwargs: Arbitrary keyword arguments passed to the Thread.__init__ method,
                      including 'name' for the consumer's identifier.
        """
        
        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.kwargs = kwargs

    def consumer_add_to_cart(self, quantity, cart_id, product_id):
        """
        Attempts to add a specified quantity of a product to the consumer's cart.
        Retries adding the product if the marketplace fails the operation,
        waiting for `retry_wait_time` between attempts.

        Args:
            quantity (int): The number of units of the product to add.
            cart_id (int): The ID of the cart to which the product should be added.
            product_id (str): The identifier of the product to add.
        """
        
        counter = 0
        # Block Logic: Loop until the desired quantity of the product has been added to the cart.
        while counter < quantity:
            # Block Logic: Attempt to add the product to the cart.
            if not self.marketplace.add_to_cart(cart_id, product_id):
                # Block Logic: If adding to cart fails, wait for a specified time before retrying.
                time.sleep(self.retry_wait_time)
            else:
                # Block Logic: If successful, increment the counter.
                counter = counter + 1

    def run(self):
        """
        Executes the consumer's shopping logic.
        - Creates a new cart in the marketplace.
        - Iterates through the list of desired items, adding or removing them from the cart.
        - Places the final order and prints the purchased products.
        """
        
        cart_id = self.marketplace.new_cart()
        
        # Block Logic: Process each cart provided to the consumer.
        # Invariant: Each 'cart' element represents a list of purchase commands.
        for cart in self.carts:
            # Block Logic: Process each entry (command) within the current cart.
            # Pre-condition: 'entry' is a dictionary containing 'type', 'quantity', and 'product'.
            for entry in cart:
                
                if entry.get("type") == "remove": # Block Logic: Handle product removal from the cart.
                    
                    for _ in range(entry.get("quantity")): # Block Logic: Remove the specified quantity of the product.
                        self.marketplace.remove_from_cart(cart_id, entry.get("product"))
                else: # Block Logic: Handle product addition to the cart.
                    
                    self.consumer_add_to_cart(entry.get("quantity"), cart_id, entry.get("product"))
        
        # Block Logic: Place the final order for all items in the cart and print the purchased products.
        for product in self.marketplace.place_order(cart_id):
            print(self.name, "bought", product)


from threading import Lock


class Marketplace:
    """
    Manages the overall marketplace operations, including producers, consumers,
    product queues, and cart processing. Ensures thread-safe access to shared resources
    using individual locks for producers and consumers.
    """
    
    def __init__(self, queue_size_per_producer):
        """
        Initializes the Marketplace with a specified queue size per producer.

        Args:
            queue_size_per_producer (int): The maximum number of products
                                           a single producer can have in its queue.
        """
        
        self.queue_size_per_producer = queue_size_per_producer
        self.producer_id = -1             # Counter for assigning unique producer IDs
        self.producer_queue = {}          # Dictionary to store product queues for each producer
        self.producer_lock = Lock()       # Lock to protect producer-related shared data
        self.consumer_id = -1             # Counter for assigning unique consumer (cart) IDs
        self.consumer_queue = {}          # Dictionary to store carts for each consumer
        self.consumer_lock = Lock()       # Lock to protect consumer-related shared data

    def register_producer(self):
        """
        Registers a new producer with the marketplace and assigns a unique ID.

        Returns:
            int: The unique ID assigned to the new producer.
        """
        
        self.producer_id = self.producer_id + 1
        
        self.producer_queue[self.producer_id] = [] # Initialize an empty product queue for the new producer
        return self.producer_id

    def publish(self, producer_id, product):
        """
        Publishes a product to the marketplace from a specific producer.
        The product is added to the producer's queue if there is space.
        Ensures thread-safe access to the producer's queue.

        Args:
            producer_id (int): The ID of the producer publishing the product.
            product (str): The name of the product to publish.

        Returns:
            bool: True if the product was successfully published, False otherwise.
        """
        
        # Block Logic: Check if the producer's queue has space.
        # Pre-condition: `producer_id` refers to a valid, registered producer.
        if len(self.producer_queue.get(producer_id)) < self.queue_size_per_producer:
            # Block Logic: Acquire producer lock before modifying shared producer queue.
            self.producer_lock.acquire()
            self.producer_queue.get(producer_id).append(product)
            self.producer_lock.release()
            return True
        return False

    def new_cart(self):
        """
        Creates a new shopping cart for a consumer and assigns a unique ID.

        Returns:
            int: The unique ID assigned to the new cart.
        """
        
        self.consumer_id = self.consumer_id + 1
        
        self.consumer_queue[self.consumer_id] = [] # Initialize an empty cart for the new consumer
        return self.consumer_id

    def add_to_cart(self, cart_id, product):
        """
        Adds a product to a specific cart.
        It searches through all producer queues for the product, removes it from the first found,
        and adds it to the consumer's cart. Ensures thread-safe operation.

        Args:
            cart_id (int): The ID of the cart to which the product should be added.
            product (str): The name of the product to add.

        Returns:
            bool: True if the product was successfully added to the cart, False otherwise.
        """
        
        # Block Logic: Iterate through all producer queues to find the desired product.
        for producer in self.producer_queue:
            for item in self.producer_queue.get(producer): # Invariant: `item` is a product in the current producer's queue.
                if item == product:
                    # Block Logic: Acquire consumer lock before modifying shared consumer queue.
                    self.consumer_lock.acquire()
                    self.consumer_queue.get(cart_id).append([product, producer]) # Add product to consumer's cart.
                    self.producer_queue.get(producer).remove(product) # Remove product from producer's queue.
                    self.consumer_lock.release()
                    return True
        return False

    def remove_from_cart(self, cart_id, product):
        """
        Removes a product from a specific cart and returns it to its original producer's queue.
        Ensures thread-safe operation.

        Args:
            cart_id (int): The ID of the cart from which the product should be removed.
            product (str): The name of the product to remove.
        """
        
        # Block Logic: Iterate through items in the specified cart to find the product.
        # Invariant: `item` is the product name, `producer` is its original producer's ID.
        for item, producer in self.consumer_queue.get(cart_id):
            if item == product:
                self.consumer_queue.get(cart_id).remove([product, producer]) # Remove from consumer's cart.
                # Block Logic: Acquire producer lock before modifying shared producer queue.
                self.producer_lock.acquire()
                self.producer_queue.get(producer).append(product) # Return to producer's queue.
                self.producer_lock.release()
                break

    def place_order(self, cart_id):
        """
        Places an order for all items currently in the specified cart.
        The items are removed from the cart and returned as a list of product names.

        Args:
            cart_id (int): The ID of the cart for which to place the order.

        Returns:
            list: A list of products (names) that were part of the placed order.
        """
        
        products = []
        # Block Logic: Extract only product names from the cart entries.
        for product, _ in self.consumer_queue.get(cart_id):
            products.append(product)
        return products


from threading import Thread
import time


class Producer(Thread):
    """
    Represents a producer (seller) in the marketplace. Inherits from `threading.Thread`.
    Producers register with the marketplace and continuously publish products.
    """
    

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        Initializes a Producer thread.

        Args:
            products (list): A list of products the producer will publish.
                             Each product is a tuple: (product_id, quantity, produce_time).
            marketplace (Marketplace): The marketplace instance the producer interacts with.
            republish_wait_time (float): Time in seconds to wait before retrying a publish operation.
            **kwargs: Arbitrary keyword arguments passed to the Thread.__init__ method.
        """
        
        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        self.kwargs = kwargs

    def produce(self, product, quantity, produce_time, producer_id):
        """
        Attempts to publish a specified quantity of a product to the marketplace.
        Retries publishing the product if the marketplace fails the operation,
        waiting for `republish_wait_time` between attempts. Once a product is
        successfully published, it waits for `produce_time`.

        Args:
            product (str): The identifier of the product to produce.
            quantity (int): The number of units of the product to produce.
            produce_time (float): The time in seconds to wait after successfully publishing one unit.
            producer_id (int): The ID of the producer.
        """
        
        counter = 0
        # Block Logic: Loop until the desired quantity of the product has been published.
        while counter < quantity:
            # Block Logic: Attempt to publish the product to the marketplace.
            if not self.marketplace.publish(producer_id, product):
                # Block Logic: If publishing fails, wait for a specified time before retrying.
                time.sleep(self.republish_wait_time)
            else:
                # Block Logic: If successful, wait for `produce_time` and increment the counter.
                time.sleep(produce_time)
                counter = counter + 1

    def run(self):
        """
        Executes the producer's main logic.
        - Registers itself with the marketplace.
        - Continuously iterates through its defined products, producing and publishing them.
        """
        # Block Logic: Main loop for continuous operation of the producer.
        while True:
            
            producer_id = self.marketplace.register_producer() # Register the producer and get its ID.
            # Block Logic: Iterate through the producer's defined products to produce and publish them.
            # Pre-condition: `product` is a tuple (product_id, quantity, produce_time).
            for product in self.products:
                
                self.produce(product[0], product[1], product[2], producer_id)


from dataclasses import dataclass


@dataclass(init=True, repr=True, order=False, frozen=True)
class Product:
    """
    Base data class for a product.

    Attributes:
        name (str): The name of the product.
        price (int): The price of the product.
    """
    
    name: str
    price: int


@dataclass(init=True, repr=True, order=False, frozen=True)
class Tea(Product):
    """
    Data class for a Tea product, inheriting from Product.

    Attributes:
        type (str): The type of tea (e.g., Green, Black, Herbal).
    """
    
    type: str


@dataclass(init=True, repr=True, order=False, frozen=True)
class Coffee(Product):
    """
    Data class for a Coffee product, inheriting from Product.

    Attributes:
        acidity (str): The acidity level of the coffee.
        roast_level (str): The roast level of the coffee.
    """
    
    acidity: str
    roast_level: str
