
"""
@96426801-926c-4770-b2e5-5ee794a7f577/consumer.py
@brief Implements a multi-threaded marketplace simulation with distinct producer and consumer behaviors.

This module defines the core components for simulating an e-commerce marketplace
where `Producer` threads publish products to a shared `Marketplace`, and `Consumer`
threads create carts, add/remove products, and place orders. This version features
a `Marketplace` that manages product availability per producer and cart states
with specific locking mechanisms for thread safety.

The simulation models concurrent interactions in a shared marketplace, focusing on
resource management and synchronization in a multi-threaded environment.

Classes:
- Consumer: Represents a customer agent that interacts with the marketplace.
- Marketplace: The central hub managing products, carts, and producer/consumer interactions.
- Producer: Represents a supplier agent that publishes products to the marketplace.
- Product: Base dataclass for all products.
- Tea, Coffee: Specialized product dataclasses.

Domain: Concurrent Programming, Producer-Consumer Problem, Multi-threading, Marketplace Simulation.
"""

from threading import Thread
from time import sleep

class Consumer(Thread):
    """
    @brief Represents a consumer agent in the marketplace simulation.

    Each `Consumer` thread simulates a customer's shopping journey, including
    creating carts, adding/removing products, and placing orders. It interacts
    with the `Marketplace` and incorporates a retry mechanism for failed 'add' operations.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        @brief Initializes a new Consumer thread.

        @param carts: A list of shopping cart operations (e.g., add, remove) for this consumer.
        @param marketplace: A reference to the shared Marketplace instance.
        @param retry_wait_time: The time in seconds to wait before retrying a failed operation.
        @param kwargs: Additional keyword arguments for the `Thread` constructor, including "name".
        """
        Thread.__init__(self, **kwargs)
        # List of cart operations to perform.
        self.carts = carts
        # Reference to the shared marketplace instance.
        self.marketplace = marketplace
        # Time to wait before retrying an operation.
        self.retry_wait_time = retry_wait_time
        # The name of the consumer thread.
        self.name = kwargs["name"]

    def run(self):
        """
        @brief The main execution logic for the Consumer thread.

        Pre-condition: `carts` contains a list of operations to perform.
        Invariant: The consumer attempts to process all operations in its carts,
                   retrying failed additions, and eventually places orders,
                   printing confirmation messages for bought products.
        """
        # Block Logic: Iterates through each defined shopping cart sequence for this consumer.
        for cart in self.carts:
            # Creates a new shopping cart in the marketplace and gets its unique ID.
            cart_id = self.marketplace.new_cart()
            # Block Logic: Processes each operation (`opp`) within the current cart.
            for opp in cart:
                # Block Logic: Attempts to fulfill the desired quantity for the current operation.
                for i in range(0, opp["quantity"]):
                    if opp["type"] == "add":
                        # Block Logic: Attempts to add a product to the cart.
                        # Invariant: Will keep retrying to add the product until successful.
                        while self.marketplace.add_to_cart(cart_id, opp["product"]) == False:
                            # Inline: If adding to cart fails (e.g., product out of stock), wait and retry.
                            sleep(self.retry_wait_time)
                    elif opp["type"] == "remove":
                        # Block Logic: Attempts to remove a product from the cart.
                        self.marketplace.remove_from_cart(cart_id, opp["product"])

            # Places the final order for the current cart and retrieves the list of purchased products.
            prod_list = self.marketplace.place_order(cart_id)

            # Block Logic: Prints a confirmation message for each product successfully bought.
            for product in prod_list:
                print(str(self.name) + " bought " + str(product))




from threading import Lock

class Marketplace:
    """
    @brief The central marketplace managing products, producers, consumers, and carts.

    This class handles producer registration, product publishing, and all cart-related
    operations. It uses `Lock` objects to ensure thread safety for shared resources.
    Products are managed per producer, and carts track products along with their original producer ID.
    """
    
    def __init__(self, queue_size_per_producer):
        """
        @brief Initializes the Marketplace.

        @param queue_size_per_producer: The maximum number of products a single producer can have published in the marketplace at any time.
        """
        # Maximum number of products a single producer can have in the marketplace.
        self.queue_size_per_producer = queue_size_per_producer
        # Counter for generating unique producer IDs.
        self.producer_id = 0
        # Counter for generating unique consumer IDs (used as cart IDs).
        self.consumer_id = 0
        # Dictionary mapping producer IDs to a list of products they currently have published.
        self.prod_dict = {}

        # Dictionary mapping cart IDs to a list of products within that cart,
        # storing (product, original_producer_id) tuples.
        self.cart_dict = {}
        # Lock to protect `add_to_cart` and `remove_from_cart` operations.
        self.lock_add_cart = Lock()
        # Lock to protect `publish` operation and `prod_dict` state.
        self.lock_publish = Lock()

        pass

    
    def register_producer(self):
        """
        @brief Registers a new producer with the marketplace and assigns a unique ID.

        @return: The unique ID assigned to the new producer.
        """
        # Increments the producer ID (implicitly thread-safe due to single `producer_id` variable and no external access).
        self.producer_id += 1

        # Initializes an empty list for the new producer's published products.
        self.prod_dict[self.producer_id] = []
        return self.producer_id
        pass

    def publish(self, producer_id, product):
        """
        @brief Attempts to publish a product from a producer to the marketplace.

        The product is published only if the producer's queue size limit is not exceeded.

        @param producer_id: The ID of the producer publishing the product.
        @param product: The product object to be published.
        @return: True if the product was published successfully, False otherwise.
        """
        # Critical Section: Acquire lock to protect `prod_dict` and ensure atomicity of publish operation.
        self.lock_publish.acquire()
        # Block Logic: Checks if the producer has reached its maximum queue size for published products.
        if len(self.prod_dict[producer_id]) < self.queue_size_per_producer:
            self.prod_dict[producer_id].append(product)
            self.lock_publish.release()
            return True
        self.lock_publish.release()
        return False
        pass

    def new_cart(self):
        """
        @brief Creates a new empty shopping cart in the marketplace and assigns a unique ID.

        @return: The unique ID assigned to the new cart.
        """
        # Increments the consumer ID (used as cart ID).
        self.consumer_id += 1

        # Initializes an empty list for the new cart, storing (product, producer_id) pairs.
        self.cart_dict[self.consumer_id] = []
        return self.consumer_id
        pass

    def add_to_cart(self, cart_id, product):
        """
        @brief Attempts to add a product to a specific shopping cart.

        This operation is thread-safe. If the product is available, it's moved
        from the marketplace inventory (producer's list) to the cart, and its
        original producer's ID is stored with it.

        @param cart_id: The ID of the cart to which the product should be added.
        @param product: The product object to add.
        @return: True if the product was successfully added, False if not available.
        """
        # Critical Section: Acquire lock to protect `prod_dict` and `cart_dict` during modification.
        self.lock_add_cart.acquire()

        # Block Logic: Iterates through all producers to find the product.
        for prod_id in self.prod_dict.keys():
            # Block Logic: Iterates through products published by the current producer.
            for p in self.prod_dict[prod_id]:
                if p == product:
                    self.prod_dict[prod_id].remove(product) # Remove from producer's inventory.
                    self.cart_dict[cart_id].append([product, prod_id]) # Add to cart with producer ID.
                    self.lock_add_cart.release()
                    return True
        self.lock_add_cart.release()

        return False
        pass

    def remove_from_cart(self, cart_id, product):
        """
        @brief Removes a product from a specific shopping cart and returns it to the original producer's inventory.

        This operation is thread-safe.

        @param cart_id: The ID of the cart from which the product should be removed.
        @param product: The product object to remove.
        """
        # Block Logic: Iterates through products in the cart to find the one to remove.
        for prod in self.cart_dict[cart_id]:
            if prod[0] == product:
                self.cart_dict[cart_id].remove(prod) # Remove from cart.
                self.prod_dict[prod[1]].append(prod[0]) # Return to original producer's inventory.
                break

    def place_order(self, cart_id):
        """
        @brief Finalizes the shopping cart, effectively placing an order.

        @param cart_id: The ID of the cart to finalize.
        @return: A list of actual product objects that were in the finalized cart.
        """
        prod_list = []
        # Block Logic: Extracts only the product objects from the (product, producer_id) pairs in the cart.
        for prod in self.cart_dict[cart_id]:
           prod_list.append(prod[0])
        return prod_list>>>> file: producer.py


from threading import Thread
from time import sleep

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
        # Registers the producer with the marketplace.
        # This call is redundant as producerID is already assigned in __init__
        producer_id = self.marketplace.register_producer() # Redundant call; producerID is set in __init__
        # Block Logic: Main loop for continuous product publishing.
        while True:
            # Block Logic: Iterates through each product defined for this producer.
            for product in self.products:
                # Inline: Waits for the product's specified publish interval before attempting to publish.
                sleep(product[2])
                # Block Logic: Attempts to publish the desired quantity of the current product.
                for i in range(0, product[1]):
                    # Invariant: Will keep retrying to publish the product until successful.
                    while self.marketplace.publish(producer_id, product[0]) == False:
                        # Inline: If publishing fails (e.g., producer's queue is full), wait and retry.
                        sleep(self.republish_wait_time)



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
