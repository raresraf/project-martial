"""
@cc7911aa-57fc-4a5c-9f81-2fb44ed50f63/consumer.py
@brief Simulates a multi-threaded e-commerce marketplace with producers and consumers.
This module defines classes for `Consumer`, `Marketplace`, and `Producer` to model
concurrent interactions in an online shopping scenario. It includes mechanisms
for product publication, cart management, order placement, and thread-safe
access to shared resources using locks.

Domain: Concurrency, Distributed Systems (simulated), E-commerce.
"""

from threading import Thread, Lock
from time import sleep


class Consumer(Thread):
    """
    @brief Represents a consumer in the e-commerce marketplace.
    Consumers create carts, add/remove products, and place orders.
    Each consumer operates as a separate thread.
    """
    
    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        @brief Initializes a Consumer thread.
        @param carts: A list of shopping cart operations to perform. Each element
                      in the list represents a sequence of add/remove commands.
        @param marketplace: The Marketplace instance with which the consumer interacts.
        @param retry_wait_time: The time (in seconds) to wait before retrying
                                a failed cart operation.
        @param kwargs: Arbitrary keyword arguments passed to the Thread constructor.
        """
        # Functional Utility: Initializes the base Thread class.
        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time

    def add_command(self, cart_id, product, quantity):
        """
        @brief Executes an 'add product' command for a given cart.
        Attempts to add a specified quantity of a product to a cart.
        If the operation fails (e.g., product unavailable), it retries
        after a specified wait time.
        @param cart_id: The ID of the shopping cart.
        @param product: The product to add.
        @param quantity: The number of units of the product to add.
        """
        # Block Logic: Iterates to add the specified quantity of the product.
        for _ in range(quantity):
            status = False
            # Block Logic: Continuously attempts to add the product until successful.
            # Invariant: `status` becomes True once `add_to_cart` succeeds.
            while not status:
                status = self.marketplace.add_to_cart(cart_id, product)
                # Block Logic: If adding to cart failed, wait before retrying.
                if not status:
                    sleep(self.retry_wait_time)

    def remove_command(self, cart_id, product, quantity):
        """
        @brief Executes a 'remove product' command for a given cart.
        Removes a specified quantity of a product from a cart.
        @param cart_id: The ID of the shopping cart.
        @param product: The product to remove.
        @param quantity: The number of units of the product to remove.
        """
        # Block Logic: Iterates to remove the specified quantity of the product.
        for _ in range(quantity):
            self.marketplace.remove_from_cart(cart_id, product)

    def run(self):
        """
        @brief The main execution loop for the Consumer thread.
        Each consumer processes its assigned list of cart operations,
        places an order, and prints the result.
        """
        # Functional Utility: Obtains a new unique cart ID from the marketplace.
        cart_id = self.marketplace.new_cart()
        # Block Logic: Iterates through each set of operations defined for this consumer's cart.
        for op in self.carts:
            command = op.get("type")
            # Block Logic: Dispatches the appropriate command handler based on the operation type.
            if command == "add":
                    self.add_command(cart_id, op.get("product"), op.get("quantity"))
            if command == "remove":
                    self.remove_command(cart_id, op.get("product"), op.get("quantity"))
            # Functional Utility: Places the final order for the completed cart.
            item_list = self.marketplace.place_order(cart_id)
            # Block Logic: Prints the items bought by this consumer.
            for prod in item_list:
                print("%s bought %s" % (self.name, prod))


class Marketplace:
    """
    @brief Manages products, producers, consumers, and cart operations in a thread-safe manner.
    The Marketplace acts as the central hub where producers publish products
    and consumers manage their shopping carts. All critical operations are
    protected by a global lock to ensure data consistency in a multi-threaded environment.
    """

    def __init__(self, queue_size_per_producer):
        """
        @brief Initializes the Marketplace.
        @param queue_size_per_producer: The maximum number of products a producer
                                        can have in the marketplace's inventory at any time.
        """
        self.queue_size_per_producer = queue_size_per_producer
        # Functional Utility: Initializes unique IDs for producers and carts.
        self.producer_id = -1
        self.cart_id = -1
        # List to store products from each producer. Each element is a list for a producer.
        self.producer_list = []
        # List to store items in each shopping cart.
        self.cart_list = []
        # Global lock to ensure thread-safe access to marketplace's shared data.
        self.lock = Lock()

    def register_producer(self):
        """
        @brief Registers a new producer with the marketplace.
        Assigns a unique producer ID and initializes an empty product list
        for the new producer.
        @return The unique ID assigned to the registered producer.
        """
        # Block Logic: Acquires a lock to safely modify shared producer registration data.
        self.lock.acquire()
        self.producer_id += 1
        # Functional Utility: Creates an empty list to hold products for this new producer.
        self.producer_list.append([])
        # Block Logic: Releases the lock after modifying shared data.
        self.lock.release()
        return self.producer_id

    def publish(self, producer_id, product):
        """
        @brief Publishes a product from a producer to the marketplace.
        Checks if the producer's queue size limit has been reached before
        adding the product.
        @param producer_id: The ID of the producer publishing the product.
        @param product: The product to publish.
        @return True if the product was successfully published, False otherwise.
        """
        # Block Logic: Acquires a lock to safely check and modify producer's product list.
        self.lock.acquire()
        count = 0
        # Block Logic: Counts currently available (not yet bought) products from this producer.
        for prod_item in self.producer_list[producer_id]:
            # Invariant: `prod_item[1]` is True if the product is available.
            if prod_item[1]:
                count += 1

        # Block Logic: Checks if the producer's queue has space for a new product.
        if (
            # Inline: Ensures `producer_list[producer_id]` is not empty before checking `count`.
            len(self.producer_list[producer_id]) > 0
            and self.queue_size_per_producer > count
        ):
            # Functional Utility: Adds the product to the producer's inventory, initially marked as available (True).
            self.producer_list[producer_id].append([product, True])
            self.lock.release()
            return True
        self.lock.release()
        return False

    def new_cart(self):
        """
        @brief Creates a new, empty shopping cart.
        Assigns a unique cart ID and returns it.
        @return The unique ID assigned to the new cart.
        """
        # Block Logic: Acquires a lock to safely assign a new cart ID and initialize the cart.
        self.lock.acquire()
        self.cart_id += 1
        self.cart_list.append([])
        self.lock.release()
        return self.cart_id

    def add_to_cart(self, cart_id, product):
        """
        @brief Adds a product to a specified shopping cart.
        Searches for the product among all producers. If found and available,
        it's moved to the consumer's cart and marked as unavailable in the
        producer's inventory.
        @param cart_id: The ID of the cart to add the product to.
        @param product: The product to add.
        @return True if the product was successfully added, False otherwise.
        """
        # Block Logic: Acquires a lock to safely modify cart and producer product data.
        self.lock.acquire()
        # Block Logic: Iterates through all producers' product lists.
        for lists in self.producer_list:
            # Block Logic: Iterates through products within each producer's list.
            for item in lists:
                # Invariant: Checks if the product matches and is currently available (`item[1]` is True).
                if item[0] == product and item[1]:
                    # Functional Utility: Adds the product to the consumer's cart.
                    self.cart_list[cart_id].append(product)
                    # Functional Utility: Marks the product as unavailable in the producer's inventory.
                    item[1] = False
                    self.lock.release()
                    return True
        self.lock.release()
        return False

    def remove_from_cart(self, cart_id, product):
        """
        @brief Removes a product from a specified shopping cart.
        Also marks the product as available again in the producer's inventory.
        @param cart_id: The ID of the cart to remove the product from.
        @param product: The product to remove.
        """
        # Block Logic: Acquires a lock to safely modify cart and producer product data.
        self.lock.acquire()
        # Functional Utility: Removes the product from the consumer's cart.
        self.cart_list[cart_id].remove(product)

        # Block Logic: Iterates through all producers' product lists to mark the product as available again.
        for lists in self.producer_list:
            for item in lists:
                # Invariant: Finds the product that was previously marked as unavailable (`item[1]` is False).
                if item[0] == product and not item[1]:
                    # Functional Utility: Marks the product as available again.
                    item[1] = True
        self.lock.release()

    def place_order(self, cart_id):
        """
        @brief Finalizes an order by returning the contents of the specified cart.
        @param cart_id: The ID of the cart for which to place the order.
        @return A list of products in the ordered cart.
        """
        return self.cart_list[cart_id]


class Producer(Thread):
    """
    @brief Represents a producer in the e-commerce marketplace.
    Producers continuously publish products to the marketplace,
    handling republishing attempts and cooldown periods.
    """
    
    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        @brief Initializes a Producer thread.
        @param products: A list of products this producer will offer, including
                         product ID, quantity to publish, and cooldown time.
        @param marketplace: The Marketplace instance with which the producer interacts.
        @param republish_wait_time: The time (in seconds) to wait before retrying
                                     a failed publish operation.
        @param kwargs: Arbitrary keyword arguments passed to the Thread constructor.
        """
        # Functional Utility: Initializes the base Thread class.
        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time

    def publish_product(self, product_id, quantity, cooldown, producer_id):
        """
        @brief Publishes a specified quantity of a single product to the marketplace.
        Includes retry logic and a cooldown period between publications.
        @param product_id: The ID of the product to publish.
        @param quantity: The number of units of the product to publish.
        @param cooldown: The time (in seconds) to wait after successfully publishing
                         a single unit of the product.
        @param producer_id: The ID of this producer in the marketplace.
        """
        # Block Logic: Iterates to publish the specified quantity of the product.
        for _ in range(quantity):
            status = False
            # Block Logic: Continuously attempts to publish the product until successful.
            # Invariant: `status` becomes True once `marketplace.publish` succeeds.
            while not status:
                status = self.marketplace.publish(producer_id, product_id)
                # Block Logic: If publishing failed, wait before retrying.
                if not status:
                    sleep(self.republish_wait_time)
            # Block Logic: Waits for a cooldown period after successfully publishing a product.
            sleep(cooldown)

    def run(self):
        """
        @brief The main execution loop for the Producer thread.
        Registers the producer with the marketplace and then continuously
        publishes its defined products.
        """
        # Functional Utility: Registers with the marketplace to obtain a unique producer ID.
        producer_id = self.marketplace.register_producer()
        # Block Logic: The continuous loop for publishing products.
        while True:
            # Block Logic: Iterates through each product defined for this producer.
            for prod in self.products:
                self.publish_product(prod[0], prod[1], prod[2], producer_id)
