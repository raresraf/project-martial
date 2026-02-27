from threading import Thread
import time


class Consumer(Thread):
    """
    A consumer thread that simulates purchasing items from a marketplace.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        Initializes the Consumer.

        Args:
            carts (list): A list of shopping sessions. Each session is a list of
                          operations (add/remove) to be performed.
            marketplace (Marketplace): The shared marketplace instance.
            retry_wait_time (float): Time to wait before retrying a failed operation.
            **kwargs: Arguments for the parent Thread class.
        """
        Thread.__init__(self, **kwargs)

        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time

    def run(self):
        """
        The main execution logic for the consumer.

        Iterates through its assigned shopping sessions, performs add/remove
        operations for each, and places an order at the end of each session.
        """
        for cart in self.carts:
            # A new cart is created for each shopping session.
            cart_id = self.marketplace.new_cart()

            for operation in cart:
                no_ops = 0
                qty = operation["quantity"]
                op_type = operation["type"]
                prod = operation["product"]

                # Perform the operation `qty` times.
                while no_ops < qty:
                    result = self.execute_operation(cart_id, op_type, prod)

                    if result is None or result:
                        # Operation succeeded or was a remove (None return).
                        no_ops += 1
                    else:
                        # Operation failed (e.g., item not available), wait and retry.
                        time.sleep(self.retry_wait_time)

            self.marketplace.place_order(cart_id)

    def execute_operation(self, cart_id, operation_type, product) -> bool:
        """
        Dispatches the appropriate marketplace method based on the operation type.

        Args:
            cart_id (int): The ID of the current cart.
            operation_type (str): "add" or "remove".
            product (any): The product to operate on.

        Returns:
            bool: The result of the marketplace operation.
        """
        if operation_type == "add":
            return self.marketplace.add_to_cart(cart_id, product)

        if operation_type == "remove":
            # Note: The original code does not capture the return value of remove.
            self.marketplace.remove_from_cart(cart_id, product)
            return None # Assuming remove always succeeds logically.

        return False


from threading import Lock, currentThread


class Marketplace:
    """
    A marketplace that facilitates transactions between producers and consumers.

    This implementation uses a centralized list of available products and separate
    counters for producer queues.
    """
    def __init__(self, queue_size_per_producer):
        """
        Initializes the marketplace.

        Args:
            queue_size_per_producer (int): Max items a single producer can publish.
        """
        self.queue_size_per_producer = queue_size_per_producer
        # Maps a product to the ID of the producer who published it.
        self.products_mapping = {}
        # A list where index=producer_id and value=number of items published.
        self.producers_queues = []
        # Maps a cart_id to a list of products in that cart.
        self.consumers_carts = {}
        # A single list of all products currently available for sale.
        self.available_products = []
        
        self.no_carts = 0
        # Locks to manage concurrent access to shared resources.
        self.consumer_cart_creation_lock = Lock()
        self.cart_operation_lock = Lock()

    def register_producer(self) -> int:
        """
        Registers a new producer and returns a unique producer ID.

        Returns:
            int: The new producer's ID.
        """
        new_producer_id = len(self.producers_queues)
        self.producers_queues.append(0) # Initialize producer's item count to 0.
        return new_producer_id

    def publish(self, producer_id, product) -> bool:
        """
        Publishes a product from a producer to the marketplace.

        Args:
            producer_id (int): The ID of the publishing producer.
            product (any): The product to publish.

        Returns:
            bool: True if successful, False if the producer's queue is full.
        """
        if self.producers_queues[producer_id] >= self.queue_size_per_producer:
            return False

        self.producers_queues[producer_id] += 1
        self.available_products.append(product)
        self.products_mapping[product] = producer_id
        return True

    def new_cart(self) -> int:
        """Atomically creates a new cart and returns its ID."""
        with self.consumer_cart_creation_lock:
            self.no_carts += 1
            self.consumers_carts[self.no_carts] = []
            return self.no_carts

    def add_to_cart(self, cart_id, product) -> bool:
        """
        Adds a product to a cart, moving it from the available pool.

        This operation is atomic.

        Args:
            cart_id (int): The ID of the cart.
            product (any): The product to add.

        Returns:
            bool: True if successful, False if the product is not available.
        """
        with self.cart_operation_lock:
            if product not in self.available_products:
                return False

            producer_id = self.products_mapping[product]
            self.producers_queues[producer_id] -= 1
            self.available_products.remove(product)
            self.consumers_carts[cart_id].append(product)
            return True

    def remove_from_cart(self, cart_id, product):
        """
        Removes a product from a cart, returning it to the available pool.

        Note: This method contains a potential race condition. It modifies
        shared lists (`consumers_carts`, `available_products`) *before* acquiring
        the lock, which could lead to inconsistent state under high contention.
        """
        self.consumers_carts[cart_id].remove(product)
        self.available_products.append(product)

        with self.cart_operation_lock:
            producer_id = self.products_mapping[product]
            self.producers_queues[producer_id] += 1

    def place_order(self, cart_id) -> list:
        """
        Finalizes an order, removing the cart and its products.

        Args:
            cart_id (int): The ID of the cart to be ordered.

        Returns:
            list: The list of products that were in the cart.
        """
        products = self.consumers_carts.pop(cart_id, None)

        for product in products:
            print(currentThread().getName() + " bought " + str(product))
        return products


from threading import Thread
import time


class Producer(Thread):
    """
    A producer thread that continuously publishes products to the marketplace.
    """
    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        Initializes the Producer.

        Args:
            products (list): A list of (product, quantity, wait_time) tuples.
            marketplace (Marketplace): The shared marketplace instance.
            republish_wait_time (float): Time to wait before retrying to publish.
            **kwargs: Arguments for the parent Thread class.
        """
        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        # The producer is registered once upon creation.
        self.producer_id = self.marketplace.register_producer()

    def run(self):
        """
        Main execution loop. Continuously produces and publishes items.
        """
        while True:
            for (product, no_products, publish_wait_time) in self.products:
                no_prod = 0
                while no_prod < no_products:
                    result = self.publish_product(product, publish_wait_time)
                    if result:
                        no_prod += 1

    def publish_product(self, product, publish_wait_time) -> bool:
        """
        Attempts to publish a product, waiting if necessary.

        Args:
            product (any): The product to publish.
            publish_wait_time (float): Time to wait after a successful publish.

        Returns:
            bool: True if the publish was successful.
        """
        result = self.marketplace.publish(self.producer_id, product)
        
        if result:
            time.sleep(publish_wait_time)
            return True

        # If publish failed, wait before the next retry.
        time.sleep(self.republish_wait_time)
        return False
