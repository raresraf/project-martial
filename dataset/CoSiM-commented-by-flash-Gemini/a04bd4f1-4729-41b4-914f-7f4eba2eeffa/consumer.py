
"""
This module defines the Consumer, Marketplace, and Producer classes
which together simulate a marketplace where producers sell products
and consumers buy them, managing carts and orders with thread-safe operations.
"""

import time
from threading import Thread, Lock


class Consumer(Thread):
    """
    Represents a consumer thread that interacts with the marketplace to
    add and remove products from carts, and eventually place orders.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        Initializes a new Consumer thread.

        Args:
            carts (list): A list of shopping lists for this consumer.
            marketplace (Marketplace): The marketplace instance to interact with.
            retry_wait_time (int): Time in seconds to wait before retrying an action.
            **kwargs: Arbitrary keyword arguments passed to the Thread constructor.
        """
        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        # Functional Utility: `print_locked` acts as a mutex to prevent interleaved
        # print statements from multiple consumer threads, ensuring readability.
        self.print_locked = Lock()

    def run(self):
        """
        The main execution method for the consumer thread. It iterates through
        assigned carts, attempts to add/remove products, and places orders.
        """

        # Block Logic: Iterates through each predefined shopping cart for the consumer.
        for cart in self.carts:
            # Functional Utility: Obtains a new unique cart ID from the marketplace
            # for the current shopping session.
            my_cart = self.marketplace.new_cart()
            # Block Logic: Processes each item in the current shopping cart.
            for to_do in cart:
                repeat = to_do['quantity']
                # Block Logic: Continuously attempts to fulfill the quantity of each product.
                while repeat > 0:
                    # Pre-condition: Checks if the product is currently available in the market's stock.
                    if to_do['product'] in self.marketplace.market_stock:
                        # Functional Utility: Executes the specific task (add/remove) for the product.
                        self.execute_task(to_do['type'], my_cart, to_do['product'])
                        repeat -= 1
                    else:
                        # Functional Utility: Pauses execution for a defined period if the product is
                        # not available, simulating a retry mechanism.
                        time.sleep(self.retry_wait_time)

            # Functional Utility: Submits the populated cart to the marketplace to finalize the order.
            order = self.marketplace.place_order(my_cart)
            # Block Logic: Ensures exclusive access to the print statement to avoid garbled output
            # when multiple consumers are printing concurrently.
            with self.print_locked:
                # Post-condition: Iterates through the confirmed order and prints each purchased product.
                for product in order:
                    print(self.getName(), "bought", product)

    def execute_task(self, task_type, cart_id, product):
        """
        Executes a specific action (add or remove) on a given product in a cart.

        Args:
            task_type (str): The type of action to perform ('add' or 'remove').
            cart_id (int): The ID of the cart to modify.
            product (str): The product to add or remove.
        """
        if task_type == 'add':
            # Functional Utility: Delegates the 'add to cart' operation to the marketplace.
            self.marketplace.add_to_cart(cart_id, product)
        else:
            # Invariant: This conditional block ensures that only 'remove' operations
            # are processed if the task_type is not 'add'.
            if task_type == 'remove':
                # Functional Utility: Delegates the 'remove from cart' operation to the marketplace.
                self.marketplace.remove_from_cart(cart_id, product)

class Marketplace:
    """
    Manages products, producers, and consumer carts in a thread-safe manner.
    It handles registration of producers, publishing products, managing carts,
    and processing orders.
    """

    def __init__(self, queue_size_per_producer):
        """
        Initializes the Marketplace with a specified queue size per producer.

        Args:
            queue_size_per_producer (int): The maximum number of products
                                           a single producer can have in the market.
        """
        self.queue_size_per_producer = queue_size_per_producer
        self.num_producers = -1
        # Functional Utility: `register_locked` acts as a mutex to ensure atomic
        # incrementation of `num_producers` during producer registration,
        # preventing race conditions.
        self.register_locked = Lock()
        self.market_stock = []
        self.product_counter = []
        self.product_owner = {}
        self.num_consumers = -1
        self.cart = [[]]
        # Functional Utility: `cart_locked` acts as a mutex to ensure atomic
        # incrementation of `num_consumers` during new cart creation,
        # preventing race conditions.
        self.cart_locked = Lock()
        # Functional Utility: `add_locked` acts as a mutex to protect `product_counter`
        # updates when adding items to a cart, ensuring data consistency.
        self.add_locked = Lock()
        # Functional Utility: `remove_locked` acts as a mutex to protect `product_counter`
        # updates when removing items from a cart, ensuring data consistency.
        self.remove_locked = Lock()
        # Functional Utility: `publish_locked` acts as a mutex to protect `product_counter`
        # and `product_owner` updates during product publication, ensuring data consistency.
        self.publish_locked = Lock()
        # Functional Utility: `market_locked` acts as a mutex to protect `market_stock`
        # modifications during product addition/removal from carts, ensuring data consistency.
        self.market_locked = Lock()

    def register_producer(self):
        """
        Registers a new producer with the marketplace and assigns a unique ID.

        Returns:
            int: The unique ID assigned to the registered producer.
        """
        with self.register_locked:
            self.num_producers += 1
            new_producer_id = self.num_producers
        self.product_counter.append(0)
        return new_producer_id

    def publish(self, producer_id, product):
        """
        Publishes a product to the marketplace by a producer.

        Args:
            producer_id (int): The ID of the producer publishing the product.
            product (str): The name of the product to publish.

        Returns:
            bool: True if the product was successfully published, False otherwise.
        """
        # Pre-condition: Checks if the producer has reached their publishing limit.
        if self.product_counter[producer_id] >= self.queue_size_per_producer:
            return False
        self.market_stock.append(product)
        with self.publish_locked:
            self.product_counter[producer_id] += 1
            self.product_owner[product] = producer_id
        return True

    def new_cart(self):
        """
        Creates a new shopping cart for a consumer and assigns a unique ID.

        Returns:
            int: The unique ID assigned to the new cart.
        """
        with self.cart_locked:
            self.num_consumers += 1
            new_consumer_cart_id = self.num_consumers
        self.cart.append([])
        return new_consumer_cart_id

    def add_to_cart(self, cart_id, product):
        """
        Adds a product to a consumer's cart.

        Args:
            cart_id (int): The ID of the cart to add the product to.
            product (str): The product to add.

        Returns:
            bool: True if the product was successfully added, False otherwise.
        """

        # Pre-condition: Checks if the product is available in the market's stock.
        if product not in self.market_stock:
            return False
        self.cart[cart_id].append(product)
        with self.add_locked:
            # Functional Utility: Decrements the count of the product held by its owner
            # producer, reflecting it being moved to a consumer's cart.
            self.product_counter[self.product_owner[product]] -= 1
        with self.market_locked:
            # Block Logic: Removes the product from the main market stock to reflect its purchase.
            if product in self.market_stock:
                element_index = self.market_stock.index(product)
                del self.market_stock[element_index]
        return True

    def remove_from_cart(self, cart_id, product):
        """
        Removes a product from a consumer's cart and returns it to the marketplace.

        Args:
            cart_id (int): The ID of the cart to remove the product from.
            product (str): The product to remove.
        """

        # Pre-condition: Ensures the product exists within the specified cart before attempting removal.
        if product in self.cart[cart_id]:
            with self.remove_locked:
                # Functional Utility: Increments the product count for the owning producer,
                # as the product is returned to the market.
                self.product_counter[self.product_owner[product]] += 1
            self.cart[cart_id].remove(product)
            self.market_stock.append(product)

    def place_order(self, cart_id):
        """
        Places an order for the items in a given cart.

        Args:
            cart_id (int): The ID of the cart to place an order for.

        Returns:
            list: A list of products in the placed order.
        """
        return self.cart[cart_id]

class Producer(Thread):
    """
    Represents a producer thread that continuously publishes products
    to the marketplace.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        Initializes a new Producer thread.

        Args:
            products (list): A list of products this producer will publish,
                             each with quantity and delay.
            marketplace (Marketplace): The marketplace instance to interact with.
            republish_wait_time (int): Time in seconds to wait if publishing fails.
            **kwargs: Arbitrary keyword arguments passed to the Thread constructor.
        """
        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        self.my_id = self.marketplace.register_producer()

    def run(self):
        """
        The main execution method for the producer thread. It continuously
        attempts to publish its products to the marketplace.
        """

        # Block Logic: Enters an infinite loop to continuously attempt to publish products.
        while True:
            # Block Logic: Iterates through each product defined for this producer.
            for (product, quantity, seconds) in self.products:
                repeat = quantity
                # Block Logic: Attempts to publish the specified quantity of a product.
                while repeat > 0:
                    # Functional Utility: Attempts to publish a product to the marketplace,
                    # receiving a boolean indicating success or failure due to queue limits.
                    wait = self.marketplace.publish(self.my_id, product)
                    # Pre-condition: Checks if the product was successfully published.
                    if wait:
                        # Functional Utility: Pauses for a specified duration after successful publication,
                        # simulating production time.
                        time.sleep(seconds)
                        repeat -= 1
                    else:
                        # Functional Utility: Pauses for a specified duration if publication fails,
                        # allowing for a retry attempt.
                        time.sleep(self.republish_wait_time)
