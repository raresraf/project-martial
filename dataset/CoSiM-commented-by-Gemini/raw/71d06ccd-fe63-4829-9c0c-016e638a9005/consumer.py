"""
A producer-consumer simulation of an e-commerce marketplace.

This script models a marketplace with producers, consumers, and products.
Producers create products and add them to the marketplace. Consumers add
products to their carts and eventually place orders. The simulation uses
multiple threads to represent concurrent producers and consumers, and employs
locking mechanisms to ensure thread-safe interactions with the marketplace
and product containers.
"""


from threading import Thread
from time import sleep


class Consumer(Thread):
    """
    Represents a consumer that interacts with the marketplace.

    Each consumer is a thread that processes a predefined list of shopping
    carts, where each cart contains a sequence of 'add' and 'remove'
    operations.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        Initializes a Consumer.

        Args:
            carts: A list of carts, where each cart is a list of operations.
            marketplace: The marketplace to interact with.
            retry_wait_time: Time to wait before retrying a failed operation.
            **kwargs: Additional arguments, including the consumer's name.
        """
        Thread.__init__(self)

        self.marketplace = marketplace
        self.carts = {self.marketplace.new_cart() : cart for cart in carts}
        self.retry_wait_time = retry_wait_time
        self.name = kwargs['name']

    def run(self):
        """
        The main execution loop for the consumer.

        Processes all assigned carts, performing add/remove operations and
        placing an order for each cart.
        """
        for cart_id, cart in self.carts.items():
            for operation in cart:
                # Process 'add' operations.
                if operation['type'] == 'add':
                    for _ in range(operation['quantity']):
                        add_status = \
                                self.marketplace.add_to_cart(
                                    cart_id,
                                    operation['product']
                                )

                        # Retry mechanism for adding to cart.
                        while not add_status:
                            sleep(self.retry_wait_time)
                            add_status = \
                                    self.marketplace.add_to_cart(
                                        cart_id,
                                        operation['product']
                                    )
                # Process 'remove' operations.
                if operation['type'] == 'remove':
                    for _ in range(operation['quantity']):
                        remove_status = \
                            self.marketplace.remove_from_cart(
                                cart_id,
                                operation['product']
                            )

                        # Retry mechanism for removing from cart.
                        while not remove_status:
                            sleep(self.retry_wait_time)
                            remove_status = \
                                self.marketplace.remove_from_cart(
                                    cart_id,
                                    operation['product']
                                )

        # Place orders for all processed carts.
        for cart_id, _ in self.carts.items():
            order = self.marketplace.place_order(cart_id)
            for product in order:
                print(self.name + " bought " + str(product[0]))


from threading import Lock
from tema.products_container import ProductsContainer

class Marketplace:
    """
    The central marketplace for producers and consumers.

    Manages product inventory, producer registration, and shopping carts.
    """
    def __init__(self, queue_size_per_producer):
        """
        Initializes the Marketplace.

        Args:
            queue_size_per_producer: The maximum number of products each
                                     producer can have in the marketplace.
        """
        self.queue_size_per_producer = queue_size_per_producer

        self.register_producer_lock = Lock()
        self.new_cart_lock = Lock()

        self.avail_producer_id = 0
        self.avail_cart_id = 0

        self.products = {}
        self.carts = {}

    def register_producer(self):
        """
        Registers a new producer with the marketplace.

        Returns:
            The ID assigned to the new producer.
        """

        with self.register_producer_lock:
            producer_id = self.avail_producer_id
            self.avail_producer_id += 1

            self.products[producer_id] = ProductsContainer(self.queue_size_per_producer)

            return producer_id

    def publish(self, producer_id, product):
        """
        Publishes a product from a producer to the marketplace.

        Args:
            producer_id: The ID of the producer.
            product: The product to be published.

        Returns:
            True if successful, False otherwise.
        """

        return self.products[producer_id].put((product, producer_id))

    def new_cart(self):
        """
        Creates a new shopping cart for a consumer.

        Returns:
            The ID of the new cart.
        """
        with self.new_cart_lock:
            cart_id = self.avail_cart_id
            self.avail_cart_id += 1

            self.carts[cart_id] = ProductsContainer()

            return cart_id

    def add_to_cart(self, cart_id, product):
        """
        Adds a product to a consumer's cart.

        This involves moving the product from a producer's inventory to the cart.

        Args:
            cart_id: The ID of the cart.
            product: The product to be added.

        Returns:
            True if successful, False otherwise.
        """
        for producer_id, products in self.products.items():
            product_data = (product, producer_id)
            if products.has(product_data):
                self.products[producer_id].remove(product_data)
                self.carts[cart_id].put(product_data)
                return True

        return False

    def remove_from_cart(self, cart_id, product):
        """
        Removes a product from a cart and returns it to the producer.

        Args:
            cart_id: The ID of the cart.
            product: The product to be removed.

        Returns:
            True if successful, False otherwise.
        """

        products_to_remove = [it for it in self.carts[cart_id].get_all() if it[0] == product]
        if len(products_to_remove) == 0:
            return False

        removed_product = products_to_remove[0]

        if removed_product[1] not in self.products:
            return False

        self.carts[cart_id].remove(removed_product)
        self.products[removed_product[1]].put(removed_product)

        return True

    def place_order(self, cart_id):
        """
        Finalizes an order for a given cart.

        Args:
            cart_id: The ID of the cart to be ordered.

        Returns:
            A list of products in the order.
        """
        return self.carts[cart_id].get_all()


from threading import Thread
from random import choice
from time import sleep


class Producer(Thread):
    """
    Represents a producer that creates products and publishes them to the marketplace.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        Initializes a Producer.

        Args:
            products: A list of products that the producer can create.
            marketplace: The marketplace to publish products to.
            republish_wait_time: Time to wait before retrying to publish.
            **kwargs: Additional arguments.
        """
        Thread.__init__(self, daemon=True)

        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        self.producer_id = self.marketplace.register_producer()

    def run(self):
        """
        The main execution loop for the producer.

        Continuously produces and publishes products to the marketplace.
        """
        while True:
            published_product = choice(self.products)

            for _ in range(published_product[1]):
                publish_status = self.marketplace.publish(self.producer_id, published_product[0])

                # Retry mechanism for publishing.
                while not publish_status:
                    sleep(self.republish_wait_time)
                    publish_status = \
                            self.marketplace.publish(self.producer_id, published_product[0])

                sleep(published_product[2])


from dataclasses import dataclass


@dataclass(init=True, repr=True, order=False, frozen=True)
class Product:
    """A base dataclass for a product."""
    name: str
    price: int


@dataclass(init=True, repr=True, order=False, frozen=True)
class Tea(Product):
    """A dataclass for a tea product, inheriting from Product."""
    type: str


@dataclass(init=True, repr=True, order=False, frozen=True)
class Coffee(Product):
    """A dataclass for a coffee product, inheriting from Product."""
    acidity: str
    roast_level: str


from threading import Lock

class ProductsContainer:
    """
    A thread-safe container for storing products.
    """
    def __init__(self, max_size=-1):
        """
        Initializes the ProductsContainer.

        Args:
            max_size: The maximum number of products the container can hold.
                      -1 means unlimited size.
        """
        self.products = []
        self.lock = Lock()
        self.max_size = max_size

    def put(self, product_data):
        """
        Adds a product to the container.

        Args:
            product_data: The product to be added.

        Returns:
            True if successful, False if the container is full.
        """
        with self.lock:
            if self.max_size == -1:
                self.products.append(product_data)
                return True

            if len(self.products) >= self.max_size:
                return False

            self.products.append(product_data)
            return True

    def remove(self, product_data):
        """
        Removes a product from the container.

        Args:
            product_data: The product to be removed.

        Returns:
            True if successful, False if the product is not found.
        """
        with self.lock:
            try:
                self.products.remove(product_data)
                return True
            except ValueError:
                return False

    def get_all(self):
        """
        Returns all products in the container.

        Returns:
            A list of all products.
        """
        with self.lock:
            return self.products

    def has(self, product_data):
        """
        Checks if a product is in the container.

        Args:
            product_data: The product to check for.

        Returns:
            True if the product is in the container, False otherwise.
        """
        with self.lock:
            return product_data in self.products