"""
This module implements a producer-consumer simulation for a marketplace
using multithreading in Python.
"""


from threading import Thread, Lock
import time
from dataclasses import dataclass


class Consumer(Thread):
    """
    Represents a consumer that buys products from the marketplace.
    Each consumer runs in its own thread.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        Initializes a Consumer instance.

        Args:
            carts (list): A list of carts, where each cart is a list of orders.
            marketplace (Marketplace): The marketplace instance to interact with.
            retry_wait_time (float): Time to wait before retrying to add a product.
            **kwargs: Additional keyword arguments, expecting 'name'.
        """
        Thread.__init__(self)

        self.name = kwargs["name"]
        self.no_product_wait_time = retry_wait_time
        self.shop = marketplace
        self.carts = carts

    def run(self):
        """
        The main execution method for the consumer thread.
        Processes each cart, places an order, and prints the items bought.
        """
        for cart in self.carts:
            cart_id = self.shop.new_cart()
            for order in cart:
                if order["type"] == "add":
                    self.add_to_cart(cart_id, order)
                elif order["type"] == "remove":
                    self.remove_from_cart(cart_id, order)
            bought_items = self.shop.place_order(cart_id)
            self.print_what_was_bought(bought_items)

    def add_to_cart(self, cart_id, order):
        """
        Adds a specified quantity of a product to a cart.
        If the product is not available, it waits and retries.

        Args:
            cart_id (int): The ID of the cart to add items to.
            order (dict): A dictionary containing the 'product' and 'quantity'.
        """
        i = order["quantity"]
        while i > 0:
            if not self.shop.add_to_cart(cart_id, order["product"]):
                # If the product is not available, wait before retrying.
                time.sleep(self.no_product_wait_time)
                continue
            i -= 1

    def remove_from_cart(self, cart_id, order):
        """
        Removes a specified quantity of a product from a cart.

        Args:
            cart_id (int): The ID of the cart to remove items from.
            order (dict): A dictionary containing the 'product' and 'quantity'.
        """
        for _ in range(order["quantity"]):
            self.shop.remove_from_cart(cart_id, order["product"])

    def print_what_was_bought(self, bought):
        """
        Prints the products that were bought by the consumer.

        Args:
            bought (list): A list of products.
        """
        for product in bought:
            print(self.name, "bought", product)


class Marketplace:
    """
    Represents the marketplace where producers publish products and consumers buy them.
    This class manages inventory and cart creation in a thread-safe manner.
    """

    def __init__(self, queue_size_per_producer):
        """
        Initializes a Marketplace instance.

        Args:
            queue_size_per_producer (int): The maximum number of products a single
                                           producer can have in the marketplace queue.
        """
        self.reg_prod_lock = Lock()
        self.prod_list = []
        self.prod_max_queue = queue_size_per_producer

        self.shop_items_lock = Lock()
        self.shop_items = {}

        self.carts = {}
        self.cart_id_lock = Lock()
        self.cart_id = 0

    def register_producer(self):
        """
        Registers a new producer and returns a unique producer ID.

        Returns:
            int: The ID for the newly registered producer.
        """
        with self.reg_prod_lock:
            self.prod_list.append(0)
            return len(self.prod_list) - 1

    def publish(self, producer_id, product) -> bool:
        """
        Publishes a product from a producer to the marketplace.

        Args:
            producer_id (int): The ID of the producer.
            product (Product): The product to be published.

        Returns:
            bool: True if the product was published successfully, False otherwise.
        """
        if self.prod_list[producer_id] < self.prod_max_queue:
            self.prod_list[producer_id] += 1
            with self.shop_items_lock:
                if product not in self.shop_items:
                    self.shop_items[product] = (Lock(), [])
            
            with self.shop_items[product][0]:
                self.shop_items[product][1].append(producer_id)
            return True
        return False

    def new_cart(self):
        """
        Creates a new, empty cart and returns its ID.

        Returns:
            int: The ID of the new cart.
        """
        with self.cart_id_lock:
            cart_id_var = self.cart_id
            self.carts[cart_id_var] = []
            self.cart_id += 1
            return cart_id_var

    def add_to_cart(self, cart_id, product) -> bool:
        """
        Adds a product to a specified cart.

        Args:
            cart_id (int): The ID of the cart.
            product (Product): The product to add.

        Returns:
            bool: True if the product was added successfully, False otherwise.
        """
        with self.shop_items_lock:
            if product not in self.shop_items:
                return False

        with self.shop_items[product][0]:
            if len(self.shop_items[product][1]) > 0:
                prod_id = self.shop_items[product][1].pop(0)
                if prod_id != -1:
                    self.prod_list[prod_id] -= 1
                self.carts[cart_id].append(product)
                return True
        return False

    def remove_from_cart(self, cart_id, product):
        """
        Removes a product from a cart and returns it to the marketplace inventory.

        Args:
            cart_id (int): The ID of the cart.
            product (Product): The product to remove.
        """
        try:
            product_index = self.carts[cart_id].index(product)
            self.carts[cart_id].pop(product_index)
            with self.shop_items[product][0]:
                self.shop_items[product][1].append(-1)  # -1 indicates a returned item
        except ValueError:
            return

    def place_order(self, cart_id):
        """
        Finalizes an order and returns the list of items in the cart.

        Args:
            cart_id (int): The ID of the cart to place an order for.

        Returns:
            list: The list of products in the cart.
        """
        return self.carts[cart_id]


class Producer(Thread):
    """
    Represents a producer that creates products and adds them to the marketplace.
    Each producer runs in its own thread.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        Initializes a Producer instance.

        Args:
            products (list): A list of products to be produced. Each item is a tuple
                             of (product, quantity, production_time).
            marketplace (Marketplace): The marketplace to publish products to.
            republish_wait_time (float): Time to wait if the marketplace queue is full.
            **kwargs: Additional keyword arguments.
        """
        Thread.__init__(self, daemon=True)
        self.production_list = products
        self.shop = marketplace
        self.prod_id = marketplace.register_producer()
        self.shop_full_wait_time = republish_wait_time

    def run(self):
        """
        The main execution method for the producer thread.
        Continuously produces and publishes products to the marketplace.
        """
        while True:
            for product, quantity, production_time in self.production_list:
                produced_count = 0
                while produced_count < quantity:
                    if self.shop.publish(self.prod_id, product):
                        produced_count += 1
                        time.sleep(production_time)
                    else:
                        time.sleep(self.shop_full_wait_time)


@dataclass(init=True, repr=True, order=False, frozen=True)
class Product:
    """A base class for a product with a name and price."""
    name: str
    price: int

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other) -> bool:
        if not isinstance(other, Product):
            return NotImplemented
        return self.name == other.name


@dataclass(init=True, repr=True, order=False, frozen=True)
class Tea(Product):
    """A dataclass for Tea, inheriting from Product."""
    type: str


@dataclass(init=True, repr=True, order=False, frozen=True)
class Coffee(Product):
    """A dataclass for Coffee, inheriting from Product."""
    acidity: str
    roast_level: str
