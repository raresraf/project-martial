"""
This module implements a producer-consumer simulation centered around a marketplace.
It defines Producer and Consumer threads that interact with a shared Marketplace
to publish and purchase products. The simulation uses locks to ensure thread-safe
operations on shared data structures and includes unit tests for the marketplace functionality.
"""


from threading import Thread
from time import sleep

class Consumer(Thread):
    """
    Represents a consumer that interacts with the marketplace to buy products.
    Each consumer has a list of operations (add or remove items from a cart)
    and places an order at the end.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        Initializes a Consumer thread.

        Args:
            carts (list): A list of shopping lists, where each list contains actions.
            marketplace (Marketplace): The shared marketplace object.
            retry_wait_time (float): The time to wait before retrying to add a product.
            **kwargs: Additional arguments for the Thread constructor.
        """
        super().__init__(kwargs=kwargs)

        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.kwargs = kwargs
        self._id = 0

    def run(self):
        """
        The main execution logic for the consumer. It processes a list of actions
        (adding/removing products from a cart) and finally places an order for each cart.
        """
        for act_cart in self.carts:
            self._id = self.marketplace.new_cart()

            items_count = 0



            for act_op in act_cart:
                if act_op["type"] == "add":
                    count = 0
                    while count < act_op["quantity"]:
                        if self.marketplace.add_to_cart(self._id, act_op["product"]):
                            count += 1


                            items_count += 1
                        else:
                            sleep(self.retry_wait_time)

                elif act_op["type"] == "remove":
                    count = 0
                    while count < act_op["quantity"]:
                        self.marketplace.remove_from_cart(self._id, act_op["product"])
                        count += 1
                        items_count -= 1

            items = self.marketplace.place_order(self._id)

            for item in items:
                print(self.kwargs["name"] + " bought " + str(item))


from threading import Lock
import time
import unittest
import logging
import logging.handlers

class TestMarketplace(unittest.TestCase):
    """
    Unit tests for the Marketplace class to ensure its core functionalities
    like producer registration, product publishing, and cart operations work correctly.
    """
    def setUp(self):
        """Sets up a new Marketplace instance for each test."""

        self.marketplace = Marketplace(15)

    def test_register_producer(self):
        """Tests the sequential registration of multiple producers."""

        for i in range(0, 100):
            act_id = self.marketplace.register_producer()
            self.assertEqual(i, act_id)

    def test_publish(self):
        """Tests if a product can be successfully published by a registered producer."""
        from product import Product

        prod_id = self.marketplace.register_producer()

        new_prod = Product("Ceai verde", 10)

        return_val = self.marketplace.publish(prod_id, new_prod)

        if return_val:
            self.assertEqual(new_prod, self.marketplace.goods[prod_id][-1][0])

    def test_new_cart(self):
        """Tests the creation of new carts with sequential IDs."""

        for i in range(0, 100):
            act_id = self.marketplace.new_cart()
            self.assertEqual(i, act_id)

    def test_add_to_cart(self):
        """Tests adding a product to a cart."""
        from product import Product

        cart_id = self.marketplace.new_cart()

        new_prod = Product("Ceai verde", 10)

        return_val = self.marketplace.add_to_cart(cart_id, new_prod)

        if return_val:
            self.assertEqual(new_prod, self.marketplace.carts[cart_id][-1][0])

    def test_remove_from_cart(self):
        """Tests removing a product from a cart."""
        from product import Product

        cart_id = self.marketplace.new_cart()

        new_prod = Product("Ceai verde", 10)

        return_val = self.marketplace.add_to_cart(cart_id, new_prod)

        if not return_val:
            return

        self.marketplace.remove_from_cart(cart_id, new_prod)

        self.assertEqual(0, len(self.marketplace.carts[cart_id]))

    def test_place_order(self):
        """Tests the final order placement logic."""
        from product import Product

        cart_id = self.marketplace.new_cart()

        new_prod = Product("Ceai verde", 10)

        return_val = self.marketplace.add_to_cart(cart_id, new_prod)

        if not return_val:
            return

        prod_list = self.marketplace.place_order(card_id)

        self.assertEqual(new_prod, prod_list[0])

class Marketplace:
    """
    A thread-safe marketplace that facilitates the exchange of products between
    producers and consumers. It manages product inventory, shopping carts, and logging.
    """
    def __init__(self, queue_size_per_producer):
        """
        Initializes the Marketplace.

        Args:
            queue_size_per_producer (int): The maximum number of products a single
                                           producer can have in their inventory queue.
        """

        self.queue_size_per_producer = queue_size_per_producer
        self.goods = {}
        self.carts = {}

        self.id_prod = 0
        self.id_cart = 0

        self.id_prod_lock = Lock()
        self.id_cart_lock = Lock()
        self.goods_locks = {}

        self.logger = logging.getLogger("marketLogger")
        self.logger.setLevel(logging.INFO)

        handler = logging.handlers.RotatingFileHandler(filename="marketplace.log", backupCount=15, maxBytes=5242880)

        handler.setLevel(logging.INFO)

        formatter = logging.Formatter('%(asctime)s %(message)s')
        formatter.converter = time.gmtime
        handler.setFormatter(formatter)

        self.logger.addHandler(handler)

    def register_producer(self):
        """
        Registers a new producer, assigning a unique ID and initializing their inventory space.
        This operation is thread-safe.
        Returns:
            int: The unique ID assigned to the new producer.
        """
        self.logger.info("Entry in function register_producer.")

        with self.id_prod_lock:
            new_id = self.id_prod
            self.id_prod += 1
            self.goods[new_id] = []
            self.goods_locks[new_id] = Lock()

        self.logger.info("Leave from function register_producer with return value %d.", new_id)
        return new_id

    def publish(self, producer_id, product):
        """
        Publishes a product to the marketplace inventory for a specific producer.
        Fails if the producer's inventory queue is full.
        """
        self.logger.info("Entry in function publish with params producer_id = %d, product = %s.", producer_id, str(product))
        f = True

        with self.goods_locks[producer_id]:

            if len(self.goods[producer_id]) == self.queue_size_per_producer:
                f = False
            else:
                self.goods[producer_id].append((product, 1))

        self.logger.info("Leave from function publish with return value %d.", f)
        return f

    def new_cart(self):
        """Creates a new, empty shopping cart and returns its unique ID."""
        self.logger.info("Entry in function new_cart.")

        with self.id_cart_lock:
            new_id = self.id_cart
            self.id_cart += 1
            self.carts[new_id] = []

        self.logger.info("Leave from function new_cart with return value %d", new_id)
        return new_id

    def add_to_cart(self, cart_id, product):
        """
        Adds a product to a cart if it's available in any producer's inventory.
        This is a critical section and is protected by locks to ensure atomicity.
        """
        self.logger.info("Entry in function add_to_cart with params cart_id = %d, product = %s.", cart_id, product)

        found = False

        for id_p in self.goods:
            self.goods_locks[id_p].acquire()

            for i in range(0, len(self.goods[id_p])):
                prod = self.goods[id_p][i]

                if (prod[1] == 1) & (prod[0] == product):
                    self.carts[cart_id].append((product, id_p))

                    self.goods[id_p][i] = (prod[0], 0)
                    found = True
                    break

            self.goods_locks[id_p].release()
            if found == True:
                break

        self.logger.info("Leave from function add_to_cart with return value %d.", found)
        return found

    def remove_from_cart(self, cart_id, product):
        """Removes a product from a cart and returns it to the original producer's inventory."""
        self.logger.info("Entry in function remove_from_cart with params cart_id = %d, product = %s.", cart_id, str(product))

        for item in self.carts[cart_id]:
            if item[0] == product:
                prod_id = item[1]
                _it = item

        self.carts[cart_id].remove(_it)

        self.goods_locks[prod_id].acquire()

        for i in range(0, len(self.goods[prod_id])):
            item = self.goods[prod_id][i]

            if (item[0] == product) & (item[1] == 0):
                self.goods[prod_id][i] = (product, 1)
                break

        self.goods_locks[prod_id].release() 
        self.logger.info(f"Leave from function remove_from_cart.")

    def place_order(self, cart_id):
        """Finalizes the purchase, removing items from the inventory permanently."""
        self.logger.info("Entry in function place_order with params cart_id = %d.", cart_id)

        itemList = []
        for item in self.carts[cart_id]:
            itemList.append(item[0])
            prod_id = item[1]

            self.goods_locks[prod_id].acquire()

            for it in self.goods[prod_id]:
                if (it[0] == item[0]) & (it[1] == 0):
                    my_it = it
                    break

            self.goods[prod_id].remove(my_it)

            self.goods_locks[prod_id].release()

        self.logger.info("Leave from function place_order with return value %s.", str(itemList))
        return itemList


from threading import Thread
from time import sleep


class Producer(Thread):
    """
    Represents a producer that creates products and publishes them to the marketplace.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        Initializes a Producer thread.

        Args:
            products (list): A list of products to be produced, with quantities and timings.
            marketplace (Marketplace): The shared marketplace object.
            republish_wait_time (float): Time to wait before retrying to publish a product.
            **kwargs: Additional arguments for the Thread constructor.
        """
        super().__init__(kwargs=kwargs, daemon=True)

        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        self._id = 0

    def run(self):
        """
        The main execution logic for the producer. It registers itself and then
        enters a loop to continuously publish its products to the marketplace.
        """

        self._id = self.marketplace.register_producer()

        while True:
            for prod in self.products:
                count = 0

                while count < prod[1]:
                    if self.marketplace.publish(self._id, prod[0]):
                        count += 1
                        sleep(prod[2])
                    else:
                        sleep(self.republish_wait_time)


from dataclasses import dataclass


@dataclass(init=True, repr=True, order=False, frozen=True)
class Product:
    """A simple data class for a generic product with a name and a price."""
    name: str
    price: int


@dataclass(init=True, repr=True, order=False, frozen=True)
class Tea(Product):
    """A data class representing Tea, inheriting from Product and adding a 'type' attribute."""
    type: str


@dataclass(init=True, repr=True, order=False, frozen=True)
class Coffee(Product):
    """A data class representing Coffee, inheriting from Product and adding acidity and roast level."""
    acidity: str
    roast_level: str