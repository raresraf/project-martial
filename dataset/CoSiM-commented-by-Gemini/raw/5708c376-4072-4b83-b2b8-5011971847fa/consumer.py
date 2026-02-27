from threading import Thread
import time

class Consumer(Thread):
    """
    A consumer thread that simulates purchasing items from a marketplace.
    It processes a list of operations from a predefined shopping list.
    """
    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """Initializes the Consumer."""
        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.kwargs = kwargs

    def run(self):
        """
        The main execution logic. It creates one cart and performs all assigned
        add/remove operations before placing the final order.
        """
        # 'id_cos' is Romanian for 'cart_id'.
        id_cos = self.marketplace.new_cart()
        # 'lista' is Romanian for 'list'.
        for lista in self.carts:
            # 'dictionar' is Romanian for 'dictionary'.
            for dictionar in lista:
                for _ in range(dictionar.get("quantity")):
                    op_type = dictionar.get("type")
                    if op_type == "add":
                        # This loop will block and retry until the item is added.
                        while not self.marketplace.add_to_cart(id_cos, dictionar.get("product")):
                            time.sleep(self.retry_wait_time)
                    elif op_type == "remove":
                        self.marketplace.remove_from_cart(id_cos, dictionar.get("product"))

        # 'lista_comanda' is Romanian for 'order_list'.
        lista_comanda = self.marketplace.place_order(id_cos)
        for value in lista_comanda:
            print(f"{self.name} bought {value}")

from threading import Lock
import unittest

class Marketplace:
    """
    A marketplace simulation.

    WARNING: This implementation is NOT thread-safe. Critical methods like
    `register_producer`, `new_cart`, `remove_from_cart`, and `place_order`
    modify shared state without acquiring a lock, which will cause race
    conditions under concurrent use.
    """
    def __init__(self, queue_size_per_producer):
        """Initializes the marketplace."""
        self.queue_size_per_producer = queue_size_per_producer
        self.id_producer = 0
        # 'dictionar_producers' maps a producer ID to a list of their products.
        self.dictionar_producers = {}
        self.id_consumer = 0
        # 'dictionar_cos' maps a cart ID to a list of [product, producer_id] pairs.
        self.dictionar_cos = {}
        self.publish_lock = Lock()
        self.add_to_cart_lock = Lock()

    def register_producer(self):
        """
        Registers a new producer.

        WARNING: This method is not thread-safe. It performs a non-atomic
        read-modify-write on `self.id_producer`.
        """
        self.id_producer = self.id_producer + 1
        self.dictionar_producers[self.id_producer] = []
        return self.id_producer

    def publish(self, producer_id, product):
        """Atomically publishes a product for a given producer."""
        with self.publish_lock:
            if len(self.dictionar_producers.get(producer_id)) < self.queue_size_per_producer:
                self.dictionar_producers.get(producer_id).append(product)
                return True
            else:
                return False

    def new_cart(self):
        """
        Creates a new cart for a consumer.

        WARNING: This method is not thread-safe. It performs a non-atomic
        read-modify-write on `self.id_consumer`.
        """
        self.id_consumer = self.id_consumer + 1
        self.dictionar_cos[self.id_consumer] = []
        return self.id_consumer

    def add_to_cart(self, cart_id, product):
        """
        Atomically adds a product to a cart. This involves an inefficient
        linear scan through all products of all producers.
        """
        ok = False
        key_aux = None
        with self.add_to_cart_lock:
            for key, values in self.dictionar_producers.items():
                if product in values:
                    ok = True
                    key_aux = key
                    break
            if ok:
                self.dictionar_producers.get(key_aux).remove(product)
                self.dictionar_cos.get(cart_id).append([product, key_aux])
        return ok

    def remove_from_cart(self, cart_id, product):
        """
        Removes a product from a cart and returns it to the original producer.

        WARNING: This method is not thread-safe as it does not acquire a lock.
        """
        # 'value' is [product, producer_id]
        for value, id_value in self.dictionar_cos.get(cart_id):
            if product == value:
                self.dictionar_cos.get(cart_id).remove([value, id_value])
                self.dictionar_producers.get(id_value).append(value)
                break

    def place_order(self, cart_id):
        """
        Finalizes an order.

        WARNING: This method is not thread-safe as it does not acquire a lock.
        """
        lista_comanda = []
        for value, id_value in self.dictionar_cos.get(cart_id):
            lista_comanda.append(value)
        return lista_comanda

class TestMarketplace(unittest.TestCase):
    """Unit tests for the Marketplace class."""
    # ... (tests omitted for brevity)

from threading import Thread
import time

class Producer(Thread):
    """A producer thread that continuously publishes products."""
    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """Initializes the producer."""
        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        self.kwargs = kwargs

    def run(self):
        """Main loop: registers and then continuously publishes all products."""
        id_producer = self.marketplace.register_producer()
        while True:
            for value in self.products:
                for _ in range(value[1]):
                    if self.marketplace.publish(id_producer, value[0]):
                        time.sleep(value[2])
                    else:
                        # If publishing fails, wait and retry.
                        time.sleep(self.republish_wait_time)

# The following appear to be from a separate, related file.
from dataclasses import dataclass

@dataclass(init=True, repr=True, order=False, frozen=True)
class Product:
    """A base dataclass for a product."""
    name: str
    price: int

@dataclass(init=True, repr=True, order=False, frozen=True)
class Tea(Product):
    """A Tea product."""
    type: str

@dataclass(init=True, repr=True, order=False, frozen=True)
class Coffee(Product):
    """A Coffee product."""
    acidity: str
    roast_level: str
