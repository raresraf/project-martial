"""
This module simulates a marketplace with producers and consumers running in separate threads.

It defines three main classes:
- Producer: Simulates a producer that creates products and publishes them to the marketplace.
- Consumer: Simulates a consumer that adds products to a cart and buys them.
- Marketplace: Acts as a central hub where producers publish products and consumers
  purchase them. It handles the inventory and transactions in a thread-safe manner.
"""

from __future__ import annotations

from dataclasses import dataclass
from threading import Thread
from time import sleep
from typing import Dict, List, Tuple

from .marketplace import Marketplace
from .product import Product

class Consumer(Thread):
    """Represents a consumer that buys products from the marketplace."""
    
    @dataclass
    class Operation():
        """Represents an operation in a consumer's cart (add or remove)."""
        type: str
        product: Product
        quantity: int

        @classmethod
        def from_dict(
            cls,
            dict: Dict
        ) -> Operation:
            """Creates an Operation object from a dictionary."""
            return cls(
                type=dict['type'],
                product=dict['product'],
                quantity=dict['quantity']
            )

    def __init__(
        self,
        carts: List[List[Dict]],
        marketplace: Marketplace,
        retry_wait_time: int,
        **kwargs
    ):
        """Initializes a Consumer thread.

        Args:
            carts (List[List[Dict]]): A list of carts, where each cart is a list of operations.
            marketplace (Marketplace): The marketplace instance to interact with.
            retry_wait_time (int): Time to wait before retrying to add a product to the cart.
            **kwargs: Arguments for the Thread base class.
        """
        Thread.__init__(self, **kwargs)
        self.operations = [[self.Operation.from_dict(op) for op in cart] for cart in carts]
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.kwargs = kwargs

    def run(self):
        """The main execution logic for the consumer thread."""
        for cart in self.operations:
            c_id = self.marketplace.new_cart()

            for op in cart:
                if op.type == 'add':
                    # Persistently try to add the product to the cart
                    while op.quantity:
                        if self.marketplace.add_to_cart(c_id, op.product):
                            op.quantity -= 1
                        else:
                            # If the product is not available, wait and retry
                            sleep(self.retry_wait_time)
                elif op.type == 'remove':
                    # Remove the product from the cart
                    while op.quantity:
                        self.marketplace.remove_from_cart(c_id, op.product)
                        op.quantity -= 1

            # Finalize the purchase and print the bought products
            for p in self.marketplace.place_order(c_id):
                print(f'{self.name} bought {p}')



from __future__ import annotations

from collections import defaultdict
from threading import Lock
from typing import Dict, List, NamedTuple, Optional
from uuid import UUID, uuid4

from .product import Product

class Marketplace:
    """A thread-safe marketplace for producers and consumers."""
    
    BrandedProduct = NamedTuple('BrandedProduct', [
        ('producer_id', UUID),
        ('product', Product)
    ])

    def __init__(
        self,
        queue_size_per_producer: int
    ):
        """Initializes the marketplace.

        Args:
            queue_size_per_producer (int): The maximum number of products a single
                                           producer can have for sale at one time.
        """
        self.queue_size_per_producer: int = queue_size_per_producer
        self.producer_lot: Dict[UUID, List[Product]] = defaultdict(list)
        self.consumers: Dict[UUID, List[self.BrandedProduct]] = defaultdict(list)

        # Locks for ensuring thread-safe access to shared resources
        self.p_lock = Lock()
        self.add_to_cart_lock = Lock()

    def register_producer(self) -> UUID:
        """Generates a unique ID for a new producer."""
        return uuid4()

    def publish(
        self,
        producer_id: UUID,
        product: Product
    ) -> bool:
        """Publishes a product from a producer to the marketplace.

        Args:
            producer_id (UUID): The ID of the producer.
            product (Product): The product to be published.

        Returns:
            bool: True if publishing was successful, False if the producer's lot is full.
        """
        with self.p_lock:
            if len(self.producer_lot[producer_id]) == self.queue_size_per_producer:
                return False

            self.producer_lot[producer_id].append(product)
            return True

    def new_cart(self) -> UUID:
        """Creates a new, unique cart ID for a consumer."""
        return uuid4()

    def add_to_cart(
        self,
        cart_id: UUID,
        product: Product
    ) -> bool:
        """Adds a product to a consumer's cart.

        This method is thread-safe. It searches for the product across all
        producers and moves it to the consumer's cart if found.

        Args:
            cart_id (UUID): The ID of the consumer's cart.
            product (Product): The product to add.

        Returns:
            bool: True if the product was added successfully, False otherwise.
        """
        with self.add_to_cart_lock:
            for p_id, products in self.producer_lot.items():
                if product in products:
                    self.consumers[cart_id].append(self.BrandedProduct(p_id, product))
                    products.remove(product)
                    return True
            return False

    def remove_from_cart(
        self,
        cart_id: UUID,
        product: Product
    ):
        """Removes a product from a consumer's cart and returns it to the producer.

        Args:
            cart_id (UUID): The ID of the consumer's cart.
            product (Product): The product to remove.
        """
        cart = self.consumers[cart_id]
        for bp in cart:
            if bp.product == product:
                self.producer_lot[bp.producer_id].append(bp.product)
                cart.remove(bp)
                break

    def place_order(
        self,
        cart_id: UUID
    ) -> List[Product]:
        """Finalizes an order and returns the products in the cart.

        Args:
            cart_id (UUID): The ID of the cart to be ordered.

        Returns:
            List[Product]: A list of products that were in the cart.
        """
        return [bp.product for bp in self.consumers[cart_id]]

from __future__ import annotations

from dataclasses import dataclass
from threading import Thread
from time import sleep
from typing import List, Tuple

from .marketplace import Marketplace
from .product import Product


class Producer(Thread):
    """Represents a producer that creates products and sells them in the marketplace."""
    
    @dataclass
    class ProductionLine():
        """Defines a production line for a specific product."""
        product: Product
        count: int
        time: float

        @classmethod
        def from_tuple(
            cls,
            tup: Tuple[Product, int, float]
        ):
            """Creates a ProductionLine object from a tuple."""
            return cls(
                product=tup[0],
                count=tup[1],
                time=tup[2]
            )

    def __init__(
        self,
        products: List[Tuple[Product, int, float]],
        marketplace: Marketplace,
        republish_wait_time: float,
        **kwargs
    ):
        """Initializes a Producer thread.

        Args:
            products (List[Tuple[Product, int, float]]): A list of products to be produced.
            marketplace (Marketplace): The marketplace instance to interact with.
            republish_wait_time (float): Time to wait before retrying to publish a product.
            **kwargs: Arguments for the Thread base class.
        """
        Thread.__init__(self, **kwargs)
        self.production = [self.ProductionLine.from_tuple(pl) for pl in products]
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        self.kwargs = kwargs

    def run(self):
        """The main execution logic for the producer thread."""
        p_id = self.marketplace.register_producer()

        # Infinite loop to continuously produce and publish products
        while True:
            for prod_line in self.production:
                # Simulate production time
                sleep(prod_line.time)

                # Publish the produced items
                _count = prod_line.count
                while _count:
                    if self.marketplace.publish(p_id, prod_line.product):
                        _count -= 1
                    else:
                        # If the marketplace lot is full, wait and retry
                        sleep(self.republish_wait_time)
