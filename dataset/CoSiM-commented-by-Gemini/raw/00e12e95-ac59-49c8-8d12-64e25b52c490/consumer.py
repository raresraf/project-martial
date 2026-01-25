"""
This module implements a producer-consumer simulation using a marketplace model.
It defines `Consumer` and `Producer` threads that interact with a central `Marketplace`
to exchange products. The implementation leverages modern Python features like
`dataclasses` and `UUIDs` for managing products and entities.
"""

from __future__ import annotations

from dataclasses import dataclass
from threading import Thread
from time import sleep
from typing import Dict, List, Tuple

from .marketplace import Marketplace
from .product import Product

class Consumer(Thread):
    """
    Represents a consumer in the marketplace simulation. Each consumer thread
    processes a list of shopping carts, where each cart contains a series of
    operations (add or remove products).
    """
    @dataclass
    class Operation():
        """A data class representing a single operation in a shopping cart."""
        type: str
        product: Product
        quantity: int

        @classmethod
        def from_dict(
            cls,
            dict: Dict
        ) -> Operation:
            """Creates an Operation instance from a dictionary."""
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
        """
        Initializes a Consumer thread.

        Args:
            carts: A list of carts, where each cart is a list of operation dictionaries.
            marketplace: The shared Marketplace instance.
            retry_wait_time: Time in seconds to wait before retrying to add an item.
            **kwargs: Additional keyword arguments for the Thread constructor.
        """
        Thread.__init__(self, **kwargs)
        self.operations = [[self.Operation.from_dict(op) for op in cart] for cart in carts]
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.kwargs = kwargs

    def run(self):
        """
        The main execution logic for the consumer thread. It iterates through its
        assigned carts, performs the operations in each, and places an order.
        """
        for cart in self.operations:
            c_id = self.marketplace.new_cart()

            # Block Logic: Process all operations for a single cart.
            for op in cart:
                if op.type == 'add':
                    
                    # Invariant: The loop continues until the desired quantity of the product is added.
                    while op.quantity:
                        if self.marketplace.add_to_cart(c_id, op.product):
                            op.quantity -= 1
                        else:
                            # If adding fails (e.g., product unavailable), wait and retry.
                            sleep(self.retry_wait_time)
                elif op.type == 'remove':
                    
                    # Invariant: The loop continues until the desired quantity is removed.
                    while op.quantity:
                        self.marketplace.remove_from_cart(c_id, op.product)
                        op.quantity -= 1

            
            # After processing all operations, place the order and print the purchased items.
            for p in self.marketplace.place_order(c_id):
                print(f'{self.name} bought {p}')



from __future__ import annotations

from collections import defaultdict
from threading import Lock
from typing import Dict, List, NamedTuple, Optional
from uuid import UUID, uuid4

from .product import Product

class Marketplace:
    """
    A thread-safe marketplace that facilitates the exchange of products between
    producers and consumers. It uses locks to protect shared data structures.
    """
    BrandedProduct = NamedTuple('BrandedProduct', [
        ('producer_id', UUID),
        ('product', Product)
    ])

    def __init__(
        self,
        queue_size_per_producer: int
    ):
        """
        Initializes the Marketplace.

        Args:
            queue_size_per_producer: The maximum number of products a producer can have listed.
        """

        
        self.queue_size_per_producer: int = queue_size_per_producer
        self.producer_lot: Dict[UUID, List[Product]] = defaultdict(list)
        self.consumers: Dict[UUID, List[self.BrandedProduct]] = defaultdict(list)

        
        # Lock for protecting producer-related shared data.
        self.p_lock = Lock()
        
        # Lock specifically for the 'add_to_cart' operation to ensure atomicity.
        self.add_to_cart_lock = Lock()

    def register_producer(self) -> UUID:
        """Generates a new unique ID for a producer."""
        
        return uuid4()

    def publish(
        self,
        producer_id: UUID,
        product: Product
    ) -> bool:
        """
        Publishes a product from a producer to the marketplace. The operation is
        thread-safe.
        Returns:
            bool: True if publishing was successful, False if the producer's lot was full.
        """
        
        
        with self.p_lock:
            
            # Pre-condition: Check if the producer's inventory has space.
            if len(self.producer_lot[producer_id]) == self.queue_size_per_producer:
                return False

            self.producer_lot[producer_id].append(product)

            return True

    def new_cart(self) -> UUID:
        """Creates a new unique ID for a consumer's shopping cart."""
        
        return uuid4()

    def add_to_cart(
        self,
        cart_id: UUID,
        product: Product
    ) -> bool:
        """
        Atomically finds a product from any producer and adds it to the specified cart.
        """
        
        with self.add_to_cart_lock:
            
            # Searches for the product across all producer inventories.
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
        """
        Removes a product from a cart and returns it to the original producer's inventory.
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
        """
        Returns the list of products currently in the specified cart without removing them.
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
    """
    Represents a producer that generates products and publishes them to the marketplace.
    """
    
    @dataclass
    class ProductionLine():
        """A data class representing a single production line for a product."""
        product: Product
        count: int
        time: float


        @classmethod
        def from_tuple(
            cls,
            tup: Tuple[Product, int, float]
        ):
            """Creates a ProductionLine instance from a tuple."""
            
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
        """
        Initializes a Producer thread.

        Args:
            products: A list of production tasks, each defined by a product, quantity, and production time.
            marketplace: The shared Marketplace instance.
            republish_wait_time: Time in seconds to wait before retrying to publish.
            **kwargs: Additional keyword arguments for the Thread constructor.
        """
        
        Thread.__init__(self, **kwargs)
        self.production = [self.ProductionLine.from_tuple(pl) for pl in products]
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        self.kwargs = kwargs

    def run(self):
        """
        The main execution logic for the producer thread. It registers with the
        marketplace and then enters an infinite loop to produce and publish products.
        """
        p_id = self.marketplace.register_producer()

        
        # Block Logic: Infinitely cycles through its production lines.
        while True:
            
            for prod_line in self.production:
                
                sleep(prod_line.time)

                
                # Invariant: The loop continues until the specified quantity for the production line is published.
                _count = prod_line.count
                while _count:
                    if self.marketplace.publish(p_id, prod_line.product):
                        _count -= 1
                    else:
                        # If publishing fails (e.g., inventory is full), wait and retry.
                        sleep(self.republish_wait_time)