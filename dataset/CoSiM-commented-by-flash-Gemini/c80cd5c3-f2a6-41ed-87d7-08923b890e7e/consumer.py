"""
This module implements a multi-threaded producer-consumer marketplace simulation.
It defines three core entities:
- `Consumer`: Represents a buyer that interacts with the marketplace by adding and removing products from carts, and eventually placing orders. Each consumer runs in its own thread.
- `Marketplace`: Acts as the central hub where producers publish products and consumers manage their carts and place orders. It handles product inventory and cart management, including synchronization to ensure thread safety.
- `Producer`: Represents a seller that continuously produces and publishes products to the marketplace. Each producer operates on its own thread, simulating a production line with varying product types and production times.

The simulation models concurrent access to shared resources (product inventory and carts) using threading and locking mechanisms.
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
    Represents a consumer (buyer) in the marketplace simulation.
    Each consumer runs as a separate thread, executing a series of operations
    (add/remove products to/from a cart) and eventually placing an order.
    """
    
    @dataclass
    class Operation():
        """
        Represents a single operation a consumer performs on their cart.
        
        Attributes:
            type (str): The type of operation, either 'add' or 'remove'.
            product (Product): The product involved in the operation.
            quantity (int): The number of times this operation should be performed for the product.
        """
        type: str
        product: Product
        quantity: int

        @classmethod
        def from_dict(
            cls,
            dict: Dict
        ) -> Operation:
            """
            Creates an Operation instance from a dictionary.
            
            Args:
                dict (Dict): A dictionary containing 'type', 'product', and 'quantity'.
            
            Returns:
                Operation: An initialized Operation object.
            """
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
            carts (List[List[Dict]]): A list of shopping carts, where each cart is a list of operation dictionaries.
            marketplace (Marketplace): The marketplace instance the consumer interacts with.
            retry_wait_time (int): Time in seconds to wait before retrying an operation if it fails.
            **kwargs: Arbitrary keyword arguments passed to the Thread constructor.
        """
        Thread.__init__(self, **kwargs)
        # Converts raw dictionary representations of operations into Operation objects.
        self.operations = [[self.Operation.from_dict(op) for op in cart] for cart in carts]
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.kwargs = kwargs

    def run(self):
        """
        The main execution method for the consumer thread.
        Each consumer processes its list of carts:
        1. Creates a new cart in the marketplace.
        2. Executes 'add' and 'remove' operations for each product in the cart.
           - 'add' operations retry if the product is not immediately available.
           - 'remove' operations are executed directly.
        3. Places the order for the completed cart.
        """
        # Iterates through each shopping cart defined for this consumer.
        for cart in self.operations:
            c_id = self.marketplace.new_cart()

            # Processes each operation (add or remove) within the current cart.
            for op in cart:
                # Handles 'add' operations.
                if op.type == 'add':
                    # Continues adding the product until the desired quantity is reached.
                    while op.quantity:
                        # Attempts to add the product to the cart.
                        if self.marketplace.add_to_cart(c_id, op.product):
                            op.quantity -= 1
                        else:
                            # If adding fails (product not available), waits before retrying.
                            sleep(self.retry_wait_time)
                # Handles 'remove' operations.
                elif op.type == 'remove':
                    # Continues removing the product until the desired quantity is reached.
                    while op.quantity:
                        self.marketplace.remove_from_cart(c_id, op.product)
                        op.quantity -= 1

            # Places the order for the current cart and prints the bought products.
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
    Manages products from producers and facilitates consumer shopping.
    It acts as a central hub for product publishing, cart management, and order placement,
    ensuring thread-safe operations through the use of locks.
    """
    
    BrandedProduct = NamedTuple('BrandedProduct', [
        ('producer_id', UUID),
        ('product', Product)
    ])
    """
    A named tuple representing a product along with the ID of the producer who published it.
    This helps in tracking product origin within the marketplace.
    """

    def __init__(
        self,
        queue_size_per_producer: int
    ):

        # Initializes the marketplace with a specified queue size for each producer.
        self.queue_size_per_producer: int = queue_size_per_producer
        # Stores products published by each producer, mapped by producer UUID.
        self.producer_lot: Dict[UUID, List[Product]] = defaultdict(list)
        # Stores products added to consumer carts, mapped by cart UUID.
        self.consumers: Dict[UUID, List[self.BrandedProduct]] = defaultdict(list)

        # Lock to ensure thread-safe access to the producer_lot (for publishing).
        self.p_lock = Lock()
        # Lock to ensure thread-safe access when adding items to a consumer's cart.
        self.add_to_cart_lock = Lock()

    def register_producer(self) -> UUID:
        """
        Registers a new producer with the marketplace and returns a unique producer ID.
        
        Returns:
            UUID: A unique identifier for the registered producer.
        """
        return uuid4()

    def publish(
        self,
        producer_id: UUID,
        product: Product
    ) -> bool:
        """
        Publishes a product to the marketplace under the given producer ID.
        
        Args:
            producer_id (UUID): The ID of the producer publishing the product.
            product (Product): The product to publish.
            
        Returns:
            bool: True if the product was successfully published, False if the producer's queue is full.
        """
        # Ensures exclusive access to the producer_lot for publishing operations.
        with self.p_lock:
            # Checks if the producer's product queue has reached its maximum size.
            if len(self.producer_lot[producer_id]) == self.queue_size_per_producer:
                return False

            self.producer_lot[producer_id].append(product)
            return True

    def new_cart(self) -> UUID:
        """
        Creates a new shopping cart and returns its unique cart ID.
        
        Returns:
            UUID: A unique identifier for the new cart.
        """
        return uuid4()

    def add_to_cart(
        self,
        cart_id: UUID,
        product: Product
    ) -> bool:
        """
        Adds a product to a specific shopping cart.
        
        Args:
            cart_id (UUID): The ID of the cart to add the product to.
            product (Product): The product to add.
            
        Returns:
            bool: True if the product was successfully added, False if the product was not found in any producer's stock.
        """
        # Ensures exclusive access to cart modification operations.
        with self.add_to_cart_lock:
            # Iterates through all producers' product lists to find the requested product.
            for p_id, products in self.producer_lot.items():
                # If the product is found in a producer's stock:
                if product in products:
                    self.consumers[cart_id].append(self.BrandedProduct(p_id, product))
                    products.remove(product) # Removes the product from the producer's stock.
                    return True # Product successfully added to cart.
            return False # Product not found in any producer's stock.

    def remove_from_cart(
        self,
        cart_id: UUID,
        product: Product
    ):
        """
        Removes a product from a specific shopping cart and returns it to the original producer's stock.
        
        Args:
            cart_id (UUID): The ID of the cart to remove the product from.
            product (Product): The product to remove.
        """
        cart = self.consumers[cart_id]
        # Iterates through items in the cart to find the product to remove.
        for bp in cart:
            if bp.product == product:
                # Returns the product to the producer's inventory.
                self.producer_lot[bp.producer_id].append(bp.product)
                cart.remove(bp) # Removes the product from the cart.
                break # Assumes only one instance needs to be removed per call.

    def place_order(
        self,
        cart_id: UUID
    ) -> List[Product]:
        """
        Places an order for all items currently in the specified cart.
        
        Args:
            cart_id (UUID): The ID of the cart for which to place the order.
            
        Returns:
            List[Product]: A list of products that were successfully ordered.
        """
        # Retrieves and returns the list of products from the consumer's cart.
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
    Represents a producer (seller) in the marketplace simulation.
    Each producer runs as a separate thread, continuously producing products
    and publishing them to the marketplace.
    """
    
    @dataclass
    class ProductionLine():
        """
        Represents a single production line item for a producer.
        
        Attributes:
            product (Product): The type of product to produce.
            count (int): The number of units of this product to produce per cycle.
            time (float): The time it takes to produce one batch of 'count' products.
        """
        product: Product
        count: int
        time: float


        @classmethod
        def from_tuple(
            cls,
            tup: Tuple[Product, int, float]
        ):
            """
            Creates a ProductionLine instance from a tuple.
            
            Args:
                tup (Tuple[Product, int, float]): A tuple containing product, count, and time.
                
            Returns:
                ProductionLine: An initialized ProductionLine object.
            """
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
            products (List[Tuple[Product, int, float]]): A list of production line configurations.
            marketplace (Marketplace): The marketplace instance the producer interacts with.
            republish_wait_time (float): Time in seconds to wait before attempting to republish a product if the marketplace queue is full.
            **kwargs: Arbitrary keyword arguments passed to the Thread constructor.
        """
        Thread.__init__(self, **kwargs)
        # Converts raw tuple representations of production lines into ProductionLine objects.
        self.production = [self.ProductionLine.from_tuple(pl) for pl in products]
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        self.kwargs = kwargs

    def run(self):
        """
        The main execution method for the producer thread.
        A producer continuously produces and publishes products to the marketplace.
        It registers itself with the marketplace and then enters an infinite loop
        to cycle through its production lines.
        """
        p_id = self.marketplace.register_producer()

        # Enters an infinite loop to continuously produce and publish products.
        while True:
            # Iterates through each defined production line.
            for prod_line in self.production:
                # Simulates the time taken to produce a batch of products.
                sleep(prod_line.time)

                # Attempts to publish the product until the required count is met.
                _count = prod_line.count
                while _count:
                    # Tries to publish the product to the marketplace.
                    if self.marketplace.publish(p_id, prod_line.product):
                        _count -= 1
                    else:
                        # If publishing fails (marketplace queue full), waits before retrying.
                        sleep(self.republish_wait_time)
