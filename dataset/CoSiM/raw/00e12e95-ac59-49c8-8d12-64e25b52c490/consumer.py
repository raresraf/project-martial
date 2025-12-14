

from __future__ import annotations

from dataclasses import dataclass
from threading import Thread
from time import sleep
from typing import Dict, List, Tuple

from .marketplace import Marketplace
from .product import Product

class Consumer(Thread):
    
    @dataclass
    class Operation():
        type: str
        product: Product
        quantity: int

        @classmethod
        def from_dict(
            cls,
            dict: Dict
        ) -> Operation:
            
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
        
        Thread.__init__(self, **kwargs)
        self.operations = [[self.Operation.from_dict(op) for op in cart] for cart in carts]
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.kwargs = kwargs

    def run(self):
        
        for cart in self.operations:
            c_id = self.marketplace.new_cart()

            for op in cart:
                if op.type == 'add':
                    
                    while op.quantity:
                        if self.marketplace.add_to_cart(c_id, op.product):
                            op.quantity -= 1
                        else:
                            sleep(self.retry_wait_time)
                elif op.type == 'remove':
                    
                    while op.quantity:
                        self.marketplace.remove_from_cart(c_id, op.product)
                        op.quantity -= 1

            
            for p in self.marketplace.place_order(c_id):
                print(f'{self.name} bought {p}')



from __future__ import annotations

from collections import defaultdict
from threading import Lock
from typing import Dict, List, NamedTuple, Optional
from uuid import UUID, uuid4

from .product import Product

class Marketplace:
    
    BrandedProduct = NamedTuple('BrandedProduct', [
        ('producer_id', UUID),
        ('product', Product)
    ])

    def __init__(
        self,
        queue_size_per_producer: int
    ):

        
        self.queue_size_per_producer: int = queue_size_per_producer
        self.producer_lot: Dict[UUID, List[Product]] = defaultdict(list)
        self.consumers: Dict[UUID, List[self.BrandedProduct]] = defaultdict(list)

        
        self.p_lock = Lock()
        
        self.add_to_cart_lock = Lock()

    def register_producer(self) -> UUID:
        
        return uuid4()

    def publish(
        self,
        producer_id: UUID,
        product: Product
    ) -> bool:
        
        
        with self.p_lock:
            
            if len(self.producer_lot[producer_id]) == self.queue_size_per_producer:
                return False

            self.producer_lot[producer_id].append(product)

            return True

    def new_cart(self) -> UUID:
        
        return uuid4()

    def add_to_cart(
        self,
        cart_id: UUID,
        product: Product
    ) -> bool:
        
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
        

        return [bp.product for bp in self.consumers[cart_id]]
from __future__ import annotations


from dataclasses import dataclass
from threading import Thread
from time import sleep
from typing import List, Tuple


from .marketplace import Marketplace
from .product import Product


class Producer(Thread):
    
    @dataclass
    class ProductionLine():
        product: Product
        count: int
        time: float


        @classmethod
        def from_tuple(
            cls,
            tup: Tuple[Product, int, float]
        ):
            
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
        
        Thread.__init__(self, **kwargs)
        self.production = [self.ProductionLine.from_tuple(pl) for pl in products]
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        self.kwargs = kwargs

    def run(self):
        p_id = self.marketplace.register_producer()

        
        while True:
            
            for prod_line in self.production:
                
                sleep(prod_line.time)

                
                _count = prod_line.count
                while _count:
                    if self.marketplace.publish(p_id, prod_line.product):
                        _count -= 1
                    else:
                        sleep(self.republish_wait_time)
