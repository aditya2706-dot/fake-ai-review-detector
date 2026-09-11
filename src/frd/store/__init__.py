"""The commerce side of the shop: accounts, carts and orders.

Kept apart from `frd.api` on purpose. Everything under `frd.api` is a view over
the detector -- stateless, derived from the corpus and the model. This package
is the opposite: it owns mutable state that belongs to a person, and it is the
only thing in the repo that writes to a database. The separation is what lets
the detector keep its "no hidden state" property while the shop has customers.
"""
