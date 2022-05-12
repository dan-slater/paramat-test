Usage
=====

.. _installation:

Installation
------------

To use ParaMat, first install it using pip:

.. code-block:: console

   pip install paramat

Creating recipes
----------------

To retrieve a list of random ingredients,
you can use the ``lumache.get_random_ingredients()`` function:

.. autofunction:: lumache.get_random_ingredients(kind=None)



The ``kind`` parameter should be either ``"meat"``, ``"fish"``,
or ``"veggies"``. Otherwise, :func:`lumache.get_random_ingredients`
will raise an exception.

.. autoexception:: lumache.InvalidKindError






