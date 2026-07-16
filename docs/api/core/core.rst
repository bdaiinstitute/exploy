Core Exporter Functionality
===========================
This section contains detailed API documentation for the core exporter functionality of exploy.



Actor
-----

Abstract interface and utilities for exportable actors.

This module defines the ``ExportableActor`` base class for policy networks that can be
exported to ONNX, along with helpers like ``make_exportable_actor`` and ``add_actor_memory``.

.. automodule:: exploy.exporter.core.actor
   :members:
   :undoc-members:
   :show-inheritance:


Exportable Environment
----------------------

Abstract base class for exportable environments.

This module defines the ``ExportableEnvironment`` base class that provides
the standardized interface required by the exporter to trace observation
computation, action processing, and simulation stepping.

.. automodule:: exploy.exporter.core.exportable_environment
   :members:
   :undoc-members:
   :show-inheritance:


Components
----------

Core building blocks for environment export.

This module defines the component abstractions (``Input``, ``Output``, ``Memory``, ``Group``, ``Connection``)
used to structure and manage data flow during policy export.

.. automodule:: exploy.exporter.core.components
   :members:
   :undoc-members:
   :show-inheritance:


Context Manager
---------------

Manages components for environment export.

The ``ContextManager`` class organizes and manages all inputs, outputs, memory components,
and connections needed during the ONNX export process.

.. automodule:: exploy.exporter.core.context_manager
   :members:
   :undoc-members:
   :show-inheritance:


Evaluator
---------

Validation and testing utilities for exported ONNX models.

This module provides the ``evaluate`` function to compare ONNX model outputs
against the original environment and PyTorch policy for correctness verification.

.. automodule:: exploy.exporter.core.evaluator
   :members:
   :undoc-members:
   :show-inheritance:


Exporter Module
---------------

Core ONNX export functionality for RL policies.

This module provides the main entry point for exporting trained policies to ONNX format,
including the ``export_environment_as_onnx`` function and the ``OnnxEnvironmentExporter`` class.

.. automodule:: exploy.exporter.core.exporter
   :members:
   :undoc-members:
   :show-inheritance:


Session Wrapper
---------------

ONNX Runtime inference session management.

The ``SessionWrapper`` class provides a convenient interface for loading and running
ONNX models with ONNX Runtime, managing input/output handling and session configuration.

.. automodule:: exploy.exporter.core.session_wrapper
   :members:
   :undoc-members:
   :show-inheritance:


Tensor Proxy
------------

Tensor list abstraction for improved ONNX export.

The ``TensorProxy`` class manages lists of tensors and exposes them as a single stacked tensor,
improving the structure of exported computational graphs.

.. automodule:: exploy.exporter.core.tensor_proxy
   :members:
   :undoc-members:
   :show-inheritance:


Derived Tensors
---------------

Materialization of derived quantities as ONNX-traceable leaf tensors.

The ``DerivedTensor`` class bundles a derived quantity's compute function and split dimension.
Materializing it against a data-source proxy stores the computed value as a leaf
tensor so it can become an ONNX graph input; calling it returns the stored leaf during export and
computes the quantity live otherwise.

.. automodule:: exploy.exporter.core.derived_tensors
   :members:
   :undoc-members:
   :show-inheritance:
