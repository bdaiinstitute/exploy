C++ Controller API
==================

High-performance C++ controller for real-time policy execution with ONNX Runtime.

Overview
--------

The C++ controller provides a real-time inference system for deploying RL policies
exported to ONNX format. It features:

* **ONNX Runtime Integration** - Efficient model execution
* **Extensible Interface Design** - Easy integration with different robot platforms
* **Component-Based Architecture** - Modular input/output handling
* **Automatic Component Matching** - Maps ONNX model I/O to robot interfaces

Core Classes
------------

Controller
~~~~~~~~~~

.. doxygenclass:: exploy::control::OnnxRLController
   :members:
   :undoc-members:

Context Management
~~~~~~~~~~~~~~~~~~

.. doxygenclass:: exploy::control::OnnxContext
   :members:
   :undoc-members:

ONNX Runtime
~~~~~~~~~~~~

.. doxygenclass:: exploy::control::OnnxRuntime
   :members:
   :undoc-members:

Interfaces
----------

These abstract interfaces define the contract for robot integration.

State Interface
~~~~~~~~~~~~~~~

.. doxygenclass:: exploy::control::RobotStateInterface
   :members:
   :undoc-members:

Command Interface
~~~~~~~~~~~~~~~~~

.. doxygenclass:: exploy::control::CommandInterface
   :members:
   :undoc-members:

Data Collection Interface
~~~~~~~~~~~~~~~~~~~~~~~~~

.. doxygenclass:: exploy::control::DataCollectionInterface
   :members:
   :undoc-members:

Components
----------

Input and output components for data flow management.

.. doxygenstruct:: exploy::control::Input
   :members:
   :undoc-members:

.. doxygenstruct:: exploy::control::Output
   :members:
   :undoc-members:

Data Types
----------

Core data structures and type aliases used throughout the controller.

.. doxygentypedef:: exploy::control::Position

.. doxygentypedef:: exploy::control::Quaternion

.. doxygentypedef:: exploy::control::LinearVelocity

.. doxygentypedef:: exploy::control::AngularVelocity

.. doxygentypedef:: exploy::control::SE2Velocity

.. doxygenstruct:: exploy::control::SE3Pose
   :members:

.. doxygenstruct:: exploy::control::HeightScan
   :members:

Configuration
-------------

.. doxygenstruct:: exploy::control::OnnxRuntimeOptions
   :members:

.. doxygenstruct:: exploy::control::SE2VelocityConfig
   :members:

.. doxygenstruct:: exploy::control::SE2VelocityRanges
   :members:

Metadata
--------

Metadata structures for component configuration.

.. doxygenstruct:: exploy::control::metadata::JointMetadata
   :members:

.. doxygenstruct:: exploy::control::metadata::JointOutputMetadata
   :members:

.. doxygenstruct:: exploy::control::metadata::SE2VelocityCommandMetadata
   :members:

.. doxygenstruct:: exploy::control::metadata::HeightScanMetadata
   :members:

.. doxygenstruct:: exploy::control::metadata::SphericalImageMetadata
   :members:

.. doxygenstruct:: exploy::control::metadata::PinholeImageMetadata
   :members:

Matchers
--------

Component matchers for automatic ONNX I/O mapping.

.. doxygenclass:: exploy::control::Matcher
   :members:
   :undoc-members:

.. doxygenstruct:: exploy::control::Match
   :members:
