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

Interface Info Structs
----------------------

Every public :cpp:class:`~exploy::control::CommandInterface` and
:cpp:class:`~exploy::control::RobotStateInterface` method takes a dedicated info
struct instead of loose parameters. This keeps call sites and overrides stable as
new fields are added over time.

Command Interface
~~~~~~~~~~~~~~~~~

.. doxygenstruct:: exploy::control::Se2VelocityCommandInfo
   :members:

.. doxygenstruct:: exploy::control::Se3PoseCommandInfo
   :members:

.. doxygenstruct:: exploy::control::BooleanSelectorCommandInfo
   :members:

.. doxygenstruct:: exploy::control::FloatValueCommandInfo
   :members:

.. doxygenstruct:: exploy::control::JointPositionCommandInfo
   :members:

State Interface
~~~~~~~~~~~~~~~

.. doxygenstruct:: exploy::control::BasePosWInfo
   :members:

.. doxygenstruct:: exploy::control::BaseQuatWInfo
   :members:

.. doxygenstruct:: exploy::control::BaseLinVelBInfo
   :members:

.. doxygenstruct:: exploy::control::BaseAngVelBInfo
   :members:

.. doxygenstruct:: exploy::control::JointPositionInfo
   :members:

.. doxygenstruct:: exploy::control::JointVelocityInfo
   :members:

.. doxygenstruct:: exploy::control::JointEffortInfo
   :members:

.. doxygenstruct:: exploy::control::JointOutputInfo
   :members:

.. doxygenstruct:: exploy::control::SetJointPositionInfo
   :members:

.. doxygenstruct:: exploy::control::SetJointVelocityInfo
   :members:

.. doxygenstruct:: exploy::control::SetJointEffortInfo
   :members:

.. doxygenstruct:: exploy::control::SetJointPGainInfo
   :members:

.. doxygenstruct:: exploy::control::SetJointDGainInfo
   :members:

.. doxygenstruct:: exploy::control::Se2VelocityInfo
   :members:

.. doxygenstruct:: exploy::control::ImuLinearVelocityImuInfo
   :members:

.. doxygenstruct:: exploy::control::ImuAngularVelocityImuInfo
   :members:

.. doxygenstruct:: exploy::control::ImuOrientationWInfo
   :members:

.. doxygenstruct:: exploy::control::BodyPositionWInfo
   :members:

.. doxygenstruct:: exploy::control::BodyOrientationWInfo
   :members:

.. doxygenstruct:: exploy::control::BodyLinearVelocityBInfo
   :members:

.. doxygenstruct:: exploy::control::BodyAngularVelocityBInfo
   :members:

.. doxygenstruct:: exploy::control::HeightScanInfo
   :members:

.. doxygenstruct:: exploy::control::SphericalImageInfo
   :members:

.. doxygenstruct:: exploy::control::PinholeImageInfo
   :members:

Components
----------

Input, output, and observer components for data flow management.
Inputs and outputs read/write ONNX buffers and the robot interfaces; observers
inspect ONNX buffers read-only (e.g. for debugging or visualization).

.. doxygenstruct:: exploy::control::Input
   :members:
   :undoc-members:

.. doxygenstruct:: exploy::control::Output
   :members:
   :undoc-members:

.. doxygenstruct:: exploy::control::Observer
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

Workers
-------

Execution strategies for the read → inference → write pipeline.
Select a strategy via :cpp:enum:`exploy::control::WorkerMode` when calling
``OnnxRLController::init()``.

.. doxygenenum:: exploy::control::WorkerMode

.. doxygenclass:: exploy::control::Worker
   :members:
   :undoc-members:

.. doxygenclass:: exploy::control::SyncWorker
   :members:
   :undoc-members:

.. doxygenclass:: exploy::control::AsyncWorker
   :members:
   :undoc-members:

Matchers
--------

Component matchers for automatic ONNX I/O mapping. A matcher's ``matches()`` claims a
tensor and produces components via ``createInputs()``, ``createOutputs()``, and
``createObservers()``. Ownership is enforced at the component level: every ONNX input and
output tensor must be served by exactly one input/output component, as declared through
:cpp:func:`exploy::control::Input::tensorNames` /
:cpp:func:`exploy::control::Output::tensorNames`. Observers are read-only, do not declare
tensor ownership, and are exempt from this check, so they may observe tensors already
served by another component.

.. doxygenclass:: exploy::control::Matcher
   :members:
   :undoc-members:

.. doxygenstruct:: exploy::control::Match
   :members:
