# Serialization documentation

## Libraries
We use ONNX and Protobuf to Serialize our data in C#.
- [Protobuf C# doc](https://developers.google.com/protocol-buffers/docs/reference/csharp-generated)
- [ONNX repo](https://github.com/onnx/onnx)

## Contributing
If you plan to add a serialization capacity to a function, check first that this function has a well defined spec [here](https://github.com/onnx/onnx/blob/master/docs/Operators.md) and implements it accordingly. If not, just implement the new serialization process and mark it with a comment as "experimental".
