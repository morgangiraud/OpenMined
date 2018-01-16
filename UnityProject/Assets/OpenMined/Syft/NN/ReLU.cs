using System.Collections.Generic;
using OpenMined.Network.Controllers;
using OpenMined.Syft.Tensor;
using UnityEngine;
using Newtonsoft.Json.Linq;
using OpenMined.Protobuf.Onnx;

namespace OpenMined.Syft.Layer
{
    public class ReLU : Layer, LayerDefinition
    {

        [SerializeField] string name = "relu";

        public ReLU(SyftController controller)
        {
            init(this.name);

#pragma warning disable 420
            id = System.Threading.Interlocked.Increment(ref nCreated);
            controller.addModel(this);
        }

        public override FloatTensor Forward(FloatTensor input)
        {
            FloatTensor output = input.ReLU();
            activation = output.Id;

            return output;
        }

        public override int getParameterCount() { return 0; }

        // Serialization
        public string GetLayerDefinition()
        {
            return JsonUtility.ToJson(this);
        }

        public override JToken GetConfig()
        {
            var config = new JObject
            {
                { "name", name }           
            };
            
            return config;
        }

        // See https://github.com/onnx/onnx/blob/master/docs/Operators.md#Gemm
        public override GraphProto GetProto(int input_tensor_id, SyftController ctrl)
        {
            FloatTensor input_tensor = ctrl.floatTensorFactory.Get(input_tensor_id);
            this.Forward(input_tensor);

            NodeProto node = new NodeProto
            {
                Input = { input_tensor_id.ToString() },
                Output = { activation.ToString() },
                OpType = "relu",
            };

            GraphProto g =  new GraphProto
            {
                Node = { node },
                Initializer = {  },
                Input = {  },
                Output = { ctrl.floatTensorFactory.Get(activation).GetValueInfoProto() },
            };

            return g;
        }
    }
}