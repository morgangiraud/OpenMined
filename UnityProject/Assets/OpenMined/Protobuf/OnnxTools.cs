using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using OpenMined.Network.Controllers;
using UnityEngine;
using OpenMined.Network.Utils;
using OpenMined.Protobuf.Onnx;

namespace OpenMined.Protobuf.Onnx
{
    public abstract partial class ONNXTools
    {
        public static GraphProto GetSubGraphFromNodeAndMainGraph(NodeProto node, GraphProto mainGraph)
        {
            List<TensorProto> inits = new List<TensorProto>();
            List<ValueInfoProto> infos = new List<ValueInfoProto>();

            List<TensorProto> allInits = new List<TensorProto>(mainGraph.Initializer);
            List<ValueInfoProto> allInfos = new List<ValueInfoProto>(mainGraph.Input);

            foreach (string id in node.Input)
            {
                // If not find it returns a null value
                TensorProto init = allInits.Find(x => x.Name == id);
                // We need to cehck the mainGraph before
                // But assuming the graph is well made
                // a null value here means that it's not a tensor needed to create the operation
                if (init == null)
                {
                    continue;
                }
                inits.Add(init);
                infos.Add(allInfos.Find(x => x.Name == id));
            }
            
            GraphProto g =  new GraphProto
            {
                Node = { node },
                Initializer = { inits },
                Input = { infos },
            };

            return g;
        }

        public static AttributeProto FindAttribute(NodeProto node, string name)
        {
            List<AttributeProto> allAttr = new List<AttributeProto>(node.Attribute);   

            return allAttr.Find(x => x.Name == name);
        }
    }
}