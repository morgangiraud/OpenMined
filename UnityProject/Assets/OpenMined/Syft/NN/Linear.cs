using System;
using System.Collections.Generic;
using OpenMined.Network.Controllers;
using OpenMined.Syft.Tensor;
using UnityEngine;
using OpenMined.Network.Servers;
using Newtonsoft.Json.Linq;
using OpenMined.Protobuf.Onnx;
using Google.Protobuf.Collections;

namespace OpenMined.Syft.Layer
{
    
    public class Linear: Layer, LayerDefinition
	{
		private int _input;
		private int _output;

        [SerializeField] string name = "linear";
        [SerializeField] public FloatTensor _weights;
        [SerializeField] FloatTensor _bias;
        private bool _biased;

        public Linear (SyftController _controller, int input, int output, string initializer="Xavier",
            bool biased = false, float[] weights = null, float[] bias = null)
		{
            init(name);

			this.controller = _controller;
			
			_input = input;
			_output = output;

            _biased = biased || bias != null;

            int[] weightShape = { input, output };
            if (weights == null)
            {
                weights = initializer == "Xavier" ? controller.RandomWeights(input * output, input) : controller.RandomWeights(input * output);
            };
            _weights = controller.floatTensorFactory.Create(_shape: weightShape, _data: weights, _autograd: true, _keepgrads: true);

            parameters.Add(_weights.Id);

            if (_biased)
            {
                int[] biasShape = { 1, output };
                _bias = controller.floatTensorFactory.Create(_data: bias, _shape: biasShape, _autograd: true);
                parameters.Add(_bias.Id);
            };

            #pragma warning disable 420
            id = System.Threading.Interlocked.Increment(ref nCreated);
            controller.addModel(this);
        }

        public Linear (SyftController _controller, int input, int output, FloatTensor weights, FloatTensor bias = null, string initializer="Xavier")
        {
            init(this.name);

            this.controller = _controller;

            _input = input;
            _output = output;

            _weights = weights;
            _bias = bias;

            parameters.Add(_weights.Id);

            if (_bias != null)
            {
                parameters.Add(_bias.Id);
            }

            #pragma warning disable 420
            id = System.Threading.Interlocked.Increment(ref nCreated);
            controller.addModel(this);
        }

		// Overloading the constructor to load from an ONNX proto
		public Linear (SyftController _controller, GraphProto graph)
		{
      init(this.name);

			this.controller = _controller;

			long[] longWeightShape = new long[graph.Initializer[0].Dims.Count];
			graph.Initializer[0].Dims.CopyTo(longWeightShape, 0);
			int[] weightShape = Array.ConvertAll(longWeightShape, val => (int) val);
			Array.Reverse(weightShape);
			float[] weightData;
			if (graph.Initializer[0].FloatData.Count == 0)
			{
				byte[] tmpData = graph.Initializer[0].RawData.ToByteArray();
				weightData = new float[tmpData.Length / 4];
				Buffer.BlockCopy(tmpData, 0, weightData, 0, tmpData.Length);
			}
			else
			{
				weightData = new float[graph.Initializer[0].FloatData.Count];
				graph.Initializer[0].FloatData.CopyTo(weightData, 0);
			}

			if(graph.Initializer[1].Dims.Count == 1)
			{
				graph.Initializer[1].Dims.Insert(0, 1);
			}
			long[] longBiasShape = new long[graph.Initializer[1].Dims.Count];
			graph.Initializer[1].Dims.CopyTo(longBiasShape, 0);
			int[] biasShape = Array.ConvertAll(longBiasShape, val => (int) val);
			float[] biasData;
			if (graph.Initializer[0].FloatData.Count == 0)
			{
				byte[] tmpData = graph.Initializer[1].RawData.ToByteArray();
				biasData = new float[tmpData.Length / 4];
				Buffer.BlockCopy(tmpData, 0, biasData, 0, tmpData.Length);
			}
			else
			{
				biasData = new float[graph.Initializer[1].FloatData.Count];
				graph.Initializer[1].FloatData.CopyTo(biasData, 0);
			}
			
			_input = weightShape[0];
			_output = weightShape[1];
			
			_weights = controller.floatTensorFactory.Create(_shape: weightShape, _data: weightData, _autograd: true, _keepgrads: true);
			_bias = controller.floatTensorFactory.Create(_shape:biasShape, _data: biasData, _autograd: true);

			parameters.Add(_weights.Id);
			parameters.Add(_bias.Id);
			
			#pragma warning disable 420
			id = System.Threading.Interlocked.Increment(ref nCreated);
			controller.addModel(this);
		}


    public override FloatTensor Forward(FloatTensor input)
		{	
			FloatTensor output = input.MM(_weights);
            if (_biased)
            {
                output = output.Add(_bias.Expand(output.Shape).Contiguous());
            };
            activation = output.Id;
		
			return output;
		}

        public string GetLayerDefinition()
        {
            return JsonUtility.ToJson(this);
        }

        public override int getParameterCount()
        {
            return _biased ? _weights.Size + _bias.Size : _weights.Size;
        }

	  public override JToken GetConfig()
    {
		  var config = new JObject
			{
			    { "name", "linear" },
				{ "trainable", true },
				{ "dtype", "float32" }, 
				{ "output", _output },
                { "input", _input },
                { "bias", _bias?.GetConfig() },
                { "weights", _weights?.GetConfig() },
				{ "activation", "linear" },
				{ "use_bias", true },
				{
				    "kernel_initializer", new JObject
					{
					    { "class_name", "VarianceScaling" },
						{ 
						    "config", new JObject
						  	{
							    { "scale", 1.0 },
							  	{ "mode", "fan_avg" },
							  	{ "distribution", "uniform" },
							  	{ "seed", null }
						  	}
						}
					}
				},
				{ 
				    "bias_initializer", new JObject
					{
		          	    { "class_name", "Zeros"},
		          		{ "config", new JObject() }
		          	}
		        },
				{ "kernel_regularizer", null },
		        { "bias_regularizer", null },
		        { "activity_regularizer", null },
		        { "kernel_constraint", null },
		        { "bias_constraint", null }
				};

			return config;
		}

		// See https://github.com/onnx/onnx/blob/master/docs/Operators.md#Gemm
		public override GraphProto GetProto (int input_tensor_id, SyftController ctrl)
		{
			FloatTensor input_tensor = ctrl.floatTensorFactory.Get(input_tensor_id);
			this.Forward(input_tensor);

			NodeProto node = new NodeProto
			{
				Input = { input_tensor_id.ToString(), _weights.Id.ToString(), _bias.Id.ToString() },
				Output = { activation.ToString() },
				Name = this.name,
				OpType = "Gemm",
				DocString = ""
			};
			node.Attribute.Add(new AttributeProto{
				Name = "alpha",
				Type = AttributeProto.Types.AttributeType.Float,
				F = 1.0f
			});
			node.Attribute.Add(new AttributeProto{
				Name = "beta",
				Type = AttributeProto.Types.AttributeType.Float,
				F = 1.0f
			});
			node.Attribute.Add(new AttributeProto{
				Name = "broadcast",
				Type = AttributeProto.Types.AttributeType.Int,
				I = 1
			});
			node.Attribute.Add(new AttributeProto{
				Name = "transB",
				Type = AttributeProto.Types.AttributeType.Int,
				I = 1
			});

			TensorProto w_init = _weights.GetProto();
			TensorProto b_init = _bias.GetProto();

			ValueInfoProto input_info = input_tensor.GetValueInfoProto();
			ValueInfoProto w_info = _weights.GetValueInfoProto();
			ValueInfoProto b_info = _bias.GetValueInfoProto();

			GraphProto g =  new GraphProto
			{
				Node = { node },
				Initializer = { w_init, b_init },
				Input = { input_info, w_info, b_info },
				Output = { ctrl.floatTensorFactory.Get(activation).GetValueInfoProto() },
			};

			return g;
		}

	}
}

