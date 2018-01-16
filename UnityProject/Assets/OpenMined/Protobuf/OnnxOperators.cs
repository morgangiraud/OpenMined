// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: onnx-operators.proto3
#pragma warning disable 1591, 0612, 3021
#region Designer generated code

using pb = global::Google.Protobuf;
using pbc = global::Google.Protobuf.Collections;
using pbr = global::Google.Protobuf.Reflection;
using scg = global::System.Collections.Generic;
namespace OpenMined.Protobuf.Onnx {

  /// <summary>Holder for reflection information generated from onnx-operators.proto3</summary>
  public static partial class OnnxOperatorsReflection {

    #region Descriptor
    /// <summary>File descriptor for onnx-operators.proto3</summary>
    public static pbr::FileDescriptor Descriptor {
      get { return descriptor; }
    }
    private static pbr::FileDescriptor descriptor;

    static OnnxOperatorsReflection() {
      byte[] descriptorData = global::System.Convert.FromBase64String(
          string.Concat(
            "ChVvbm54LW9wZXJhdG9ycy5wcm90bzMSF09wZW5NaW5lZC5Qcm90b2J1Zi5v",
            "bm54Ggtvbm54LnByb3RvMyLCAQoNT3BlcmF0b3JQcm90bxIPCgdvcF90eXBl",
            "GAEgASgJEhUKDXNpbmNlX3ZlcnNpb24YAiABKAMSRQoGc3RhdHVzGAMgASgO",
            "MjUuT3Blbk1pbmVkLlByb3RvYnVmLm9ubnguT3BlcmF0b3JQcm90by5PcGVy",
            "YXRvclN0YXR1cxISCgpkb2Nfc3RyaW5nGAogASgJIi4KDk9wZXJhdG9yU3Rh",
            "dHVzEhAKDEVYUEVSSU1FTlRBTBAAEgoKBlNUQUJMRRABIuQBChBPcGVyYXRv",
            "clNldFByb3RvEg0KBW1hZ2ljGAEgASgJEhIKCmlyX3ZlcnNpb24YAiABKAUS",
            "HQoVaXJfdmVyc2lvbl9wcmVyZWxlYXNlGAMgASgJEhkKEWlyX2J1aWxkX21l",
            "dGFkYXRhGAcgASgJEg4KBmRvbWFpbhgEIAEoCRIVCg1vcHNldF92ZXJzaW9u",
            "GAUgASgDEhIKCmRvY19zdHJpbmcYBiABKAkSOAoIb3BlcmF0b3IYCCADKAsy",
            "Ji5PcGVuTWluZWQuUHJvdG9idWYub25ueC5PcGVyYXRvclByb3RvYgZwcm90",
            "bzM="));
      descriptor = pbr::FileDescriptor.FromGeneratedCode(descriptorData,
          new pbr::FileDescriptor[] { global::OpenMined.Protobuf.Onnx.OnnxReflection.Descriptor, },
          new pbr::GeneratedClrTypeInfo(null, new pbr::GeneratedClrTypeInfo[] {
            new pbr::GeneratedClrTypeInfo(typeof(global::OpenMined.Protobuf.Onnx.OperatorProto), global::OpenMined.Protobuf.Onnx.OperatorProto.Parser, new[]{ "OpType", "SinceVersion", "Status", "DocString" }, null, new[]{ typeof(global::OpenMined.Protobuf.Onnx.OperatorProto.Types.OperatorStatus) }, null),
            new pbr::GeneratedClrTypeInfo(typeof(global::OpenMined.Protobuf.Onnx.OperatorSetProto), global::OpenMined.Protobuf.Onnx.OperatorSetProto.Parser, new[]{ "Magic", "IrVersion", "IrVersionPrerelease", "IrBuildMetadata", "Domain", "OpsetVersion", "DocString", "Operator" }, null, null, null)
          }));
    }
    #endregion

  }
  #region Messages
  /// <summary>
  /// An OperatorProto represents the immutable specification of the signature
  /// and semantics of an operator.
  ///
  /// Operators are declared as part of an OperatorSet, which also defines the
  /// domain name for the set.
  ///
  /// Operators are uniquely identified by a three part identifier
  ///   (domain, op_type, since_version)
  /// where
  ///   *domain* is the domain of an operator set that
  ///      contains this operator specification.
  ///
  ///   *op_type* is the name of the operator as referenced by a
  ///      NodeProto.op_type
  ///
  ///   *since_version* is the version of the operator set that
  ///      this operator was initially declared in.
  /// </summary>
  public sealed partial class OperatorProto : pb::IMessage<OperatorProto> {
    private static readonly pb::MessageParser<OperatorProto> _parser = new pb::MessageParser<OperatorProto>(() => new OperatorProto());
    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public static pb::MessageParser<OperatorProto> Parser { get { return _parser; } }

    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public static pbr::MessageDescriptor Descriptor {
      get { return global::OpenMined.Protobuf.Onnx.OnnxOperatorsReflection.Descriptor.MessageTypes[0]; }
    }

    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    pbr::MessageDescriptor pb::IMessage.Descriptor {
      get { return Descriptor; }
    }

    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public OperatorProto() {
      OnConstruction();
    }

    partial void OnConstruction();

    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public OperatorProto(OperatorProto other) : this() {
      opType_ = other.opType_;
      sinceVersion_ = other.sinceVersion_;
      status_ = other.status_;
      docString_ = other.docString_;
    }

    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public OperatorProto Clone() {
      return new OperatorProto(this);
    }

    /// <summary>Field number for the "op_type" field.</summary>
    public const int OpTypeFieldNumber = 1;
    private string opType_ = "";
    /// <summary>
    /// The name of the operator within a domain.
    /// This field MUST be present in this version of the IR.
    /// </summary>
    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public string OpType {
      get { return opType_; }
      set {
        opType_ = pb::ProtoPreconditions.CheckNotNull(value, "value");
      }
    }

    /// <summary>Field number for the "since_version" field.</summary>
    public const int SinceVersionFieldNumber = 2;
    private long sinceVersion_;
    /// <summary>
    /// The version of the operator set that first introduced this
    /// operator. This value MUST be the same value as the
    /// opset_version of the operator set that first published this operator.
    /// Subsequent versions of the operator set MUST NOT alter the signature
    /// or semantics of the operator once published as STABLE.
    /// This field MUST be present in this version of the IR.
    /// </summary>
    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public long SinceVersion {
      get { return sinceVersion_; }
      set {
        sinceVersion_ = value;
      }
    }

    /// <summary>Field number for the "status" field.</summary>
    public const int StatusFieldNumber = 3;
    private global::OpenMined.Protobuf.Onnx.OperatorProto.Types.OperatorStatus status_ = 0;
    /// <summary>
    /// This field indicates whether the syntax, semantics, or presence
    /// of this operator is in an experimental or stable stage. Once an
    /// operator is published as STABLE, it's syntax and semantics MUST NOT
    /// change in subsequent versions of the operator set.
    /// When an operator is published as EXPERIMENTAL, the syntax and semantics
    /// of the operator MAY change across operator set versions.
    /// Operators "become" stable by deprecating the experimental version and
    /// introducing a new stable operator with the same op_type.
    /// </summary>
    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public global::OpenMined.Protobuf.Onnx.OperatorProto.Types.OperatorStatus Status {
      get { return status_; }
      set {
        status_ = value;
      }
    }

    /// <summary>Field number for the "doc_string" field.</summary>
    public const int DocStringFieldNumber = 10;
    private string docString_ = "";
    /// <summary>
    /// A human-readable documentation for this operator. Markdown is allowed.
    /// </summary>
    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public string DocString {
      get { return docString_; }
      set {
        docString_ = pb::ProtoPreconditions.CheckNotNull(value, "value");
      }
    }

    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public override bool Equals(object other) {
      return Equals(other as OperatorProto);
    }

    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public bool Equals(OperatorProto other) {
      if (ReferenceEquals(other, null)) {
        return false;
      }
      if (ReferenceEquals(other, this)) {
        return true;
      }
      if (OpType != other.OpType) return false;
      if (SinceVersion != other.SinceVersion) return false;
      if (Status != other.Status) return false;
      if (DocString != other.DocString) return false;
      return true;
    }

    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public override int GetHashCode() {
      int hash = 1;
      if (OpType.Length != 0) hash ^= OpType.GetHashCode();
      if (SinceVersion != 0L) hash ^= SinceVersion.GetHashCode();
      if (Status != 0) hash ^= Status.GetHashCode();
      if (DocString.Length != 0) hash ^= DocString.GetHashCode();
      return hash;
    }

    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public override string ToString() {
      return pb::JsonFormatter.ToDiagnosticString(this);
    }

    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public void WriteTo(pb::CodedOutputStream output) {
      if (OpType.Length != 0) {
        output.WriteRawTag(10);
        output.WriteString(OpType);
      }
      if (SinceVersion != 0L) {
        output.WriteRawTag(16);
        output.WriteInt64(SinceVersion);
      }
      if (Status != 0) {
        output.WriteRawTag(24);
        output.WriteEnum((int) Status);
      }
      if (DocString.Length != 0) {
        output.WriteRawTag(82);
        output.WriteString(DocString);
      }
    }

    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public int CalculateSize() {
      int size = 0;
      if (OpType.Length != 0) {
        size += 1 + pb::CodedOutputStream.ComputeStringSize(OpType);
      }
      if (SinceVersion != 0L) {
        size += 1 + pb::CodedOutputStream.ComputeInt64Size(SinceVersion);
      }
      if (Status != 0) {
        size += 1 + pb::CodedOutputStream.ComputeEnumSize((int) Status);
      }
      if (DocString.Length != 0) {
        size += 1 + pb::CodedOutputStream.ComputeStringSize(DocString);
      }
      return size;
    }

    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public void MergeFrom(OperatorProto other) {
      if (other == null) {
        return;
      }
      if (other.OpType.Length != 0) {
        OpType = other.OpType;
      }
      if (other.SinceVersion != 0L) {
        SinceVersion = other.SinceVersion;
      }
      if (other.Status != 0) {
        Status = other.Status;
      }
      if (other.DocString.Length != 0) {
        DocString = other.DocString;
      }
    }

    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public void MergeFrom(pb::CodedInputStream input) {
      uint tag;
      while ((tag = input.ReadTag()) != 0) {
        switch(tag) {
          default:
            input.SkipLastField();
            break;
          case 10: {
            OpType = input.ReadString();
            break;
          }
          case 16: {
            SinceVersion = input.ReadInt64();
            break;
          }
          case 24: {
            status_ = (global::OpenMined.Protobuf.Onnx.OperatorProto.Types.OperatorStatus) input.ReadEnum();
            break;
          }
          case 82: {
            DocString = input.ReadString();
            break;
          }
        }
      }
    }

    #region Nested types
    /// <summary>Container for nested types declared in the OperatorProto message type.</summary>
    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public static partial class Types {
      public enum OperatorStatus {
        [pbr::OriginalName("EXPERIMENTAL")] Experimental = 0,
        [pbr::OriginalName("STABLE")] Stable = 1,
      }

    }
    #endregion

  }

  /// <summary>
  /// An OperatorSetProto represents an immutable set of immutable operator
  /// specifications.
  ///
  /// The domain of the set (OperatorSetProto.domain) is a reverse-DNS name
  /// that disambiguates operator sets defined by independent entities.
  ///
  /// The version of the set (opset_version) is a monotonically increasing
  /// integer that indicates changes to the membership of the operator set.
  ///
  /// Operator sets are uniquely identified by a two part identifier (domain, opset_version)
  ///
  /// Like ModelProto, OperatorSetProto is intended as a top-level file/wire format,
  /// and thus has the standard format headers in addition to the operator set information.
  /// </summary>
  public sealed partial class OperatorSetProto : pb::IMessage<OperatorSetProto> {
    private static readonly pb::MessageParser<OperatorSetProto> _parser = new pb::MessageParser<OperatorSetProto>(() => new OperatorSetProto());
    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public static pb::MessageParser<OperatorSetProto> Parser { get { return _parser; } }

    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public static pbr::MessageDescriptor Descriptor {
      get { return global::OpenMined.Protobuf.Onnx.OnnxOperatorsReflection.Descriptor.MessageTypes[1]; }
    }

    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    pbr::MessageDescriptor pb::IMessage.Descriptor {
      get { return Descriptor; }
    }

    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public OperatorSetProto() {
      OnConstruction();
    }

    partial void OnConstruction();

    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public OperatorSetProto(OperatorSetProto other) : this() {
      magic_ = other.magic_;
      irVersion_ = other.irVersion_;
      irVersionPrerelease_ = other.irVersionPrerelease_;
      irBuildMetadata_ = other.irBuildMetadata_;
      domain_ = other.domain_;
      opsetVersion_ = other.opsetVersion_;
      docString_ = other.docString_;
      operator_ = other.operator_.Clone();
    }

    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public OperatorSetProto Clone() {
      return new OperatorSetProto(this);
    }

    /// <summary>Field number for the "magic" field.</summary>
    public const int MagicFieldNumber = 1;
    private string magic_ = "";
    /// <summary>
    /// All OperatorSetProtos start with a distingushed byte sequence to disambiguate
    /// protobuf files containing OperatorSets from other content.
    /// This field MUST be "ONNXOPSET"
    /// This field MUST be present in this version of the IR
    /// </summary>
    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public string Magic {
      get { return magic_; }
      set {
        magic_ = pb::ProtoPreconditions.CheckNotNull(value, "value");
      }
    }

    /// <summary>Field number for the "ir_version" field.</summary>
    public const int IrVersionFieldNumber = 2;
    private int irVersion_;
    /// <summary>
    /// All OperatorSetProtos indicate the version of the IR syntax and semantics
    /// they adhere to. It is always IR_VERSION.
    /// This field MUST be present in this version of the IR
    /// </summary>
    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public int IrVersion {
      get { return irVersion_; }
      set {
        irVersion_ = value;
      }
    }

    /// <summary>Field number for the "ir_version_prerelease" field.</summary>
    public const int IrVersionPrereleaseFieldNumber = 3;
    private string irVersionPrerelease_ = "";
    /// <summary>
    /// The prerelease component of the SemVer of the IR.
    /// This field MAY be absent in this version of the IR
    /// </summary>
    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public string IrVersionPrerelease {
      get { return irVersionPrerelease_; }
      set {
        irVersionPrerelease_ = pb::ProtoPreconditions.CheckNotNull(value, "value");
      }
    }

    /// <summary>Field number for the "ir_build_metadata" field.</summary>
    public const int IrBuildMetadataFieldNumber = 7;
    private string irBuildMetadata_ = "";
    /// <summary>
    /// The build metadata component of the SemVer of the IR.
    /// This field MAY be absent in this version of the IR
    /// </summary>
    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public string IrBuildMetadata {
      get { return irBuildMetadata_; }
      set {
        irBuildMetadata_ = pb::ProtoPreconditions.CheckNotNull(value, "value");
      }
    }

    /// <summary>Field number for the "domain" field.</summary>
    public const int DomainFieldNumber = 4;
    private string domain_ = "";
    /// <summary>
    /// Domain name of the operator set, in reverse DNS form (e.g., com.acme.dnnops).
    /// </summary>
    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public string Domain {
      get { return domain_; }
      set {
        domain_ = pb::ProtoPreconditions.CheckNotNull(value, "value");
      }
    }

    /// <summary>Field number for the "opset_version" field.</summary>
    public const int OpsetVersionFieldNumber = 5;
    private long opsetVersion_;
    /// <summary>
    /// The version of the set of operators. This is a simple int value
    /// that is monotonically increasing as new versions of operator set
    /// are published. All operators in this set MUST have version
    /// numbers no greater than opset_version.
    /// </summary>
    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public long OpsetVersion {
      get { return opsetVersion_; }
      set {
        opsetVersion_ = value;
      }
    }

    /// <summary>Field number for the "doc_string" field.</summary>
    public const int DocStringFieldNumber = 6;
    private string docString_ = "";
    /// <summary>
    /// A human-readable documentation for this set of operators. Markdown is allowed.
    /// </summary>
    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public string DocString {
      get { return docString_; }
      set {
        docString_ = pb::ProtoPreconditions.CheckNotNull(value, "value");
      }
    }

    /// <summary>Field number for the "operator" field.</summary>
    public const int OperatorFieldNumber = 8;
    private static readonly pb::FieldCodec<global::OpenMined.Protobuf.Onnx.OperatorProto> _repeated_operator_codec
        = pb::FieldCodec.ForMessage(66, global::OpenMined.Protobuf.Onnx.OperatorProto.Parser);
    private readonly pbc::RepeatedField<global::OpenMined.Protobuf.Onnx.OperatorProto> operator_ = new pbc::RepeatedField<global::OpenMined.Protobuf.Onnx.OperatorProto>();
    /// <summary>
    /// The operators specified by this operator set.
    /// The (name, version) MUST be unique across all OperatorProtos in operator
    /// </summary>
    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public pbc::RepeatedField<global::OpenMined.Protobuf.Onnx.OperatorProto> Operator {
      get { return operator_; }
    }

    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public override bool Equals(object other) {
      return Equals(other as OperatorSetProto);
    }

    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public bool Equals(OperatorSetProto other) {
      if (ReferenceEquals(other, null)) {
        return false;
      }
      if (ReferenceEquals(other, this)) {
        return true;
      }
      if (Magic != other.Magic) return false;
      if (IrVersion != other.IrVersion) return false;
      if (IrVersionPrerelease != other.IrVersionPrerelease) return false;
      if (IrBuildMetadata != other.IrBuildMetadata) return false;
      if (Domain != other.Domain) return false;
      if (OpsetVersion != other.OpsetVersion) return false;
      if (DocString != other.DocString) return false;
      if(!operator_.Equals(other.operator_)) return false;
      return true;
    }

    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public override int GetHashCode() {
      int hash = 1;
      if (Magic.Length != 0) hash ^= Magic.GetHashCode();
      if (IrVersion != 0) hash ^= IrVersion.GetHashCode();
      if (IrVersionPrerelease.Length != 0) hash ^= IrVersionPrerelease.GetHashCode();
      if (IrBuildMetadata.Length != 0) hash ^= IrBuildMetadata.GetHashCode();
      if (Domain.Length != 0) hash ^= Domain.GetHashCode();
      if (OpsetVersion != 0L) hash ^= OpsetVersion.GetHashCode();
      if (DocString.Length != 0) hash ^= DocString.GetHashCode();
      hash ^= operator_.GetHashCode();
      return hash;
    }

    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public override string ToString() {
      return pb::JsonFormatter.ToDiagnosticString(this);
    }

    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public void WriteTo(pb::CodedOutputStream output) {
      if (Magic.Length != 0) {
        output.WriteRawTag(10);
        output.WriteString(Magic);
      }
      if (IrVersion != 0) {
        output.WriteRawTag(16);
        output.WriteInt32(IrVersion);
      }
      if (IrVersionPrerelease.Length != 0) {
        output.WriteRawTag(26);
        output.WriteString(IrVersionPrerelease);
      }
      if (Domain.Length != 0) {
        output.WriteRawTag(34);
        output.WriteString(Domain);
      }
      if (OpsetVersion != 0L) {
        output.WriteRawTag(40);
        output.WriteInt64(OpsetVersion);
      }
      if (DocString.Length != 0) {
        output.WriteRawTag(50);
        output.WriteString(DocString);
      }
      if (IrBuildMetadata.Length != 0) {
        output.WriteRawTag(58);
        output.WriteString(IrBuildMetadata);
      }
      operator_.WriteTo(output, _repeated_operator_codec);
    }

    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public int CalculateSize() {
      int size = 0;
      if (Magic.Length != 0) {
        size += 1 + pb::CodedOutputStream.ComputeStringSize(Magic);
      }
      if (IrVersion != 0) {
        size += 1 + pb::CodedOutputStream.ComputeInt32Size(IrVersion);
      }
      if (IrVersionPrerelease.Length != 0) {
        size += 1 + pb::CodedOutputStream.ComputeStringSize(IrVersionPrerelease);
      }
      if (IrBuildMetadata.Length != 0) {
        size += 1 + pb::CodedOutputStream.ComputeStringSize(IrBuildMetadata);
      }
      if (Domain.Length != 0) {
        size += 1 + pb::CodedOutputStream.ComputeStringSize(Domain);
      }
      if (OpsetVersion != 0L) {
        size += 1 + pb::CodedOutputStream.ComputeInt64Size(OpsetVersion);
      }
      if (DocString.Length != 0) {
        size += 1 + pb::CodedOutputStream.ComputeStringSize(DocString);
      }
      size += operator_.CalculateSize(_repeated_operator_codec);
      return size;
    }

    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public void MergeFrom(OperatorSetProto other) {
      if (other == null) {
        return;
      }
      if (other.Magic.Length != 0) {
        Magic = other.Magic;
      }
      if (other.IrVersion != 0) {
        IrVersion = other.IrVersion;
      }
      if (other.IrVersionPrerelease.Length != 0) {
        IrVersionPrerelease = other.IrVersionPrerelease;
      }
      if (other.IrBuildMetadata.Length != 0) {
        IrBuildMetadata = other.IrBuildMetadata;
      }
      if (other.Domain.Length != 0) {
        Domain = other.Domain;
      }
      if (other.OpsetVersion != 0L) {
        OpsetVersion = other.OpsetVersion;
      }
      if (other.DocString.Length != 0) {
        DocString = other.DocString;
      }
      operator_.Add(other.operator_);
    }

    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public void MergeFrom(pb::CodedInputStream input) {
      uint tag;
      while ((tag = input.ReadTag()) != 0) {
        switch(tag) {
          default:
            input.SkipLastField();
            break;
          case 10: {
            Magic = input.ReadString();
            break;
          }
          case 16: {
            IrVersion = input.ReadInt32();
            break;
          }
          case 26: {
            IrVersionPrerelease = input.ReadString();
            break;
          }
          case 34: {
            Domain = input.ReadString();
            break;
          }
          case 40: {
            OpsetVersion = input.ReadInt64();
            break;
          }
          case 50: {
            DocString = input.ReadString();
            break;
          }
          case 58: {
            IrBuildMetadata = input.ReadString();
            break;
          }
          case 66: {
            operator_.AddEntriesFrom(input, _repeated_operator_codec);
            break;
          }
        }
      }
    }

  }

  #endregion

}

#endregion Designer generated code
