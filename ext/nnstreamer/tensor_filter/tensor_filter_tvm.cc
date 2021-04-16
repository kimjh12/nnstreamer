/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * NNStreamer tensor_filter, sub-plugin for TVM
 * Copyright (C) 2021 Junhwan Kim <jejudo.kim@samsung.com>
 */
/**
 * @file    unittest_filter_tvm.cc
 * @date    16 Apr 2021
 * @brief   NNStreamer tensor-filter sub-plugin for Apache TVM
 * @author  Junhwan Kim <jejudo.kim@samsung.com>
 * @see     http://github.com/nnstreamer/nnstreamer
 * @bug     No known bugs
 * 
 * This is the per-NN-framework plugin (TVM) for tensor_filter.
 * 
 * @todo    Only supports tvm.contrib.graph_executor model
 */

#include <glib.h>
#include <nnstreamer_cppplugin_api_filter.hh>
#include <nnstreamer_log.h>
#include <tensor_common.h>

#include <dlpack/dlpack.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>


namespace nnstreamer
{
namespace tensorfilter_tvm
{

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */
void init_filter_tvm (void) __attribute__ ((constructor));
void fini_filter_tvm (void) __attribute__ ((destructor));
#ifdef __cplusplus
}
#endif /* __cplusplus */


/**
 * @brief Tensor data information to execute the model
 */
typedef struct {
  std::vector<int64_t> shape;
  DLDataType dtype;
} tvm_data_info;

/**
 * @brief Class for TVM subplugin.
 */
class tvm_subplugin final : public tensor_filter_subplugin
{
  private:
  bool empty_model;
  char *model_path;
  GstTensorsInfo inputInfo;
  GstTensorsInfo outputInfo;

  DLDevice device;
  tvm::runtime::Module mod_factory;
  tvm::runtime::Module gmod;
  std::vector<tvm_data_info> input_info_list;
  std::vector<tvm_data_info> output_info_list;

  static const char *name;
  static const accl_hw hw_list[];
  static tvm_subplugin *registeredRepresentation;

  bool parse_custom_prop (const char *custom_prop);
  bool convert_nns_type (tensor_type nns_type, DLDataType &tvm_type);
  void cleanup();
  bool configure_meta (GstTensorsInfo &tensor_meta, std::vector<tvm_data_info> &tensor_list);
  bool set_tensor_info (GstTensorsInfo &dest_info, const GstTensorsInfo &src_info, unsigned int num_tensors);

  public:
  static void init_filter_tvm ();
  static void fini_filter_tvm ();

  tvm_subplugin ();
  ~tvm_subplugin ();

  tensor_filter_subplugin &getEmptyInstance ();
  void configure_instance (const GstTensorFilterProperties *prop);
  void invoke (const GstTensorMemory *input, GstTensorMemory *output);
  void getFrameworkInfo (GstTensorFilterFrameworkInfo &info);
  int getModelInfo (model_info_ops ops, GstTensorsInfo &in_info, GstTensorsInfo &out_info);
  int eventHandler (event_ops ops, GstTensorFilterFrameworkEventData &data);
};

const char *tvm_subplugin::name = "tvm";
const accl_hw tvm_subplugin::hw_list[] = { ACCL_CPU, ACCL_GPU };

/**
 * @brief Construct a new tvm subplugin::tvm subplugin object
 */
tvm_subplugin::tvm_subplugin ()
  : tensor_filter_subplugin (), empty_model (true), model_path (nullptr),
    device (DLDevice{kDLCPU, 0}), mod_factory (nullptr), gmod (nullptr)
{
  inputInfo.num_tensors = 0;
  outputInfo.num_tensors = 0;
}

/**
 * @brief Cleanup method for tvm subplugin
 */
void tvm_subplugin::cleanup ()
{
  if (empty_model)
    return;
  if (model_path)
    delete model_path;
  
  input_info_list.clear ();
  output_info_list.clear ();

  model_path = nullptr;
  inputInfo.num_tensors = 0;
  outputInfo.num_tensors = 0;
  empty_model = true;
}

/**
 * @brief Destroy the tvm subplugin::tvm subplugin object
 */
tvm_subplugin::~tvm_subplugin ()
{
  cleanup ();
}

/**
 * @brief Method to get an empty object
 */
tensor_filter_subplugin & tvm_subplugin::getEmptyInstance ()
{
  return *(new tvm_subplugin ());
}

/**
 * @brief Internal method to parse custom properties
 * @param custom_prop Given c_str value of 'custom' property,
 *                    which contains device info
 */
bool tvm_subplugin::parse_custom_prop (const char *custom_prop)
{
  gchar **options;
  bool invalid_option = false;

  if (custom_prop == nullptr) {
    /* no custom properties */
    return true;
  }

  options = g_strsplit (custom_prop, ",", -1);
  
  for (guint op = 0; op < g_strv_length (options); ++op) {
    gchar **option = g_strsplit (options[op], ":", -1);

    if (g_strv_length (option) > 1) {
      g_strstrip (option[0]);
      g_strstrip (option[1]);

      if (g_ascii_strcasecmp (option[0], "device") == 0) {
        if (g_ascii_strcasecmp (option[1], "CPU") == 0) {
          device = DLDevice{kDLCPU, 0};
        } else if (g_ascii_strcasecmp (option[1], "GPU") == 0) {
          device = DLDevice{kDLGPU, 0};
        } else {
          nns_loge ("Unknown device (%s).", option[1]);
          invalid_option = true;
          
        }
      } else {
        nns_logw ("Unknown option (%s).", options[op]);
      }
    }
    if (invalid_option)
      break;
    g_strfreev (option);
  }
  g_strfreev (options);
  return !invalid_option;
}

/**
 * @brief Configure tvm instance
 */
void tvm_subplugin::configure_instance (const GstTensorFilterProperties *prop)
{
  if (!parse_custom_prop (prop->custom_properties)) {
    nns_loge ("Failed to parse custom property.");
    throw std::invalid_argument ("Failed to parse custom property.");
  }
  
  if (!empty_model) {
    if (!prop->model_files[0] || prop->model_files[0][0] == '\0') {
      std::cerr << "Model path is not given." << std::endl;
      throw std::invalid_argument ("Model path is not given.");
    }
    cleanup ();
  }

  /* read model */
  model_path = g_strdup (prop->model_files[0]);
  mod_factory = tvm::runtime::Module::LoadFromFile (model_path, "so");
  gmod = mod_factory.GetFunction ("default") (device);

  if (!set_tensor_info (inputInfo, prop->input_meta, (int) gmod.GetFunction ("get_num_inputs") ()) || 
      !set_tensor_info (outputInfo, prop->output_meta, (int) gmod.GetFunction ("get_num_outputs") ())) {
    nns_loge ("Failed to set tensor info.");
    throw std::invalid_argument ("Failed to set tensor info.");
  }
  if (!configure_meta (inputInfo, input_info_list) ||
      !configure_meta (outputInfo, output_info_list)) {
    nns_loge ("Failed to configure tensor meta.");
    throw std::invalid_argument ("Failed to configure tensor meta.");
  }
  empty_model = false;
}

/**
 * @brief Internal method to set tensor information for model input & output
 */
bool tvm_subplugin::configure_meta (GstTensorsInfo &tensor_meta, std::vector<tvm_data_info> &tensor_info_list)
{
  for (unsigned int i = 0; i < tensor_meta.num_tensors; ++i) {
    tvm_data_info tvm_info;
    auto &info = tensor_meta.info[i];
    
    tvm_info.shape = std::vector<int64_t> (std::begin(info.dimension), std::end(info.dimension));
    if (!convert_nns_type (info.type, tvm_info.dtype)) {
      nns_loge ("Failed to convert input type.");
      return false;
    }
    tensor_info_list.push_back (tvm_info);
  }
  return true;
}

/**
 * @brief Internal method to copy user defined tensor properties
 */
bool tvm_subplugin::set_tensor_info (GstTensorsInfo &dest_info, const GstTensorsInfo &src_info, unsigned int num_tensors)
{
  if (num_tensors > NNS_TENSOR_SIZE_LIMIT)
  {
    nns_loge ("The number of tensors required by the given model exceeds the nnstreamer tensor limit (16 by default).");
    return false;
  }

  dest_info.num_tensors = num_tensors;
  for (unsigned int i = 0; i < num_tensors; i++) {
    gst_tensor_info_copy (std::addressof(dest_info.info[i]), std::addressof(src_info.info[i]));
  }
  return true;
}

/**
 * @brief Invoke tvm instance
 */
void tvm_subplugin::invoke (const GstTensorMemory *input, GstTensorMemory *output) 
{
  assert (!empty_model);
  assert (gmod.defined());
  assert (input != NULL && output != NULL);
  unsigned int i;
  tvm::runtime::NDArray tensor;

  /* read functions */
  tvm::runtime::PackedFunc set_input = gmod.GetFunction("set_input");
  if (set_input == nullptr) 
    throw std::runtime_error ("packed function `set_input` not defined in model");
  tvm::runtime::PackedFunc get_output = gmod.GetFunction("get_output");
  if (set_input == nullptr) 
    throw std::runtime_error ("packed function `get_output` not defined in model");
  tvm::runtime::PackedFunc run = gmod.GetFunction("run");
  if (set_input == nullptr) 
    throw std::runtime_error ("packed function `run` not defined in model");
  
  for (i = 0; i < inputInfo.num_tensors; ++i) {
    tensor = tvm::runtime::NDArray::Empty(input_info_list[i].shape, input_info_list[i].dtype, device);
    tensor.CopyFromBytes(input[i].data, input[i].size);
    set_input (i, tensor);
  }

  run ();

  for (i = 0; i < outputInfo.num_tensors; ++i) {
    tensor = tvm::runtime::NDArray::Empty(output_info_list[i].shape, output_info_list[i].dtype, device);
    get_output (i, tensor);
    tensor.CopyToBytes(output[i].data, output[i].size);
  }
}

/**
 * @brief Internal method to convert tensor type to DLPack data type
 */
bool
tvm_subplugin::convert_nns_type (tensor_type nns_type, DLDataType &tvm_type)
{
  tvm_type.lanes = 1;
  switch (nns_type) {
    case _NNS_FLOAT32:
      tvm_type.code = kDLFloat;
      tvm_type.bits = 32;
      return true;
    case _NNS_FLOAT64:
      tvm_type.code = kDLFloat;
      tvm_type.bits = 64;
      return true;
    case _NNS_INT8:
    case _NNS_UINT8:
      tvm_type.code = kDLInt;
      tvm_type.bits = 8;
      return true;
    case _NNS_INT16:
    case _NNS_UINT16:
      tvm_type.code = kDLInt;
      tvm_type.bits = 16;
      return true;
    case _NNS_INT32:
    case _NNS_UINT32:
      tvm_type.code = kDLFloat;
      tvm_type.bits = 32;
      return true;
    case _NNS_INT64:
    case _NNS_UINT64:
      tvm_type.code = kDLFloat;
      tvm_type.bits = 64;
      return true;
    default:
      nns_loge ("The tensor type %d is not supported.", nns_type);
  }
  return false;
}

/**
 * @brief Get tvm frameworks info
 */
void tvm_subplugin::getFrameworkInfo (GstTensorFilterFrameworkInfo &info) 
{
  info.name = name;
  info.allow_in_place = 0;
  info.allocate_in_invoke = 0;
  info.run_without_model = 0;
  info.verify_model_path = 1;
}

/**
 * @brief Get tvm model information
 */
int tvm_subplugin::getModelInfo (model_info_ops ops, GstTensorsInfo &in_info, GstTensorsInfo &out_info)
{
  if (ops == GET_IN_OUT_INFO) {
    gst_tensors_info_copy (std::addressof (in_info), std::addressof (inputInfo));
    gst_tensors_info_copy (std::addressof (out_info), std::addressof (outputInfo));
    return 0;
  }

  return -ENOENT;
}

/**
 * @brief Method to handle the event
 */
int tvm_subplugin::eventHandler (event_ops ops, GstTensorFilterFrameworkEventData &data)
{
  return -ENOENT;
}

tvm_subplugin *tvm_subplugin::registeredRepresentation = nullptr;

/**
 * @brief Initialize the object for runtime register
 */
void tvm_subplugin::init_filter_tvm (void)
{
  registeredRepresentation
    = tensor_filter_subplugin::register_subplugin<tvm_subplugin> ();
}

/**
 * @brief Destruct the subplugin
 */
void tvm_subplugin::fini_filter_tvm (void)
{
  assert (registeredRepresentation != nullptr);
  tensor_filter_subplugin::unregister_subplugin (registeredRepresentation);
}

/**
 * @brief initializer
 */
void init_filter_tvm ()
{
  tvm_subplugin::init_filter_tvm ();
}

/**
 * @brief finalizer
 */
void fini_filter_tvm ()
{
  tvm_subplugin::fini_filter_tvm ();
}

} /* namespace nnstreamer::tensor_filter_tvm */
} /* namespace nnstreamer */
