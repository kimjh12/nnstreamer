/**
 * @file    unittest_filter_tvm.cc
 * @date    16 Apr 2021
 * @brief   Unit test for TVM tensor filter sub-puglin
 * @author  Junhwan Kim <jejudo.kim@samsung.com>
 * @see     http://github.com/nnstreamer/nnstreamer
 * @bug     No known bugs
 * 
 */
#include <gtest/gtest.h>
#include <glib.h>

#include <nnstreamer_plugin_api_filter.h>

/**
 * @brief Construct a new TEST object
 */
TEST (nnstreamerFilterTvm, checkExistence)
{
  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("tvm");
  EXPECT_NE (sp, (void *)NULL);
}

/**
 * @brief Test getModelInfo method for tvm subplugin
 */
TEST (nnstreamerFilterTvm, getModelInfo)
{
  int ret;
  void *data = NULL;
  gchar *model_file;
  const gchar *root_path = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");

  model_file = g_build_filename (
    root_path, "tests", "test_models", "models", "tvm_add_one", NULL
  );
  const gchar *model_files[] = {
    model_file, NULL,
  };
  GstTensorFilterProperties prop = {
    .fwname = "tvm", .fw_opened = 0, .model_files = model_files, .num_models = 1,
  };

  GstTensorMemory input, output;
  GstTensorsInfo in_info, out_info;
  prop.input_meta.info[0].dimension[0] = 1;
  prop.input_meta.info[0].dimension[1] = 2;
  prop.input_meta.info[0].dimension[2] = 3;
  prop.input_meta.info[0].dimension[3] = 4;
  prop.input_meta.info[0].type = _NNS_FLOAT32;
  prop.output_meta.info[0].dimension[0] = 4;
  prop.output_meta.info[0].dimension[1] = 3;
  prop.output_meta.info[0].dimension[2] = 2;
  prop.output_meta.info[0].dimension[3] = 1;
  prop.output_meta.info[0].type = _NNS_INT8;

  output.size = input.size = sizeof (uint) * 1;

  input.data = g_malloc (input.size);
  output.data = g_malloc (output.size);

  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("tvm");
  EXPECT_NE (sp, (void *)NULL);

  /* before open */
  ret = sp->getModelInfo (NULL, NULL, NULL, GET_IN_OUT_INFO, &in_info, &out_info);
  EXPECT_NE (ret, 0);

  ret = sp->open (&prop, &data);
  EXPECT_EQ (ret, 0);
  EXPECT_NE (data, (void *)NULL);

  /* unsucessful call */
  ret = sp->getModelInfo (NULL, NULL, data, SET_INPUT_INFO, &in_info, &out_info);
  EXPECT_NE (ret, 0);
  ret = sp->getModelInfo (NULL, NULL, NULL, GET_IN_OUT_INFO, &in_info, &out_info);
  EXPECT_NE (ret, 0);

  /* sucessful call */
  ret = sp->getModelInfo (NULL, NULL, data, GET_IN_OUT_INFO, &in_info, &out_info);
  EXPECT_EQ (ret, 0);
  EXPECT_EQ (in_info.num_tensors, 1);
  EXPECT_EQ (in_info.info[0].dimension[0], 1);
  EXPECT_EQ (in_info.info[0].dimension[1], 2);
  EXPECT_EQ (in_info.info[0].dimension[2], 3);
  EXPECT_EQ (in_info.info[0].dimension[3], 4);
  EXPECT_EQ (in_info.info[0].type, _NNS_FLOAT32);
  EXPECT_EQ (out_info.num_tensors, 1);
  EXPECT_EQ (out_info.info[0].dimension[0], 4);
  EXPECT_EQ (out_info.info[0].dimension[1], 3);
  EXPECT_EQ (out_info.info[0].dimension[2], 2);
  EXPECT_EQ (out_info.info[0].dimension[3], 1);
  EXPECT_EQ (out_info.info[0].type, _NNS_INT8);

  sp->close (&prop, &data);
}

/**
 * @brief Test tvm subplugin with successful open
 */
TEST (nnstreamerFilterTvm, openClose)
{
  int ret;
  void *data = NULL;
  gchar *model_file;
  GstTensorMemory input, output;
  const gchar *root_path = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");

  model_file = g_build_filename (
    root_path, "tests", "test_models", "models", "invalid_file_name", NULL
  );
  const gchar *model_files[] = {
    model_file, NULL,
  };
  GstTensorFilterProperties prop = {
    .fwname = "tvm", .fw_opened = 0, .model_files = model_files, .num_models = 1,
  };

  prop.input_meta.info[0].dimension[0] = 1;
  prop.input_meta.info[0].dimension[1] = 1;
  prop.input_meta.info[0].dimension[2] = 1;
  prop.input_meta.info[0].dimension[3] = 1;
  prop.input_meta.info[0].type = _NNS_FLOAT32;
  prop.output_meta.info[0].dimension[0] = 1;
  prop.output_meta.info[0].dimension[1] = 1;
  prop.output_meta.info[0].dimension[2] = 1;
  prop.output_meta.info[0].dimension[3] = 1;
  prop.output_meta.info[0].type = _NNS_FLOAT32;

  output.size = input.size = sizeof (float) * 1;

  input.data = g_malloc (input.size);
  output.data = g_malloc (output.size);

  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("tvm");
  EXPECT_NE (sp, (void *)NULL);

  /* close before open */
  sp->close (&prop, &data);

  sp = nnstreamer_filter_find ("tvm");
  EXPECT_NE (sp, (void *)NULL);

  /* open with wrong model file */
  ret = sp->open (&prop, &data);
  EXPECT_NE (ret, 0);

  model_file = g_build_filename (
    root_path, "tests", "test_models", "models", "tvm_add_one", NULL
  );
  const gchar *model_files2[] = {
    model_file, NULL,
  };
  prop.model_files = model_files2;
  /* successful open */
  ret = sp->open (&prop, &data);
  EXPECT_EQ (ret, 0);
  EXPECT_NE (data, (void *)NULL);

  sp->close (&prop, &data);
  /* double close */
  sp->close (&prop, &data);
}

/**
 * @brief Test tvm subplugin invoke with simple addition model
 */
TEST (nnstreamerFilterTvm, invoke)
{
  int ret;
  void *data = NULL;
  gchar *model_file;
  GstTensorMemory input, output;
  const gchar *root_path = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");

  model_file = g_build_filename (
    root_path, "tests", "test_models", "models", "tvm_add_one", NULL
  );
  const gchar *model_files[] = {
    model_file, NULL,
  };
  GstTensorFilterProperties prop = {
    .fwname = "tvm", .fw_opened = 0, .model_files = model_files, .num_models = 1,
  };

  prop.input_meta.info[0].dimension[0] = 1;
  prop.input_meta.info[0].dimension[1] = 1;
  prop.input_meta.info[0].dimension[2] = 1;
  prop.input_meta.info[0].dimension[3] = 1;
  prop.input_meta.info[0].type = _NNS_FLOAT32;
  prop.output_meta.info[0].dimension[0] = 1;
  prop.output_meta.info[0].dimension[1] = 1;
  prop.output_meta.info[0].dimension[2] = 1;
  prop.output_meta.info[0].dimension[3] = 1;
  prop.output_meta.info[0].type = _NNS_FLOAT32;

  output.size = input.size = sizeof (float) * 1;

  input.data = g_malloc (input.size);
  output.data = g_malloc (output.size);

  const GstTensorFilterFramework *sp = nnstreamer_filter_find ("tvm");
  EXPECT_NE (sp, (void *)NULL);

  /* before open */
  ret  = sp->invoke (NULL, NULL, data, &input, &output);
  EXPECT_NE (ret, 0);

  ret = sp->open (&prop, &data);
  EXPECT_EQ (ret, 0);
  EXPECT_NE (data, (void *)NULL);

  ((float *)input.data)[0] = 10.0;
  
  /* unsucessful invoke */
  ret = sp->invoke (NULL, NULL, NULL, &input, &output);
  EXPECT_NE (ret, 0);
  /* below throws assertion error */
  EXPECT_DEATH (sp->invoke (NULL, NULL, data, NULL, &output), "");
  EXPECT_DEATH (sp->invoke (NULL, NULL, data, &input, NULL), "");

  /* sucesssful invoke */
  ret = sp->invoke (NULL, NULL, data, &input, &output);
  EXPECT_EQ (ret, 0);
  EXPECT_EQ (*((float *)output.data), 11.0);

  ((float *)input.data)[0] = 1.0;
  ret = sp->invoke (NULL, NULL, data, &input, &output);
  EXPECT_EQ (ret, 0);
  EXPECT_EQ (*((float *)output.data), 2.0);

  g_free (input.data);
  g_free (output.data);
  sp->close (&prop, &data);
}

/**
 * @brief Main gtest
 */
int main (int argc, char **argv) 
{
  int result = -1;

  try {
    testing::InitGoogleTest (&argc, argv);
  } catch (...) {
    g_warning ("catch 'testing::internal::<unnamed>::ClassUniqueToAlwaysTrue'");
  }

  try {
    result = RUN_ALL_TESTS ();
  } catch (...) {
    g_warning ("catch `testing::internal::GoogleTestFailureException`");
  }

  return result;
}
