
#include <gtest/gtest.h>
#include <glib.h>
#include <gst/gst.h>
#include <unittest_util.h>


static gchar model_name1[] = "mobilenet_v1_1.0_224_quant.tflite";
static gchar model_name2[] = "mobilenet_v2_1.0_224_quant.tflite";
static gchar data_name[] = "orange.png";

TEST (nnstreamerFilterSharedModel, reload)
{
  const gchar *src_root = g_getenv ("NNSTREAMER_SOURCE_ROOT_PATH");
  gchar *root_path = src_root ? g_strdup (src_root) : g_get_current_dir ();

  gchar *model_path1 = g_build_filename (
      root_path, "tests", "test_models", "models", model_name1, NULL);
  gchar *model_path2 = g_build_filename (
      root_path, "tests", "test_models", "models", model_name2, NULL);
  gchar *image_path = g_build_filename (
      root_path, "tests", "test_models", "data", data_name, NULL);
  // gchar *path;

  gchar *pipeline_str = g_strdup_printf (
      "filesrc location=%s ! pngdec ! videoscale ! imagefreeze ! videoconvert ! "
      "video/x-raw,format=RGB,framerate=30/1"" ! tensor_converter ! "
      "tensor_filter name=filter1 framework=tensorflow-lite model=%s is-updatable=TRUE "
      "shared-tensor-filter-key=aa ! appsink "
      "filesrc location=%s ! pngdec ! videoscale ! imagefreeze ! videoconvert ! "
      "video/x-raw,format=RGB,framerate=30/1"" ! tensor_converter ! "
      "tensor_filter name=filter2 framework=tensorflow-lite model=%s is-updatable=TRUE "
      "shared-tensor-filter-key=aa ! appsink",
      image_path, model_path1, image_path, model_path1);
  
  GstElement *pipeline, *filter1, *filter2;

  g_free (root_path);
  pipeline = gst_parse_launch (pipeline_str, NULL);
  g_free (pipeline_str);

  filter1 = gst_bin_get_by_name (GST_BIN (pipeline), "filter1");
  ASSERT_TRUE (filter1 != NULL);
  filter2 = gst_bin_get_by_name (GST_BIN (pipeline), "filter2");
  ASSERT_TRUE (filter2 != NULL);

  g_object_set (filter1, "model", model_path2, NULL);
  // g_object_get (filter2, "model", &path, NULL);
  // EXPECT_STREQ (model_path2, path);
  // g_free (path);

  gst_object_unref (filter1);
  gst_object_unref (filter2);
  gst_object_unref (pipeline);
}


/**
 * @brief Main gtest
 */
int
main (int argc, char **argv)
{
  int result = -1;

  try {
    testing::InitGoogleTest (&argc, argv);
  } catch (...) {
    g_warning ("catch 'testing::internal::<unnamed>::ClassUniqueToAlwaysTrue'");
  }

  gst_init (&argc, &argv);

  try {
    result = RUN_ALL_TESTS ();
  } catch (...) {
    g_warning ("catch `testing::internal::GoogleTestFailureException`");
  }

  return result;
}
