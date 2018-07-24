void
do_ray_base(int tscheduler,
            int tdevices,
            uint check,
            int wsize,
            int chunksize,
            bool use_binaries,
            vector<float>& props,
            string scene_path)
{

  string source_str;
  try {
    source_str = file_read("support/kernels/ray.cl");
  } catch (std::ios::failure& e) {
    cout << "io failure: " << e.what() << "\n";
  }

  srand(0);

  data_t data;
  data_t_init(&data);

  data.width = wsize;
  data.height = wsize;
  auto image_size = wsize * wsize;
  data.total_size = image_size;
  data.scene = scene_path.c_str();

  int depth = data.depth;
  int fast_norm = data.fast_norm;
  int buil_norm = data.buil_norm;
  int nati_sqrt = data.nati_sqrt;
  int buil_dot = data.buil_dot;
  int buil_len = data.buil_len;
  int width = data.width;
  int height = data.height;
  float viewp_w = data.viewp_w;
  float viewp_h = data.viewp_h;
  float camera_x = data.camera_x;
  float camera_y = data.camera_y;
  float camera_z = data.camera_z;

  ray_begin(&data);

  int n_primitives = data.n_primitives;

  auto in_prim_list = make_shared<vector<Primitive>>(n_primitives);
  in_prim_list.get()->assign(data.A, data.A + n_primitives);
  auto in_ptr = reinterpret_cast<Primitive*>(in_prim_list.get()->data());

  auto out_pixels = make_shared<vector<Pixel>>(image_size);
  out_pixels.get()->assign(data.C, data.C + image_size);
  auto out_ptr = reinterpret_cast<Pixel*>(out_pixels.get()->data());

  auto lws = 128;
  auto gws = image_size;

  auto sel_platform = PLATFORM;
  auto sel_device = DEVICE;

  CUnits cunits;
  set_cunits(cunits, use_binaries, tdevices, "ray", true, false);

  auto time_init = std::chrono::system_clock::now().time_since_epoch();

  set_cunits(cunits, use_binaries, tdevices, "ray", false, false);

  vector<char> kernel_bin = move(cunits.kernel_bin);

  sel_platform = cunits.sel_platform;
  sel_device = cunits.sel_device;

  auto in_bytes = n_primitives * sizeof(Primitive);
  auto out_bytes = image_size * sizeof(Pixel);

  vector<cl::Platform> platforms;
  vector<vector<cl::Device>> platfordevices;
  cl::Device device;

  IF_LOGGING(cout << "discoverDevices\n");
  cl::Platform::get(&platforms);
  IF_LOGGING(cout << "platforms: " << platforms.size() << "\n");
  auto i = 0;
  for (auto& platform : platforms) {
    vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
    IF_LOGGING(cout << "platform: " << i++ << " devices: " << devices.size() << "\n");
    platfordevices.push_back(move(devices));
  }

  auto last_platform = platforms.size() - 1;
  if (sel_platform > last_platform) {
    throw runtime_error("invalid platform selected");
  }

  auto last_device = platfordevices[sel_platform].size() - 1;
  if (sel_device > last_device) {
    throw runtime_error("invalid device selected");
  }

  device = move(platfordevices[sel_platform][sel_device]);

  cl_int cl_err = CL_SUCCESS;
  cl::Context context(device);

  cl::CommandQueue queue(context, device, 0, &cl_err);
  CL_CHECK_ERROR(cl_err, "CommandQueue queue");

  IF_LOGGING(cout << "initBuffers\n");

  cl_int buffer_in_flags = CL_MEM_READ_WRITE;
  cl_int buffer_out_flags = CL_MEM_READ_WRITE;

  cl::Buffer in_buffer(context, buffer_in_flags, in_bytes, NULL);
  CL_CHECK_ERROR(cl_err, "in buffer ");
  cl::Buffer out_buffer(context, buffer_out_flags, out_bytes, NULL);
  CL_CHECK_ERROR(cl_err, "out buffer ");

  CL_CHECK_ERROR(queue.enqueueWriteBuffer(in_buffer, CL_FALSE, 0, in_bytes, in_ptr, NULL));

  IF_LOGGING(cout << "initKernel\n");

  cl::Program::Sources sources;
  cl::Program::Binaries binaries;

  cl::Program program;
  if (use_binaries) {
    binaries.push_back({ kernel_bin.data(), kernel_bin.size() });
    vector<cl_int> status = { -1 };
    program = std::move(cl::Program(context, { device }, binaries, &status, &cl_err));
    CL_CHECK_ERROR(cl_err, "building program from binary failed for device ");
  } else {
    sources.push_back({ source_str.c_str(), source_str.length() });
    program = std::move(cl::Program(context, sources));
  }

  string options;
  options.reserve(32);
  options += "-DECL_KERNEL_GLOBAL_WORK_OFFSET_SUPPORTED=" +
    to_string(ECL_KERNEL_GLOBAL_WORK_OFFSET_SUPPORTED);

  cl_err = program.build({ device }, options.c_str());
  if (cl_err != CL_SUCCESS) {
    IF_LOGGING(cout << " Error building: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device)
               << "\n");
    CL_CHECK_ERROR(cl_err);
  }

  string kernel_str = "raytracer_kernel";
  cl::Kernel kernel(program, kernel_str.c_str(), &cl_err);

  cl_err = kernel.setArg(0, out_buffer);
  CL_CHECK_ERROR(cl_err, "kernel arg 0");

  cl_err = kernel.setArg(1, width);
  CL_CHECK_ERROR(cl_err, "kernel arg 1");

  cl_err = kernel.setArg(2, height);
  CL_CHECK_ERROR(cl_err, "kernel arg 2");

  cl_err = kernel.setArg(3, camera_x);
  CL_CHECK_ERROR(cl_err, "kernel arg 3");

  cl_err = kernel.setArg(4, camera_y);
  CL_CHECK_ERROR(cl_err, "kernel arg 4");

  cl_err = kernel.setArg(5, camera_z);
  CL_CHECK_ERROR(cl_err, "kernel arg 5");

  cl_err = kernel.setArg(6, viewp_w);
  CL_CHECK_ERROR(cl_err, "kernel arg 6");

  cl_err = kernel.setArg(7, viewp_h);
  CL_CHECK_ERROR(cl_err, "kernel arg 7");

  cl_err = kernel.setArg(8, in_buffer);
  CL_CHECK_ERROR(cl_err, "kernel arg 8");

  cl_err = kernel.setArg(9, n_primitives);
  CL_CHECK_ERROR(cl_err, "kernel arg 9");

  cl_err = kernel.setArg(10, n_primitives * sizeof(Primitive), NULL);
  CL_CHECK_ERROR(cl_err, "kernel arg 10");

  auto offset = 0;
  queue.enqueueNDRangeKernel(
                             kernel, cl::NDRange(offset), cl::NDRange(gws), cl::NDRange(lws), NULL, NULL);

  queue.enqueueReadBuffer(out_buffer, CL_TRUE, 0, out_bytes, out_ptr);

  auto t2 = std::chrono::system_clock::now().time_since_epoch();
  size_t diff_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - time_init).count();

  cout << "time: " << diff_ms << "\n";

  string m_info_buffer;
  m_info_buffer.reserve(128);
  CL_CHECK_ERROR(platforms[sel_platform].getInfo(CL_PLATFORM_NAME, &m_info_buffer));

  if (m_info_buffer.size() && m_info_buffer[m_info_buffer.size() - 1] == '\0')
    m_info_buffer.erase(m_info_buffer.size() - 1, 1);
  cout << "Selected platform: " << m_info_buffer << "\n";
  CL_CHECK_ERROR(device.getInfo(CL_DEVICE_NAME, &m_info_buffer));
  if (m_info_buffer.size() && m_info_buffer[m_info_buffer.size() - 1] == '\0')
    m_info_buffer.erase(m_info_buffer.size() - 1, 1);
  cout << "Selected device: " << m_info_buffer << "\n";

  cout << "program type: " << (use_binaries ? "binary" : "source") << "\n";
  cout << "kernel: " << kernel_str << "\n";

  if (check) {
    data.C = out_pixels.get()->data();
    data.out_file = "ray_base.bmp";

    auto threshold = 0.01f;
    auto pos = check_ray(&data);
    auto ok = pos == -1;

    if (ok) {
      success(diff_ms);
    } else {
      failure(diff_ms);
    }
    if (check == 2) {
      ray_end(&data);
      cout << "Writing to ray_base.bmp\n";
    }
  } else {
    cout << "Done\n";
  }

  free(data.C);
  free(data.A);
}
