void
do_gaussian_base(int tscheduler,
                 int tdevices,
                 uint check,
                 uint image_width,
                 int chunksize,
                 bool use_binaries,
                 vector<float>& props,
                 uint filter_width)
{
  uint image_height = image_width;

  int worksize = chunksize;

  IF_LOGGING(cout << image_width << "\n");

  Gaussian gaussian(image_width, image_height, filter_width);

  string source_str = file_read("support/kernels/gaussian.cl");

  int size = gaussian._total_size;

  auto a_array = shared_ptr<vector<cl_uchar4>>(&gaussian._a);
  auto b_array = shared_ptr<vector<cl_float>>(&gaussian._b);
  auto c_array = shared_ptr<vector<cl_uchar4>>(&gaussian._c);

  auto sel_platform = PLATFORM;
  auto sel_device = DEVICE;

  CUnits cunits;
  set_cunits(cunits, use_binaries, tdevices, "gaussian", true, false);

  auto time_init = std::chrono::system_clock::now().time_since_epoch();

  set_cunits(cunits, use_binaries, tdevices, "gaussian", false, false);

  vector<char> kernel_bin = move(cunits.kernel_bin);

  sel_platform = cunits.sel_platform;
  sel_device = cunits.sel_device;

  vector<cl::Platform> platforms;
  vector<vector<cl::Device>> platform_devices;
  cl::Device device;

  IF_LOGGING(cout << "discoverDevices\n");
  cl::Platform::get(&platforms);
  IF_LOGGING(cout << "platforms: " << platforms.size() << "\n");
  auto i = 0;
  for (auto& platform : platforms) {
    vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
    IF_LOGGING(cout << "platform: " << i++ << " devices: " << devices.size() << "\n");
    platform_devices.push_back(move(devices));
  }

  auto last_platform = platforms.size() - 1;
  if (sel_platform > last_platform) {
    throw runtime_error("invalid platform selected");
  }

  auto last_device = platform_devices[sel_platform].size() - 1;
  if (sel_device > last_device) {
    throw runtime_error("invalid device selected");
  }

  device = move(platform_devices[sel_platform][sel_device]);

  cl_int cl_err = CL_SUCCESS;
  cl::Context context(device);

  cl::CommandQueue queue(context, device, 0, &cl_err);
  CL_CHECK_ERROR(cl_err, "CommandQueue queue");

  IF_LOGGING(cout << "initBuffers\n");

  cl_int buffer_in_flags = CL_MEM_READ_WRITE;
  cl_int buffer_out_flags = CL_MEM_READ_WRITE;

  IF_LOGGING(cout << a_array.get()->size() << "\n");
  IF_LOGGING(cout << b_array.get()->size() << "\n");
  IF_LOGGING(cout << c_array.get()->size() << "\n");

  cl::Buffer a_buffer(context, buffer_in_flags, sizeof(cl_uchar4) * a_array.get()->size(), NULL);
  CL_CHECK_ERROR(cl_err, "in1 buffer ");
  cl::Buffer b_buffer(context, buffer_in_flags, sizeof(cl_float) * b_array.get()->size(), NULL);
  CL_CHECK_ERROR(cl_err, "in2 buffer ");
  cl::Buffer c_buffer(context, buffer_out_flags, sizeof(cl_uchar4) * c_array.get()->size(), NULL);
  CL_CHECK_ERROR(cl_err, "out buffer ");

  CL_CHECK_ERROR(queue.enqueueWriteBuffer(
                                          a_buffer, CL_FALSE, 0, sizeof(cl_uchar4) * a_array.get()->size(), a_array.get()->data(), NULL));

  CL_CHECK_ERROR(queue.enqueueWriteBuffer(
                                          b_buffer, CL_FALSE, 0, sizeof(cl_float) * b_array.get()->size(), b_array.get()->data(), NULL));

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

  string kernel_str = "gaussian_blur";
  cl::Kernel kernel(program, kernel_str.c_str(), &cl_err);

  cl_err = kernel.setArg(0, c_buffer);
  CL_CHECK_ERROR(cl_err, "kernel arg 0");

  cl_err = kernel.setArg(1, a_buffer);
  CL_CHECK_ERROR(cl_err, "kernel arg 1");

  cl_err = kernel.setArg(2, image_height);
  CL_CHECK_ERROR(cl_err, "kernel arg 2");

  cl_err = kernel.setArg(3, image_width);
  CL_CHECK_ERROR(cl_err, "kernel arg 3");

  cl_err = kernel.setArg(4, b_buffer);
  CL_CHECK_ERROR(cl_err, "kernel arg 1");

  cl_err = kernel.setArg(5, filter_width);
  CL_CHECK_ERROR(cl_err, "kernel arg 3");

  cl::UserEvent end(context, &cl_err);
  CL_CHECK_ERROR(cl_err, "user event end");

  cl::Event evkernel;

  auto lws = 128;

  auto offset = 0;
  auto gws = size;
  queue.enqueueNDRangeKernel(
                             kernel, cl::NDRange(offset), cl::NDRange(gws), cl::NDRange(lws), NULL, NULL);

  cl::Event evread;
  vector<cl::Event> events({ evkernel });

  queue.enqueueReadBuffer(
                          c_buffer, CL_TRUE, 0, sizeof(cl_uchar4) * c_array.get()->size(), c_array.get()->data());

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

  auto in1 = *a_array.get();
  auto in2 = *b_array.get();
  auto out = *c_array.get();

  if (check) {

    auto ok = gaussian.compare_gaussian_blur();

    if (ok) {
      success(diff_ms);
    } else {
      failure(diff_ms);
    }
    if (check == 2) {
      auto img = write_bmp_file(out.data(), image_width, image_height, "gaussian_base.bmp");
      cout << "writing gaussian_base.bmp (" << img << ")\n";
    }
  } else {
    cout << "Done\n";
  }
}
