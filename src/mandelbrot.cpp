void
do_mandelbrot_base(int tscheduler,
                   int tdevices,
                   uint check,
                   int chunksize,
                   bool use_binaries,
                   vector<float>& props,
                   int width,
                   int height,
                   double xpos,
                   double ypos,
                   double xstep,
                   double ystep,
                   uint max_iterations)
{
  string source_str;
  try {
    source_str = file_read("support/kernels/mandelbrot.cl");
  } catch (std::ios::failure& e) {
    cout << "io failure: " << e.what() << "\n";
  }

  // Make sure width is a multiple of 4
  width = (width + 3) & ~(4 - 1);

  IF_LOGGING(cout << width << " h " << height << "\n");

  int size_matrix = width * height;
  auto size = size_matrix;

  auto lws = 256;
  auto gws = size >> 2;

  int worksize = chunksize;

  auto numDevices = 1;
  auto bench = 0;
  auto xsize = 4.0;

  auto larger = true; // the set is larger than the default
  if (larger) {
    xsize = 4 * xsize / 7;
    xpos = -0.65;
    ypos = 0.3;
  }

  auto out_array = make_shared<vector<cl_uchar4>>(size_matrix);
  cl_uchar4* out_ptr = reinterpret_cast<cl_uchar4*>(out_array.get()->data());

  double aspect = (double)width / (double)height;
  xstep = (xsize / (double)width);
  // Adjust for aspect ratio
  double ysize = xsize / aspect;
  ystep = (-(xsize / aspect) / height);
  auto leftx = (xpos - xsize / 2.0);
  auto idx = 0;
  auto topy = (ypos + ysize / 2.0 - ((double)idx * ysize) / (double)numDevices);

  float leftxF = (float)leftx;
  float topyF = (float)topy;
  float xstepF = (float)xstep;
  float ystepF = (float)ystep;

  auto sel_platform = PLATFORM;
  auto sel_device = DEVICE;

  CUnits cunits;
  set_cunits(cunits, use_binaries, tdevices, "mandelbrot", true, false);

  auto time_init = std::chrono::system_clock::now().time_since_epoch();

  set_cunits(cunits, use_binaries, tdevices, "mandelbrot", false, false);

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

  IF_LOGGING(cout << out_array.get()->size() << "\n");

  cl::Buffer out_buffer(
                        context, buffer_out_flags, sizeof(cl_uchar4) * out_array.get()->size(), NULL);
  CL_CHECK_ERROR(cl_err, "out buffer ");

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

  string kernel_str = "mandelbrot_vector_float";
  cl::Kernel kernel(program, kernel_str.c_str(), &cl_err);

  cl_err = kernel.setArg(0, out_buffer);
  CL_CHECK_ERROR(cl_err, "kernel arg 0");

  cl_err = kernel.setArg(1, leftxF);
  CL_CHECK_ERROR(cl_err, "kernel arg 1");

  cl_err = kernel.setArg(2, topyF);
  CL_CHECK_ERROR(cl_err, "kernel arg 2");

  cl_err = kernel.setArg(3, xstepF);
  CL_CHECK_ERROR(cl_err, "kernel arg 3");

  cl_err = kernel.setArg(4, ystepF);
  CL_CHECK_ERROR(cl_err, "kernel arg 4");

  cl_err = kernel.setArg(5, max_iterations);
  CL_CHECK_ERROR(cl_err, "kernel arg 5");

  cl_err = kernel.setArg(6, width);
  CL_CHECK_ERROR(cl_err, "kernel arg 6");

  cl_err = kernel.setArg(7, bench);
  CL_CHECK_ERROR(cl_err, "kernel arg 7");

  auto offset = 0;
  queue.enqueueNDRangeKernel(
                             kernel, cl::NDRange(offset), cl::NDRange(gws), cl::NDRange(lws), NULL, NULL);

  queue.enqueueReadBuffer(
                          out_buffer, CL_TRUE, 0, sizeof(cl_uchar4) * out_array.get()->size(), out_array.get()->data());

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
  auto out = *out_array.get();

  if (ECL_LOGGING) {
    cout << "out:\n";
    for (uint i = 0; i < 10; ++i) {
      cout << out_ptr[i] << " ";
    }
    cout << "\n";
    for (uint i = size_matrix - 10; i < size_matrix; ++i) {
      cout << out_ptr[i] << " ";
    }
    cout << "\n";
  }

  auto image_width = width;
  auto image_height = height;
  if (check) {
    auto threshold = 0.001f;

    auto ok = check_mandelbrot(
                               out_ptr, leftxF, topyF, xstepF, ystepF, max_iterations, width, height, bench, threshold);

    if (ok) {
      success(diff_ms);
    } else {
      failure(diff_ms);
    }
    if (check == 2) {
      transform_image(out.data(), width, height);
      auto img = write_bmp_file(out.data(), width, height, "mandelbrot_base.bmp");
      cout << "writing mandelbrot_base.bmp (" << img << ")\n";
    }
  } else {
    cout << "Done\n";
  }
}
