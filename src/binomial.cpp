void
do_binomial_base(int tscheduler,
                 int tdevices,
                 uint check,
                 int samples,
                 int chunksize,
                 bool use_binaries,
                 vector<float>& props)
{
  string source_str;
  try {
    source_str = file_read("support/kernels/binomial.cl");
  } catch (std::ios::failure& e) {
    cout << "io failure: " << e.what() << "\n";
  }

  uint steps = 254;

  samples = (samples / 4) ? (samples / 4) * 4 : 4;

  auto steps1 = steps + 1;
  int samplesPerVectorWidth = samples / 4;
  size_t gws = steps1 * samplesPerVectorWidth;
  size_t out_size = samplesPerVectorWidth;

  int worksize = chunksize;

  auto in_size = samplesPerVectorWidth;
  auto in_array = make_shared<vector<cl_float4>>(in_size);
  float* in_ptr = reinterpret_cast<float*>(in_array.get()->data());
  for (uint i = 0; i < samples; ++i) {
    float f = (float)rand() / (float)RAND_MAX;
    in_ptr[i] = f;
  }

  auto out_array = make_shared<vector<cl_float4>>(out_size);
  float* out_ptr = reinterpret_cast<float*>(out_array.get()->data());
  for (uint i = 0; i < samples; ++i) {
    out_ptr[i] = 0.0f;
  }

  auto lws = steps1;

  auto sel_platform = PLATFORM;
  auto sel_device = DEVICE;

  CUnits cunits;
  set_cunits(cunits, use_binaries, tdevices, "binomial", true, false);

  auto time_init = std::chrono::system_clock::now().time_since_epoch();

  set_cunits(cunits, use_binaries, tdevices, "binomial", false, false);

  vector<char> kernel_bin = move(cunits.kernel_bin);

  sel_platform = cunits.sel_platform;
  sel_device = cunits.sel_device;

  auto in_bytes = in_size * sizeof(cl_float4);
  auto out_bytes = out_size * sizeof(cl_float4);

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

  string kernel_str = "binomial_options";
  cl::Kernel kernel(program, kernel_str.c_str(), &cl_err);

  cl_err = kernel.setArg(0, steps);
  CL_CHECK_ERROR(cl_err, "kernel arg 0");

  cl_err = kernel.setArg(1, in_buffer);
  CL_CHECK_ERROR(cl_err, "kernel arg 1");

  cl_err = kernel.setArg(2, out_buffer);
  CL_CHECK_ERROR(cl_err, "kernel arg 2");

  cl_err = kernel.setArg(3, steps1 * sizeof(cl_float4), NULL);
  CL_CHECK_ERROR(cl_err, "kernel arg 3");

  cl_err = kernel.setArg(4, steps * sizeof(cl_float4), NULL);
  CL_CHECK_ERROR(cl_err, "kernel arg 4");

  auto offset = 0;
  cl_err = queue.enqueueNDRangeKernel(
                                      kernel, cl::NDRange(offset), cl::NDRange(gws), cl::NDRange(lws), NULL, NULL);
  CL_CHECK_ERROR(cl_err, "enqueue kernel");

  cl_err = queue.enqueueReadBuffer(out_buffer, CL_TRUE, 0, out_bytes, out_ptr);
  CL_CHECK_ERROR(cl_err, "read buffer");

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
    auto threshold = 0.01f;
    auto pos = check_binomial(in_ptr, out_ptr, samplesPerVectorWidth, samples, steps, threshold);
    auto ok = pos == -1;

    if (ok) {
      success(diff_ms);
    } else {
      failure(diff_ms);
    }
  } else {
    cout << "Done\n";
  }
}
