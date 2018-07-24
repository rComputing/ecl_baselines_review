void
do_nbody_base(int tscheduler,
              int tdevices,
              uint check,
              uint num_particles,
              int chunksize,
              bool use_binaries,
              vector<float>& props)
{
  auto group_size = GROUP_SIZE;

  cl_float delT = DEL_T;
  cl_float espSqr = ESP_SQR;

  int worksize = chunksize;

  string source_str = file_read("support/kernels/nbody.cl");

  num_particles = (uint)(((size_t)num_particles < group_size) ? group_size : num_particles);
  num_particles = (uint)((num_particles / group_size) * group_size);

  uint num_bodies = num_particles;

  auto pos_in_array = make_shared<vector<cl_float4>>(num_bodies);
  auto vel_in_array = make_shared<vector<cl_float4>>(num_bodies);
  auto pos_out_array = make_shared<vector<cl_float4>>(num_bodies);
  auto vel_out_array = make_shared<vector<cl_float4>>(num_bodies);

  cl_float4* pos_in_ptr = reinterpret_cast<cl_float4*>(pos_in_array.get()->data());
  cl_float4* vel_in_ptr = reinterpret_cast<cl_float4*>(vel_in_array.get()->data());
  cl_float4* pos_out_ptr = reinterpret_cast<cl_float4*>(pos_out_array.get()->data());
  cl_float4* vel_out_ptr = reinterpret_cast<cl_float4*>(vel_out_array.get()->data());

  float* pos_in = reinterpret_cast<float*>(pos_in_ptr);
  float* vel_in = reinterpret_cast<float*>(vel_in_ptr);
  float* pos_out = reinterpret_cast<float*>(pos_out_ptr);
  float* vel_out = reinterpret_cast<float*>(vel_out_ptr);

  srand(0);
  for (uint i = 0; i < num_bodies; ++i) {
    int index = 4 * i;

    // First 3 values are position in x,y and z direction
    for (int j = 0; j < 3; ++j) {
      pos_in[index + j] = random(3, 50);
    }

    // Mass value
    pos_in[index + 3] = random(1, 1000);

    for (int j = 0; j < 4; ++j) {
      // init to 0
      vel_in[index + j] = 0.0f;
    }
  }

  auto lws = group_size;
  auto gws = num_bodies;

  auto sel_platform = PLATFORM;
  auto sel_device = DEVICE;

  CUnits cunits;
  set_cunits(cunits, use_binaries, tdevices, "nbody", true, false);

  auto time_init = std::chrono::system_clock::now().time_since_epoch();

  set_cunits(cunits, use_binaries, tdevices, "nbody", false, false);

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

  size_t buffer_size = num_bodies * sizeof(cl_float4);

  cl::Buffer pos_in_buffer(context, buffer_in_flags, buffer_size, 0, &cl_err);
  CL_CHECK_ERROR(cl_err, "pos in1 buffer ");
  cl::Buffer pos_out_buffer(context, buffer_out_flags, buffer_size, 0, &cl_err);
  CL_CHECK_ERROR(cl_err, "pos out1 buffer ");
  cl::Buffer vel_in_buffer(context, buffer_in_flags, buffer_size, 0, &cl_err);
  CL_CHECK_ERROR(cl_err, "vel in1 buffer ");
  cl::Buffer vel_out_buffer(context, buffer_out_flags, buffer_size, 0, &cl_err);
  CL_CHECK_ERROR(cl_err, "vel out1 buffer ");

  IF_LOGGING(cout << "x\n");
  CL_CHECK_ERROR(
                 queue.enqueueWriteBuffer(pos_in_buffer, CL_FALSE, 0, buffer_size, pos_in_ptr, NULL, NULL));

  CL_CHECK_ERROR(
                 queue.enqueueWriteBuffer(vel_in_buffer, CL_FALSE, 0, buffer_size, vel_in_ptr, NULL, NULL));

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

  string kernel_str = "nbody_sim";
  cl::Kernel kernel(program, kernel_str.c_str(), &cl_err);

  cl_err = kernel.setArg(0, pos_in_buffer);
  CL_CHECK_ERROR(cl_err, "kernel arg 0");

  cl_err = kernel.setArg(1, vel_in_buffer);
  CL_CHECK_ERROR(cl_err, "kernel arg 1");

  cl_err = kernel.setArg(2, num_bodies);
  CL_CHECK_ERROR(cl_err, "kernel arg 2");

  cl_err = kernel.setArg(3, delT);
  CL_CHECK_ERROR(cl_err, "kernel arg 3");

  cl_err = kernel.setArg(4, espSqr);
  CL_CHECK_ERROR(cl_err, "kernel arg 4");

  cl_err = kernel.setArg(5, pos_out_buffer);
  CL_CHECK_ERROR(cl_err, "kernel arg 5");

  cl_err = kernel.setArg(6, vel_out_buffer);
  CL_CHECK_ERROR(cl_err, "kernel arg 6");

  if (ECL_LOGGING) {
    cout << "pos in:\n";
    for (uint i = 0; i < 10; ++i) {
      cout << pos_in_ptr[i] << " ";
    }
    cout << "\n";
    for (uint i = num_bodies - 10; i < num_bodies; ++i) {
      cout << pos_in_ptr[i] << " ";
    }
    cout << "\n";
    cout << "vel in:\n";
    for (uint i = 0; i < 10; ++i) {
      cout << vel_in_ptr[i] << " ";
    }
    cout << "\n";
    for (uint i = num_bodies - 10; i < num_bodies; ++i) {
      cout << vel_in_ptr[i] << " ";
    }
    cout << "\n";
  }

  auto offset = 0;
  queue.enqueueNDRangeKernel(
                             kernel, cl::NDRange(offset), cl::NDRange(gws), cl::NDRange(lws), NULL, NULL);

  CL_CHECK_ERROR(
                 queue.enqueueReadBuffer(pos_out_buffer, CL_TRUE, 0, buffer_size, pos_out_ptr, NULL, NULL));
  CL_CHECK_ERROR(
                 queue.enqueueReadBuffer(vel_out_buffer, CL_TRUE, 0, buffer_size, vel_out_ptr, NULL, NULL));

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

  if (ECL_LOGGING) {
    cout << "pos out:\n";
    for (uint i = 0; i < 10; ++i) {
      cout << pos_out_ptr[i] << " ";
    }
    cout << "\n";
    for (uint i = num_bodies - 10; i < num_bodies; ++i) {
      cout << pos_out_ptr[i] << " ";
    }
    cout << "\n";
    cout << "vel out:\n";
    for (uint i = 0; i < 10; ++i) {
      cout << vel_out_ptr[i] << " ";
    }
    cout << "\n";
    for (uint i = num_bodies - 10; i < num_bodies; ++i) {
      cout << vel_out_ptr[i] << " ";
    }
    cout << "\n";
  }
  if (check) {
    auto threshold = 0.001f;
    auto ok = do_nbody_check(num_bodies, delT, espSqr, pos_in, vel_in, pos_out, vel_out, threshold);

    if (ok) {
      success(diff_ms);
    } else {
      failure(diff_ms);
    }
  } else {
    cout << "Done\n";
  }
}
