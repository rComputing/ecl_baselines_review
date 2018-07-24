# Overhead OpenCL Baselines for Review

The important part in the code provided is between the measured lines (`diff_ms`).

The same operations are performed in both sides. The only differences are those specific OpenCL API calls/variables.

As an example, this:

```cpp
// ...

auto in_size = samplesPerVectorWidth;
auto in_bytes = in_size * sizeof(cl_float4);
auto in_array = make_shared<vector<cl_float4>>(in_size);
float* in_ptr = reinterpret_cast<float*>(in_array.get()->data());
for (uint i = 0; i < samples; ++i) {
  float f = (float)rand() / (float)RAND_MAX;
  in_ptr[i] = f;
}

auto out_size = samplesPerVectorWidth;
auto out_bytes = out_size * sizeof(cl_float4);
auto out_array = make_shared<vector<cl_float4>>(out_size);
float* out_ptr = reinterpret_cast<float*>(out_array.get()->data());
for (uint i = 0; i < samples; ++i) {
  out_ptr[i] = 0.0f;
}

// ...

cl::Buffer in_buffer(context, buffer_in_flags, in_bytes, NULL);
CL_CHECK_ERROR(cl_err, "in buffer");

cl::Buffer out_buffer(context, buffer_out_flags, out_bytes, NULL);
CL_CHECK_ERROR(cl_err, "out buffer");

CL_CHECK_ERROR(queue.enqueueWriteBuffer(in_buffer, CL_FALSE, 0, in_bytes, in_ptr, NULL));
```

Will be used like:

```cpp
// ...

auto in_size = samplesPerVectorWidth;
auto in_array = make_shared<vector<cl_float4>>(in_size);
float* in_ptr = reinterpret_cast<float*>(in_array.get()->data());
for (uint i = 0; i < samples; ++i) {
  float f = (float)rand() / (float)RAND_MAX;
  in_ptr[i] = f;
}

auto out_size = samplesPerVectorWidth;
auto out_array = make_shared<vector<cl_float4>>(out_size);
float* out_ptr = reinterpret_cast<float*>(out_array.get()->data());
for (uint i = 0; i < samples; ++i) {
  out_ptr[i] = 0.0f;
}

// ...

ecl::Program program;
program.in(in_array);
program.out(out_array);
```
