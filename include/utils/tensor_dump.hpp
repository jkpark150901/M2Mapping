#pragma once
// 디버그용 텐서 덤프 (헤더-온리). 노트북에서 numpy 로 읽기 쉬운 단순 바이너리.
//
// 포맷 (little-endian):
//   int64 ndim;  int64 shape[ndim];  float32 data[prod(shape)]
// numpy 읽기:
//   ndim=np.fromfile(f,'<i8',1)[0]; shape=np.fromfile(f,'<i8',ndim)
//   data=np.fromfile(f,'<f4',int(np.prod(shape))).reshape(shape)

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <torch/torch.h>

namespace tdump {

inline void save(const std::string &path, const torch::Tensor &t_in) {
  if (!t_in.defined() || t_in.numel() == 0)
    return; // 미정의/빈 텐서는 건너뜀
  auto t = t_in.detach().to(torch::kCPU).to(torch::kFloat32).contiguous();
  std::ofstream f(path, std::ios::binary | std::ios::trunc);
  if (!f) {
    std::cerr << "[tdump] save 실패: " << path << "\n";
    return;
  }
  int64_t ndim = t.dim();
  f.write(reinterpret_cast<const char *>(&ndim), sizeof(ndim));
  for (int64_t d = 0; d < ndim; ++d) {
    int64_t s = t.size(d);
    f.write(reinterpret_cast<const char *>(&s), sizeof(s));
  }
  f.write(reinterpret_cast<const char *>(t.data_ptr<float>()),
          (std::streamsize)(t.numel() * sizeof(float)));
}

} // namespace tdump
