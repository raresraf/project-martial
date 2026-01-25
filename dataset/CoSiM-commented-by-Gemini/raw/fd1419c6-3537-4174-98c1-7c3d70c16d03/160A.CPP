#include <algorithm>
#include <iostream>
#include <vector>

int main() {
  int N;
  std::cin >> N;
  std::vector<int> vec(N);
  int lim = 0;
  for (int i = 0; i < N; ++i) {
    std::cin >> vec[i];
    lim += vec[i];
  }
  lim /= 2;
  std::sort(vec.begin(), vec.end(), std::greater<int>());
  int a = 0, i = 0;
  while (a <= lim)
    a += vec[i++];
  std::cout << i << std::endl;

  return 0;
}
