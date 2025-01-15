#include <iostream>
#include <vector>

using namespace std;

int main() {
  int n;
  std::cin >> n;
  std::string sv, s;
  vector<string> v;

  for (int i = 0; i < n; i++) {
    std::cin >> sv;
    v.push_back(sv);
  }
  for (int j = 0; j < v.size(); j++) {
    if (v[j].length() > 10) {
      int in = v[j].length() - 2;
      s = v[j][0] + std::to_string(in) + v[j][v[j].length() - 1];
      v[j] = s;
    }
  }
  for (int l = 0; l < v.size(); l++) {
    std::cout << v[l] << std::endl;
  }
  return 0;
}