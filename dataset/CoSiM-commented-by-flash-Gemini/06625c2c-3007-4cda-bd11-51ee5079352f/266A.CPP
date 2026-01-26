#include <iostream>
#include <string>

using namespace std;
int main() {
  int n;
  std::cin >> n;
  std::string s;
  std::cin >> s;
  int count = 0;
  for (int i = 0; i < n - 1; i++) {
    if (s[i] == s[i + 1]) {
      count += 1;
    }
  }
  std::cout << count;
  return 0;
}