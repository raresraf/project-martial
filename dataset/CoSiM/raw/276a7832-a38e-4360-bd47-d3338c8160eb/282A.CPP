#include <iostream>

using namespace std;

int main() {
  int n, c = 0;
  string a;
  std::cin >> n;
  for (int i = 0; i < n; i++) {
    std::cin >> a;
    if (a == "X++" || a == "++X")
      c++;
    if (a == "--X" || a == "X--")
      c--;
  }
  std::cout << c;
  return 0;
}