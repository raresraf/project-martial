#include <iostream>

using std::cin;
using std::cout;
using std::endl;

int main() {
  int n;
  cin >> n;
  int a, b, c;
  int problemSolved = 0;

  for (int i = 0; i != n; ++i) {
    cin >> a >> b >> c;
    if ((a + b + c) >= 2) {
      problemSolved++;
    }
  }
  cout << problemSolved;
  return 0;
}
