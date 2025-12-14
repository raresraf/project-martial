#include <iostream>

using std::cin;
using std::cout;
using std::endl;
int main() {
  int n, out, in;
  int space = 0;
  cin >> n;
  int maxSpace = 0;
  for (int i = 0; i != n; ++i) {
    cin >> out >> in;
    space -= out;
    space += in;
    if (maxSpace < space) {
      maxSpace = space;
    }
  }
  cout << maxSpace;
  return 0;
}
