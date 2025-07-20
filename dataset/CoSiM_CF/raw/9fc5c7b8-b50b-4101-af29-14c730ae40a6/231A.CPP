#include <iostream>

using namespace std;

int main() {
  int test, count = 0;
  cin >> test;

  while (test--) {
    int x, y, z;
    cin >> x >> y >> z;

    auto sum = x + y + z;
    if (sum >= 2)
      count++;
  }

  cout << count << endl;

  return 0;
}
