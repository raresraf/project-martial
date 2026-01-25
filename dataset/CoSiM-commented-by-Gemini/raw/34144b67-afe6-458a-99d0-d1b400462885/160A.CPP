#include <algorithm>
#include <iostream>
using namespace std;
int main() {
  int manus, capacity, T, n = 0, v;
  cin >> T;
  while (T--) {
    cin >> manus >> capacity;
    v = capacity - manus;
    if (v >= 2) {
      n++;
    }
  }
  cout << n << endl;
  return 0;
}
