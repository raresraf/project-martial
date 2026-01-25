// 116A
#include <iostream>
using namespace std;
int main() {
  int n;
  int a, b;
  int cap = 0;
  cin >> n;
  int x = 0;
  for (int i = 0; i < n; i++) {
    cin >> a >> b;
    x = x - a;
    x = x + b;
    if (x > cap)
      cap = x;
  }
  cout << cap;
  return 0;
}
