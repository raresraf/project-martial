#include <iostream>
using namespace std;
int main() {
  int n;
  cin >> n;
  int cap = 0, curr = 0;
  int a, b;
  while (n--) {
    cin >> a >> b;
    curr = curr - a + b;
    if (curr > cap)
      cap = curr;
  }
  cout << cap;
  return 0;
}