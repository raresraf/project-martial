#include <iostream>
using namespace std;
int main() {
  int t;
  cin >> t;
  string x;
  int p = 0, m = 0;
  while (t > 0) {
    cin >> x;
    if (x[1] == '+') {
      p++;
    } else {
      m++;
    }
    t--;
  }
  cout << p - m;
  return 0;
}
