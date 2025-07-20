#include <iostream>
using namespace std;
int main() {
  int n;
  cin >> n;
  int count = 0;
  for (int i = 0; i < n; i++) {
    int a, b, c;
    cin >> a >> b >> c;
    int c1 = 0;
    if (a == 1) {
      c1++;
    }
    if (b == 1) {
      c1++;
    }
    if (c == 1) {
      c1++;
    }
    if (c1 >= 2) {
      count++;
    }
  }
  cout << count << endl;
}
