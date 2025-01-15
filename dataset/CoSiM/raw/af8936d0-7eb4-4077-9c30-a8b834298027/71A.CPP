#include <iostream>
using namespace std;

int main() {
  int n;
  string c;
  cin >> n;
  for (int i = 0; i < n; i++) {
    cin >> c;
    if (c.length() > 10) {
      cout << c.front() << c.length() - 2 << c.back() << endl;
    } else {
      cout << c << endl;
    }
  }
}