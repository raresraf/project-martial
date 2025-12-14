#include <iostream>

using namespace std;

int main() {
  int n;
  int count;
  cin >> n;
  count = 0;

  for (int i = 0; i < n; i++) {
    int a, b, c;
    cin >> a >> b >> c;
    if ((a == 1 && b == 1)) {
      count = count + 1;
    } else if ((a == 1 && c == 1)) {
      count = count + 1;
    } else if ((b == 1 && c == 1)) {
      count = count + 1;
    }
  }
  cout << count << "\n";
  return 0;
}