#include <iostream>
#include <string>
using namespace std;

int main() {
  int count = 0, n, a, b, c;

  cin >> n;

  while (n-- > 0) {
    cin >> a >> b >> c;

    if (a == 1 && b == 1 && c == 1) {
      count++;
    } else if (a == 1 && b == 1 && c == 0) {
      count++;
    } else if (a == 0 && b == 1 && c == 1) {
      count++;
    } else if (a == 1 && b == 0 && c == 1) {
      count++;
    }
  }
  cout << count << endl;
  return 0;
}
