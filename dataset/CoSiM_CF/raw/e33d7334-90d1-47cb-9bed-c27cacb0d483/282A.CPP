#include <iostream>
#include <string>

using namespace std;

int main() {
  int n, x = 0;
  cin >> n;
  while (n--) {
    string op;
    cin >> op;
    if (op.front() == '+' || op.back() == '+') {
      x++;
    } else {
      x--;
    }
  }
  cout << x << endl;
  return 0;
}