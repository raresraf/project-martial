#include <iostream>

using namespace std;

int main() {
  int n = 0, x = 0;
  string op = "";
  cin >> n;
  for (int i = 0; i < n; i++) {
    cin >> op;
    if (op == "X++" || op == "++X") {
      x++;
    } else if (op == "X--" || op == "--X") {
      x--;
    }
  }
  cout << x << endl;
  return 0;
}
